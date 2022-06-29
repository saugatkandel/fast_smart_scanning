#
# Authors 06/29/22:
# Saugat Kandel: for the SLADS parts
# Tao Zhou: for the EPICS and the Tkinter parts
# Mathew Cherukara: General consulting
#
# Uses tkinter and uses epics triggers generated in ives (Sector26 computer) to run the measurement.
#
import argparse
import configparser
import logging
import sys
import time
import tkinter as tk
from pathlib import Path

import epics
import numpy as np
from paramiko import SFTPClient

import helper
from sladsnet.code.base import ExperimentalSample
from sladsnet.code.sampling import check_stopping_criteria
from sladsnet.utils import logger
from sladsnet.utils.readMDA import readMDA
from sladsnet.utils.sftp import setup_sftp, sftp_put, sftp_get

REMOTE_PATH = Path('/home/sector26/2022R2/20220621/Analysis/')
REMOTE_IP = 'ives.cnm.aps.anl.gov'
REMOTE_USERNAME = 'user26id'

logger.setup_logger()


class MainWindow:
    """Uses tkinter to spin up a monitor that responds to epics triggers."""

    def __init__(self, master, sample: ExperimentalSample,
                 scan_sizex: int, scan_sizey: int,
                 xpos: int, ypos: int, xfactor: int,
                 scan_stepsize: int,
                 current_file_suffix: int,
                 mda_file_offset: int,
                 sftp: SFTPClient,
                 store_file_scan_points_num: int,
                 points_per_route: int = 50,
                 debug=False):
        self.sftp = sftp
        self.sample = sample
        self.store_file_scan_points_num = store_file_scan_points_num
        self.completed_run_flag = False

        monitor1 = epics.PV("26idbPBS:sft01:ph01:ao13.VAL", callback=self.monitor_function)
        monitor2 = epics.PV("26idbSOFT:scan1.WCNT", callback=self.monitor_function)

        self.scan_file_offset = int(sys.argv[1])
        self.is_monitor_initialized = False
        self.debug = debug

        self.monitor_start_time_prev = None
        self.route_full = np.empty((0, 2), dtype='int')
        self.new_idxs_to_write = np.empty((0, 2), dtype='int')

        self.current_file_suffix = current_file_suffix
        self.scan_sizex, self.scan_sizey = scan_sizex, scan_sizey
        self.scan_centerx, self.scan_centery = scan_sizex // 2, scan_sizey // 2
        self.xpos, self.ypos = xpos, ypos
        self.xfactor = xfactor

        self.points_per_route = points_per_route
        self.scan_stepsize = scan_stepsize

        self.scan_file_offset = mda_file_offset - self.current_file_suffix
        self.current_file_position = 0

        self.checkpoint_skipped = 0

    def sftp_put(self, lpath, rpath):
        sftp_put(self.sftp, lpath, rpath)

    def sftp_get(self, rpath, lpath):
        sftp_get(self.sftp, rpath, lpath)

    def monitor_function(self, value=None, **kw):
        if value == 1:
            if not self.is_monitor_initialized:
                self.is_monitor_initialized = True
                self.monitor_start_time_prev = time.time()
                if self.debug:
                    logging.info("Monitor has been triggered...")
                return

            t00 = time.time()
            time_diff = t00 - self.monitor_start_time_prev

            if self.debug:
                logging.info('Delay between monitor updates is %.3f s.' % time_diff)
            if time_diff < 20:
                # requests.post(webhook, json={"text":"{0}: AGX : Delay between trigger times was less than 20s. Ignoring the trigger.".format(datetime.now())})
                if self.debug:
                    logging.info("WARNING: Delay between trigger times was less than 20s. Ignoring the trigger.")
                return

            self.monitor_start_time_prev = t00

            t0 = time.time()

            n1 = self.scan_file_offset
            n2 = self.current_file_suffix

            mda_file_name = '/home/sector26/2022R2/20220621/mda/26idbSOFT_%04d.mda' % (n1 + n2)
            logging.info("MDA file name is " + mda_file_name)

            self.sftp_get(mda_file_name, 'mda_current.mda')

            t1 = time.time()
            logging.info('AGX received data from detector (time elapsed) %.3f s.' % (t1 - t0))

            mda = readMDA('mda_current.mda', verbose=False)
            data = np.array(mda[1].d[3].data)
            xx = np.round((np.array(mda[1].d[32].data) - self.xpos) / self.scan_stepsize, 0) + self.scan_centerx
            yy = np.round((np.array(mda[1].d[31].data) - self.ypos) / self.scan_stepsize, 0) + self.scan_centery
            curr_pt = mda[1].curr_pt
            points_of_interest = curr_pt % self.points_per_route
            if points_of_interest == 0:
                points_of_interest = self.points_per_route

            expected_shape = (self.current_file_position + 1) * self.points_per_route
            self.checkpoint_skipped = 0

            if curr_pt != expected_shape:
                if self.debug:
                    logging.info("Warning: At iteration", self.current_file_position,
                             "data shape is", mda[1].curr_pt, ", but the expected shape is", expected_shape)

            # This is for the case when we receive a faulty epics trigger without ives actually having
            # the new position indices.
            if curr_pt > expected_shape + 30:
                self.current_file_position += 1
                self.checkpoint_skipped = 1
                points_of_interest += self.points_per_route

            xpoints = xx[curr_pt - points_of_interest:curr_pt - 1]
            ypoints = yy[curr_pt - points_of_interest:curr_pt - 1]
            route_idxs = np.array((ypoints, xpoints), dtype='int').T

            new_intensities = data[curr_pt - points_of_interest + 1:curr_pt]

            if np.shape(route_idxs)[0] != np.shape(new_intensities)[0]:
                logging.info('Mismatch between shapes of route %d and ' % xpoints.shape[0] + 'and intensities %d.' % (
                    np.shape(new_intensities)[0]))

            if curr_pt == self.store_file_scan_points_num:
                if self.completed_run_flag:
                    # requests.post(webhook, json={"text":"{0}: AGX : Done!".format(datetime.now())})
                    sys.exit()
                self.current_file_suffix += 1
                self.new_idxs_to_write = np.empty((0, 2), dtype='int')
                self.current_file_position = 0
            else:
                if self.debug:
                    logging.info('Incrementing current file position')
                self.current_file_position += 1

            if self.debug:
                logging.info('Shape of full route is', self.route_full.shape)
            self.update(route_idxs, new_intensities)

    def update(self, route_idxs, new_intensities):

        self.sample.measurement_interface.finalize_external_measurement(new_intensities)
        self.sample.perform_measurements(route_idxs)

        t0 = time.time()
        self.sample.reconstruct_and_compute_erd()

        total_erd = self.sample.ERD.sum()
        logging.info('Total ERD is %4.3f.' % total_erd)

        self.completed_run_flag = check_stopping_criteria(self.sample, 0)

        new_idxs_to_write = np.array(self.sample.find_new_measurement_idxs()).copy()
        if self.checkpoint_skipped:
            self.new_idxs_to_write = np.concatenate((self.new_idxs_to_write, new_idxs_to_write), axis=0)[
                                     :self.store_file_scan_points_num]
        else:
            self.new_idxs_to_write = np.concatenate((self.new_idxs_to_write[:-self.points_per_route],
                                                     new_idxs_to_write), axis=0)[
                                     :self.store_file_scan_points_num]
        t1 = time.time()
        logging.info('Time required to calculate the new positions is %.3f sec.' % (t1 - t0))

        recon_local_fname = 'recon.npy'
        np.save(recon_local_fname, self.sample.recon_image)

        percent_measured = self.sample.ratio_measured
        logging.info('Surface covered: %.3f' % (percent_measured))

        if len(new_idxs_to_write) != 0:
            # local_idx_path = self.sample.measurement_interface.initialize_external_measurement(new_idxs)
            local_fname = self.write_instructions_file()
            # Send epics output here
            if self.debug:
                logging.info('Generating new location file %s.' % (local_fname))
            self.sftp_put(local_fname, str(Path(REMOTE_PATH) / local_fname))
        else:
            logging.info('No new scan position found. Stopping scan.')
            self.completed_run_flag = True
        t2 = time.time()
        self.sftp_put(recon_local_fname, str(REMOTE_PATH / 'recon_latest.npy'))
        logging.info('Sent new smart scan positions (time elapsed) %.3f s.\n' % (t2 - t1))

    def write_instructions_file(self):
        fname = 'instructions_%03d.csv' % self.current_file_suffix
        np.savetxt(fname, self.new_idxs_to_write, delimiter=',', fmt='%10d')
        return fname

    def write_route_file_and_update_suffix(self):
        # When we replace the points from that read from the current points, we might not have 500,
        # but 499 or something like that
        # so we need to have a margin for safety.
        # Keep track of number of points either through the read intensities or the route.
        if np.shape(self.route_full)[0] == self.store_file_scan_points_num:
            fname = 'route_%03d.csv' % self.current_file_suffix
            if self.debug:
                logging.info('Saving route file to' + fname)
            np.savetxt(fname, self.route_full, delimiter=',', fmt='%10d')
            self.current_file_suffix += 1

            self.route_full = np.empty((0, 2), dtype='int')
            self.new_idxs_to_write = np.empty((0, 2), dtype='int')

            self.current_file_position = 0
        else:
            if self.debug:
                logging.info('Incrementing current file position')
            self.current_file_position += 1


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('m', help='Current mda file offset.')
    argparser.add_argument('-c', default='config.txt',
                           help='Name of config file to download from remote.')
    argparser.add_argument('-r', '--stop_ratio', type=float, default=0.35,
                           help='Stop ratio.')
    argparser.add_argument('-g', '--indices_to_generate', type=int, default=100,
                           help='Number of position indices to generate from slads at every measurement step.')
    argparser.add_argument('-p', '--points_to_scan', type=int, default=50,
                           help='Number of points to actual scan at every measurement step.')
    argparser.add_argument('-t', '--points_to_store', type=int, default=500,
                           help='Number of points to store in every instructions file.')
    sysargs = argparser.parse_args()

    if sysargs.p < 2 * sysargs.g:
        logging.ERROR('Number of points generated by SLADS should be at least 2x the number of points ' \
                      'scanned at every step.')
        sys.exit()
    if sysargs.t <= sysargs.p:
        logging.ERROR('Number of points stored in every instructions file should be greater than the number of points' \
                      'scanned at every step.')
        sys.exit()

    config_fname = sysargs.c

    ssh, sftp = setup_sftp(REMOTE_IP, REMOTE_USERNAME)

    config_path_remote = REMOTE_PATH / config_fname
    sftp_get(sftp, config_path_remote, config_fname)

    cparser = configparser.ConfigParser()
    cparser.read(config_fname)
    cargs = cparser.defaults()

    celems = ['scan_sizex', 'scan_sizey', 'xpos', 'ypos', 'xfactor', 'scan_stepsize']
    if set(celems).isdisjoint(cargs):
        logging.ERROR('Config file does not contain required input parameters.')

    sample = helper.create_experiment_sample(numx=celems['scan_sizex'], numy=celems['scan_sizey'],
                                             inner_batch_size=sysargs.g,
                                             stop_ratio=sysargs.s,
                                             c_value=2.0,
                                             full_erd_recalculation_frequency=1,
                                             affected_neighbors_window_min=5,
                                             affected_neighbors_window_max=15)

    init_data_dir = 'initial_data'
    helper.clean_data_directory(init_data_dir)
    helper.get_init_npzs_from_remote(sftp=sftp, remote_dir=REMOTE_PATH, data_dir=init_data_dir)
    n_init, initial_idxs, initial_intensities = helper.load_idxs_and_intensities(init_data_dir)

    root = tk.Tk()
    gui = MainWindow(master=root, sample=sample,
                     scan_sizex=celems['scan_sizex'],
                     scan_sizey=celems['scan_sizey'],
                     xpos=celems['xpos'],
                     ypos=celems['ypos'],
                     xfactor=celems['xfactor'],
                     scan_stepsize=celems['scan_stepsize'],
                     current_file_suffix=n_init,
                     mda_file_offset=sysargs.m,
                     sftp=sftp,
                     store_file_scan_points_num=sysargs.t,
                     points_per_route=sysargs.p)

    gui.update(initial_idxs, initial_intensities)

    root.wm_attributes("-topmost", 1)
    root.mainloop()
