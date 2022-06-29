import logging
from datetime import datetime
import sys
import requests

level    = logging.INFO
format   = '%(message)s'
handlers = [logging.FileHandler('LOGS/smart_scan_%s.log' %datetime.now().strftime("%m_%d_%Y_%H:%M:%S")), logging.StreamHandler()]

logging.basicConfig(level = level, format = format, handlers = handlers)

from sladsnet.code.erd import SladsSklearnModel
from sladsnet.code.measurement_interface import ExternalMeasurementInterface
from sladsnet.input_params import ERDInputParams, GeneralInputParams, SampleParams
from sladsnet.code.results import Result
from sladsnet.code.base import ExperimentalSample
from sladsnet.code.sampling import check_stopping_criteria

import numpy as np
import tkinter as tk
from pathlib import Path
import epics
from tqdm import tqdm
import tifffile as tif
import time
from readMDA import readMDA

import paramiko
import os
from pathlib import Path

REMOTE_PATH = Path('/home/sector26/2022R2/20220621/Analysis/') 
REMOTE_IP = 'ives.cnm.aps.anl.gov'
REMOTE_USERNAME = 'user26id'

#def write_local(fname, array, fmt): 
    #path = self.local_write_path / self._get_current_fname()
    #np.savetxt(path, self._list_idxs, delimiter=',', fmt='%10d')
    
    
def sftp_put(sftp, lpath, rpath):
    sftp.put(str(lpath), str(rpath))

def sftp_get(sftp, rpath, lpath):
    sftp.get(str(rpath), str(lpath))



class MainWindow:
    def __init__(self, master, sample, sftp, store_file_scan_points_num, debug=False):
        self.logfile = open('log.info', 'a+')
        self.sftp = sftp
        self.sample = sample
        self.store_file_scan_points_num = store_file_scan_points_num
        self.completed_run_flag = False

        #self.pbar = tqdm(total=sample.params_sample.stop_ratio * 100, desc='% sampled', leave=True, ascii=True)
        
        percent_measured = round(sample.ratio_measured * 100, 2)
        #self.pbar.n = np.clip(percent_measured, 0, sample.params_sample.stop_ratio * 100)
        #self.pbar.refresh()

        monitor1 = epics.PV("26idbPBS:sft01:ph01:ao13.VAL", callback = self.monitor_function)
        monitor2 = epics.PV("26idbSOFT:scan1.WCNT", callback = self.monitor_function)
        # need to duplicate mon
        self.scan_file_offset = int(sys.argv[1])
        self.is_monitor_initialized = False
        self.debug = debug

        self.monitor_start_time_prev = None
        self.route_full = np.empty((0, 2), dtype='int')
        self.new_idxs_to_write = np.empty((0,2), dtype='int')
        
        #remember to set your file_index back to 1 (since we are restarting)
        self.current_file_suffix = 1
        self.current_file_position = 0

        self.attoz0 = -21.347
        self.samy0 = -627.676
        self.xfactor = 0.366
        self.scan_stepsize = 0.1
        self.scan_centerx = 200
        self.scan_centery = 40

        self.checkpoint_skipped = 0

    def sftp_put(self, p1, p2):
        sftp_put(self.sftp, p1, p2)

    def sftp_get(self, p1, p2):
        sftp_get(self.sftp, p1, p2)

    def monitor_function(self, value = None, **kw):
        if value == 1:
            if not self.is_monitor_initialized:
                self.is_monitor_initialized = True
                self.monitor_start_time_prev = time.time()
                if self.debug:
                    logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+ " Monitor has been triggered...")

                return
            
            t00 = time.time()
            time_diff = t00 - self.monitor_start_time_prev
            
            if self.debug:
                logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Delay between monitor updates is %.3f s.' %time_diff)
            if time_diff < 20:
            #    requests.post(webhook, json={"text":"{0}: AGX : Delay between trigger times was less than 20s. Ignoring the trigger.".format(datetime.now())})
                #logging.info("WARNING: Delay between trigger times was less than 20s. Ignoring the trigger.")
                return

            self.monitor_start_time_prev = t00

            t0 = time.time()
            #logging.info(datetime.now().strftime("%m_%d_%Y_%H:%M:%S")+ " Receiving optimized route file.")
            # self.sftp_get(str(REMOTE_PATH / 'route.npz'), 'route.npz')

            n1 = self.scan_file_offset
            #n2 = self.sample.measurement_interface.current_file_suffix 
            n2 = self.current_file_suffix

            mda_file_name = '/home/sector26/2022R2/20220621/mda/26idbSOFT_%04d.mda' %(n1+n2)
            logging.info(datetime.now().strftime("%m_%d_%Y_%H:%M:%S") + " MDA file name is" + mda_file_name)
            
            self.sftp_get(mda_file_name, 'mda_current.mda')

            t1 = time.time()
            logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' AGX received data from detector (time elapsed) %.3f s.' %(t1-t0))
            
            mda = readMDA('mda_current.mda', verbose=False)
            data = np.array(mda[1].d[3].data)
            # 32 for attoz
            xx = np.round((np.array(mda[1].d[35].data)-self.attoz0)/self.scan_stepsize/self.xfactor,0) +self.scan_centerx
            # 31 for samy
            yy = np.round((np.array(mda[1].d[36].data)-self.samy0)/self.scan_stepsize,0) +self.scan_centery
            curr_pt = mda[1].curr_pt
            points_of_interest = curr_pt % 50
            if points_of_interest == 0:
                points_of_interest = 50

            expected_shape =  (self.current_file_position  + 1) * 50
            self.checkpoint_skipped = 0
            #self.sample.measurement_interface.current_file_position * 50
            if curr_pt != expected_shape:
                if self.debug:
                    logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + " Warning: At iteration", self.current_file_position,  
                    "data shape is", mda[1].curr_pt, ", but the expected shape is", expected_shape)
            if curr_pt > expected_shape + 30:
                #logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")  + "WARNING: Possible epics communication error.")
                #requests.post(webhook, json={"text":"{0}: AGX : Possible epics communication error.".format(datetime.now())})
                self.current_file_position += 1
                self.checkpoint_skipped = 1
                points_of_interest += 50

            #route = np.load('route.npz')
            xpoints = xx[curr_pt-points_of_interest:curr_pt-1]
            ypoints = yy[curr_pt-points_of_interest:curr_pt-1]
            route_idxs = np.array((ypoints, xpoints), dtype='int').T
            #route_shape = np.shape(route_idxs)[0]

            new_intensities = data[curr_pt-points_of_interest+1:curr_pt]
            #print(new_intensities.min(), new_intensities.mean(), data[curr_pt])
            #print(xpoints, xpoints.mean(), xx[curr_pt])

            if np.shape(route_idxs)[0] != np.shape(new_intensities)[0]:
                logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Mismatch between shapes of route %d and ' %xpoints.shape[0] + 'and intensities %d.' %(np.shape(new_intensities)[0]) )
            
            #route_this = np.array((route['y'], route['x']), dtype='int').T
            #self.route_full = np.concatenate((self.route_full, route_this), axis=0)
            #self.write_route_file_and_update_suffix()

            if curr_pt == self.store_file_scan_points_num:
                if self.completed_run_flag :
                    #requests.post(webhook, json={"text":"{0}: AGX : Done!".format(datetime.now())})
                    sys.exit()
                self.current_file_suffix += 1
                self.new_idxs_to_write = np.empty((0, 2), dtype='int')
                self.current_file_position = 0
            else:
                if self.debug:
                    logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Incrementing current file position')
                self.current_file_position += 1

            if self.debug:
                logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S")+ ' Shape of full route is', self.route_full.shape)
            self.update(route_idxs, new_intensities)

        
    def update(self, route_idxs, new_intensities):
        
        #self.iteration += 1

        self.sample.measurement_interface.finalize_external_measurement(new_intensities)
        self.sample.perform_measurements(route_idxs)

        t0 = time.time()
        self.sample.reconstruct_and_compute_erd()    
        percent_measured = round(self.sample.ratio_measured * 100, 2)

        total_erd = self.sample.ERD.sum()
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Total ERD is %4.3f.' %total_erd)
        #self.pbar.set_postfix({'total ERD': total_erd})
        #self.pbar.n = np.clip(percent_measured, 0, self.sample.params_sample.stop_ratio * 100)
        #self.pbar.refresh()

        self.completed_run_flag = check_stopping_criteria(self.sample, 0)

        new_idxs_to_write = np.array(self.sample.find_new_measurement_idxs()).copy()
        if self.checkpoint_skipped:
            self.new_idxs_to_write = np.concatenate((self.new_idxs_to_write, new_idxs_to_write), axis=0)[:self.store_file_scan_points_num]
        else:
            self.new_idxs_to_write = np.concatenate((self.new_idxs_to_write[:-50], new_idxs_to_write), axis=0)[:self.store_file_scan_points_num]
        t1 = time.time()
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Time required to calculate the new positions is %.3f sec.' %(t1-t0))

        recon_local_fname = 'recon.npy'
        np.save(recon_local_fname, self.sample.recon_image)
        
        percent_measured = self.sample.ratio_measured 
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Surface covered: %.3f' %(percent_measured))

        #recon_remote_fname1 = 'recon_%03d.npy' %self.current_file_suffix
        #self.sftp_put(recon_local_fname, str(REMOTE_PATH / recon_remote_fname1))

        if len(new_idxs_to_write) != 0:
            #local_idx_path = self.sample.measurement_interface.initialize_external_measurement(new_idxs)
            local_fname = self.write_instructions_file()
            # Send epics output here
            if self.debug:
                logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + " Generating new location file %s." %(local_fname))
            #self.sftp_put(local_idx_path,  str(Path(REMOTE_PATH) / local_idx_path))
            self.sftp_put(local_fname,  str(Path(REMOTE_PATH) / local_fname))
        else:
            logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' No new scan position found. Stopping scan.')
            self.completed_run_flag = True
        t2 = time.time()
        self.sftp_put(recon_local_fname, str(REMOTE_PATH / 'recon_latest.npy'))
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Sent new smart scan positions (time elapsed) %.3f s.\n' %(t2-t1))
        
        #if completed_run_flag:
        #    self.pbar.close()

    def write_instructions_file(self):
        
        fname = 'instructions_%03d.csv' %self.current_file_suffix
        np.savetxt(fname, self.new_idxs_to_write, delimiter=',', fmt='%10d')
        return fname
    
    def write_route_file_and_update_suffix(self):
        # When we replace the points from that read from the current points, we might not have 500, but 499 or something like that
        # so we need to have a margin for safety.
        # Keep track of number of points either through the read intensities or the route.
        if np.shape(self.route_full)[0] == self.store_file_scan_points_num:
            fname = 'route_%03d.csv' %self.current_file_suffix
            if self.debug:
                logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Saving route file to' + fname)
            np.savetxt(fname, self.route_full, delimiter=',', fmt='%10d')
            self.current_file_suffix += 1

            self.route_full = np.empty((0, 2), dtype='int')
            self.new_idxs_to_write = np.empty((0, 2), dtype='int')
            
            self.current_file_position = 0
        else:
            if self.debug:
                logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Incrementing current file position')
            self.current_file_position += 1


if __name__ == '__main__':
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(REMOTE_IP, username=REMOTE_USERNAME, look_for_keys=True, allow_agent=False)
    sftp = ssh.open_sftp()

    C_VALUE = 2
    train_base_path = Path.cwd().parent / 'ResultsAndData/TrainingData/cameraman/'
    erd_model = SladsSklearnModel(load_path=train_base_path / f'c_{C_VALUE}/erd_model_relu.pkl')

    inner_batch_size = 100
    store_file_scan_points_num = 2000
    num_iterative_idxs = inner_batch_size

    stop_ratio = 0.65
    store_results_percentage = 1

    affected_neighbors_window_min = 5
    affected_neighbors_window_max = 15
    full_erd_recalculation_frequency = 1

    params_erd = ERDInputParams(c_value=C_VALUE,
                                full_erd_recalculation_frequency=full_erd_recalculation_frequency,
                                affected_neighbors_window_min=affected_neighbors_window_min,
                                affected_neighbors_window_max=affected_neighbors_window_max)
    params_gen = GeneralInputParams()


    initial_data_path = Path('initial_new')
    npz_files = list(initial_data_path.glob('init_*.npz'))
    initial_idxs = np.empty((0,2), dtype='int')
    initial_intensities = np.empty(0, dtype='int')
    for fnpz in npz_files:
        fname = str(fnpz)
        logging.info(datetime.now().strftime("%m/%d/%Y, %H:%M:%S") + ' Loading initial data from %s'%fname)
        initial_data = np.load(fname)
        # Need to shift the idxs so that it starts from 0.
        xpoints = initial_data['x'][:-1]
        ypoints = initial_data['y'][:-1]
        idxs_this = np.array((ypoints, xpoints), dtype='int').T
        intensities = initial_data['I'][1:]
        
        initial_idxs = np.concatenate((initial_idxs, idxs_this), axis=0)
        initial_intensities = np.concatenate((initial_intensities, intensities))
    
    sample_params = SampleParams(image_shape=(80, 400),
                                inner_batch_size=inner_batch_size,
                                initial_idxs=initial_idxs,
                                stop_ratio=stop_ratio,
                                random_seed=11)

    measurement_interface = ExternalMeasurementInterface()#num_initial_idxs= initial_scan_points_num,
                                                            #store_file_scan_points_num=store_file_scan_points_num,
                                                            #num_iterative_idxs=num_iterative_idxs, 
                                                            #is_initialized=False)
    sample = ExperimentalSample(sample_params=sample_params,
                general_params=params_gen,
                erd_params=params_erd,
                measurement_interface=measurement_interface,
                erd_model=erd_model)
    
    root = tk.Tk()
    gui = MainWindow(root, sample, sftp, store_file_scan_points_num, debug=False)
    
    gui.update(initial_idxs, initial_intensities)
    

    root.wm_attributes("-topmost",1)
    root.mainloop()
