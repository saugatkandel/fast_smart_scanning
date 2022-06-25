import logging
  
level    = logging.INFO
format   = '%(message)s'
handlers = [logging.FileHandler('smart_scan.log'), logging.StreamHandler()]

logging.basicConfig(level = level, format = format, handlers = handlers)

from sladsnet.code.erd import SladsSklearnModel
from sladsnet.code.measurement_interface import ExperimentMeasurementInterface
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
from datetime import datetime
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

        self.pbar = tqdm(total=sample.params_sample.stop_ratio * 100, desc='% sampled', leave=True, ascii=True)
        
        percent_measured = round(sample.ratio_measured * 100, 2)
        self.pbar.n = np.clip(percent_measured, 0, sample.params_sample.stop_ratio * 100)
        self.pbar.refresh()

        monitor1 = epics.PV("26idbPBS:sft01:ph01:ao13.VAL", callback = self.monitor_function)
        monitor2 = epics.PV("26idbSOFT:scan1.WCNT", callback = self.monitor_function)
        # need to duplicate mon
        self.scan_file_offset = 175
        self.is_monitor_initialized = False
        self.debug = debug

        self.monitor_start_time_prev = None
        self.route_full = np.empty((0, 2), dtype='int')
        self.new_idxs_to_write = np.empty((0,2), dtype='int')
        
        self.current_file_suffix = 1
        self.current_file_position = 0

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
                    print(datetime.now(), "Monitor has been triggered...")

                return
            
            t00 = time.time()
            time_diff = t00 - self.monitor_start_time_prev
            
            if self.debug:
                print(datetime.now(), f'Delay between monitor updates is {time_diff:.3f} s.')
            if time_diff < 20:
                print("WARNING: Delay between trigger times was less than 20s. Ignoring the trigger.")
                return

            self.monitor_start_time_prev = t00

            t0 = time.time()
            self.print_twice(datetime.now(), "Receiving optimized route file.")
            self.sftp_get(str(REMOTE_PATH / 'route.npz'), 'route.npz')

            n1 = self.scan_file_offset
            #n2 = self.sample.measurement_interface.current_file_suffix 
            n2 = self.current_file_suffix

            mda_file_name = f'/home/sector26/2022R2/20220621/mda/26idbSOFT_{n1+n2:04d}.mda'
            print(datetime.now(), "MDA file name is", mda_file_name)
            
            self.sftp_get(mda_file_name, 'mda_current.mda')

            t1 = time.time()
            print(datetime.now(), f'Time required to transfer files from detector is {t1-t0:.3f} s.')
            
            mda = readMDA('mda_current.mda', verbose=False)
            data = np.array(mda[1].d[3].data)
            curr_pt = mda[1].curr_pt
            points_of_interest = curr_pt % 50
            if points_of_interest == 0:
                points_of_interest = 50

            if self.debug:
                expected_shape =  (self.current_file_position  + 1) * 50
                #self.sample.measurement_interface.current_file_position * 50
                if mda[1].curr_pt != expected_shape:
                    if self.debug:
                        print(datetime.now(), "Warning: At iteration", self.current_file_position,  
                        "data shape is", mda[1].curr_pt, ", but the expected shape is", expected_shape)

            route = np.load('route.npz')
            xpoints = route['x'][:points_of_interest - 1]
            ypoints = route['y'][:points_of_interest - 1]
            route_idxs = np.array((ypoints, xpoints), dtype='int').T
            route_shape = np.shape(route_idxs)[0]

            new_intensities = data[-points_of_interest + 1:]

            if np.shape(route_idxs)[0] != np.shape(new_intensities)[0]:
                if self.debug:
                    print(datetime.now(), f'Mismatch between shapes of route {route_shape} and '\
                        f'and intensities {np.shape(new_intensities)[0]}.')
            
            route_this = np.array((route['y'], route['x']), dtype='int').T
            self.route_full = np.concatenate((self.route_full, route_this), axis=0)
            self.write_route_file_and_update_suffix()
            if self.debug:
                print(datetime.now(), 'Shape of full route is', self.route_full.shape)
            self.update(route_idxs, new_intensities)

        
    def update(self, route_idxs, new_intensities):
        
        #self.iteration += 1

        self.sample.measurement_interface.finalize_external_measurement(new_intensities)
        self.sample.perform_measurements(route_idxs)

        t0 = time.time()
        self.sample.reconstruct_and_compute_erd()    
        percent_measured = round(self.sample.ratio_measured * 100, 2)

        total_erd = self.sample.ERD.sum()
        print(datetime.now(), f'Total ERD is {total_erd:4.3g}.')
        self.pbar.set_postfix({'total ERD': total_erd})
        self.pbar.n = np.clip(percent_measured, 0, self.sample.params_sample.stop_ratio * 100)
        self.pbar.refresh()


        completed_run_flag = check_stopping_criteria(self.sample, 0)

        new_idxs_to_write = np.array(self.sample.find_new_measurement_idxs()).copy()
        self.new_idxs_to_write = np.concatenate((self.new_idxs_to_write, new_idxs_to_write), axis=0)
        t1 = time.time()
        print(datetime.now(), f'Time required to calculate the new positions is {t1-t0:.3f} sec.')

        recon_local_fname = 'recon.npy'
        np.save(recon_local_fname, self.sample.recon_image)
        self.sftp_put(recon_local_fname, str(REMOTE_PATH / 'recon_latest.npy'))

        recon_remote_fname1 = f'recon_{self.current_file_suffix:03d}.npy'
        self.sftp_put(recon_local_fname, str(REMOTE_PATH / recon_remote_fname1))

        if len(new_idxs_to_write) != 0:
            #local_idx_path = self.sample.measurement_interface.initialize_external_measurement(new_idxs)
            local_fname = self.write_instructions_file()
            # Send epics output here
            if self.debug:
                print(datetime.now(), f"Generating new location file {local_fname}.")
            #self.sftp_put(local_idx_path,  str(Path(REMOTE_PATH) / local_idx_path))
            self.sftp_put(local_fname,  str(Path(REMOTE_PATH) / local_fname))
        else:
            print(datetime.now(), 'No new scan position found. Stopping scan.')
            completed_run_flag = True
        t2 = time.time()
        print(datetime.now(), f'Time required to transfer files to detector is {t2-t1:.3f} s.')
        
        if completed_run_flag:
            self.pbar.close()

    def write_instructions_file(self):
        fname = f'instructions_{self.current_file_suffix:03d}.csv'
        np.savetxt(fname, self.new_idxs_to_write, delimiter=',', fmt='%10d')
        return fname
    
    def write_route_file_and_update_suffix(self):
        if np.shape(self.route_full)[0] == self.store_file_scan_points_num:
            fname = f'route_{self.current_file_suffix:03d}.csv'
            if self.debug:
                print(datetime.now(), 'Saving route file to', fname)
            np.savetxt(fname, self.route_full, delimiter=',', fmt='%10d')
            self.current_file_suffix += 1

            self.route_full = np.empty((0, 2), dtype='int')
            self.new_idxs_to_write = np.empty((0, 2), dtype='int')
            
            self.current_file_position = 0
        else:
            if self.debug:
                print(datetime.now(), 'Incrementing current file position')
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

    inner_batch_size = 50
    store_file_scan_points_num = 2000
    num_iterative_idxs = inner_batch_size

    stop_ratio = 0.3
    store_results_percentage = 1

    affected_neighbors_window_min = 5
    affected_neighbors_window_max = 15
    full_erd_recalculation_frequency = 1

    params_erd = ERDInputParams(c_value=C_VALUE,
                                full_erd_recalculation_frequency=full_erd_recalculation_frequency,
                                affected_neighbors_window_min=affected_neighbors_window_min,
                                affected_neighbors_window_max=affected_neighbors_window_max)
    params_gen = GeneralInputParams()


    # These array indices only apply for the current scan, and should be changed before the next
    # initial scan. For future we only need [:-1]
    initial_route = np.load('initial_route.npz')
    xpoints = initial_route['x'][1:-1]
    ypoints = initial_route['y'][1:-1]
    
    initial_idxs = np.array((ypoints, xpoints), dtype='int').T
    
    sample_params = SampleParams(image_shape=(600, 400),
                                inner_batch_size=inner_batch_size,
                                initial_idxs=initial_idxs,
                                stop_ratio=stop_ratio,
                                random_seed=11)

    measurement_interface = ExperimentMeasurementInterface()#num_initial_idxs= initial_scan_points_num,
                                                            #store_file_scan_points_num=store_file_scan_points_num,
                                                            #num_iterative_idxs=num_iterative_idxs, 
                                                            #is_initialized=False)
    sample = ExperimentalSample(sample_params=sample_params,
                general_params=params_gen,
                erd_params=params_erd,
                measurement_interface=measurement_interface,
                erd_model=erd_model)

    # These array indices only apply for the current scan, and should be changed before the next
    # initial scan.
    initial_intensities = np.load('initial_intensities.npy')[1:]
    #sample.measurement_interface._external_measurement_initialized = True
    #sample.measurement_interface._initialized = True
    #sample.measurement_interface.current_file_suffix = 1

    root = tk.Tk()
    gui = MainWindow(root, sample, sftp, store_file_scan_points_num, debug=False)
    
    gui.update(initial_idxs, initial_intensities)

    root.wm_attributes("-topmost",1)
    root.mainloop()
