import abc

import numpy as np


class MeasurementInterface(abc.ABC):

    @abc.abstractmethod
    def perform_measurement(self, idxs):
        pass


class TransmissionSimulationMeasurementInterface(MeasurementInterface):
    def __init__(self, image: np.ndarray = None, image_path: str = None):
        import tifffile as tif
        assert (image is not None) or (image_path is not None)
        self.image_path = image_path
        if image is not None:
            self.image = image
        else:
            self.image = tif.imread(self.image_path)

    def perform_measurement(self, idxs):
        return self.image[idxs[:, 0], idxs[:, 1]]

class ExperimentMeasurementInterface(MeasurementInterface):
    def __init__(self, local_write_path: str, 
    remote_write_path: str = '/home/sector26/2022R2/20220621/Analysis/instructions.csv', 
    remote_ip: str = 'ives.cnm.aps.anl.gov',
    remote_username: str='user26id',
    num_initial_idxs: int = 2000, 
    num_iterative_idxs: int=50):
        from collections import deque
        from pathlib import Path
        import paramiko
        import os
        
        local_write_path = Path(local_write_path)
        if local_write_path.suffix != '.csv':
            local_write_path = local_write_path.parent / (local_write_path.name + '.csv')
        self.local_write_path = local_write_path
        
        
        self.remote_write_path = remote_write_path
        self.remote_ip = remote_ip
        self.remote_username = remote_username
        self._ssh = paramiko.SSHClient()
        self._ssh.load_host_keys(os.path.expanduser(os.path.join('~', '.ssh', 'known_hosts')))
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(self.remote_ip, username=self.remote_username, look_for_keys=True, allow_agent=False)
        self._sftp = self._ssh.open_sftp()


        self._num_initial_idxs = num_initial_idxs
        self._num_iterative_idxs = num_iterative_idxs
        self._deque_array_idxs = deque(maxlen=num_initial_idxs)
        self._array_idxs = np.array(self._deque_array_idxs, dtype='int32')

        self._initialized = False

    def _write_idx_local(self):
        np.savetxt(self.local_write_path, self._array_idxs, delimiter=',', fmt='%10d')

    def _write_idx_remote(self):
        self._sftp.put(self.local_write_path, self.remote_write_path)

    def _update_deque_and_array(self, idxs):
        if not self._initialized:
            if np.shape(idxs)[0] != self._num_initial_idxs:
                raise ValueError(f'Number of idxs for first measurement must match with the "num_initial_idxs"'\
                    f'supplied when creating the measurement interface.')
            for idx in idxs:
                self._deque_array_idxs.append(idx)
            self._array_idxs = np.array(self._deque_array_idxs, dtype='int32')
            self._initialized = True
        else:
            if np.shape(idxs)[0] > self._num_initial_idxs:
                raise ValueError(f'Number of idxs for new measurement must not be greater than "num_iterative_idxs"'\
                    f'supplied when creating the measurement interface.')
            for idx in idxs:
                self._deque_array_idxs.append(idx)
            self._array_idxs = np.array(self._deque_array_idxs, dtype='int32')


    def perform_measurement(self, idxs):
        self._update_deque_and_array(idxs)
        self._write_idx_local()
        #self._write_idx_remote()
        #pass
        #return self.image[idxs[:, 0], idxs[:, 1]]