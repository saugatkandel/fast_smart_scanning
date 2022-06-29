import paramiko
from pathlib import Path


def setup_sftp(remote_ip: str, remote_username: str, look_for_keys: bool=True):
    ssh = paramiko.SSHClient()
    ssh.load_host_keys(Path.expanduser('~/.ssh/known_hosts'))
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(remote_ip, username=remote_username, look_for_keys=look_for_keys, allow_agent=False)
    sftp = ssh.open_sftp()
    return ssh, sftp


def sftp_put(sftp, lpath, rpath):
    sftp.put(str(lpath), str(rpath))


def sftp_get(sftp, rpath, lpath):
    sftp.get(str(rpath), str(lpath))