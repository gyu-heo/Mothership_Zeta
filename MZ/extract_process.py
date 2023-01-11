import os
import sys
from pathlib import Path

import paramiko
from scp import SCPClient

import getpass
import gc
import time
import logging
import pickle
import natsort

import Mothership_Zeta as MZ
from Mothership_Zeta.MZ.source_extraction.s2p_on_o2 import util

class extract_multi_run:
    def __init__(self,
    dir_data = Path("/n/data1/hms/neurobio/sabatini/gyu/data/abduction").resolve(),
    dir_output = Path("/n/data1/hms/neurobio/sabatini/gyu/analysis/suite2p_output/").resolve(),
    dir_fastDisk = Path("/n/data1/hms/neurobio/sabatini/gyu/analysis/suite2p_fastDisk/").resolve(),
    path_dispatcher = Path(MZ.__path__[0]).resolve() / 'MZ/source_extraction/s2p_on_o2/dispatcher.py',
    path_s2pScript_remote = Path(MZ.__path__[0]).resolve() / 'MZ/source_extraction/s2p_on_o2/remote_run_s2p.py',
    name_job = 'jobNum_',
    ):
        """One-liner code for the batch suite2p. The whole set of code will be improved later via inheritance.
        Source code created by RH


        Args:
            dir_data (_type_, optional):  Data directory for suite2p. Assuming a specific day-folder structure.. Defaults to Path("/n/data1/hms/neurobio/sabatini/gyu/data/abduction").resolve().
            dir_output (_type_, optional): Suite2p output will be saved here. Defaults to Path("/n/data1/hms/neurobio/sabatini/gyu/analysis/suite2p_output/").resolve().
            dir_fastDisk (_type_, optional): Temporary directory for the suite2p process. Defaults to Path("/n/data1/hms/neurobio/sabatini/gyu/analysis/suite2p_fastDisk/").resolve().
            path_dispatcher (_type_, optional): Python code path to submit detailed batch suite2p job. Defaults to Path(MZ.__path__[0]).resolve()/'MZ/source_extraction/s2p_on_o2/dispatcher.py'.
            path_s2pScript_remote (_type_, optional): Python code path to run suite2p on Linux server. Defaults to Path(MZ.__path__[0]).resolve()/'MZ/source_extraction/s2p_on_o2/remote_run_s2p.py'.
            name_job (str, optional): Simple prefix. Defaults to 'jobNum_'.
        """    
        self.module_path = Path(MZ.__path__[0]).resolve().parents[0]
        self.dir_data = dir_data
        self.dir_output = dir_output
        self.dir_fastDisk = dir_fastDisk
        self.name_job = name_job
                
        self.path_dispatcher = path_dispatcher
        self.path_s2pScript_remote = path_s2pScript_remote

        self.movie_list = self.find_movie()
        self.batch_command = self.batch_maker()

    def find_movie(self):
        """List raw tif files aka 2p videos parent directory to submit batch suite2p job

        Returns:
            _type_: _description_
        """        
        movie_list = []
        for movie_path in self.dir_data.rglob('*'):
            if 'tif' in str(movie_path):
                if movie_path.parents[0] not in movie_list:
                    movie_list.append(movie_path.parents[0])
        movie_list = natsort.natsorted(movie_list)
        return movie_list

    def batch_maker(self):
        """Create slurm command to submit batch logger-alignment job

        Returns:
            _type_: _description_
        """        
        mkdir_list, command_list = [], []
        for session in self.movie_list:
            dir_data_remote = Path(session).resolve()
            name_slurm = dir_data_remote.relative_to(self.dir_data)
            dir_S2pOutput_remote = self.dir_output / dir_data_remote.relative_to(self.dir_data)
            dir_fastDisk_remote = self.dir_fastDisk / dir_data_remote.relative_to(self.dir_data)
            mkdir_list.append(f"mkdir -p {str(dir_fastDisk_remote)}")
            command_list.append(f"python3 {str(self.path_dispatcher)} {dir_S2pOutput_remote} {self.path_s2pScript_remote} {self.name_job} {dir_fastDisk_remote} {str(name_slurm)} {dir_data_remote} {str(self.module_path)}")
            logging.warning(f"{str(name_slurm)} num of movies {len(os.listdir(session))}")
        batch_command = {"mkdir_list":mkdir_list, "command_list":command_list}
        return batch_command

    def multi_run(self):
        """Submit batch suite2p job
        """        
        for cmd_mkdir in self.batch_command['mkdir_list']:
            os.system(cmd_mkdir)

        for s2p_run in self.batch_command['command_list']:
            os.system(s2p_run)
        