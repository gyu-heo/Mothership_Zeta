import os
import sys
from pathlib import Path

import os
import pickle
import natsort
import logging

import Mothership_Zeta

class logger_multi_run:
    def __init__(
        self,
        dir_data = Path("/n/data1/hms/neurobio/sabatini/gyu/data/abduction").resolve(),
        dir_output = Path("/n/data1/hms/neurobio/sabatini/gyu/analysis/BWAIN_output/").resolve(),
        path_aligner = Path("/n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/behavior/logger_align/BMIaligner.sh").resolve()
    ):
        """_summary_

        Args:
            dir_data (_type_): _description_. Defaults to None.
        """    
        self.dir_data = dir_data
        self.dir_output = dir_output
        self.path_aligner = path_aligner
        
        logging.warning("Finding logger list...")
        self.logger_list = self.find_logger()
        # logging.warning(self.logger_list)
        self.batch_command = self.batch_maker()

    def find_logger(self):
        logger_list = []
        for logger_path in self.dir_data.rglob('*'):
            if ('logger.mat' in str(logger_path)) or ('logger_valsROIs.mat' in str(logger_path)):
                if logger_path.parents[0] not in logger_list:
                    logger_list.append(logger_path.parents[0])
        logger_list = natsort.natsorted(logger_list)
        return logger_list

    def batch_maker(self):
        mkdir_list, command_list = [], []
        for session in self.logger_list:
            dir_data_remote = Path(session).resolve()
            movie_path = dir_data_remote.parents[0] / 'scanimage_data/exp'
            output_path = str(self.dir_output / dir_data_remote.relative_to(self.dir_data))
            cmdline_input = f"sbatch {str(self.path_aligner)} {str(movie_path)}matdelim{session}matdelim{output_path}"
            mkdir_list.append(f"mkdir -p {output_path}")
            command_list.append(cmdline_input)
            # logging.warning(cmdline_input)

        batch_command = {"mkdir_list":mkdir_list, "command_list":command_list}
        return batch_command

    def multi_run(self):
        for cmd_mkdir in self.batch_command['mkdir_list']:
            os.system(cmd_mkdir)

        for logger_align_run in self.batch_command['command_list']:
            logging.warning(f"Submitting jobs: {logger_align_run}")
            os.system(logger_align_run)