import os
from pathlib import Path
import sys
import argparse
import natsort
import logging

import Mothership_Zeta as MZ

class abduction:
    def __init__(
        self,
        dir_data = "/n/data1/hms/neurobio/sabatini/gyu/data/abduction",
        save_dir = None,
        source_extraction = "suite2p",
        classify = "ROICaT",
        tracking = "ROICaT",
        analysis = "seqNMF",
    ):
        """One-liner code for the batch suite2p and logger-alignment. The whole set of code will be improved later via inheritance.

        Args:
            dir_data (str, optional): Data directory for suite2p and logger-alignment. Assuming a specific day-folder structure. Defaults to "/n/data1/hms/neurobio/sabatini/gyu/data/abduction".
            save_dir (_type_, optional): Not functional rn. Defaults to None.
            source_extraction (str, optional): Not functional rn. Defaults to "suite2p".
            classify (str, optional): Not functional rn. Defaults to "ROICaT".
            tracking (str, optional): Not functional rn. Defaults to "ROICaT".
            analysis (str, optional): Not functional rn. Defaults to "seqNMF".
        """    
        dir_data = Path(dir_data).resolve()
        self.dir_data = dir_data
        self.save_dir = save_dir
        self.source_extraction = source_extraction
        self.classify = classify
        self.tracking = tracking
        self.analysis = analysis

    def journey(self):
        """Submit both suite2p and logger-alignment job
        """        
        self.extraction()
        self.logger_align()

    def extraction(self):
        """Submit batch suite2p job
        """        
        logging.warning("Journey 1: source extraction")
        logging.warning(self.dir_data)
        extraction_runner = MZ.extract_multi_run(dir_data = self.dir_data)
        extraction_runner.multi_run()

    def logger_align(self):
        """Submit batch logger-alignment job
        """        
        logging.warning("Journey 2: logger alignment")
        logging.warning(self.dir_data)
        logger_align_runner = MZ.logger_multi_run(dir_data = self.dir_data)
        logger_align_runner.multi_run()

def journey_process():
    args = cmdline_parser()
    handle = abduction(**args.__dict__)
    handle.journey()

def extraction_process():
    args = cmdline_parser()
    handle = abduction(**args.__dict__)
    handle.extraction()

def logger_align_process():
    args = cmdline_parser()
    handle = abduction(**args.__dict__)
    handle.logger_align()

def cmdline_parser():
    parser = argparse.ArgumentParser(
        description="MZ_runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir-data",
        dest = "dir_data",
        default="/n/data1/hms/neurobio/sabatini/gyu/data/abduction",
        help="Data directory for suite2p and logger-alignment. Assuming a specific day-folder structure.",
    )

    return parser.parse_args()