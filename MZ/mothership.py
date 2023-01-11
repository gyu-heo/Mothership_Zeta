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
        classify = "BMI_IDAP",
        tracking = "ROICaT",
        analysis = "seqNMF",
    ):
        dir_data = Path(dir_data).resolve()
        self.dir_data = dir_data
        self.save_dir = save_dir
        self.source_extraction = source_extraction
        self.classify = classify
        self.tracking = tracking
        self.analysis = analysis

    def journey(self):
        self.extraction()
        self.logger_align()

    def extraction(self):
        logging.warning("Journey 1: source extraction")
        logging.warning(self.dir_data)
        extraction_runner = MZ.extract_multi_run(dir_data = self.dir_data)
        extraction_runner.multi_run()

    def logger_align(self):
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
        help="Data_saved_here",
    )
    # parser.add_argument(
    #     "--seqNMF-param",
    #     dest = "seqNMF_param",
    #     default="/n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/analysis/seqNMF/params.yml",
    #     help="seqNMF param list",
    # )
    # parser.add_argument(
    #     "--reference-session",
    #     dest = "reference_session",
    #     default=None,
    #     help="Choose a reference session to calculate W_init",
    # )
    # parser.add_argument(
    #     "--cascade",
    #     dest = "cascade",
    #     default=False,
    #     action="store_true",
    #     help="Cascade seqNMF?",
    # )
    # parser.add_argument(
    #     "--unfinished",
    #     dest = "unfinished",
    #     default=False,
    #     action="store_true",
    #     help="Any missing job, trigger True",
    # )

    return parser.parse_args()