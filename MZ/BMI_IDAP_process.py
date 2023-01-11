import os
import sys
import argparse
from pathlib import Path

import itertools
import re
import pickle
import yaml
import natsort
import logging
from datetime import datetime

from Mothership_Zeta.MZ.classify.BMI_IDAP import util as BMI_IDAP_util
from Mothership_Zeta.MZ.classify.BMI_IDAP import ROI_classification, trace_quality_inclusion
from Mothership_Zeta.MZ.analysis.seqNMF import batch_run_seqNMF

class BMI_IDAP_multi_run:
    def __init__(
        self,
        dir_s2p,
        seqNMF_param = None,
        reference_session = None,
        cascade = False,
        unfinished = False,
    ):
        """One-liner code to run ROI classification and subsequent analysis. Will be deprecated if we successfully implement ROICaT classification process.
        Source code created by RH

        Args:
            dir_s2p (_type_): _description_
            seqNMF_param (_type_, optional): _description_. Defaults to None.
            reference_session (_type_, optional): _description_. Defaults to None.
            cascade (bool, optional): _description_. Defaults to False.
            unfinished (bool, optional): _description_. Defaults to False.
        """
        logging.warning(dir_s2p)
        self.dir_s2p = Path(dir_s2p).resolve()
        self.seqNMF_param = seqNMF_param
        self.reference_session = reference_session
        self.cascade = cascade
        self.unfinished = unfinished

    def cmd_maker(self, cmd_dir_s2p=None):
        logging.warning(self.reference_session)
        logging.warning(self.cascade)

        if cmd_dir_s2p is None:
            cmd_dir_s2p = f" --dir-s2p {self.dir_s2p}"

        cmd_seqNMF_param = f" --seqNMF-param {self.seqNMF_param}"

        if self.reference_session is not None:
            cmd_reference_session = f" --reference-session {self.reference_session}"
        else:
            cmd_reference_session = ""

        if self.cascade:
            cmd_cascade = " --cascade"
        else:
            cmd_cascade = ""

        if self.unfinished:
            cmd_unfinished = " --unfinished"
        else:
            cmd_unfinished = ""

        logging.warning(cmd_reference_session)
        logging.warning(cmd_cascade)

        cmd_accessory = cmd_dir_s2p + cmd_reference_session + cmd_cascade + cmd_unfinished
        return cmd_accessory

    def submit_multi_job(self):
        cmd_base = "sbatch /n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/classify/BMI_IDAP/BMI_IDAP_single_process.sh"
        for session in self.dir_s2p.rglob('*'):
            if "stat.npy" in str(session):
                cmd_dir_s2p = f" --dir-s2p {str(session.parents[0])}"
                cmd = cmd_base + self.cmd_maker(cmd_dir_s2p)
                logging.warning(cmd)
                os.system(cmd)

    def submit_single_job(self):
        path_list = BMI_IDAP_util.BMI_IDAP_path_loader(self.dir_s2p)
        iscell = ROI_classification.ROI_classification(path_list)
        trace_quality_inclusion.tqi(path_list, iscell)
        if self.seqNMF_param is not None:
            logging.warning("Start seqNMF")
            cmd_base = "sbatch /n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/analysis/seqNMF/seqNMF_multi_process.sh"
            cmd = cmd_base + self.cmd_maker(cmd_dir_s2p = None)
            logging.warning(cmd)
            os.system(cmd)


class seqNMF_multi_run:
    def __init__(
        self,
        dir_s2p,
        seqNMF_param = "/n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/analysis/seqNMF/params.yml",
        reference_session = None,
        cascade = False,
        unfinished = False,
    ):
        logging.warning(dir_s2p)
        logging.warning(seqNMF_param)
        self.dir_s2p = Path(dir_s2p).resolve()
        self.seqNMF_param = seqNMF_param
        self.params = self.load_params()
        self.reference_session = reference_session
        self.cascade = cascade
        self.unfinished = unfinished

        self.jobs_list_file = "jobs_list.p"


    def load_params(self):
        with open(self.seqNMF_param, 'r') as handle:
            params = yaml.safe_load(handle)
        return params

    def create_jobs_list(self):
        sweep_keys = list(self.params['sweep_params'].keys())
        sweep_list = list(itertools.product(*list(self.params['sweep_params'].values())))
        jobs_list = [dict(zip(sweep_keys, sweep_values)) for sweep_values in sweep_list]
        return jobs_list

    def save_jobs_list(self, jobs_list):
        jobs_list_file = self.dir_s2p / self.jobs_list_file
        with open(str(jobs_list_file), "wb") as file:
            pickle.dump(jobs_list, file)

    def load_jobs_list(self):
        jobs_list_file = self.dir_s2p / self.jobs_list_file
        with open(str(jobs_list_file), "rb") as file:
            jobs_list = pickle.load(file)
        return jobs_list

    def cmd_maker(self, cmd_dir_s2p=None):
        if cmd_dir_s2p is None:
            cmd_dir_s2p = f" --dir-s2p {self.dir_s2p}"
            
        cmd_seqNMF_param = f" --seqNMF-param {self.seqNMF_param}"

        if self.reference_session is not None:
            cmd_reference_session = f" --reference-session {self.reference_session}"
        else:
            cmd_reference_session = ""

        if self.cascade:
            cmd_cascade = " --cascade"
        else:
            cmd_cascade = ""

        if self.unfinished:
            cmd_unfinished = " --unfinished"
        else:
            cmd_unfinished = ""

        cmd_accessory = cmd_dir_s2p + cmd_reference_session + cmd_cascade + cmd_unfinished
        return cmd_accessory

    def save_cmd(self, cmd):
        cmd_save = (
            str(self.dir_s2p)
            + "/cmd_"
            + datetime.now().strftime("%Y%m%d-%H%M%S")
            + ".p"
        )
        with open(cmd_save, "wb") as file:
            pickle.dump(cmd, file)

    def missing_link(self, jobs_list):
        """_Create array list to submit batch jobs: if unfinished, run missing jobs only_
        """
        # Get finished_jobs list by listing "pose__.mat"
        perfect_list = [sweep for sweep in os.listdir(self.dir_s2p) if sweep[:5] == "sweep" and sweep[-1:] == "p"]
        array_list = list(range(len(jobs_list)))

        # If there are no finished_jobs, run whole jobs_list again!
        # If there are finished jobs...
        if perfect_list:
            # List & sort missing jobs index
            perfect_jobs = [int(re.findall("[0-9]+", jobs)[0]) for jobs in perfect_list]
            perfect_jobs.sort()
            array_list = [
                imperfect_jobs
                for imperfect_jobs in array_list
                if imperfect_jobs not in perfect_jobs
            ]
            strtemp = list(map(str, array_list))
            delim = ","
            cmd_array_list = delim.join(strtemp)
        else:
            cmd_array_list = [array_list[0], array_list[-1]]

        return cmd_array_list

    def submit_multi_job(self):
        if self.unfinished:
            jobs_list = self.load_jobs_list()
            cmd_array_list = self.missing_link(jobs_list)
        else:
            jobs_list = self.create_jobs_list()
            cmd_array_list = [0, len(jobs_list) - 1]

        cmd_accessory = self.cmd_maker()

        if isinstance(cmd_array_list, str):
            cmd = "sbatch --array=%s /n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/analysis/seqNMF/seqNMF_single_process.sh" % (
                cmd_array_list
            ) + cmd_accessory
        else:                
            cmd = "sbatch --array=%d-%d /n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/analysis/seqNMF/seqNMF_single_process.sh" % (
                cmd_array_list[0],
                cmd_array_list[-1]
            ) + cmd_accessory
        logging.warning(cmd)
        
        self.save_cmd(cmd)
        self.save_jobs_list(jobs_list)
        sys.exit(os.WEXITSTATUS(os.system(cmd)))

    def submit_single_job(self, kwargs):
        path_list = BMI_IDAP_util.BMI_IDAP_path_loader(self.dir_s2p)
        batch_run_seqNMF.seqNMF_cascade(path_list, kwargs, self.reference_session, self.cascade)


def BMI_IDAP_multi_process():
    args = cmdline_parser()
    handle = BMI_IDAP_multi_run(**args.__dict__)
    handle.submit_multi_job()

def BMI_IDAP_single_process():
    args = cmdline_parser()
    handle = BMI_IDAP_multi_run(**args.__dict__)
    handle.submit_single_job()

def seqNMF_multi_process():
    args = cmdline_parser()
    handle = seqNMF_multi_run(**args.__dict__)
    handle.submit_multi_job()

def seqNMF_single_process():
    args = cmdline_parser()
    handle = seqNMF_multi_run(**args.__dict__)

    jobs_list = handle.load_jobs_list()

    job_id = int(os.getenv("SLURM_ARRAY_TASK_ID"))
    job_params = jobs_list[job_id]

    kwargs = {**handle.params["kwarg_params"], **job_params}
    logging.warning(kwargs)

    handle.submit_single_job(kwargs)

def arg_tester():
    args = cmdline_parser()
    logging.warning(args.__dict__)


def cmdline_parser():
    parser = argparse.ArgumentParser(
        description="BMI_IDAP_runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir-s2p",
        dest = "dir_s2p",
        default=None,
        help="Suite2p output path: BMI_IDAP output will also be saved here",
    )
    parser.add_argument(
        "--seqNMF-param",
        dest = "seqNMF_param",
        default="/n/data1/hms/neurobio/sabatini/gyu/github_clone/Mothership_Zeta/MZ/analysis/seqNMF/params.yml",
        help="seqNMF param list",
    )
    parser.add_argument(
        "--reference-session",
        dest = "reference_session",
        default=None,
        help="Choose a reference session to calculate W_init",
    )
    parser.add_argument(
        "--cascade",
        dest = "cascade",
        default=False,
        action="store_true",
        help="Cascade seqNMF?",
    )
    parser.add_argument(
        "--unfinished",
        dest = "unfinished",
        default=False,
        action="store_true",
        help="Any missing job, trigger True",
    )

    return parser.parse_args()