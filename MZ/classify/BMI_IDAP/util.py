from pathlib import Path
import numpy as np
import logging
import copy
import torch
import pickle
import matplotlib.pyplot as plt

import Mothership_Zeta.MZ as MZ

## Path loader
class BMI_IDAP_path_loader:
    def __init__(
        self,
        dir_s2p,
        dir_save = None,
    ):
        logging.warning("Loading Paths...")

        ## Directory with F.npy, stat.npy etc.
        try:
            self.dir_s2p = Path(dir_s2p).resolve()
        except:
            logging.warning("dir_s2p is not defined!")
        
        ## Directory to save outputs
        if dir_save is None:
            self.dir_save = dir_s2p
        else:
            self.dir_save = dir_save

        ## NN fileNames
        self.fileName_NN_pth = 'ResNet18_simCLR_model_202112078_EOD_transfmod=norm.pth' # name of pth file in dir_NNmodels directory
        self.fileName_NN_py  = 'ResNet18_simCLR_model_202112078_EOD_transfmod=norm' # EXCLUDE THE .PY AT THE END. name of py file in dir_NNmodels directory.
        self.fileName_classifier = 'logreg_model_0.01.pkl' # path to logististic classifier pickle file in dir_classifiers

        ## Directories of Classifier stuff
        self.dir_github = Path(MZ.__path__[0]).resolve()
        self.dir_analysis = self.dir_github / 'analysis'
        self.dir_classify = self.dir_github / 'classify'

        
        self.dir_GRC_repo = self.dir_classify / 'GCaMP_ROI_classifier'
        self.dir_GRC_EndUser = self.dir_GRC_repo / 'End_User'
        self.dir_NNmodels = self.dir_GRC_EndUser / 'simclr-models'
        self.dir_classifiers = self.dir_GRC_EndUser / 'classifier-models'
        self.dir_GRC_util = self.dir_GRC_repo / 'new_stuff'

        ## Paths to NN and LR classifiers
        self.path_NN_pth = self.dir_NNmodels / self.fileName_NN_pth
        self.path_NN_py = self.dir_NNmodels / self.fileName_NN_py
        self.path_classifier = self.dir_classifiers / self.fileName_classifier

        self.path_statFile = self.dir_s2p / 'stat.npy'
        self.path_opsFile = self.dir_s2p / 'ops.npy'