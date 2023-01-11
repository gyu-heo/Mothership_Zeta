from pathlib import Path
import sys
import importlib

import numpy as np
import copy
import torch

import logging
import pickle
import matplotlib.pyplot as plt

from Mothership_Zeta.MZ.utils.basic_neural_processing_modules import ca2p_preprocessing, file_helpers

## Main
def tqi(path_list, iscell):
    logging.warning(f'Trace quality inclusion')
    
    dir_s2p = path_list.dir_s2p
    dir_save = path_list.dir_save    

    ## == IMPORT DATA ==
    logging.warning(f'Import Suite2p output data')

    F = np.load(dir_s2p / 'F.npy') # masks multiplied by data
    Fneu = np.load(dir_s2p / 'Fneu.npy') # estimated neuropil signal (Fns = F - Fneu; Fo = ptile(Fns, 30); dFoF=Fns-Fo/Fo)
    ops = np.load(dir_s2p / 'ops.npy', allow_pickle=True) # parameters for the suite2p
    spks_s2p = np.load(dir_s2p / 'spks.npy') # blind deconvolution
    stat = np.load(dir_s2p / 'stat.npy', allow_pickle=True) # statistics for individual neurons 

    num_frames_S2p = F.shape[1]
    Fs = ops[()]['fs']


    F_toUse = F[iscell]
    Fneu_toUse = Fneu[iscell]

    channelOffset_correction = 300
    percentile_baseline = 10
    neuropil_fraction=0.7

    logging.warning(f'Generate dFoF')
    dFoF , dF , F_neuSub , F_baseline = ca2p_preprocessing.make_dFoF(F=F_toUse + channelOffset_correction,
                                                                    Fneu=Fneu_toUse + channelOffset_correction,
                                                                    neuropil_fraction=neuropil_fraction,
                                                                    percentile_baseline=percentile_baseline,
                                                                    multicore_pref=True,
                                                                    verbose=True)

    dFoF_params = {
        "channelOffset_correction": channelOffset_correction,
        "percentile_baseline": percentile_baseline,
        "neuropil_fraction": neuropil_fraction,
    }
    logging.warning(f'{dFoF_params}')

    thresh = {
        'var_ratio': 1,
        'EV_F_by_Fneu': 0.6,
        'base_FneuSub': 0,
        'base_F': 25,
        'peter_noise_levels': 12,
        'rich_nsr': 50,
        'max_dFoF': 50,
        'baseline_var': 1,
        }
    logging.warning(f'{thresh}')

    tqm, iscell_tqm = ca2p_preprocessing.trace_quality_metrics(
        F_toUse,
        Fneu_toUse,
        dFoF,
        dF,
        F_neuSub,
        F_baseline,
        percentile_baseline=percentile_baseline,
        Fs=Fs,
        plot_pref=False,
        thresh=thresh,
    )
    idxROI_tqm_toInclude = np.where(iscell_tqm)[0]
    idxROI_tqm_toExclude = np.where(~iscell_tqm)[0]

    iscell_new = copy.copy(iscell)
    iscell_new[iscell_new] = iscell_tqm

    file_helpers.pickle_save(
        obj={
            "tqm": tqm,
            "iscell_tqm": iscell_tqm,
            "dFoF_params": dFoF_params
        },
        path_save=dir_save / 'trace_quality.pkl'
    )

    np.save(
        file= dir_save / 'iscell_NN_tqm.npy',
        arr=iscell_new
    )