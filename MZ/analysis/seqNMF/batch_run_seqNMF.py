import os
import sys
import argparse
from pathlib import Path

import numpy as np
import copy
import pickle
import natsort
import logging

# from Mothership_Zeta.MZ.analysis.seqNMF import seqnmf
# from Mothership_Zeta.MZ.utils.basic_neural_processing_modules import file_helpers,torch_helpers,timeSeries,ca2p_preprocessing,welford_moving_2D,linear_regression,similarity
# from Mothership_Zeta.MZ.utils.Big_Ugly_ROI_Tracker.multiEps.multiEps_modules import *

## Main
def seqNMF_cascade(path_list, kwargs, reference_session, cascade):
    if cascade:
        seqnmf_result, within_session_list = seqNMF_single(path_list, kwargs, reference_session, cascade)
        seqNMF_daughter(path_list, seqnmf_result, within_session_list, kwargs, reference_session)
    else:
        seqNMF_single(path_list, kwargs, reference_session)

def seqNMF_daughter(path_list, seqnmf_result, within_session_list, kwargs, reference_session):
    subordinate_path = copy.deepcopy(path_list)
    for subordinate in within_session_list:
        if str(subordinate) not in str(path_list.dir_s2p):
            subordinate_path.dir_s2p = subordinate
            seqNMF_single(path_list,
            kwargs,
            reference_session = subordinate,
            cascade = True,
            W_init = seqnmf_result['W'])


def seqNMF_single(path_list, kwargs, reference_session = None, cascade = False, W_init = None, H_init = None):
    sys.path.append(str(path_list.dir_github))
    # sys.path.append(str(path_list.dir_analysis))

    dir_s2p = path_list.dir_s2p
    logging.warning(f"dir_s2p {dir_s2p}")
    if reference_session is None:
        logging.warning(f"No reference session, solo que")
    else:
        logging.warning(f"Cascade running: {reference_session}")
        if str(reference_session) not in str(dir_s2p):
            logging.warning(f"Not a reference session. Stop seqNMF for this condition")
            sys.exit()


    from analysis.seqNMF import seqnmf
    from tracking.ROICaT import ROICaT_util
    from utils.basic_neural_processing_modules import file_helpers,torch_helpers,timeSeries,ca2p_preprocessing,welford_moving_2D,linear_regression,similarity
    from utils.Big_Ugly_ROI_Tracker.multiEps.multiEps_modules import import_and_convert_to_CellReg_spatialFootprints

    DEVICE = torch_helpers.set_device(use_GPU=True)

    path_iscell = path_list.dir_s2p / 'iscell_NN_tqm.npy'
    path_tqm = path_list.dir_s2p / 'trace_quality.pkl'

    iscell = np.load(path_iscell)
    dFoF_params = file_helpers.pickle_load(path_tqm)['dFoF_params']

    ## == IMPORT DATA ==
    logging.warning("Importing Data...")
    F = np.load(dir_s2p / 'F.npy') # masks multiplied by data
    Fneu = np.load(dir_s2p / 'Fneu.npy') # estimated neuropil signal (Fns = F - Fneu; Fo = ptile(Fns, 30); dFoF=Fns-Fo/Fo)
    ops = np.load(dir_s2p / 'ops.npy', allow_pickle=True) # parameters for the suite2p
    spks_s2p = np.load(dir_s2p / 'spks.npy') # blind deconvolution
    stat = np.load(dir_s2p / 'stat.npy', allow_pickle=True) # statistics for individual neurons 

    num_frames_S2p = F.shape[1]
    Fs = ops[()]['fs']

    frame_height = ops[()]['meanImg'].shape[0]
    frame_width = ops[()]['meanImg'].shape[1]

    sf = import_and_convert_to_CellReg_spatialFootprints([dir_s2p / 'stat.npy'], frame_height=frame_height, frame_width=frame_width, dtype=np.float32)[0]

    if cascade:
        # ## Get session date and mouse name
        # session_index = [parts.isdigit() for parts in dir_s2p.parts]
        # session_date_parts = np.nonzero(session_index)[0][0]
        # session_date = dir_s2p.parts[session_date_parts]
        # logging.warning(f"Session date {session_date}")
        # name_save = dir_s2p.parts[np.nonzero(session_index)[0][0]-1]
        # ## Load ROICaT
        # logging.warning("Cascade running")
        # logging.warning("Loading ROICaT result...")
        # dir_roicat = dir_s2p.parents[len(dir_s2p.parts) - session_date_parts - 1]
        # tracker_path = dir_roicat / (name_save + '.ROICaT.results' + '.pkl')
        # logging.warning(tracker_path)
        # with open(tracker_path, "rb") as handle:
        #     tracker = pickle.load(handle)

        # ## Load same-day sessions
        # logging.warning(f"Loading day {session_date} UCIDs")
        # roicat_bool = [session_date in str(track) for track in tracker["Paths"]]
        # roi_session_list, within_session_list = [], []
        # for index, track in enumerate(tracker["Paths"]):
        #     if session_date in str(track):
        #         within_session_list.append(track)
        #         roi_session_list.append(tracker["UCIDs_bySession"][index])

        # UCIDs, UCIDs_counts = np.unique(np.concatenate(roi_session_list), return_counts=True)
        # ## Retrieve rois tracked across a day, full sessions
        # tracked_UCIDs = UCIDs[UCIDs_counts==len(roi_session_list)]
        # roi_istracked = []
        # for roi_session in roi_session_list:
        #     tracked = [roi in tracked_UCIDs for roi in roi_session]
        #     roi_istracked.append(tracked)

        # ## Boolean of iscell
        # reference_session_bool = [str(reference_session) in str(track) for track in within_session_list]

        roi_istracked, reference_session_bool = ROICaT_util.ROICaT_loader(dir_s2p, reference_session, same_day = True, track_reference = None)
        logging.warning(f"Num of ROIs before tracking {np.sum(iscell)}")
        iscell = np.multiply(iscell,roi_istracked[np.nonzero(reference_session_bool)[0][0]])
        logging.warning(f"Num of ROIs after tracking {np.sum(iscell)}")
    else:
        logging.warning(f"Num of ROIs {len(iscell)}")


    F_toUse = F[iscell]
    Fneu_toUse = Fneu[iscell]

    win_smooth = 4
    kernel_smoothing = np.zeros(win_smooth*2)
    kernel_smoothing[win_smooth:] = 1
    kernel_smoothing /= kernel_smoothing.sum()

    # Pipeline for the NMF Strategy 
    # Smooth F
    logging.warning(f"Smoothing calcium trace")
    F_smooth = timeSeries.convolve_along_axis(F_toUse,
    kernel=kernel_smoothing,
    axis=1,
    mode='same',
    multicore_pref=True,
    verbose=True).astype(np.float32)
    # logging.warning(F_smooth.shape)

    # dFoF with reduced percentile for baseline
    channelOffset_correction = 500
    percentile_baseline = 5
    neuropil_fraction=0.7

    logging.warning(f"Calculating dFoF")
    dFoF , dF , F_neuSub , F_baseline = ca2p_preprocessing.make_dFoF(F=F_smooth + channelOffset_correction,
    Fneu=Fneu_toUse + channelOffset_correction,
    neuropil_fraction=neuropil_fraction,
    percentile_baseline=percentile_baseline,
    multicore_pref=True,
    verbose=True)

    # # Threshold for nonnegativity
    # dFoF_z = dFoF / np.std(dFoF,axis=1,keepdims=True)

    # Test out rolling subtraction of the 10th percentile of the data to remove microscope movement artifacts
    ptile = 10
    window = int(Fs*60*1)

    # dFoF_sub_ptile = dFoF - timeSeries.rolling_percentile_pd(dFoF, ptile=ptile, window=window)
    dFoF_sub_ptile = dFoF - timeSeries.rolling_percentile_rq_multicore(dFoF, ptile=ptile, window=window)
    dFoF_sub_ptile_clipped = np.clip(dFoF_sub_ptile, a_min=0, a_max=None)

    # neural_data_toUse = (dFoF_sub_ptile / np.std(dFoF_sub_ptile,axis=1,keepdims=True))
    neural_data_toUse = (dFoF_sub_ptile_clipped / np.std(dFoF_sub_ptile,axis=1,keepdims=True))


    # Run SeqNMF
    logging.warning("Run seqNMF...")
    if W_init is None:
        logging.warning(f"No W initialization")
    else:
        logging.warning(f"W_init {W_init.shape}")

    logging.warning(kwargs)

    W, H, cost, loadings, power = seqnmf.seqnmf(X = neural_data_toUse,
    W_init = W_init,
    H_init = H_init,
    tol = -np.inf,
    M = None,
    **kwargs)

    seqnmf_result = {"W":W, "H":H, "cost":cost, "loadings":loadings, "power":power, "params":kwargs}

    saver_name = f"{kwargs['K']}_{kwargs['Lambda']}_{kwargs['L']}_seqNMF_factors.p"
    seqnmf_saver = path_list.dir_s2p / saver_name
    logging.warning("Saving seqNMF factors...")
    logging.warning(seqnmf_saver)

    with open(seqnmf_saver, 'wb') as handle:
        pickle.dump(seqnmf_result,handle)

    return seqnmf_result, within_session_list