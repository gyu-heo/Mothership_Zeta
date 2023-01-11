import os
import sys
import numpy as np
import logging
import pickle

def ROICaT_loader(dir_s2p, reference_session, same_day = False, track_reference = None):
    """_summary_

    Args:
        dir_s2p (_type_): _description_
        same_day (bool, optional): _description_. Defaults to False.
        track_reference (_type_, optional): _description_. Defaults to None. If int, track cells based on "int"-index session.

    Returns:
        _type_: _description_
    """
    session_index = [parts.isdigit() for parts in dir_s2p.parts]
    session_date_parts = np.nonzero(session_index)[0][0]
    session_date = dir_s2p.parts[session_date_parts]
    logging.warning(f"Session date {session_date}")
    name_save = dir_s2p.parts[np.nonzero(session_index)[0][0]-1]
    ## Load ROICaT
    logging.warning("Cascade running")
    logging.warning("Loading ROICaT result...")
    dir_roicat = dir_s2p.parents[len(dir_s2p.parts) - session_date_parts - 1]
    tracker_path = dir_roicat / (name_save + '.ROICaT.results' + '.pkl')
    logging.warning(tracker_path)
    with open(tracker_path, "rb") as handle:
        tracker = pickle.load(handle)

    ## Load sessions
    roi_session_list, within_session_list = [], []
    for index, track in enumerate(tracker["Paths"]):
        if same_day:
            # Load same-day sessions
            logging.warning(f"Loading day {session_date} UCIDs")
            if session_date in str(track):
                within_session_list.append(track)
                roi_session_list.append(tracker["UCIDs_bySession"][index])
        else:
            # Full load
            logging.warning(f"Loading full UCIDs")
            within_session_list.append(track)
            roi_session_list.append(tracker["UCIDs_bySession"][index])

    logging.warning(f"Track_reference {track_reference}")
    if track_reference is None:
        UCIDs, UCIDs_counts = np.unique(np.concatenate(roi_session_list), return_counts=True)
        ## Retrieve rois tracked across a day, full sessions
        tracked_UCIDs = UCIDs[UCIDs_counts==len(roi_session_list)]
        roi_istracked = []
        for roi_session in roi_session_list:
            tracked = [roi in tracked_UCIDs for roi in roi_session]
            roi_istracked.append(tracked)
        ## Boolean of tracked cells
        reference_session_bool = [str(reference_session) in str(track) for track in within_session_list]
        return roi_istracked, reference_session_bool

    elif isinstance(track_reference, int):
        ref_UCID = roi_session_list[track_reference]
        tracked_UCIDs = []
        for index, this_session in enumerate(roi_session_list):
            if index == track_reference:
                tracked_UCIDs.append(list(range(len(this_session))))
            else:
                mapper = []
                ts = list(this_session)
                for roi in ref_UCID:
                    try:
                        mapper.append(ts.index(roi))
                    except ValueError:
                        mapper.append(-1)
                tracked_UCIDs.append(mapper)
        return tracked_UCIDs