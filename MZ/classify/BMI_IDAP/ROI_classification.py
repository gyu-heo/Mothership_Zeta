from pathlib import Path
import sys
import importlib

import numpy as np
import copy
import torch

import logging
import pickle
import matplotlib.pyplot as plt

## basic functions
def dataloader_to_latents(dataloader, model, DEVICE='cpu'):
    def subset_to_latents(data):
        return model.get_head(model.base_model(data[0][0].to(DEVICE))).detach().cpu()
    return torch.cat([subset_to_latents(data) for data in dataloader], dim=0)

def load_classifier_model(classifier_name):
    with open(classifier_name, 'rb') as classifier_model_file:
        classifier = pickle.load(classifier_model_file)
    return classifier

## Main
def ROI_classification(path_list):
    logging.warning(f'ROI classification')

    logging.warning(f'Add {path_list.dir_github}')
    sys.path.append(str(path_list.dir_github))
    logging.warning(f'Add {path_list.dir_classify}')
    sys.path.append(str(path_list.dir_classify))
    # sys.path.append(path_list.dir_GRC_repo)
    logging.warning(f'Add {path_list.dir_NNmodels}')
    sys.path.append(str(path_list.dir_NNmodels))

    from GCaMP_ROI_classifier.new_stuff import util
    from utils.basic_neural_processing_modules import torch_helpers, plotting_helpers, file_helpers


    ## Device to use for NN model
    DEVICE = torch_helpers.set_device(use_GPU=True)

    ## Import suite2p stat file to spatial footprints
    logging.warning(f'Load Spatial Footprints')
    spatial_footprints = torch.as_tensor(
        util.statFile_to_spatialFootprints(path_list.path_statFile, out_height_width=[36,36], max_footprint_width=455)
    )

    spatial_footprints = spatial_footprints / torch.sum(spatial_footprints, dim=(1,2), keepdim=True)

    # spatial_footprints = drop_nan_imgs(spatial_footprints)
    print(spatial_footprints.shape[0], 'ROIs loaded.')

    # Instantiate Model
    # model_file = importlib.util.spec_from_file_location('path_NN_py')
    logging.warning(f'Instantiate Model')
    model_file = importlib.import_module(path_list.fileName_NN_py)
    model = model_file.get_model(path_list.path_NN_pth)
    model.eval()

    # Create Data Sets / Data Loaders
    logging.warning(f'Create Data Sets / Data Loaders')
    dataset, dataloader = model_file.get_dataset_dataloader(spatial_footprints, batch_size=64, device=DEVICE)

    model.to(DEVICE)

    # Get Model Latents
    logging.warning(f'Get Model Latents')
    latents = dataloader_to_latents(dataloader, model, DEVICE=DEVICE).numpy()

    # Load Logistic Model
    classifier_model = load_classifier_model(path_list.path_classifier)

    # Predict ROIs — Save to File
    proba = classifier_model.predict_proba(latents)
    preds = np.argmax(proba, axis=-1)
    uncertainty = util.loss_uncertainty(torch.as_tensor(proba), temperature=1, class_value=None).detach().cpu().numpy()
        
    params = classifier_model.get_params()

    ROI_classifier_outputs = {
        'latents': latents,
        'proba': proba,
        'preds': preds,
        'uncertainty': uncertainty,
        'LR_params': params
    }

    preds_toUse = [0,1]
    logging.warning(f'Classified as ROI: {preds_toUse}')
    iscell_NN = np.isin(preds, preds_toUse)
    iscell_NN_idx = np.where(iscell_NN)[0]

    logging.warning(f'number of included ROIs: {len(iscell_NN_idx)}')

    logging.warning(f'Saving file: {path_list.dir_save / "iscell_NN.npy"}')
    np.save(
        file=path_list.dir_save / 'iscell_NN.npy',
        arr=iscell_NN
    )

    # pickle_helpers.simple_save(
    #     obj=ROI_classifier_outputs,
    #     filename=dir_save / 'ROI_classifier_outputs.pkl'
    # )
    logging.warning(f'Saving file: {path_list.dir_save / "ROI_classifier_outputs.pkl"}')
    file_helpers.pickle_save(
        obj=ROI_classifier_outputs,
        path_save=path_list.dir_save / 'ROI_classifier_outputs.pkl'
    )

    return iscell_NN