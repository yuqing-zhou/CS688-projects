##############################################
#            Author: Yuqing Zhou
##############################################

import random
import numpy as np
import torch

DATASET = 'civilcomments'

DATASET_INFO = {
    'civilcomments': {
        'num_classes': 2,
        'dataset_path': '',
        'transform': None,
        'model': 'bert-base-uncased',
    },
    'MultiNLI': {
        'num_classes': 3,
        'dataset_path': 'multinli_bert_features/',
        'transform': None,
        'model': 'bert-base-uncased',
    },
    'waterbirds': {
        'num_classes': 2,
        'dataset_path': 'datasets/waterbird_complete95_forest2water2/',
        'transform': 'AugWaterbirdsCelebATransform',
        'model': 'imagenet_resnet50_pretrained',
    },
    'celeba': {
        'num_classes': 2,
        'dataset_path': 'datasets/celebA/',
        'transform': 'AugWaterbirdsCelebATransform',
        'model': 'imagenet_resnet50_pretrained',
    },
}

seeds = [42]

def set_seed(seed_value=42):
    random.seed(seed_value)  # Python random module
    np.random.seed(seed_value)  # Numpy module
    torch.manual_seed(seed_value)  # PyTorch
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def wandb_project_name(dataset, bert_version, cpns_version, reweight_version):
    project_name = f"CS688-robust-learning-{dataset}"
    return project_name


def wandb_exp_name(DATASET_NAME, bert_version, cpns_version, reweight_version, n_exp, lr, feature_size, reg_disentangle, reg_causal, gamma, weight_decay=0):
    # name = f"{DATASET_NAME}_{n_exp}_disentangle_cov_reg_{bert_version}_{cpns_version}_{reweight_version}_{feature_size}_{lr}_{reg_disentangle}_{reg_causal}_{gamma}_{weight_decay}"
    name = f"{DATASET_NAME}_{n_exp}-AFR-bert_{bert_version}_{cpns_version}_{reweight_version}_{feature_size}_{lr}_{reg_disentangle}_{reg_causal}_{gamma}_{weight_decay}"
    return name