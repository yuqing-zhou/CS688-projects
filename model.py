##############################################
#            Author: Yuqing Zhou
##############################################

import os.path
import wandb
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer, BertConfig, BertForSequenceClassification, BertPreTrainedModel
import numpy as np
import random
from wilds import get_dataset
from wilds.common.data_loaders import get_train_loader, get_eval_loader
from wilds.datasets.wilds_dataset import WILDSSubset
import wilds_exps.transforms as transforms_
from wilds_exps_utils.wilds_configs import datasets as dataset_configs
from types import SimpleNamespace
import torch.nn.functional as F
from gdro_fork.data.confounder_utils import *
from gdro_fork.loss import LossComputer

from myutils import *
import torchvision
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import  SequenceClassifierOutput

from torch.utils.data import Dataset, DataLoader


class TextDataset(Dataset):
    def __init__(self, file_path, transform=None, split=False):
        data_df = pd.read_csv(file_path, encoding='latin1')

        if split == True:
            ind = np.arange(len(data_df))
            rng = np.random.default_rng(0)
            rng.shuffle(ind)
            n_train = int(0.8 * len(ind))
            ind1 = ind[:n_train]
            ind2 = ind[n_train:]
            self.data = data_df[ind1]
        else:
            self.data = data_df

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        label = row['label']
        text = row['text']
        group_id = row['group_ids']

        label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            text = self.transform(text)

        return text, label, group_id



def load_data(file_path, batch_size=32, transform=None, shuffle=True):
    dataset = TextDataset(file_path, transform)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    return dataloader



class BertClassifierWithCovReg(BertPreTrainedModel):
    def __init__(self, model_name, num_labels, feature_size, device, reg, reg_causal=0, disentangle_en=False,
                 counterfactual_en=False, config=None):
        super().__init__(config)
        self.model_device = device
        self.num_labels = num_labels
        # self.feature_size = feature_size


        self.bert = BertModel.from_pretrained(model_name, from_tf=False, config=config)
        # self.linear = nn.Linear(self.bert.config.hidden_size, self.feature_size)
        # self.activation = nn.Tanh()
        self.feature_size = self.bert.config.hidden_size
        self.classifier = nn.Linear(self.feature_size, num_labels)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

        classifier_dropout = (config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob)
        self.dropout = nn.Dropout(classifier_dropout)

        self.mask = ~torch.eye(self.feature_size, dtype=bool).to(device)

        self.reg = reg
        self.disentangle_en = disentangle_en

        self.reg_causal = reg_causal
        self.counterfactual_en = counterfactual_en

    def forward(self, input_ids, attention_mask, labels=None, weights=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        feature = self.dropout(pooled_output)
        # feature = self.linear(pooled_output)
        # feature = self.activation(feature)
        logits = self.classifier(feature)

        total_loss = 0
        causal_regularization = None
        if labels is not None:

            if self.disentangle_en == True:
                covariance = self.compute_covariance(feature)
                regularization = torch.norm(covariance - torch.diag_embed(torch.diagonal(covariance)), p='fro')
                total_loss += self.reg * regularization

            if self.counterfactual_en == True:
                causal_regularization = self.counterfact(feature, labels, logits)
                # total_loss += self.reg_causal * causal_regularization

            # reg = self.classifier.weight.pow(2).sum() + self.classifier.bias.pow(2).sum()
            # total_loss += 2 * reg
            total_loss += self.get_total_loss(logits, labels, weights, causal_regularization)

        return logits, total_loss

    def get_total_loss(self, logits, labels, weights, causal_regularization):
        total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
        if causal_regularization is not None:
            total_loss += self.reg_causal * causal_regularization
        if weights is not None:
            total_loss *= weights

        total_loss = total_loss.mean()  # for BERT
        # total_loss = total_loss.sum() # for AFR

        return total_loss


    def compute_covariance(self, features):
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        features = features - feature_mean
        covariance_matrix = features.T @ features / (features.size(0) - 1)
        return covariance_matrix

    def counterfact(self, feature, labels, logits):
        labels = labels.clone().detach()

        ind = F.one_hot(labels, self.num_labels)
        prob_raw = torch.sum(F.softmax(logits, dim=-1) * ind, 1).clone().detach()
        prob_raw = prob_raw.repeat_interleave(self.feature_size).view(-1)

        feature = feature.view(-1, 1, self.feature_size)
        feature = feature * self.mask
        feature = feature.view(-1, self.feature_size)
        logits_counterfactual = self.classifier(feature)
        labels = labels.repeat_interleave(self.feature_size).view(-1)
        prob_sub = F.softmax(logits_counterfactual, dim=-1)[torch.arange(labels.shape[0]), labels]

        z = prob_raw - prob_sub + 1
        z = torch.where(z > 1, z, torch.tensor(1.0).to(self.model_device)).view(-1, self.feature_size)
        log_cpns = torch.mean(torch.log(z), dim=-1)
        causal_constraints = -log_cpns

        return causal_constraints

    def get_representations(self, input_ids, attention_mask, labels=None, weights=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        feature = pooled_output
        # feature = self.linear(pooled_output)
        # feature = self.activation(feature)
        logits = self.classifier(feature)

        return logits, feature


class Bert(nn.Module):
    def __init__(self, model_name, num_labels, feature_size, device, reg, reg_causal=0, disentangle_en=False,
                 counterfactual_en=False, hidden_dropout_prob=0.1):
        super().__init__()
        self.device = device
        self.num_labels = num_labels

        self.bert = BertModel.from_pretrained(model_name)
        self.feature_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(hidden_dropout_prob)
        self.classifier = nn.Linear(self.feature_size, num_labels)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

        self.mask = ~torch.eye(self.feature_size, dtype=bool).to(device)

        self.reg = reg
        self.disentangle_en = disentangle_en

        self.reg_causal = reg_causal
        self.counterfactual_en = counterfactual_en

    def forward(self, input_ids, attention_mask, labels=None, weights=None, token_type_ids=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        feature = outputs.pooler_output
        feature = self.dropout(feature)
        logits = self.classifier(feature)

        total_loss = 0
        causal_regularization = None
        if labels is not None:

            if self.disentangle_en == True:
                covariance = self.compute_covariance(feature)
                regularization = torch.norm(covariance - torch.diag_embed(torch.diagonal(covariance)), p='fro')
                total_loss += self.reg * regularization

            if self.counterfactual_en == True:
                causal_regularization = self.counterfact(feature, labels, logits)
                # total_loss += self.reg_causal * causal_regularization

            total_loss += self.get_total_loss(logits, labels, weights, causal_regularization)

        return logits, total_loss

    def get_total_loss(self, logits, labels, weights, causal_regularization):
        total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
        if causal_regularization is not None:
            total_loss += self.reg_causal * causal_regularization
        if weights is not None:
            total_loss *= weights

        total_loss = total_loss.mean()

        return total_loss

    # def get_loss(self, logits, labels, weights, causal_regularization):
    #     total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
    #     if weights is not None:
    #         total_loss *= weights
    #         total_loss = total_loss.sum()
    #     else:
    #         total_loss = total_loss.mean()
    #
    #     if causal_regularization is not None:
    #         total_loss += self.reg_causal * causal_regularization.mean()
    #
    #     return total_loss

    def compute_covariance(self, features):
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        features = features - feature_mean
        covariance_matrix = features.T @ features / (features.size(0) - 1)
        return covariance_matrix

    def counterfact(self, feature, labels, logits):
        labels = labels.clone().detach()

        ind = F.one_hot(labels, self.num_labels)
        prob_raw = torch.sum(F.softmax(logits, dim=-1) * ind, 1).clone().detach()
        prob_raw = prob_raw.repeat_interleave(self.feature_size).view(-1)

        feature = feature.view(-1, 1, self.feature_size)
        feature = feature * self.mask
        feature = feature.view(-1, self.feature_size)
        logits_counterfactual = self.classifier(feature)
        labels = labels.repeat_interleave(self.feature_size).view(-1)
        prob_sub = F.softmax(logits_counterfactual, dim=-1)[torch.arange(labels.shape[0]), labels]

        z = prob_raw - prob_sub + 1
        z = torch.where(z > 1, z, torch.tensor(1.0).to(self.device)).view(-1, self.feature_size)
        log_cpns = torch.mean(torch.log(z), dim=-1)
        causal_constraints = -log_cpns

        return causal_constraints


def get_civil_data(task_config, dataset):
    full_dataset = get_dataset(dataset=dataset, download=False, root_dir=task_config.root_dir)
    transform = transforms_.initialize_transform(
        transform_name=task_config.transform,
        config=task_config,
        dataset=full_dataset,
        additional_transform_name=None,
        is_training=False)

    test_data = full_dataset.get_subset("test", transform=transform)
    val_data = full_dataset.get_subset("val", transform=transform)
    train_data = full_dataset.get_subset("train", transform=transform)

    if task_config.dfr_reweighting_drop:
        idx = train_data.indices.copy()
        rng = np.random.default_rng(task_config.dfr_reweighting_seed)
        rng.shuffle(idx)
        n_train = int((1 - task_config.dfr_reweighting_frac) * len(idx))
        train_idx = idx[:n_train]
        val_idx = idx[n_train:]
        train_data = WILDSSubset(
            full_dataset,
            indices=train_idx,
            transform=transform
        )
        reweighting_data = WILDSSubset(
            full_dataset,
            indices=val_idx,
            transform=transform
        )
    else:
        reweighting_data = val_data
    return train_data, val_data, test_data, reweighting_data


def get_mnli_data(args, train=False, return_full_dataset=False):
    train_data, val_data, test_data = prepare_confounder_data(args, train, return_full_dataset)
    if args.dfr_reweighting_drop:
        print(f'Dropping DFR reweighting data, seed {args.dfr_reweighting_seed}')

        idx = train_data.dataset.indices.copy()
        rng = np.random.default_rng(args.dfr_reweighting_seed)
        rng.shuffle(idx)
        n_train = int((1 - args.dfr_reweighting_frac) * len(idx))
        train_idx = idx[:n_train]

        print(f'Original dataset size: {len(train_data.dataset.indices)}')
        train_data.dataset = torch.utils.data.dataset.Subset(
            train_data.dataset.dataset,
            indices=train_idx)
        print(f'New dataset size: {len(train_data.dataset.indices)}')

    return train_data, val_data, test_data


def get_datasets(task_config, dataset, train=False, return_full_dataset=False):
    if dataset == 'civilcomments':
        return get_civil_data(task_config, dataset)

    elif dataset == 'MultiNLI':
        return prepare_confounder_data(task_config, train, return_full_dataset)


def compute_accuracy(logits, labels):
    predictions = torch.argmax(logits, dim=-1)
    correct = (predictions == labels).float()
    accuracy = correct.sum() / len(correct)
    return accuracy


def compute_group_avg(losses, group_idx, n_groups):
    group_map = (group_idx == torch.arange(n_groups).unsqueeze(1).long()).float()
    group_count = group_map.sum(1)
    group_denom = group_count + (group_count == 0).float()  # avoid nans
    group_loss = (group_map @ losses.view(-1)) / group_denom
    return group_loss, group_count


def model_parameters_freeze(model):
    # Freeze layers of the pretrained model except the last linear layer
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        nn.init.normal_(param, mean=0, std=1)
        param.requires_grad = True

    print("\nAfter fixing the layers before the last linear layer:")
    for name, param in model.named_parameters():
        print(name, param.requires_grad)

    return model


def compute_weights(logits, labels, gamma, balance_classes, all_group_ids=None):
    # AFR
    with torch.no_grad():
        p = logits.softmax(-1)
        y_onehot = torch.zeros_like(logits).scatter_(-1, labels.unsqueeze(-1), 1)
        p_true = (p * y_onehot).sum(-1)
        weights = (-gamma * p_true).exp()
        n_classes = torch.unique(labels).numel()
        if balance_classes:
            if n_classes == 2:
                w1 = (labels == 0).sum()
                w2 = (labels == 1).sum()
                weights[labels == 0] *= w2 / w1
            else:
                class_count = []
                for y in range(n_classes):
                    class_count.append((labels == y).sum())
                for y in range(1, n_classes):
                    weights[labels == y] *= class_count[0] / class_count[y]
        weights = weights.detach()
        weights /= weights.sum()

    return weights

def compute_weights1(logits, labels, gamma, balance_classes, all_group_ids=None):
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct_predictions = predictions == labels
        n_classes = torch.unique(labels).numel()
        correct_counts = torch.zeros(n_classes, dtype=torch.int32)
        incorrect_counts = torch.zeros(n_classes, dtype=torch.int32)
        total_counts = len(labels)
        weights = torch.ones_like(labels, dtype=torch.float32)

        for i in range(n_classes):
            correct_counts[i] = (correct_predictions & (labels == i)).sum()
            incorrect_counts[i] = (~correct_predictions & (labels == i)).sum()
        for i in range(n_classes):
            print(f"Class {i}: Correct: {correct_counts[i]}, Incorrect: {incorrect_counts[i]}")

        correct_ratios = correct_counts.float() / total_counts
        incorrect_ratios = incorrect_counts.float() / total_counts
        correct_weights = torch.where(correct_counts > 0, 1.0 / correct_ratios, torch.tensor(1.0))
        incorrect_weights = torch.where(incorrect_counts > 0, 1.0 / incorrect_ratios, torch.tensor(1.0))

        for i in range(n_classes):
            weights[(labels == i) & correct_predictions] = correct_weights[i]
            weights[(labels == i) & (~correct_predictions)] = incorrect_weights[i]


    return weights


def compute_weights2(logits, labels, gamma, balance_classes, all_group_ids=None):
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct_predictions = predictions == labels
        n_classes = torch.unique(labels).numel()
        correct_counts = torch.zeros(n_classes, dtype=torch.int32)
        incorrect_counts = torch.zeros(n_classes, dtype=torch.int32)
        total_counts = len(labels)
        weights = torch.ones_like(labels, dtype=torch.float32)

        for i in range(n_classes):
            correct_counts[i] = (correct_predictions & (labels == i)).sum()
            incorrect_counts[i] = (~correct_predictions & (labels == i)).sum()
        for i in range(n_classes):
            print(f"Class {i}: Correct: {correct_counts[i]}, Incorrect: {incorrect_counts[i]}")

        correct_ratios = correct_counts.float() / (correct_counts + incorrect_counts)
        incorrect_ratios = incorrect_counts.float() / (correct_counts + incorrect_counts)
        correct_weights = torch.where(correct_counts > 0, 1.0 / correct_ratios, torch.tensor(1.0))
        incorrect_weights = torch.where(incorrect_counts > 0, 1.0 / incorrect_ratios, torch.tensor(1.0))

        for i in range(n_classes):
            weights[(labels == i) & correct_predictions] = correct_weights[i]
            weights[(labels == i) & (~correct_predictions)] = incorrect_weights[i]

    return weights


def compute_weights3(logits, labels, gamma, balance_classes, all_group_ids):
    with torch.no_grad():
        group_ids, group_sizes = torch.unique(all_group_ids, return_counts=True)
        total_counts = len(labels)
        group_ratios = group_sizes/total_counts
        weights = torch.ones_like(labels, dtype=torch.float32)

        for i, group_id in enumerate(group_ids):
            weights[(all_group_ids == group_id)] = 1.0 / group_ratios[i]

    return weights

def compute_weights4(logits, labels, gamma, balance_classes=None, all_group_ids=None):
    # JTT
    with torch.no_grad():
        predictions = torch.argmax(logits, dim=1)
        correct_predictions = predictions == labels
        n_classes = torch.unique(labels).numel()

        weights = torch.ones_like(labels, dtype=torch.float32)

        for i in range(n_classes):
            weights[(labels == i) & correct_predictions] = 1
            weights[(labels == i) & (~correct_predictions)] = gamma

    return weights


def evaluation(model, dataset, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_predictions, all_y_true, all_metadata, all_logits = [], [], [], []
        for batch in dataloader:
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            labels = batch[1].to(device)
            metadata = batch[2]

            logits, test_loss = model(input_ids, attention_mask, labels)
            predictions = torch.argmax(logits, axis=1)

            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            all_y_true.append(labels.cpu())
            all_metadata.append(metadata.cpu())

        all_logits = torch.cat(all_logits, axis=0)
        all_predictions = torch.cat(all_predictions, axis=0)
        all_y_true = torch.cat(all_y_true, axis=0)
        all_metadata = torch.cat(all_metadata, axis=0)

        total_loss = model.crossEntropyLoss(all_logits.view(-1, model.num_labels), all_y_true.view(-1)).mean()
        total_accuracy = compute_accuracy(all_logits, all_y_true)
        results = dataset.eval(all_predictions.cpu(), all_y_true.cpu(), all_metadata.cpu())

    return results, total_loss, total_accuracy


def evaluation_nli(model, dataset_n_groups, dataloader, device):
    model.eval()
    with torch.no_grad():
        all_predictions, all_y_true, all_group_idx, all_losses, all_logits = [], [], [], [], []
        for batch in dataloader:
            input_ids = batch[0][:, :, 0].to(device)
            attention_mask = batch[0][:, :, 1].to(device)
            labels = batch[1].to(device)
            group_ids = batch[2]

            logits, test_loss = model(input_ids, attention_mask, labels)
            loss = model.crossEntropyLoss(logits.view(-1, model.num_labels), labels.view(-1))
            predictions = torch.argmax(logits, axis=1)

            all_logits.append(logits.cpu())
            all_predictions.append(predictions.cpu())
            all_y_true.append(labels.cpu())
            all_group_idx.append(group_ids.cpu())
            all_losses.append(loss.cpu())


        all_logits = torch.cat(all_logits, axis=0)
        all_predictions = torch.cat(all_predictions, axis=0)
        all_y_true = torch.cat(all_y_true, axis=0)
        all_group_idx = torch.cat(all_group_idx, axis=0)
        all_losses = torch.cat(all_losses, axis=0)

        total_loss = all_losses.mean()
        total_accuracy = compute_accuracy(all_logits, all_y_true)

        group_acc, _ = compute_group_avg((all_predictions == all_y_true).float(), all_group_idx, dataset_n_groups)
        group_losses, _ = compute_group_avg(all_losses, all_group_idx, dataset_n_groups)

    return group_acc, total_loss, total_accuracy, group_losses


def get_data_loader(dataset_name, data, task_config, train=True, loader_type="standard", **loader_kwargs):
    data_loader = None
    if dataset_name == 'civilcomments':
        if train == True:
            data_loader = get_train_loader(loader_type, data, batch_size=task_config.batch_size,
                                           uniform_over_groups=False)
        else:
            data_loader = get_eval_loader(loader_type, data, batch_size=task_config.batch_size)

    elif dataset_name == 'MultiNLI':
        if data is not None:
            if train == True:
                data_loader = data.get_loader(train=train, reweight_groups=task_config.reweight_groups, **loader_kwargs)
            else:
                data_loader = data.get_loader(train=train, reweight_groups=None, **loader_kwargs)

    return data_loader


class ResNet50withCovReg(nn.Module):
    def __init__(self, reduce_dim, output_dim, device, reg=0, reg_causal=0, disentangle_en=False,
                 counterfactual_en=False):
        super(ResNet50withCovReg, self).__init__()
        self.device = device
        self.num_labels = output_dim
        # self.feature_size = reduce_dim

        self.resnet50 = torchvision.models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, output_dim)
        self.feature_size = self.resnet50.fc.in_features
        # self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, reduce_dim)
        # self.activation = nn.Tanh()
        # self.classifier = nn.Linear(reduce_dim, output_dim)
        self.crossEntropyLoss = nn.CrossEntropyLoss(reduction='none')

        self.mask = ~torch.eye(self.feature_size, dtype=bool).to(device)

        self.reg = reg
        self.disentangle_en = disentangle_en

        self.reg_causal = reg_causal
        self.counterfactual_en = counterfactual_en

    def forward(self, x, labels=None, weights=None):
        feature = self.resnet50(x)
        # feature = self.activation(feature)
        # logits = self.classifier(feature)
        logits = feature

        total_loss = 0
        causal_regularization = None
        if labels is not None:

            if self.disentangle_en == True:
                regularization = self.compute_disentangle_loss(feature)
                total_loss += self.reg * regularization

            if self.counterfactual_en == True:
                causal_regularization = self.counterfact(feature, labels, logits)
                # total_loss += self.reg_causal * causal_regularization

            total_loss += self.get_total_loss(logits, labels, weights, causal_regularization)

        return logits, total_loss

    def get_total_loss(self, logits, labels, weights, causal_regularization):
        total_loss = self.crossEntropyLoss(logits.view(-1, self.num_labels), labels.view(-1))
        if causal_regularization is not None:
            total_loss += self.reg_causal * causal_regularization
        if weights is not None:
            total_loss *= weights

        total_loss = total_loss.mean() # For resnet
        # total_loss = total_loss.sum() # For AFR

        return total_loss

    def compute_disentangle_loss(self, features):
        feature_mean = torch.mean(features, dim=0, keepdim=True)
        centered_features = features - feature_mean
        covariance_matrix = centered_features.T @ centered_features / (centered_features.size(0) - 1)
        disentangle_loss = torch.norm(covariance_matrix - torch.diag_embed(torch.diagonal(covariance_matrix)), p='fro')
        return disentangle_loss

    def counterfact(self, feature, labels, logits):
        labels = labels.clone().detach()

        ind = F.one_hot(labels, self.num_labels)
        prob_raw = torch.sum(F.softmax(logits, dim=-1) * ind, 1).clone().detach()
        prob_raw = prob_raw.repeat_interleave(self.feature_size).view(-1)

        feature = feature.view(-1, 1, self.feature_size)
        feature = feature * self.mask
        feature = feature.view(-1, self.feature_size)
        logits_counterfactual = self.classifier(feature)
        labels = labels.repeat_interleave(self.feature_size).view(-1)
        prob_sub = F.softmax(logits_counterfactual, dim=-1)[torch.arange(labels.shape[0]), labels]

        z = prob_raw - prob_sub + 1
        z = torch.where(z > 1, z, torch.tensor(1.0).to(self.device)).view(-1, self.feature_size)
        log_cpns = torch.mean(torch.log(z), dim=-1)
        causal_constraints = -log_cpns

        return causal_constraints

