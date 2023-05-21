import numpy as np
import time
from sklearn.metrics import auc
from sklearn.model_selection import KFold
from scipy import interpolate
import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from datasets.LFWDataset import LFWDataset

import pickle

# torch.multiprocessing.set_start_method('spawn')

from datasets.CCDataset import CCDataset
from losses.triplet_loss import TripletLoss
from datasets.TripletLossDataset import TripletFaceDataset
# from validate_on_LFW import evaluate_lfw
from plot import plot_roc_lfw, plot_accuracy_lfw
from tqdm import tqdm
from models.inceptionresnetv2 import InceptionResnetV2Triplet
from models.inceptionresnetv1 import InceptionResnetV1Triplet
from models.mobilenetv2 import MobileNetV2Triplet
from models.resnet import (
    Resnet18Triplet,
    Resnet34Triplet,
    Resnet50Triplet,
    Resnet101Triplet,
    Resnet152Triplet
)


""""-------------------------------------------------------MODEL LOADING START----------------------------------------------"""

model_architecture = "inceptionresnetv1"
model = InceptionResnetV1Triplet(
            embedding_dimension=512,
            pretrained="vggface2")
class InceptionResnetV1Triplet(nn.Module):
    """Constructs an Inception-ResNet-V2 model for FaceNet training using triplet loss.

    Args:
        embedding_dimension (int): Required dimension of the resulting embedding layer that is outputted by the model.
                                    using triplet loss. Defaults to 512.
        pretrained (bool): If True, returns a model pre-trained on the ImageNet dataset from a PyTorch repository.
                            Defaults to False.
    """
    def __init__(self, embedding_dimension=512, pretrained=False):
        super(InceptionResnetV1Triplet, self).__init__()
        if pretrained:
            self.model = inceptionresnetv1(pretrained='vggface2')
        else:
            self.model = inceptionresnetv1(pretrained=pretrained)

        # Output embedding
        self.model.last_linear = nn.Linear(1792, embedding_dimension, bias=False)

    def forward(self, images):
        """Forward pass to output the embedding vector (feature vector) after l2-normalization."""
        embedding = self.model(images)
        # From: https://github.com/timesler/facenet-pytorch/blob/master/models/inception_resnet_v1.py#L301
        embedding = F.normalize(embedding, p=2, dim=1)

        return embedding
    import os
import requests
from requests.adapters import HTTPAdapter

import torch
from torch import nn
from torch.nn import functional as F



class BasicConv2d(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_planes, out_planes,
            kernel_size=kernel_size, stride=stride,
            padding=padding, bias=False
        ) # verify bias false
        self.bn = nn.BatchNorm2d(
            out_planes,
            eps=0.001, # value found in tensorflow
            momentum=0.1, # default pytorch value
            affine=True
        )
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1)
        )

        self.conv2d = nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(nn.Module):

    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1,7), stride=1, padding=(0,3)),
            BasicConv2d(128, 128, kernel_size=(7,1), stride=1, padding=(3,0))
        )

        self.conv2d = nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(nn.Module):

    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1,3), stride=1, padding=(0,1)),
            BasicConv2d(192, 192, kernel_size=(3,1), stride=1, padding=(1,0))
        )

        self.conv2d = nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(nn.Module):

    def __init__(self):
        super().__init__()

        self.branch0 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2)
        )

        self.branch1 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch2 = nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2)
        )

        self.branch3 = nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class inceptionresnetv1(nn.Module):
    """Inception Resnet V1 model with optional loading of pretrained weights.
    Model parameters can be loaded based on pretraining on the VGGFace2 or CASIA-Webface
    datasets. Pretrained state_dicts are automatically downloaded on model instantiation if
    requested and cached in the torch cache. Subsequent instantiations use the cache rather than
    redownloading.
    Keyword Arguments:
        pretrained {str} -- Optional pretraining dataset. Either 'vggface2' or 'casia-webface'.
            (default: {None})
        classify {bool} -- Whether the model should output classification probabilities or feature
            embeddings. (default: {False})
        num_classes {int} -- Number of output classes. If 'pretrained' is set and num_classes not
            equal to that used for the pretrained model, the final linear layer will be randomly
            initialized. (default: {None})
        dropout_prob {float} -- Dropout probability. (default: {0.6})
    """
    def __init__(self, pretrained=None, classify=False, num_classes=None, dropout_prob=0.6, device=None):
        super().__init__()

        # Set simple attributes
        self.pretrained = pretrained
        self.classify = classify
        self.num_classes = num_classes

        if pretrained == 'vggface2':
            tmp_classes = 8631
        elif pretrained == 'casia-webface':
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception('If "pretrained" is not specified and "classify" is True, "num_classes" must be specified')


        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout_prob)
        self.last_linear = nn.Linear(1792, 512, bias=False)
        self.last_bn = nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)

        if pretrained is not None:
            self.logits = nn.Linear(512, tmp_classes)
            load_weights(self, pretrained)

        if self.classify and self.num_classes is not None:
            self.logits = nn.Linear(512, self.num_classes)

        self.device = torch.device('cpu')
        if device is not None:
            self.device = device
            self.to(device)

    def forward(self, x):
        """Calculate embeddings or logits given a batch of input image tensors.
        Arguments:
            x {torch.tensor} -- Batch of image tensors representing faces.
        Returns:
            torch.tensor -- Batch of embedding vectors or multinomial logits.
        """
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.last_linear(x.view(x.shape[0], -1))
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            x = F.normalize(x, p=2, dim=1)
        return x


def load_weights(mdl, name):
    """Download pretrained state_dict and load into model.
    Arguments:
        mdl {torch.nn.Module} -- Pytorch model.
        name {str} -- Name of dataset that was used to generate pretrained state_dict.
    Raises:
        ValueError: If 'pretrained' not equal to 'vggface2' or 'casia-webface'.
    """
    if name == 'vggface2':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180402-114759-vggface2.pt'
    elif name == 'casia-webface':
        path = 'https://github.com/timesler/facenet-pytorch/releases/download/v2.2.9/20180408-102900-casia-webface.pt'
    else:
        raise ValueError('Pretrained models only exist for "vggface2" and "casia-webface"')

    model_dir = os.path.join(get_torch_home(), 'checkpoints')
    os.makedirs(model_dir, exist_ok=True)

    cached_file = os.path.join(model_dir, os.path.basename(path))
    if not os.path.exists(cached_file):
        download_url_to_file(path, cached_file)

    state_dict = torch.load(cached_file)
    mdl.load_state_dict(state_dict)


def get_torch_home():
    torch_home = os.path.expanduser(
        os.getenv(
            'TORCH_HOME',
            os.path.join(os.getenv('XDG_CACHE_HOME', '~/.cache'), 'torch')
        )
    )
    return torch_home

model = InceptionResnetV1Triplet(pretrained="vggface2")

checkpoint = torch.load('./model_pretrained_triplet_epoch_0.pt', map_location='cuda')

model_state_dic = checkpoint['model_state_dict']

model.load_state_dict(model_state_dic)

model = nn.DataParallel(model)
model.cuda()

""""-------------------------------------------------------MODEL LOADING END----------------------------------------------"""




"""-------------------------------------------------------Evaluate LFW----------------------------------------------------"""
def evaluate_lfw_sf(distances, labels, far_target=1e-3):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    distance as a metric.
    Note: "TAR@FAR=0.001" means the rate that faces are successfully accepted (True Acceptance Rate) (TP/(TP+FN)) when
    the rate that faces are incorrectly accepted (False Acceptance Rate) (FP/(TN+FP)) is 0.001 (The less the FAR value
    the mode difficult it is for the model). i.e: 'What is the True Positive Rate of the model when only one false image
    in 1000 images is allowed?'.
        https://github.com/davidsandberg/facenet/issues/288#issuecomment-305961018
    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.
        far_target (float): The False Acceptance Rate to calculate the True Acceptance Rate (TAR) at,
                             defaults to 1e-3.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        precision: Array of precision values per each fold in cross validation set.
        recall: Array of recall values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (AUROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tar: Array that contains True Acceptance Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
        far: Array that contains False Acceptance Rate values per each fold in cross validation set.
    """

    # Calculate ROC metrics
    thresholds_roc = np.arange(0, 3.0, 0.001)
    true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy = \
        calculate_roc_values_sf(
            thresholds=thresholds_roc, distances=distances, labels=labels
        )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc


def calculate_roc_values_sf(thresholds, distances, labels):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)

    true_positive_rates = np.zeros((num_thresholds))
    false_positive_rates = np.zeros((num_thresholds))
    false_negative_rates = np.zeros((num_thresholds))

    test_set = np.arange(num_pairs)
    
    print(test_set)
    

        # Test on test set using the best distance threshold
    for threshold_index, threshold in enumerate(thresholds):
        print("threshold_index: ",threshold_index)
        print("threshold: ", threshold)
        true_positive_rates[threshold_index], false_positive_rates[threshold_index], false_negative_rates[threshold_index], _, _,\
            _ = calculate_metrics_sf(
                threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set])

    _, _, _, precision, recall, accuracy = calculate_metrics_sf(
        threshold=thresholds[threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
    )


    return true_positive_rates, false_positive_rates, false_negative_rates, precision, recall, accuracy


def calculate_metrics_sf(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
        
    false_negative_rate = 0 if (false_negatives + true_positives == 0) else \
        float(false_negatives) / float(false_negatives + true_positives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy







def evaluate_lfw(distances, labels, num_folds=5, far_target=1e-3):
    """Evaluates on the Labeled Faces in the Wild dataset using KFold cross validation based on the Euclidean
    distance as a metric.
    Note: "TAR@FAR=0.001" means the rate that faces are successfully accepted (True Acceptance Rate) (TP/(TP+FN)) when
    the rate that faces are incorrectly accepted (False Acceptance Rate) (FP/(TN+FP)) is 0.001 (The less the FAR value
    the mode difficult it is for the model). i.e: 'What is the True Positive Rate of the model when only one false image
    in 1000 images is allowed?'.
        https://github.com/davidsandberg/facenet/issues/288#issuecomment-305961018
    Args:
        distances: numpy array of the pairwise distances calculated from the LFW pairs.
        labels: numpy array containing the correct result of the LFW pairs belonging to the same identity or not.
        num_folds (int): Number of folds for KFold cross-validation, defaults to 10 folds.
        far_target (float): The False Acceptance Rate to calculate the True Acceptance Rate (TAR) at,
                             defaults to 1e-3.
    Returns:
        true_positive_rate: Mean value of all true positive rates across all cross validation folds for plotting
                             the Receiver operating characteristic (ROC) curve.
        false_positive_rate: Mean value of all false positive rates across all cross validation folds for plotting
                              the Receiver operating characteristic (ROC) curve.
        accuracy: Array of accuracy values per each fold in cross validation set.
        precision: Array of precision values per each fold in cross validation set.
        recall: Array of recall values per each fold in cross validation set.
        roc_auc: Area Under the Receiver operating characteristic (AUROC) metric.
        best_distances: Array of Euclidean distance values that had the best performing accuracy on the LFW dataset
                         per each fold in cross validation set.
        tar: Array that contains True Acceptance Rate values per each fold in cross validation set
              when far (False Accept Rate) is set to a specific value.
        far: Array that contains False Acceptance Rate values per each fold in cross validation set.
    """

    # Calculate ROC metrics
    thresholds_roc = np.arange(0, 3.0, 0.001)
    true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, best_distances = \
        calculate_roc_values(
            thresholds=thresholds_roc, distances=distances, labels=labels, num_folds=num_folds
        )

    roc_auc = auc(false_positive_rate, true_positive_rate)

    #Calculate validation rate
    thresholds_val = np.arange(0, 3, 0.001)
    tar, far = calculate_val(
        thresholds_val=thresholds_val, distances=distances, labels=labels, far_target=far_target, num_folds=num_folds
    )

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc, best_distances, tar, far


def calculate_roc_values(thresholds, distances, labels, num_folds=2):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    true_positive_rates = np.zeros((num_folds, num_thresholds))
    false_positive_rates = np.zeros((num_folds, num_thresholds))
    false_negative_rates = np.zeros((num_folds, num_thresholds))
    precision = np.zeros(num_folds)
    recall = np.zeros(num_folds)
    accuracy = np.zeros(num_folds)
    best_distances = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the best distance threshold for the k-fold cross validation using the train set
        accuracies_trainset = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds):
            # print(threshold)
            _, _, _, _,_, accuracies_trainset[threshold_index] = calculate_metrics(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        best_threshold_index = np.argmax(accuracies_trainset)

        # Test on test set using the best distance threshold
        for threshold_index, threshold in enumerate(thresholds):
            true_positive_rates[fold_index, threshold_index], false_positive_rates[fold_index, threshold_index], false_negative_rates[fold_index, threshold_index], _, _,\
                _ = calculate_metrics(
                    threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
                )

        _, _, _, precision[fold_index], recall[fold_index], accuracy[fold_index] = calculate_metrics(
            threshold=thresholds[best_threshold_index], dist=distances[test_set], actual_issame=labels[test_set]
        )

        true_positive_rate = np.mean(true_positive_rates, 0)
        false_positive_rate = np.mean(false_positive_rates, 0)
        false_negative_rate = np.mean(false_negative_rates, 0)
        best_distances[fold_index] = thresholds[best_threshold_index]

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, best_distances


def calculate_metrics(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_positives = np.sum(np.logical_and(predict_issame, actual_issame))
    false_positives = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))
    true_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame)))
    false_negatives = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame))

    # For dealing with Divide By Zero exception
    true_positive_rate = 0 if (true_positives + false_negatives == 0) else \
        float(true_positives) / float(true_positives + false_negatives)

    false_positive_rate = 0 if (false_positives + true_negatives == 0) else \
        float(false_positives) / float(false_positives + true_negatives)
        
    false_negative_rate = 0 if (false_negatives + true_positives == 0) else \
        float(false_negatives) / float(false_negatives + true_positives)

    precision = 0 if (true_positives + false_positives) == 0 else\
        float(true_positives) / float(true_positives + false_positives)

    recall = 0 if (true_positives + false_negatives) == 0 else \
        float(true_positives) / float(true_positives + false_negatives)

    accuracy = float(true_positives + true_negatives) / dist.size

    return true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy


def calculate_val(thresholds_val, distances, labels, far_target=1e-3, num_folds=10):
    num_pairs = min(len(labels), len(distances))
    num_thresholds = len(thresholds_val)
    k_fold = KFold(n_splits=num_folds, shuffle=False)

    tar = np.zeros(num_folds)
    far = np.zeros(num_folds)

    indices = np.arange(num_pairs)

    for fold_index, (train_set, test_set) in enumerate(k_fold.split(indices)):
        # Find the euclidean distance threshold that gives false acceptance rate (far) = far_target
        far_train = np.zeros(num_thresholds)
        for threshold_index, threshold in enumerate(thresholds_val):
            _, far_train[threshold_index] = calculate_val_far(
                threshold=threshold, dist=distances[train_set], actual_issame=labels[train_set]
            )
        if np.max(far_train) >= far_target:
            f = interpolate.interp1d(far_train, thresholds_val, kind='slinear')
            threshold = f(far_target)
        else:
            threshold = 0.0

        tar[fold_index], far[fold_index] = calculate_val_far(
            threshold=threshold, dist=distances[test_set], actual_issame=labels[test_set]
        )

    return tar, far


def calculate_val_far(threshold, dist, actual_issame):
    # If distance is less than threshold, then prediction is set to True
    predict_issame = np.less(dist, threshold)

    true_accept = np.sum(np.logical_and(predict_issame, actual_issame))
    false_accept = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame)))

    num_same = np.sum(actual_issame)
    num_diff = np.sum(np.logical_not(actual_issame))

    if num_diff == 0:
        num_diff = 1
    if num_same == 0:
        return 0, 0

    tar = float(true_accept) / float(num_same)
    far = float(false_accept) / float(num_diff)

    return tar, far























""""-------------------------------------------------------VALIDATION----------------------------------------------"""

def predict_ijb(model, cc_dataloader, model_architecture, epoch, logfname, ts):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        print("Testing on IJB! ...")
        progress_bar = enumerate(tqdm(cc_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.to('cuda', non_blocking=True)
            data_b = data_b.to('cuda', non_blocking=True)
#             print('is cuda:',data_a.is_cuda)
            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())
    return distances ,labels
    


def validate_ijb(labels, distances, model_architecture, epoch, logfname, ts, fold_type):

    if fold_type == 'sf' or fold_type == 'both':
        true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc = \
        evaluate_lfw_sf(
            distances=distances,
            labels=labels,
            far_target=1e-3
        )
        tpr_1e3 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-03))]
        tpr_1e4 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-04))]
        fpr_95 = false_positive_rate[np.argmin(np.abs(true_positive_rate - 0.95))]
        fnr = false_negative_rate
        fpr = false_positive_rate
        print("fnr",fnr,"fpr",fpr)
        sub = np.abs(fnr - fpr)
        h = np.min(sub[np.nonzero(sub)])
        h = np.where(sub == h)[0][0]
        
        print("------------------------------------------------Single fold---------------------------------------")
 
        # Print statistics and add to log
        print("Accuracy on CC: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\t".format(
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc
                )
        )
	

        print('fpr at tpr 0.95: {},  tpr at fpr 0.001: {}, tpr at fpr 0.0001: {}'.format(fpr_95,tpr_1e3,tpr_1e4))
        print('At FNR = FPR: FNR = {}, FPR = {}'.format(fnr[h],fpr[h]))
# with open('logs/cc_tpr_fpr_{}_{}.txt'.format(logfname, ts), 'a') as f:
#             f.writelines(''.format()
	
        with open('logs/cc_{}_log_triplet_{}.txt'.format(model_architecture,ts), 'a') as f:

            f.writelines("--------------------------single fold-------------------------------"
              "Epoch {}: {}: Accuracy on CC: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\t"
              "fpr at tpr 0.95: {},  tpr at fpr 0.001: {} | At FNR = FPR: FNR = {}, FPR = {}".format(
                    epoch,
                    logfname,
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    fpr_95,
                    tpr_1e3,
                    fnr[h],
                    fpr[h]
                ) + '\n'
            )

    try:
        # Plot cc curve
        plot_roc_lfw(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name="plots/roc_plots/roc_cc_{}_epoch_{}_triplet_{}_{}.png".format(model_architecture, epoch, logfname,ts)
        )
        # Plot cc accuracies plot
#         plot_accuracy_lfw(
#             log_file='logs/cc_{}_log_triplet_{}_{}.txt'.format(model_architecture,logfname,ts),
#             epochs=epoch,
#             figure_name="plots/accuracies_plots/cc_accuracies_{}_epoch_{}_triplet_{}_{}.png".format(model_architecture, epoch, logfname,ts)
#         )
    except Exception as e:
        print(e)

        
    if fold_type == 'mf' or fold_type == 'both':
    
        true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw_mf(
            distances=distances,
            labels=labels,
            far_target=1e-3
        )
        tpr_1e3 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-03))]
        tpr_1e4 = true_positive_rate[np.argmin(np.abs(false_positive_rate - 1e-04))]
        fpr_95 = false_positive_rate[np.argmin(np.abs(true_positive_rate - 0.95))]
        fnr = false_negative_rate
        fpr = false_positive_rate
        sub = np.abs(fnr - fpr)
        h = np.min(sub[np.nonzero(sub)])
        h = np.where(sub == h)[0][0]


        print("------------------------------------------------multiple fold---------------------------------------")
        
        # Print statistics and add to log
        print("Accuracy on IJB: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
              "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f}".format(
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar),
                    np.std(tar),
                    np.mean(far)
                )
        )


        print('fpr at tpr 0.95: {},  tpr at fpr 0.001: {}, tpr at fpr 0.0001: {}'.format(fpr_95,tpr_1e3,tpr_1e4))
        print('At FNR = FPR: FNR = {}, FPR = {}'.format(fnr[h],fpr[h]))
    # with open('logs/cc_tpr_fpr_{}_{}.txt'.format(logfname, ts), 'a') as f:
    #             f.writelines(''.format()

        with open('logs/IJB_{}_log_triplet_{}.txt'.format(model_architecture,ts), 'a') as f:

            f.writelines("-----------------------multi-fold---------------------"
              "Epoch {}: {}: Accuracy on IJB: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
              "ROC Area Under Curve: {:.4f}\tBest distance threshold: {:.2f}+-{:.2f}\t"
              "TAR: {:.4f}+-{:.4f} @ FAR: {:.4f} \n fpr at tpr 0.95: {},  tpr at fpr 0.001: {} | At FNR = FPR: FNR = {}, FPR = {}".format(
                    epoch,
                    logfname,
                    np.mean(accuracy),
                    np.std(accuracy),
                    np.mean(precision),
                    np.std(precision),
                    np.mean(recall),
                    np.std(recall),
                    roc_auc,
                    np.mean(best_distances),
                    np.std(best_distances),
                    np.mean(tar),
                    np.std(tar),
                    np.mean(far),
                    fpr_95,
                    tpr_1e3,
                    fnr[h],
                    fpr[h]
                ) + '\n'
            )

        try:
            # Plot cc curve
            plot_roc_lfw(
                false_positive_rate=false_positive_rate,
                true_positive_rate=true_positive_rate,
                figure_name="plots/roc_plots/roc_IJB_{}_epoch_{}_triplet_{}_{}.png".format(model_architecture, epoch, logfname,ts)
            )
            # Plot cc accuracies plot
    #         plot_accuracy_lfw(
    #             log_file='logs/cc_{}_log_triplet_{}_{}.txt'.format(model_architecture,logfname,ts),
    #             epochs=epoch,
    #             figure_name="plots/accuracies_plots/cc_accuracies_{}_epoch_{}_triplet_{}_{}.png".format(model_architecture, epoch, logfname,ts)
    #         )
        except Exception as e:
            print(e)



"""This module was imported from liorshk's 'facenet_pytorch' github repository:
        https://github.com/liorshk/facenet_pytorch/blob/master/LFWDataset.py

    It was modified to support lfw .png files for loading by using the code here:
        https://github.com/davidsandberg/facenet/blob/master/src/lfw.py#L46
"""

"""MIT License

Copyright (c) 2017 liorshk

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""



import torchvision.datasets as datasets
import os
import numpy as np
# from pynvml import *

class CCDataset(datasets.ImageFolder):
    def __init__(self, dir, pairs_path, transform=None):

        super(CCDataset, self).__init__(dir, transform)

        self.pairs_path = pairs_path

        # LFW dir contains 2 folders: faces and lists
        self.validation_images = self.get_cc_paths(dir)

    def read_cc_pairs(self, pairs_filename):
        pairs = []
        with open(pairs_filename, 'r') as f:
            for line in f.readlines()[1:]:
                pair = line.strip().split()
                pairs.append(pair)

        return np.array(pairs, dtype=object)

    def get_cc_paths(self, cc_dir):
        pairs = self.read_cc_pairs(self.pairs_path)

        nrof_skipped_pairs = 0
        path_list = []
        issame_list = []
        for pair in pairs:
            if len(pair) == 3:
                path0 = self.add_extension(os.path.join(cc_dir, pair[0], pair[1]))
                #print('path0:',path0)
                path1 = self.add_extension(os.path.join(cc_dir, pair[0], pair[2]))
                #print('path1:',path1)
                issame = True
            elif len(pair) == 4:
                path0 = self.add_extension(os.path.join(cc_dir, pair[0],pair[1]))
                path1 = self.add_extension(os.path.join(cc_dir, pair[2],pair[3]))
                issame = False
            if os.path.exists(path0) and os.path.exists(path1):  # Only add the pair if both paths exist
                path_list.append((path0, path1, issame))
                issame_list.append(issame)
            else:
                nrof_skipped_pairs += 1
        if nrof_skipped_pairs > 0:
            print('Skipped %d image pairs' % nrof_skipped_pairs)

        return path_list

    # Modified here
    def add_extension(self, path):
        if os.path.exists(path + '.jpg'):
            return path + '.jpg'
        elif os.path.exists(path + '.png'):
            return path + '.png'
        else:
            raise RuntimeError('No file "%s" with extension png or jpg.' % path)

    def __getitem__(self, index):
        """
        Args:
            index: Index of the triplet or the matches - not of a single image
        Returns:
        """

        def transform(img_path):
            """Convert image into numpy array and apply transformation
               Doing this so that it is consistent with all other datasets
               to return a PIL Image.
            """

            img = self.loader(img_path)
            return self.transform(img)

        (path_1, path_2, issame) = self.validation_images[index]
        img1, img2 = transform(path_1), transform(path_2)
        return img1, img2, issame

    def __len__(self):
        return len(self.validation_images)




def main():
    torch.multiprocessing.set_sharing_strategy('file_system')
#     torch.cuda.empty_cache()
    ts = time.time()
    num_workers=4
    lfw_batch_size = 200
    cc_dataroot = './datasets/casual_conv'
    
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y

    cc_transforms = transforms.Compose([
    transforms.Resize((160,160)),
    np.float32,
    transforms.Lambda(prewhiten),
    transforms.ToTensor()
    ])
    
    
#     ijb_dataloader_12 = torch.utils.data.DataLoader(
#         dataset=IJBDataset(
#             dir='datasets/faster_preproc',
#             pairs_path='datasets/IJB_pairs_12.txt',
#             transform=None,
#             preproc=True
#         ),
#         batch_size=2000,
#         num_workers=4,
#         shuffle=False,
#         pin_memory=True
#     )
    


    RFW_dataloader_full = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir='./datasets/rfw/test_aligned/',
            pairs_path='datasets/rfw/test_aligned/pairs_full.txt',
            transform=cc_transforms
        ),
        batch_size=1000,
        num_workers=4,
        shuffle=False,
        pin_memory=False
    )
    
    RFW_dataloader_asian = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir='./datasets/rfw/test_aligned/Asian',
            pairs_path='datasets/rfw/test_aligned/pairs_asian_parallel.txt',
            transform=cc_transforms
        ),
        batch_size=1000,
        num_workers=4,
        shuffle=False,
        pin_memory=False
    )

    RFW_dataloader_caucasian = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir='./datasets/rfw/test_aligned/Caucasian',
            pairs_path='datasets/rfw/test_aligned/pairs_caucasian_parallel.txt',
            transform=cc_transforms
        ),
        batch_size=1000,
        num_workers=4,
        shuffle=False,
        pin_memory=False
    )

    RFW_dataloader_indian = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir='./datasets/rfw/test_aligned/Indian',
            pairs_path='datasets/rfw/test_aligned/pairs_indian_parallel.txt',
            transform=cc_transforms
        ),
        batch_size=1000,
        num_workers=4,
        shuffle=False,
        pin_memory=False
    )
    
    RFW_dataloader_african = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir='./datasets/rfw/test_aligned/African',
            pairs_path='datasets/rfw/test_aligned/pairs_african_parallel.txt',
            transform=cc_transforms
        ),
        batch_size=1000,
        num_workers=4,
        shuffle=False,
        pin_memory=False
    )


    
#     ijb_dataloader_34 = torch.utils.data.DataLoader(
#         dataset=IJBDataset(
#             dir='datasets/faster_preproc',
#             pairs_path='datasets/IJB_pairs_34.txt',
#             transform=None,
#             preproc=True
#         ),
#         batch_size=2000,
#         num_workers=4,
#         shuffle=False,
#         pin_memory=True
#     )
    
    def ID(x):
        for i in range(len(x)):
            if '/' == x[i]:
                idx = i
        return idx+1
    
    epoch = 0
    
    logfname_full = RFW_dataloader_full.dataset.pairs_path[ID(RFW_dataloader_full.dataset.pairs_path):][:-4]
    
    
    distances_full, labels_full = predict_ijb(model=model, cc_dataloader=RFW_dataloader_full, model_architecture=model_architecture, epoch=epoch, logfname=logfname_full, ts=ts)
    
    with open('dist_full.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([distances_full, labels_full], f)
        
    logfname_asian = RFW_dataloader_asian.dataset.pairs_path[ID(RFW_dataloader_asian.dataset.pairs_path):][:-4]
    
    
    distances_asian, labels_asian = predict_ijb(model=model, cc_dataloader=RFW_dataloader_asian, model_architecture=model_architecture, epoch=epoch, logfname=logfname_asian, ts=ts)
    
    with open('dist_asian.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([distances_asian, labels_asian], f)
    
    logfname_indian = RFW_dataloader_indian.dataset.pairs_path[ID(RFW_dataloader_indian.dataset.pairs_path):][:-4]
    
    
    distances_indian, labels_indian = predict_ijb(model=model, cc_dataloader=RFW_dataloader_indian, model_architecture=model_architecture, epoch=epoch, logfname=logfname_indian, ts=ts)
    
    with open('dist_indian.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([distances_indian, labels_indian], f)
    

    logfname_caucasian = RFW_dataloader_caucasian.dataset.pairs_path[ID(RFW_dataloader_caucasian.dataset.pairs_path):][:-4]
    
    
    distances_caucasian, labels_caucasian = predict_ijb(model=model, cc_dataloader=RFW_dataloader_caucasian, model_architecture=model_architecture, epoch=epoch, logfname=logfname_caucasian, ts=ts)
    
    with open('dist_caucasian.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([distances_caucasian, labels_caucasian], f)
        
    logfname_african = RFW_dataloader_african.dataset.pairs_path[ID(RFW_dataloader_african.dataset.pairs_path):][:-4]
    
    
    distances_african, labels_african = predict_ijb(model=model, cc_dataloader=RFW_dataloader_african, model_architecture=model_architecture, epoch=epoch, logfname=logfname_african, ts=ts)
    
    with open('dist_african.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([distances_african, labels_african], f)

    


    # obj0, obj1, obj2 are created here...

    # Saving the objects:
#     with open('dist.pkl', 'w') as f:  # Python 3: open(..., 'wb')
#         pickle.dump([distances_ijb_12, labels_ijb_12,labels_ijb_56, labels_ijb_56], f)
        
#     with open('dist.pkl') as f:  # Python 3: open(..., 'rb')
#         distances_ijb_12, labels_ijb_12,labels_ijb_56, labels_ijb_56 = pickle.load(f)

#     def prewhiten(x):
#         mean = np.mean(x)
#         std = np.std(x)
#         std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
#         y = np.multiply(np.subtract(x, mean), 1/std_adj)
#         return y

#     cc_transforms = transforms.Compose([
#     np.float32,
#     transforms.Lambda(prewhiten),
#     transforms.ToTensor()
#     ])
    
#     cc_dataloader_1 = torch.utils.data.DataLoader(
#         dataset=CCDataset(
#             dir=cc_dataroot,
#             pairs_path='datasets/VAL_pairs_1.txt',
#             transform=cc_transforms
#         ),
#         batch_size=lfw_batch_size,
#         num_workers=num_workers,
#         shuffle=False
#     )
    
#     cc_dataloader_2 = torch.utils.data.DataLoader(
#         dataset=CCDataset(
#             dir=cc_dataroot,
#             pairs_path='datasets/VAL_pairs_2.txt',
#             transform=cc_transforms
#         ),
#         batch_size=lfw_batch_size,
#         num_workers=num_workers,
#         shuffle=False
#     )
    
#     cc_dataloader_3 = torch.utils.data.DataLoader(
#         dataset=CCDataset(
#             dir=cc_dataroot,
#             pairs_path='datasets/VAL_pairs_3.txt',
#             transform=cc_transforms
#         ),
#         batch_size=lfw_batch_size,
#         num_workers=num_workers,
#         shuffle=False
#     )
    
#     cc_dataloader_4 = torch.utils.data.DataLoader(
#         dataset=CCDataset(
#             dir=cc_dataroot,
#             pairs_path='datasets/VAL_pairs_4.txt',
#             transform=cc_transforms
#         ),
#         batch_size=lfw_batch_size,
#         num_workers=num_workers,
#         shuffle=False
#     )

#     cc_dataloader_5 = torch.utils.data.DataLoader(
#         dataset=CCDataset(
#             dir=cc_dataroot,
#             pairs_path='datasets/VAL_pairs_5.txt',
#             transform=cc_transforms
#         ),
#         batch_size=lfw_batch_size,
#         num_workers=num_workers,
#         shuffle=False
#     )

#     cc_dataloader_6 = torch.utils.data.DataLoader(
#         dataset=CCDataset(
#             dir=cc_dataroot,
#             pairs_path='datasets/VAL_pairs_6.txt',
#             transform=cc_transforms
#         ),
#         batch_size=lfw_batch_size,
#         num_workers=num_workers,
#         shuffle=False
#     )
    
# , distances, model_architecture, epoch, logfname, ts, fold_type    
    
    
    
#     _, _, _, _ = validate_ijb(
#                 labels=labels_ijb_12,
#                 distances=distances_ijb_12,
#                 model_architecture=model_architecture,
#                 epoch=0,
#                 logfname=logfname_12,
#                 ts=ts,
#                 fold_type = "both"
#             )
    
#     _, _, _, _ = validate_ijb(
#                 labels=labels_ijb_56,
#                 distances=distances_ijb_56,
#                 model_architecture=model_architecture,
#                 epoch=0,
#                 logfname=logfname_56,
#                 ts=ts,
#                 fold_type = "both"
#             )
    
#     logfname_1 = cc_dataloader_1.dataset.pairs_path[ID(cc_dataloader_1.dataset.pairs_path):][:-4]
    
#     logfname_2 = cc_dataloader_2.dataset.pairs_path[ID(cc_dataloader_2.dataset.pairs_path):][:-4]
    
#     logfname_3 = cc_dataloader_3.dataset.pairs_path[ID(cc_dataloader_3.dataset.pairs_path):][:-4]
    
#     logfname_4 = cc_dataloader_4.dataset.pairs_path[ID(cc_dataloader_4.dataset.pairs_path):][:-4]

#     logfname_5 = cc_dataloader_5.dataset.pairs_path[ID(cc_dataloader_5.dataset.pairs_path):][:-4]

#     logfname_6 = cc_dataloader_6.dataset.pairs_path[ID(cc_dataloader_6.dataset.pairs_path):][:-4]
    
#     tpr_1e3_1, tpr_1e4_1, fpr_95_1, best_distances = validate_cc(
#             model=model,
#             cc_dataloader=cc_dataloader_1,
#             model_architecture=model_architecture,
#             epoch=0,
#             logfname=logfname_1,
#             ts=ts
#         )
    
#     tpr_1e3_2, tpr_1e4_2, fpr_95_2, best_distances = validate_cc(
#             model=model,
#             cc_dataloader=cc_dataloader_2,
#             model_architecture=model_architecture,
#             epoch=0,
#             logfname=logfname_2,
#             ts=ts
#         )
    
#     tpr_1e3_3, tpr_1e4_3, fpr_95_3, best_distances = validate_cc(
#             model=model,
#             cc_dataloader=cc_dataloader_3,
#             model_architecture=model_architecture,
#             epoch=0,
#             logfname=logfname_3,
#             ts=ts
#         )

#     tpr_1e3_4, tpr_1e4_4, fpr_95_4, best_distances = validate_cc(
#             model=model,
#             cc_dataloader=cc_dataloader_4,
#             model_architecture=model_architecture,
#             epoch=0,
#             logfname=logfname_4,
#             ts=ts
#         )

#     tpr_1e3_5, tpr_1e4_5, fpr_95_5, best_distances = validate_cc(
#             model=model,
#             cc_dataloader=cc_dataloader_5,
#             model_architecture=model_architecture,
#             epoch=0,
#             logfname=logfname_5,
#             ts=ts
#         )
    
#     tpr_1e3_6, tpr_1e4_6, fpr_95_6, best_distances = validate_cc(
#             model=model,
#             cc_dataloader=cc_dataloader_6,
#             model_architecture=model_architecture,
#             epoch=0,
#             logfname=logfname_6,
#             ts=ts
#         )
    
    
if __name__ == '__main__':
    main()
