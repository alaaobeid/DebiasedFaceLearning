import numpy as np
import argparse
import os
import torch
import torch.nn as nn
import time
import torch.optim as optim
import torchvision.transforms as transforms
from torch.nn.modules.distance import PairwiseDistance
from datasets.LFWDataset import LFWDataset
from datasets.CCDataset import CCDataset
from datasets.IJBDataset import IJBDataset
from losses.triplet_loss import TripletLoss
from datasets.TripletLossDataset import TripletFaceDataset
from validate_on_LFW import evaluate_lfw
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


parser = argparse.ArgumentParser(description="Training a FaceNet facial recognition model using Triplet Loss.")
parser.add_argument('--dataroot', '-d', type=str, required=True,
                    help="(REQUIRED) Absolute path to the training dataset folder"
                    )
parser.add_argument('--lfw', type=str, required=True,
                    help="(REQUIRED) Absolute path to the labeled faces in the wild dataset folder"
                    )
parser.add_argument('--cc', type=str, required=True,
                    help="(REQUIRED) Absolute path to the cc dataset folder"
                    )
parser.add_argument('--training_dataset_csv_path', type=str, default='datasets/glint360k.csv',
                    help="Path to the csv file containing the image paths of the training dataset"
                    )
parser.add_argument('--epochs', default=150, type=int,
                    help="Required training epochs (default: 150)"
                    )
parser.add_argument('--iterations_per_epoch', default=5000, type=int,
                    help="Number of training iterations per epoch (default: 5000)"
                    )
parser.add_argument('--model_architecture', type=str, default="resnet34", choices=["resnet18", "resnet34", "resnet50", "resnet101", "resnet152", "inceptionresnetv2", "mobilenetv2","inceptionresnetv1"],
                    help="The required model architecture for training: ('resnet18','resnet34', 'resnet50', 'resnet101', 'resnet152', 'inceptionresnetv2', 'mobilenetv2', 'inceptionresnetv1'), (default: 'resnet34')"
                    )
parser.add_argument('--pretrained', default=False, type=bool,
                    help="Download a model pretrained on the ImageNet dataset (Default: False)"
                    )
parser.add_argument('--embedding_dimension', default=512, type=int,
                    help="Dimension of the embedding vector (default: 512)"
                    )
parser.add_argument('--num_human_identities_per_batch', default=32, type=int,
                    help="Number of set human identities per generated triplets batch. (Default: 32)."
                    )
parser.add_argument('--batch_size', default=544, type=int,
                    help="Batch size (default: 544)"
                    )
parser.add_argument('--lfw_batch_size', default=200, type=int,
                    help="Batch size for LFW dataset (6000 pairs) (default: 200)"
                    )
parser.add_argument('--cc_batch_size', default=200, type=int,
                    help="Batch size for LFW dataset (6000 pairs) (default: 200)"
                    )
parser.add_argument('--resume_path', default='',  type=str,
                    help='path to latest model checkpoint: (model_training_checkpoints/model_resnet34_epoch_1.pt file) (default: None)'
                    )
parser.add_argument('--num_workers', default=4, type=int,
                    help="Number of workers for data loaders (default: 4)"
                    )
parser.add_argument('--optimizer', type=str, default="adagrad", choices=["sgd", "adagrad", "rmsprop", "adam"],
                    help="Required optimizer for training the model: ('sgd','adagrad','rmsprop','adam'), (default: 'adagrad')"
                    )
parser.add_argument('--learning_rate', default=0.075, type=float,
                    help="Learning rate for the optimizer (default: 0.075)"
                    )
parser.add_argument('--margin', default=0.2, type=float,
                    help='margin for triplet loss (default: 0.2)'
                    )
parser.add_argument('--image_size', default=140, type=int,
                    help='Input image size (default: 140 (140x140))'
                    )
parser.add_argument('--use_semihard_negatives', default=False, type=bool,
                    help="If True: use semihard negative triplet selection. Else: use hard negative triplet selection (Default: False)"
                    )
parser.add_argument('--training_triplets_path', default=None, type=str,
                    help="Path to training triplets numpy file in 'datasets/generated_triplets' folder to skip training triplet generation step for the first epoch."
                    )
args = parser.parse_args()


def set_model_architecture(model_architecture, pretrained, embedding_dimension):
    if model_architecture == "resnet18":
        model = Resnet18Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet34":
        model = Resnet34Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet50":
        model = Resnet50Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet101":
        model = Resnet101Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "resnet152":
        model = Resnet152Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv2":
        model = InceptionResnetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "mobilenetv2":
        model = MobileNetV2Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained
        )
    elif model_architecture == "inceptionresnetv1":
        model = InceptionResnetV1Triplet(
            embedding_dimension=embedding_dimension,
            pretrained=pretrained)
    print("Using {} model architecture.".format(model_architecture))

    return model


def set_model_gpu_mode(model):
    flag_train_gpu = torch.cuda.is_available()
    flag_train_multi_gpu = False

    if flag_train_gpu and torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        model.cuda()
        flag_train_multi_gpu = True
        print('Using multi-gpu training.')

    elif flag_train_gpu and torch.cuda.device_count() == 1:
        model.cuda()
        print('Using single-gpu training.')

    return model, flag_train_multi_gpu


def set_optimizer(optimizer, model, learning_rate):
    if optimizer == "sgd":
        optimizer_model = optim.SGD(
            params=model.parameters(),
            lr=learning_rate,
            momentum=0.9,
            dampening=0,
            nesterov=False,
            weight_decay=1e-5
        )

    elif optimizer == "adagrad":
        optimizer_model = optim.Adagrad(
            params=model.parameters(),
            lr=learning_rate,
            lr_decay=0,
            initial_accumulator_value=0.1,
            eps=1e-10,
            weight_decay=1e-5
        )

    elif optimizer == "rmsprop":
        optimizer_model = optim.RMSprop(
            params=model.parameters(),
            lr=learning_rate,
            alpha=0.99,
            eps=1e-08,
            momentum=0,
            centered=False,
            weight_decay=1e-5
        )

    elif optimizer == "adam":
        optimizer_model = optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            amsgrad=False,
            weight_decay=1e-5
        )

    return optimizer_model


def validate_lfw(model, lfw_dataloader, model_architecture, epoch):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        print("Validating on LFW! ...")
        progress_bar = enumerate(tqdm(lfw_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.cuda()
            data_b = data_b.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
            distances=distances,
            labels=labels,
            far_target=1e-3
        )
        # Print statistics and add to log
        print("Accuracy on LFW: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
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
        with open('logs/lfw_{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch,
                np.mean(accuracy),
                np.std(accuracy),
                np.mean(precision),
                np.std(precision),
                np.mean(recall),
                np.std(recall),
                roc_auc,
                np.mean(best_distances),
                np.std(best_distances),
                np.mean(tar)
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

    try:
        # Plot ROC curve
        plot_roc_lfw(
            false_positive_rate=false_positive_rate,
            true_positive_rate=true_positive_rate,
            figure_name="plots/roc_plots/roc_{}_epoch_{}_triplet.png".format(model_architecture, epoch)
        )
        # Plot LFW accuracies plot
        plot_accuracy_lfw(
            log_file="logs/lfw_{}_log_triplet.txt".format(model_architecture),
            epochs=epoch,
            figure_name="plots/accuracies_plots/lfw_accuracies_{}_epoch_{}_triplet.png".format(model_architecture, epoch)
        )
    except Exception as e:
        print(e)

    return best_distances


def validate_cc(model, cc_dataloader, model_architecture, epoch, logfname, ts):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        print("Validating on CC! ...")
        progress_bar = enumerate(tqdm(cc_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.cuda()
            data_b = data_b.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
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
	
 
        # Print statistics and add to log
        print("Accuracy on CC: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
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
	
        with open('logs/cc_{}_log_triplet_{}.txt'.format(model_architecture,ts), 'a') as f:

            f.writelines("Epoch {}: {}: Accuracy on CC: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
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

    return tpr_1e3,tpr_1e4,fpr_95,best_distances

def validate_ijb(model, cc_dataloader, model_architecture, epoch, logfname, ts):
    model.eval()
    with torch.no_grad():
        l2_distance = PairwiseDistance(p=2)
        distances, labels = [], []

        print("Testing on IJB! ...")
        progress_bar = enumerate(tqdm(cc_dataloader))

        for batch_index, (data_a, data_b, label) in progress_bar:
            data_a = data_a.cuda()
            data_b = data_b.cuda()

            output_a, output_b = model(data_a), model(data_b)
            distance = l2_distance.forward(output_a, output_b)  # Euclidean distance

            distances.append(distance.cpu().detach().numpy())
            labels.append(label.cpu().detach().numpy())

        labels = np.array([sublabel for label in labels for sublabel in label])
        distances = np.array([subdist for distance in distances for subdist in distance])

        true_positive_rate, false_positive_rate, false_negative_rate, precision, recall, accuracy, roc_auc, best_distances, \
        tar, far = evaluate_lfw(
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

            f.writelines("Epoch {}: {}: Accuracy on IJB: {:.4f}+-{:.4f}\tPrecision {:.4f}+-{:.4f}\tRecall {:.4f}+-{:.4f}\t"
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

    return tpr_1e3,tpr_1e4,fpr_95,best_distances


def forward_pass(imgs, model, batch_size):
    imgs = imgs.cuda()
    embeddings = model(imgs)

    # Split the embeddings into Anchor, Positive, and Negative embeddings
    embeddings = embeddings
    embeddings = embeddings.detach()

    return embeddings, model


def main():
    dataroot = args.dataroot
    lfw_dataroot = args.lfw
    cc_dataroot = args.cc
    training_dataset_csv_path = args.training_dataset_csv_path
    epochs = args.epochs
    iterations_per_epoch = args.iterations_per_epoch
    model_architecture = args.model_architecture
    pretrained = args.pretrained
    embedding_dimension = args.embedding_dimension
    num_human_identities_per_batch = args.num_human_identities_per_batch
    batch_size = args.batch_size
    lfw_batch_size = args.lfw_batch_size
    cc_batch_size = args.lfw_batch_size
    resume_path = args.resume_path
    num_workers = args.num_workers
    optimizer = args.optimizer
    learning_rate = args.learning_rate
    margin = args.margin
    image_size = args.image_size
    use_semihard_negatives = args.use_semihard_negatives
    training_triplets_path = args.training_triplets_path
    flag_training_triplets_path = False
    ts = time.time()
    start_epoch = 0    


    if training_triplets_path is not None:
        flag_training_triplets_path = True  # Load triplets file for the first training epoch

    # Define image data pre-processing transforms
    #   ToTensor() normalizes pixel values between [0, 1]
    #   Normalize(mean=[0.6071, 0.4609, 0.3944], std=[0.2457, 0.2175, 0.2129]) according to the calculated glint360k
    #   dataset with tightly-cropped faces dataset RGB channels' mean and std values by
    #   calculate_glint360k_rgb_mean_std.py in 'datasets' folder.
    
    def fixed_image_standardization(image_tensor):
        processed_tensor = (image_tensor - 127.5) / 128.0
        return processed_tensor
    
    def prewhiten(x):
        mean = np.mean(x)
        std = np.std(x)
        std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
        y = np.multiply(np.subtract(x, mean), 1/std_adj)
        return y
    
    # data_transforms = transforms.Compose([
    #     transforms.Resize(size=image_size),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.RandomRotation(degrees=5),
    #     transforms.ToTensor(),
    #     transforms.Normalize(
    #         mean=[0.3727, 0.2802, 0.2660],
    #         std=[0.2096, 0.1790, 0.1753]
    #     )
    # ])
    
    data_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(degrees=5),
    np.float32,
    transforms.Lambda(prewhiten),
    transforms.ToTensor()
    ])
    
    cc_transforms = transforms.Compose([
    np.float32,
    transforms.Lambda(prewhiten),
    transforms.ToTensor()
    ])
    
    ijb_transforms = transforms.Compose([
    transforms.Resize((160,160)),
    np.float32,
    transforms.Lambda(prewhiten),
    transforms.ToTensor()
    ])

    lfw_transforms = transforms.Compose([
        transforms.Resize(size=image_size),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.6071, 0.4609, 0.3944],
            std=[0.2457, 0.2175, 0.2129]
        )
    ])

    lfw_dataloader = torch.utils.data.DataLoader(
        dataset=LFWDataset(
            dir=lfw_dataroot,
            pairs_path='datasets/LFW_pairs.txt',
            transform=lfw_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    ijb_dataloader_12 = torch.utils.data.DataLoader(
        dataset=IJBDataset(
            dir='datasets/faster_preproc',
            pairs_path='datasets/IJB_pairs_12.txt',
            transform=None,
            preproc=True
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False,
        pin_memory=False
    )
    
    ijb_dataloader_34 = torch.utils.data.DataLoader(
        dataset=IJBDataset(
            dir='datasets/faster_preproc',
            pairs_path='datasets/IJB_pairs_34.txt',
            transform=None,
            preproc=True
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    ijb_dataloader_56 = torch.utils.data.DataLoader(
        dataset=IJBDataset(
            dir='datasets/faster_preproc',
            pairs_path='datasets/IJB_pairs_56.txt',
            transform=None,
            preproc=True
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    cc_dataloader_1 = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir=cc_dataroot,
            pairs_path='datasets/VAL_pairs_1.txt',
            transform=cc_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    cc_dataloader_2 = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir=cc_dataroot,
            pairs_path='datasets/VAL_pairs_2.txt',
            transform=cc_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    cc_dataloader_3 = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir=cc_dataroot,
            pairs_path='datasets/VAL_pairs_3.txt',
            transform=cc_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    cc_dataloader_4 = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir=cc_dataroot,
            pairs_path='datasets/VAL_pairs_4.txt',
            transform=cc_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    cc_dataloader_5 = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir=cc_dataroot,
            pairs_path='datasets/VAL_pairs_5.txt',
            transform=cc_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )

    cc_dataloader_6 = torch.utils.data.DataLoader(
        dataset=CCDataset(
            dir=cc_dataroot,
            pairs_path='datasets/VAL_pairs_6.txt',
            transform=cc_transforms
        ),
        batch_size=lfw_batch_size,
        num_workers=num_workers,
        shuffle=False
    )
    
    def ID(x):
        for i in range(len(x)):
            if '/' == x[i]:
                idx = i
        return idx+1

    logfname_1 = cc_dataloader_1.dataset.pairs_path[ID(cc_dataloader_1.dataset.pairs_path):][:-4]
    
    logfname_2 = cc_dataloader_2.dataset.pairs_path[ID(cc_dataloader_2.dataset.pairs_path):][:-4]
    
    logfname_3 = cc_dataloader_3.dataset.pairs_path[ID(cc_dataloader_3.dataset.pairs_path):][:-4]
    
    logfname_4 = cc_dataloader_4.dataset.pairs_path[ID(cc_dataloader_4.dataset.pairs_path):][:-4]

    logfname_5 = cc_dataloader_5.dataset.pairs_path[ID(cc_dataloader_5.dataset.pairs_path):][:-4]

    logfname_6 = cc_dataloader_6.dataset.pairs_path[ID(cc_dataloader_6.dataset.pairs_path):][:-4]
    
    logfname_12 = ijb_dataloader_12.dataset.pairs_path[ID(ijb_dataloader_12.dataset.pairs_path):][:-4]
    logfname_34 = ijb_dataloader_34.dataset.pairs_path[ID(ijb_dataloader_34.dataset.pairs_path):][:-4]
    logfname_56 = ijb_dataloader_56.dataset.pairs_path[ID(ijb_dataloader_56.dataset.pairs_path):][:-4]


    # Instantiate model
    model = set_model_architecture(
        model_architecture=model_architecture,
        pretrained=pretrained,
        embedding_dimension=embedding_dimension
    )

    # Load model to GPU or multiple GPUs if available
    model, flag_train_multi_gpu = set_model_gpu_mode(model)


    optimizer_model = set_optimizer(
        optimizer=optimizer,
        model=model,
        learning_rate=learning_rate
    )

    # Resume from a model checkpoint
    if resume_path:
        if os.path.isfile(resume_path):
            print("Loading checkpoint {} ...".format(resume_path))
            checkpoint = torch.load(resume_path)
            start_epoch = checkpoint['epoch'] + 1
            optimizer_model = optim.Adam(
            params=model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.999),
            eps=0.1,
            amsgrad=False,
            weight_decay=1e-5
        )

            # In order to load state dict for optimizers correctly, model has to be loaded to gpu first
            if flag_train_multi_gpu:
                model.module.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded: start epoch from checkpoint = {}".format(start_epoch))
        else:
            print("WARNING: No checkpoint found at {}!\nTraining from scratch.".format(resume_path))

    if use_semihard_negatives:
        print("Using Semi-Hard negative triplet selection!")
    else:
        print("Using Hard negative triplet selection!")

    start_epoch = start_epoch

    print("Training using triplet loss starting for {} epochs:\n".format(epochs - start_epoch))
    
    _, _, _, _ = validate_ijb(
                model=model,
                cc_dataloader=ijb_dataloader_12,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_12,
                ts=ts
            )
        
    _, _, _, _ = validate_ijb(
                model=model,
                cc_dataloader=ijb_dataloader_34,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_34,
                ts=ts
            )
        
    _, _, _, _ = validate_ijb(
                model=model,
                cc_dataloader=ijb_dataloader_56,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_56,
                ts=ts
            )

    tpr_1e3_1, tpr_1e4_1, fpr_95_1, best_distances = validate_cc(
            model=model,
            cc_dataloader=cc_dataloader_1,
            model_architecture=model_architecture,
            epoch=0,
            logfname=logfname_1,
            ts=ts
        )
    
    tpr_1e3_2, tpr_1e4_2, fpr_95_2, best_distances = validate_cc(
            model=model,
            cc_dataloader=cc_dataloader_2,
            model_architecture=model_architecture,
            epoch=0,
            logfname=logfname_2,
            ts=ts
        )
    
    tpr_1e3_3, tpr_1e4_3, fpr_95_3, best_distances = validate_cc(
            model=model,
            cc_dataloader=cc_dataloader_3,
            model_architecture=model_architecture,
            epoch=0,
            logfname=logfname_3,
            ts=ts
        )

    tpr_1e3_4, tpr_1e4_4, fpr_95_4, best_distances = validate_cc(
            model=model,
            cc_dataloader=cc_dataloader_4,
            model_architecture=model_architecture,
            epoch=0,
            logfname=logfname_4,
            ts=ts
        )

    tpr_1e3_5, tpr_1e4_5, fpr_95_5, best_distances = validate_cc(
            model=model,
            cc_dataloader=cc_dataloader_5,
            model_architecture=model_architecture,
            epoch=0,
            logfname=logfname_5,
            ts=ts
        )
    
    tpr_1e3_6, tpr_1e4_6, fpr_95_6, best_distances = validate_cc(
            model=model,
            cc_dataloader=cc_dataloader_6,
            model_architecture=model_architecture,
            epoch=0,
            logfname=logfname_6,
            ts=ts
        )
    
    #Calculate skin colour distributions based on fpr at 0.95 tpr values for first epoch
    
    fpr_95_total = fpr_95_1 + fpr_95_2 + fpr_95_3 + fpr_95_4 + fpr_95_5 + fpr_95_6

    id_dist = [round((fpr_95_1*32)/fpr_95_total), round((fpr_95_2*32)/fpr_95_total), round((fpr_95_3*32)/fpr_95_total), round((fpr_95_4*32)/fpr_95_total), round((fpr_95_5*32)/fpr_95_total), round((fpr_95_6*32)/fpr_95_total)]

    for epoch in range(start_epoch, epochs):
        num_valid_training_triplets = 0
        l2_distance = PairwiseDistance(p=2)
        _training_triplets_path = None

        if flag_training_triplets_path:
            _training_triplets_path = training_triplets_path
            flag_training_triplets_path = False  # Only load triplets file for the first epoch

        # Re-instantiate training dataloader to generate a triplet list for this training epoch
        train_dataloader = torch.utils.data.DataLoader(
            dataset=TripletFaceDataset(
                root_dir=dataroot,
                training_dataset_csv_path=training_dataset_csv_path,
                num_triplets=iterations_per_epoch * batch_size,
                num_human_identities_per_batch=num_human_identities_per_batch,
                triplet_batch_size=batch_size,
                epoch=epoch,
                training_triplets_path=_training_triplets_path,
                transform=data_transforms,
                id_dist=id_dist,
                preproc=True
            ),
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False  # Shuffling for triplets with set amount of human identities per batch is not required
        )

        # Training pass
        model.train()
        progress_bar = tqdm(total=iterations_per_epoch)

        for batch_sample in train_dataloader:

            # Forward pass - compute embeddings
            anc_imgs = batch_sample['anc_img']
            pos_imgs = batch_sample['pos_img']
            neg_imgs = batch_sample['neg_img']

            # Concatenate the input images into one tensor because doing multiple forward passes would create
            #  weird GPU memory allocation behaviours later on during training which would cause GPU Out of Memory
            #  issues
            # all_imgs = torch.cat((anc_imgs, pos_imgs, neg_imgs))  # Must be a tuple of Torch Tensors

            anc_embeddings, model = forward_pass(
                    imgs=anc_imgs,
                    model=model,
                    batch_size=batch_size
                )

            pos_embeddings, model = forward_pass(
                    imgs=pos_imgs,
                    model=model,
                    batch_size=batch_size
                )

            neg_embeddings, model = forward_pass(
                    imgs=neg_imgs,
                    model=model,
                    batch_size=batch_size
                )
            

            pos_dists = l2_distance.forward(anc_embeddings, pos_embeddings)
            neg_dists = l2_distance.forward(anc_embeddings, neg_embeddings)

            if use_semihard_negatives:
                # Semi-Hard Negative triplet selection
                #  (negative_distance - positive_distance < margin) AND (positive_distance < negative_distance)
                #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L295
                first_condition = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                second_condition = (pos_dists < neg_dists).cpu().numpy().flatten()
                all = (np.logical_and(first_condition, second_condition))
                valid_triplets = np.where(all == 1)
            else:
                # Hard Negative triplet selection
                #  (negative_distance - positive_distance < margin)
                #   Based on: https://github.com/davidsandberg/facenet/blob/master/src/train_tripletloss.py#L296
                all = (neg_dists - pos_dists < margin).cpu().numpy().flatten()
                valid_triplets = np.where(all == 1)

            anc_valid_embeddings = anc_embeddings[valid_triplets].requires_grad_()
            pos_valid_embeddings = pos_embeddings[valid_triplets].requires_grad_()
            neg_valid_embeddings = neg_embeddings[valid_triplets].requires_grad_()

            # Calculate triplet loss
            triplet_loss = TripletLoss(margin=margin).forward(
                anchor=anc_valid_embeddings,
                positive=pos_valid_embeddings,
                negative=neg_valid_embeddings
            )

            # Calculating number of triplets that met the triplet selection method during the epoch
            num_valid_training_triplets += len(anc_valid_embeddings)

            # Backward pass
            optimizer_model.zero_grad()
            triplet_loss.backward()
            optimizer_model.step()
            progress_bar.update(1)
            
        progress_bar.close()
        # Print training statistics for epoch and add to log
        print('Epoch {}:\tNumber of valid training triplets in epoch: {}'.format(
                epoch,
                num_valid_training_triplets
            )
        )

        with open('logs/{}_log_triplet.txt'.format(model_architecture), 'a') as f:
            val_list = [
                epoch,
                num_valid_training_triplets
            ]
            log = '\t'.join(str(value) for value in val_list)
            f.writelines(log + '\n')

        # Evaluation pass on LFW dataset
        best_distances = validate_lfw(
            model=model,
            lfw_dataloader=lfw_dataloader,
            model_architecture=model_architecture,
            epoch=epoch
        )
        
        _, _, _, _ = validate_ijb(
                model=model,
                cc_dataloader=ijb_dataloader_12,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_12,
                ts=ts
            )
        
        _, _, _, _ = validate_ijb(
                model=model,
                cc_dataloader=ijb_dataloader_34,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_34,
                ts=ts
            )
        
        _, _, _, _ = validate_ijb(
                model=model,
                cc_dataloader=ijb_dataloader_56,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_56,
                ts=ts
            )
        
        tpr_1e3_1, tpr_1e4_1, fpr_95_1, best_distances = validate_cc(
                model=model,
                cc_dataloader=cc_dataloader_1,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_1,
                ts=ts
            )
        
        tpr_1e3_2, tpr_1e4_2, fpr_95_2, best_distances = validate_cc(
                model=model,
                cc_dataloader=cc_dataloader_2,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_2,
                ts=ts
            )
        
        tpr_1e3_3, tpr_1e4_3, fpr_95_3, best_distances = validate_cc(
                model=model,
                cc_dataloader=cc_dataloader_3,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_3,
                ts=ts
            )
        
        tpr_1e3_4, tpr_1e4_4, fpr_95_4, best_distances = validate_cc(
                model=model,
                cc_dataloader=cc_dataloader_4,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_4,
                ts=ts
            )

        tpr_1e3_5, tpr_1e4_5, fpr_95_5, best_distances = validate_cc(
                model=model,
                cc_dataloader=cc_dataloader_5,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_5,
                ts=ts
            )

        tpr_1e3_6, tpr_1e4_6, fpr_95_6, best_distances = validate_cc(
                model=model,
                cc_dataloader=cc_dataloader_6,
                model_architecture=model_architecture,
                epoch=0,
                logfname=logfname_6,
                ts=ts
            )

        fpr_95_total = fpr_95_1 + fpr_95_2 + fpr_95_3 + fpr_95_4 + fpr_95_5 + fpr_95_6

        id_dist = [round((fpr_95_1*32)/fpr_95_total), round((fpr_95_2*32)/fpr_95_total), round((fpr_95_3*32)/fpr_95_total), round((fpr_95_4*32)/fpr_95_total), round((fpr_95_5*32)/fpr_95_total), round((fpr_95_6*32)/fpr_95_total)]

        # Save model checkpoint
        state = {
            'epoch': epoch,
            'embedding_dimension': embedding_dimension,
            'batch_size_training': batch_size,
            'model_state_dict': model.state_dict(),
            'model_architecture': model_architecture,
            'optimizer_model_state_dict': optimizer_model.state_dict(),
            'best_distance_threshold': np.mean(best_distances)
        }

        # For storing data parallel model's state dictionary without 'module' parameter
        if flag_train_multi_gpu:
            state['model_state_dict'] = model.module.state_dict()

        # Save model checkpoint
        torch.save(state, 'model_training_checkpoints/model_{}_triplet_epoch_{}.pt'.format(
                model_architecture,
                epoch
            )
        )


if __name__ == '__main__':
    main()
