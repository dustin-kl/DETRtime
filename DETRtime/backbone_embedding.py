print('0-th line')
#import pytorch_lightning as pl
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from torch.nn import functional as F
import argparse
import os

from model.backbone import build_backbone
from model.transformer import build_transformer
from model.DETRtime import build, DETRtime

print('first line')

def get_args_parser():
    parser = argparse.ArgumentParser('Set transformer detector', add_help=False)

    # Hyperparameters
    parser.add_argument('--lr', default=1e-4, type=float)
    parser.add_argument('--wandb_dir', default='noname', type=str)
    parser.add_argument('--lr_backbone', default=1e-5, type=float)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--weight_decay', default=1e-4, type=float)
    parser.add_argument('--epochs', default=300, type=int)
    parser.add_argument('--lr_drop', default=50, type=int)
    #parser.add_argument('--clip_max_norm', default=0.1, type=float,
    #                    help='gradient clipping max norm')

    # Model parameters
    # parser.add_argument('--frozen_weights', type=str, default=None,
    #                    help="Path to the pretrained model. If set, only the mask head will be trained")

    # * Backbone
    parser.add_argument('--backbone', default='inception_time', type=str,
                        help="Name of the convolutional backbone to use")
    parser.add_argument('--kernel_size', default=16, type=int)
    parser.add_argument('--nb_filters', default=16, type=int)
    parser.add_argument('--in_channels', default=129, type=int)
    parser.add_argument('--out_channels', default=1, type=int)
    parser.add_argument('--backbone_depth', default=6, type=int)
    parser.add_argument('--use_residual', default=True, type=bool)
    # parser.add_argument('--dilation', action='store_true',
    #                    help="If true, we replace stride with dilation in the last convolutional block (DC5)")
    parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
    parser.add_argument('--back_channels', default=16, type=int,
                        help='Defines the number of embedding channels')
    parser.add_argument('--back_layers', default=12, type=int,
                        help='Defines the number of layers in the backbone')
    # parser.add_argument('--maxpools', default=2, nargs='+', type=int, help="Optionally define maxpools for ConvNet classes")
    # * Transformer
    parser.add_argument('--enc_layers', default=6, type=int,
                        help="Number of encoding layers in the transformer")
    parser.add_argument('--dec_layers', default=6, type=int,
                        help="Number of decoding layers in the transformer")
    parser.add_argument('--dim_feedforward', default=2048, type=int,
                        help="Intermediate size of the feedforward layers in the transformer blocks")
    parser.add_argument('--hidden_dim', default=128, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
    parser.add_argument('--dropout', default=0.1, type=float,
                        help="Dropout applied in the transformer")
    parser.add_argument('--nheads', default=8, type=int,
                        help="Number of attention heads inside the transformer's attentions")
    parser.add_argument('--num_queries', default=20, type=int,
                        help="Number of qbackboneuery slots")
    parser.add_argument('--pre_norm', default=False) #,action='store_true')

    # * Segmentation
    # parser.add_argument('--masks', action='store_true',
    #                    help="Train segmentation head if the flag is provided")

    # Loss
    # parser.add_argument('--no_aux_loss', dest='aux_loss', action='store_false',
    #                    help="Disables auxiliary decoding losses (loss at each layer)")

    # * Matcher
    parser.add_argument('--set_cost_class', default=1, type=float,
                        help="Class coefficient in the matching cost")
    parser.add_argument('--set_cost_bbox', default=10, type=float,
                        help="L1 box coefficient in the matching cost")
    parser.add_argument('--set_cost_giou', default=2, type=float,
                        help="giou box coefficient in the matching cost")

    # * Loss coefficients
    # parser.add_argument('--mask_loss_coef', default=1, type=float)
    # parser.add_argument('--dice_loss_coef', default=1, type=float)
    parser.add_argument('--bbox_loss_coef', default=10, type=float)
    parser.add_argument('--giou_loss_coef', default=2, type=float)
    parser.add_argument('--eos_coef', default=0.2, type=float,
                        help="Relative classification weight of the no-object class")

    # dataset parameters
    parser.add_argument('--sleep', default=False, type=bool, help='sleep annotations')
    parser.add_argument('--num_classes', default=3, type=int,
                        help="number of classes to predict")
    parser.add_argument('--timestamps', default=500, type=int)
    parser.add_argument('--timestamps_output', default=500, type=int)
    # parser.add_argument('--dataset_file', default='coco')
    parser.add_argument('--data_path', type=str)
    parser.add_argument('--scaler', default=None, type=str, choices=['Standard', 'MaxMin'])
    # parser.add_argument('--coco_panoptic_path', type=str)
    # parser.add_argument('--remove_difficult', action='store_true')

    parser.add_argument('--output_dir', default='', required=False,
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', default=True) #action='store_true')
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--apply_labels', default=False, type=bool)
    # distributed training parameters
    # parser.add_argument('--world_size', default=1, type=int,
    #                    help='number of distributed processes')
    # parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    return parser

parser = argparse.ArgumentParser('DETR training and evaluation script', parents=[get_args_parser()])
args = parser.parse_args()

convert_label = {
    'L_fixation': 0,
    'L_saccade': 1,
    'L_blink': 2
}

# Loading Data
path = f'../../segmentation/ICML_zuco_min_segmentation_minseq_500_margin_1_amp_thresh_10000/train/'
EEG = []
labels = []
for i, file in enumerate(os.listdir(path)[:5]):
    data_train = np.load(path + file)
    EEG = EEG + [data_train['EEG']]
    labels += [data_train['labels']]
    # EEG = np.concatenate((EEG, data_train['EEG']), axis=0)
    # labels = np.concatenate((labels, [convert_label[x[0]] for x in data_train['labels']]), axis=0) 
    #print("shape: ", np.asarray(EEG, dtype=float).shape)

print('Number of Files: ', len(EEG))
EEG = np.array([item for sublist in EEG for item in sublist])
labels = np.array([convert_label[item[0]] for sublist in labels for item in sublist])
print('Number of Timesteps: ', len(EEG))

# Model Definition
num_classes = 3
device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
backbone = build_backbone(args)
transformer = build_transformer(args)
model = DETRtime(
            backbone,
            transformer,
            num_classes=num_classes,
            num_queries=20,
        )

# load pytorch model
model_path = f'../../Resume_dustin_supermodel_zuco_22-05-16-08-24/checkpoint_best_val.pth'
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint['model'])
backbone_model = model.backbone
backbone_model.double()

# print number of params in backbone
print('Number of parameters in backbone: ', sum(p.numel() for p in backbone_model.parameters() if p.requires_grad))

# compute embeddings
embeddings = []
samples = []
num_samples = int(len(EEG)/500)
print('Number of samples in Data: ', num_samples)
for i in range(num_samples):
    sample = EEG[i*500:(i+1)*500]    
    samples.append(sample)
    # sample = torch.tensor(sample)
    # sample = sample.unsqueeze(2) # dimension before (500, 129), after (500, 129, 1)
    # sample = sample.to(device)
    # embedding, pos = backbone_model(sample.double())
    # embeddings.append(embedding)
    # if i==0: 
    #     print(sample.shape)
    #     print(embedding.shape)

samples = torch.tensor(samples)
samples = samples.permute(1, 2, 0) # (500, 129, num_samples)
samples = samples.to(device)
embedding, pos = backbone_model(samples.double())
embedding = embedding.permute(2, 0, 1)
print('Shape of Embedding: ', embedding.shape)


counts = [0, 0, 0]
number = 1000 # number of samples to plot per class
print('Number of Drawings: ', number)
embedding_list = {
    '0': [],
    '1': [],
    '2': [],
}
while (counts[0] < number) or (counts[1] < number) or (counts[2] < number):
    rand1 = np.random.randint(0, num_samples)
    rand2 = np.random.randint(0, 500)
    # sample_time = torch.tensor(EEG[rand1])
    emb_time = embedding[rand1][rand2]
    label = int(labels[rand1 * 500 + rand2])

    if counts[label] < number:
        embedding_list[str(label)].append(emb_time)
        counts[label] += 1

# compute cosine similarity
cosine_sim = {}
for i in range(3):
    for j in range(3):
        if i <= j:
            for emb1 in embedding_list[str(i)]:
                for emb2 in embedding_list[str(j)]:
                    cos_sim = F.cosine_similarity(emb1, emb2, dim=0)

                    if str(i)+str(j) not in cosine_sim:
                        cosine_sim[str(i)+str(j)] = []
                    cosine_sim[str(i)+str(j)].append(cos_sim)

for i in range(3):
    embedding_list[str(i)] = [np.reshape(x.detach().numpy(), -1) for x in embedding_list[str(i)]]

# print average cosine similarity40406193675516794
for key in cosine_sim:
    cosine_sim[key] = [x.detach().numpy() for x in cosine_sim[key]]
    print(f'Average cosine similarity between {key[0]} and {key[1]}: {np.mean(cosine_sim[key])}')

# visualize embeddings with tsne
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
tsne_results = tsne.fit_transform(embedding_list['0'] + embedding_list['1'] + embedding_list['2'])

plt.scatter(tsne_results[:,0], tsne_results[:,1], c=([0]*number + [1]*number + [2]*number))
plt.show()
