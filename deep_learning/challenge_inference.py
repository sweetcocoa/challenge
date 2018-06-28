
import os
import pandas as pd
import numpy as np
import random

import glob
import pdb

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.transforms as transforms
from torchvision.models.resnet import resnet50

from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pymongo import MongoClient
from utils.facetransform import RatioScale
from models.resnet import *
from models.patchnet import *
from utils.facedataset import *
from utils.patchdataset import *
# from OpenFacePytorch.loadOpenFace import *

import argparse
parser = argparse.ArgumentParser(description='인퍼런스')

parser.add_argument('--checkpoint', type=str, default="/data/jongho/checkpoints/challenge/smallface_0623resnet50.pth.best", metavar='PATH',
                    help='Pretrained Network')
parser.add_argument('--output', type=str, default="./results/_result.csv", metavar='PATH',
                    help='Patch Embedding Network')
parser.add_argument('--source', type=str, default="/data/jongho/workspace/test/testall/", metavar='PATH',
                    help='DESTINATION DIR')
parser.add_argument('--seed', type=int, default=0, metavar='S',
                    help='random seed (default: 0)')
parser.add_argument('--gpu', type=int, default=0, metavar='S',
                    help='gpu (default: 0)')
parser.add_argument('--db', type=str, default="challenge", metavar='S',
                    help='db name')
parser.add_argument('--collection', type=str, default="ex1", metavar='S',
                    help='db collection name')
parser.add_argument('--mode', type=str, default="small", metavar='S',
                    help='small/align/grid/inception')
config = parser.parse_args()
config.device = torch.device(config.gpu) if torch.cuda.is_available() else "cpu"

def set_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_test_loss(net, criterion, valid_loader):
    valid_loss = 0
    net.eval()
    correct = 0
    true_labels = []
    predictions = []
    if len(valid_loader.dataset) == 0:
        return 0, 0

    with torch.no_grad():
        for step, (img, labels) in tqdm(enumerate(valid_loader)):
            img, labels = img.to(config.device), labels.to(config.device)
            pred_score = net(img).view(-1)
            loss = criterion(pred_score, labels)
            valid_loss += loss.item() * len(img)
            correct += ((labels == 1) == (pred_score > 0.5)).sum(dim=0).cpu().item()
            true_labels += list(labels.cpu().view(-1).numpy())
            predictions += list(nn.functional.sigmoid(pred_score).cpu().view(-1).numpy())

    correct = correct / len(valid_loader.dataset)
    net.train(True)
    if True:
        rocauc = roc_auc_score(true_labels, predictions)
    else:
        rocauc = 0.

    return valid_loss / (len(valid_loader.dataset)), correct, rocauc, (true_labels, predictions)


def load_model(net, optimizer, path):
    saved_dict = torch.load(path)
    net.load_state_dict(saved_dict['net.state_dict'])
    criterion = saved_dict['criterion']
    optimizer.load_state_dict(saved_dict['optimizer.state_dict'])
    epoch = saved_dict['epoch']
    best_val_loss = saved_dict['best_val_loss']
    auc_roc = saved_dict['auc_socre']
    print(f"epoch {epoch}, best_val_loss{best_val_loss}")
    return net, criterion, optimizer, epoch, best_val_loss, auc_roc


def main():
    set_seeds(config.seed)

    if False:
        test_dir = config.source

        test_swaps = glob.glob(test_dir + "*.png") + glob.glob(test_dir + "*.jpg")

        test_swaps = sorted(test_swaps)

        test_datasets = [{'_id': os.path.basename(p)[:-4], # 'asdf_0'
                          '_path': p,          # "/data/jongho/challenge/train_new/no/swap_results/0Xasdf_0Xqwer_1.png"
                          '_label': 1.
                          } for p in test_swaps]
    else:
        test_dir = "/data/jongho/challenge/test2_crop/"

        test_swaps = glob.glob(test_dir + "fake/*.png")
        test_reals = glob.glob(test_dir + 'real/*.png')

        test_swaps = sorted(test_swaps)
        test_reals = sorted(test_reals)

        test_datasets = [{'_id': os.path.basename(p)[:-4],  # 'asdf_0'
                          '_path': p,  # "/data/jongho/challenge/train_new/no/swap_results/0Xasdf_0Xqwer_1.png"
                          '_label': 1.
                          } for p in test_swaps]
        test_datasets += [
            {'_id': os.path.basename(p)[:-4],
             '_path': p,
             '_label': 0.
             } for p in test_reals]




    client = MongoClient()
    db = client[config.db]
    mongo_face = db[config.collection]

    for data in test_datasets:
        doc = mongo_face.find_one(filter={'_id': data['_id']})
        if doc is not None and 'sfd' in doc:
            data['_doc'] = doc
        else:
            if doc is not None:
                print("unexpected : ", data['_id'], 'sfd' in doc)
            else:
                print("unexpected : ", data['_id'], "doc is None")


    test_datasets = list(filter(lambda p: "_doc" in p.keys(), test_datasets))
    # pdb.set_trace()
    print(f"test ({len(test_datasets)})")

    if config.mode == 'align' or config.mode == "grid":
        test_transform = transforms.Compose([
            RatioScale((224, 224)),
            transforms.ToTensor(),
        ])
        dset_test = SmallFaceDataset(test_datasets, transform=test_transform, align=True)
    elif config.mode == 'small':
        test_transform = transforms.Compose([
            RatioScale((224, 224)),
            transforms.ToTensor(),
        ])
        dset_test = SmallFaceDataset(test_datasets, transform=test_transform, align=False)

    else:
        raise ValueError("Mode is unknown : " + config.mode)

    test_loader = torch.utils.data.DataLoader(dset_test, batch_size=32, shuffle=False, pin_memory=True,
                                              num_workers=0)

    if config.mode == 'align' or config.mode == 'small':
        net = PretrainedResNet(resnet50(pretrained=True))

    elif config.mode == 'grid':
        net = resnet18(pretrained=True)
        net.avgpool = nn.AdaptiveAvgPool2d(1)
        net = PatchNet(net, num_dim=256)
        net = PatchNet2(net)
        net = PatchClassifier(net)

    net = nn.DataParallel(
        net.to(config.device),
        device_ids=[config.gpu]
    )

    net.eval()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
    criterion = nn.BCEWithLogitsLoss()

    net, criterion, optimizer, start_epoch, best_val_loss, auc_score = load_model(net, optimizer, config.checkpoint)

    print(f"best_val_loss{best_val_loss}, auc_score{auc_score}")

    test_loss, accuracy, auc_score, results = get_test_loss(net, criterion, test_loader)
    print(f"test_loss{test_loss}, best_val_loss{best_val_loss} accuracy{accuracy}, auc_score{auc_score}")

    if False:
        df_data = {'data' : [sample['_id'] for sample in test_datasets], 'pred' : [pred for pred in results[1]]}
    else:
        df_data = {'data': [sample['_id'] for sample in test_datasets], 'real': [doc['_label'] for doc in test_datasets], 'pred': [pred for pred in results[1]]}
    df = pd.DataFrame(df_data)
    df = df.drop(df.loc[(df['data'].str.endswith("_0") == False)].index)
    df['data'] = df['data'].apply(lambda x: x[:-2])
    df = df.sort_values(by=['data'])
    df.to_csv(config.output, index=False, header=False)


if __name__ == "__main__":
    main()
