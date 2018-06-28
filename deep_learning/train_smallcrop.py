
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
from torchvision.models.resnet import resnet18, resnet50


from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pymongo import MongoClient

from tinder.dataset import BalancedDataLoader
from utils.facetransform import RatioScale
from models.resnet import *
from models.patchnet import *
from utils.facedataset import *
from utils.patchdataset import *
from OpenFacePytorch.loadOpenFace import *


class args:
    seed = 1
    ratio_train = 0.8
    ratio_valid = 0.2
    batch_size = 96 * 4
    epochs = 90
    checkpoint = {
        'inception':"/data/jongho/checkpoints/challenge/fakeopenface_0625_fullno.pth",
        'grid': "/data/jongho/checkpoints/challenge/grid_0625.pth.best",
        'align': "/data/jongho/checkpoints/challenge/alignface_0626resnet50_traintest_fullno.pth",
        "small": "/data/jongho/checkpoints/challenge/smallface_0625resnet50_traintest_fullno.pth",
    }
    patch_checkpoint = "/data/jongho/checkpoints/challenge/patchnet64_0623p_gridpatch_embed256.pth.best.best"
    pretrained = False
    device = torch.device(3) if torch.cuda.is_available() else "cpu"
    train_test = True
    mode = 'align'

def set_seeds(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

def get_valid_loss(net, criterion, valid_loader):
    valid_loss = 0
    net.eval()
    correct = 0
    true_labels = []
    predictions = []
    if len(valid_loader.dataset) == 0:
        return 0, 0

    with torch.no_grad():
        for step, (img, labels) in tqdm(enumerate(valid_loader)):
            img, labels = img.to(args.device), labels.to(args.device)
            pred_score = net(img).view(-1)
            loss = criterion(pred_score, labels)
            valid_loss += loss.item() * len(img)
            correct += ((labels == 1) == (pred_score > 0.5)).sum(dim=0).cpu().item()
            true_labels += list(labels.cpu().view(-1).numpy())
            predictions += list(nn.functional.sigmoid(pred_score).cpu().view(-1).numpy())


    correct = correct / len(valid_loader.dataset)
    net.train(True)
    rocauc = roc_auc_score(true_labels, predictions)

    return valid_loss / (len(valid_loader.dataset)), correct, rocauc


def save_checkpoint(save_dict, path, is_best):
    torch.save(save_dict, path)
    if is_best:
        torch.save(save_dict, path+".best")


def train(net, optimizer, criterion, train_loader):
    net.train(True)
    avg_loss = 0
    for step, (img, labels) in tqdm(enumerate(train_loader)):
        img.requires_grad_()
        img, labels = img.to(args.device), labels.to(args.device)
        optimizer.zero_grad()
        net.zero_grad()
        pred_score = net(img).view(-1)
        loss = criterion(pred_score, labels)
        loss.backward()
        optimizer.step()
        avg_loss += loss.item() * len(img)

    return avg_loss


def load_model(net, optimizer, path):
    saved_dict = torch.load(path)
    net.load_state_dict(saved_dict['net.state_dict'])
    criterion = saved_dict['criterion']
    optimizer.load_state_dict(saved_dict['optimizer.state_dict'])
    epoch = saved_dict['epoch']
    best_val_loss = saved_dict['best_val_loss']
    best_auc_roc = saved_dict['auc_socre']
    print(f"epoch {epoch}, best_val_loss{best_val_loss}")
    return net, criterion, optimizer, epoch, best_val_loss, best_auc_roc


test_swap_dirs = [
    "/data/jongho/challenge/test_crop/fake/",
    "/data/jongho/challenge/test2_crop/fake/",
]

test_real_dirs = [
    "/data/jongho/challenge/test_crop/real/",
    "/data/jongho/challenge/test2_crop/real/",
]


def get_test_datasets(train_dir, valid_dir, prefix, db, collection):
    # ~~~/set/train/ + fake/ + a_ + *.png
    train_swaps = glob.glob(train_dir + "fake/" + prefix + "*.png") + glob.glob(train_dir + "fake/" + prefix + "*.jpg")
    train_reals = glob.glob(train_dir + "real/" + prefix + "*.png") + glob.glob(train_dir + "real/" + prefix + "*.jpg")

    train_datasets = [{'_id': os.path.basename(p)[2:-4], # 'asdf_0'
                       '_path': p,          # "/data/jongho/challenge/train_new/no/swap_results/0Xasdf_0Xqwer_1.png"
                       '_label': 1.
                       } for p in train_swaps]
    train_datasets += [
        {'_id': os.path.basename(p)[:-4],
         '_path': p,
         '_label': 0.
         } for p in train_reals]

    valid_swaps = glob.glob(valid_dir + "fake/" + prefix + "*.png") + glob.glob(valid_dir + "fake/" + prefix + "*.jpg")
    valid_reals = glob.glob(valid_dir + "real/" + prefix + "*.png") + glob.glob(valid_dir + "real/" + prefix + "*.jpg")

    valid_datasets = [{'_id': os.path.basename(p)[2:-4], # 'asdf_0'
                       '_path': p,          # "/data/jongho/challenge/train_new/no/swap_results/0Xasdf_0Xqwer_1.png"
                       '_label': 1.
                       } for p in valid_swaps]
    valid_datasets += [
        {'_id': os.path.basename(p)[:-4],
         '_path': p,
         '_label': 0.
         } for p in valid_reals]

    client = MongoClient()
    db = client[db]
    mongo_face = db[collection]
    for data in train_datasets:
        doc = mongo_face.find_one(filter={'_id': data['_id']})
        if doc is not None and 'sfd' in doc:
            data['_doc'] = doc

    for data in valid_datasets:
        doc = mongo_face.find_one(filter={'_id': data['_id']})
        if doc is not None and 'sfd' in doc:
            data['_doc'] = doc

    train_datasets = list(filter(lambda p: "_doc" in p.keys(), train_datasets))
    valid_datasets = list(filter(lambda p: "_doc" in p.keys(), valid_datasets))

    print("train data from test set", len(train_datasets), collection)
    print("valid data from test set", len(valid_datasets), collection)

    return train_datasets, valid_datasets

def main():
    set_seeds(args.seed)
    print(f"mode {args.mode}, checkpoint {os.path.basename(args.checkpoint[args.mode])}, train_test {args.train_test}")

    train_dirs = [
        "/data/jongho/challenge/train_new/no/swap_results/",
        "/data/jongho/challenge/train_new/feather/swap_results/",
        "/data/jongho/challenge/train_new/polygon/swap_results/",
    ]

    valid_dirs = [
        "/data/jongho/challenge/valid_new/no/swap_results/",
        "/data/jongho/challenge/valid_new/feather/swap_results/",
        "/data/jongho/challenge/valid_new/polygon/swap_results/",
    ]

    face_dir = "/data/jongho/challenge/swap_unused_faces/"


    def get_original_id(path):
        return path.split('X')[1]

    def get_db_id(path, is_swap):
        if is_swap:
            return get_original_id(os.path.basename(path)[:-4])
        else:
            return os.path.basename(path)[:-4]

    def get_swaps_path(dirs):
        swaps = []
        for dir in dirs:
            if dir.find("train_new/no") != -1:
                swaps += glob.glob(dir + "*.png")
            elif dir.find("valid_new/no") != -1:
                    swaps += glob.glob(dir + "*.png")
            else:
                swaps += glob.glob(dir + "*.png")
        return swaps

    train_swaps = get_swaps_path(train_dirs)
    valid_swaps = get_swaps_path(valid_dirs)

    unused_faces = glob.glob(face_dir + "*.png")
    random.shuffle(unused_faces)
    num_train_uf = int(len(unused_faces) * args.ratio_train)
    num_valid_uf = len(unused_faces) - num_train_uf
    train_uf = unused_faces[:num_train_uf]
    valid_uf = unused_faces[num_train_uf:]

    if args.mode == "align" or args.mode == "small" or args.mode == "inception" or args.mode == "grid":
        is_swap_id = True
    # elif args.mode == "grid":
    #     is_swap_id = False

    train_datasets = [{'_id': get_db_id(p, is_swap_id), # 'asdf_0'
                       '_path': p,          # "/data/jongho/challenge/train_new/no/swap_results/0Xasdf_0Xqwer_1.png"
                       '_label': 1.
                       } for p in train_swaps]
    train_datasets += [
        {'_id': os.path.basename(p)[:-4],
         '_path': p,
         '_label': 0.
         } for p in train_uf]

    valid_datasets = [{'_id': get_db_id(p, is_swap_id), # 'asdf_0'
                       '_path': p,          # "/data/jongho/challenge/train_new/no/swap_results/0Xasdf_0Xqwer_1.png"
                       '_label': 1.
                       } for p in valid_swaps]
    valid_datasets += [
        {'_id': os.path.basename(p)[:-4],
         '_path': p,
         '_label': 0.
         } for p in valid_uf]

    client = MongoClient()
    if args.mode == "align" or args.mode == 'small'or args.mode == "inception" or args.mode == "grid":
        db = client['o2']
        mongo_face = db['face']
        for data in train_datasets:
            doc = mongo_face.find_one(filter={'_id': data['_id']})
            if doc is not None and 'sfd' in doc:
                data['_doc'] = doc

        for data in valid_datasets:
            doc = mongo_face.find_one(filter={'_id': data['_id']})
            if doc is not None and 'sfd' in doc:
                data['_doc'] = doc
    # elif args.mode == "grid":
    #     db = client['challenge']
    #     mongo_face = db['face']
    #     for data in train_datasets:
    #         doc = mongo_face.find_one(filter={'_id': data['_id']})
    #         if doc is not None and 'embed256' in doc:
    #             if len(doc['embed256']) != 64:
    #                 print("unexpected : ", data['_id'], len(doc['embed256']))
    #                 pass
    #             else:
    #                 data['_doc'] = doc
    #
    #     for data in valid_datasets:
    #         doc = mongo_face.find_one(filter={'_id': data['_id']})
    #         if doc is not None and 'embed256' in doc:
    #             if len(doc['embed256']) != 64:
    #                 print("unexpected : ", data['_id'], len(doc['embed256']))
    #                 pass
    #             else:
    #                 data['_doc'] = doc

    train_datasets = list(filter(lambda p: "_doc" in p.keys(), train_datasets))
    valid_datasets = list(filter(lambda p: "_doc" in p.keys(), valid_datasets))

    if args.train_test:
        # pdb.set_trace()
        test_dataset_dir = "/data/jongho/challenge/divided_missionset/"
        ttd1, vtd1 = get_test_datasets(train_dir=test_dataset_dir + "train/",  valid_dir=test_dataset_dir + "valid/", prefix="a_",
                                       db="challenge", collection="ex1")
        ttd2, vtd2 = get_test_datasets(train_dir=test_dataset_dir + "train/",  valid_dir=test_dataset_dir + "valid/", prefix="b_",
                                       db="challenge", collection="ex2")

        train_datasets += ttd1 + ttd2
        valid_datasets += vtd1 + vtd2

    print(f"train real ({len(train_uf)}), fake ({len(train_swaps)})")
    print(f"valid real ({len(valid_uf)}), fake ({len(valid_swaps)})")
    print(f"train dset ({len(train_datasets)}), valid dset ({len(valid_datasets)})")

    # print(f"mongo doc list ({len(docs)})")
    if args.mode == "align" or args.mode == "inception" or args.mode == "grid":
        train_transform = transforms.Compose([
            # transforms.RandomAffine(20, scale=(0.9, 1.5)),
            RatioScale((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        valid_transform = transforms.Compose([
            RatioScale((224, 224)),
            transforms.ToTensor(),
        ])
        dset_train = SmallFaceDataset(train_datasets, transform=train_transform, align=True)
        dset_valid = SmallFaceDataset(valid_datasets, transform=valid_transform, align=True)

    elif args.mode == "small":
        train_transform = transforms.Compose([
            transforms.RandomAffine(20, scale=(0.9, 1.5)),
            RatioScale((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])
        valid_transform = transforms.Compose([
            RatioScale((224, 224)),
            transforms.ToTensor(),
        ])
        dset_train = SmallFaceDataset(train_datasets, transform=train_transform, align=False)
        dset_valid = SmallFaceDataset(valid_datasets, transform=valid_transform, align=False)

    # elif args.mode == "grid":
    #     train_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #     ])
    #
    #     valid_transform = transforms.Compose([
    #         transforms.ToTensor(),
    #     ])
    #     dset_train = EmbeddingDataset(train_datasets, transform=train_transform, embed="grid")
    #     dset_valid = EmbeddingDataset(valid_datasets, transform=valid_transform, embed="grid")


    train_loader = BalancedDataLoader(dset_train, [d['_label'] for d in train_datasets], batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, num_workers=24)
    valid_loader = BalancedDataLoader(dset_valid, [d['_label'] for d in valid_datasets], batch_size=args.batch_size,
                                      shuffle=True, pin_memory=True, num_workers=24)

    # import pdb
    # # pdb.set_trace()
    # for i, (_imgs,_labels) in enumerate(train_loader):
    #     for j in range(len(_imgs)):
    #         imq = _imgs[j]
    #         imq = imq.permute(1,2,0).numpy()*256
    #         imq = Image.fromarray(imq.astype(np.uint8))
    #         imq.save(f"/data/jongho/code/challenge/align_face/{i}_{j}.png")
    #     if i > 30:
    #         break
    #
    #
    # exit()

    if args.mode == "align" or args.mode == "small":

        net = PretrainedResNet(resnet50(pretrained=True))
    elif args.mode == "inception":
        net = prepareOpenFace()

    elif args.mode == 'grid':
        """
        1. patch model load
        2. patch model의 fcn weight를 cnn weight로 변환 -> patch2 model
        3. patch2 model를 classifier에 병합
        """
        resnet = resnet18(pretrained=True)
        resnet.avgpool = nn.AdaptiveAvgPool2d(1)
        net = PatchNet(resnet, num_dim=256)
        net = nn.DataParallel(
            SiamesePatchNetwork(net).to(args.device),
            device_ids=[3]
        )
        load = torch.load(args.patch_checkpoint)
        net.load_state_dict(load['net.state_dict'])
        net = PatchNet2(net.module.net)
        net = PatchClassifier(net)

    else:
        raise ValueError("no net mode assigned")

    if args.mode != "inception":
        net = nn.DataParallel(
                net.to(args.device),
                device_ids=[3, 4, 5, 6]
        )

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()))
    criterion = nn.BCEWithLogitsLoss()

    if args.pretrained:
        net, criterion, optimizer, start_epoch, best_val_loss, best_aucroc = load_model(net, optimizer, args.checkpoint[args.mode])
    else:
        best_val_loss = 100
        best_aucroc = 0
        start_epoch = 0
    net.train(True)

    for epoch in tqdm(range(start_epoch + 1, start_epoch + args.epochs + 1)):
        avg_loss = train(net, optimizer, criterion, train_loader)

        if epoch % 1 == 0:
            val_loss, accuracy, auc_score = get_valid_loss(net, criterion, valid_loader)
            is_best = auc_score > best_aucroc
            best_val_loss = val_loss if is_best else best_val_loss
            best_aucroc = auc_score if is_best else best_aucroc
            save_dict = dict(
                {'net.state_dict': net.state_dict(),
                 'criterion': criterion,
                 'optimizer.state_dict': optimizer.state_dict(),
                 'epoch': epoch,
                 'auc_socre': auc_score,
                 'best_val_loss': best_val_loss}
            )
            save_checkpoint(save_dict, args.checkpoint[args.mode], is_best)
            print(
                f"epoch{epoch}, loss{avg_loss/(len(train_loader.dataset))}, val_loss{val_loss}, best_aucroc{best_aucroc} best_val_loss{best_val_loss} accuracy{accuracy}, auc_socre{auc_score}")

    val_loss, accuracy, auc_score = get_valid_loss(net, criterion, valid_loader)
    print(f"val_loss{val_loss}, best_val_loss{best_val_loss} accuracy{accuracy}, auc_score{auc_score}")


if __name__ == "__main__":
    main()
