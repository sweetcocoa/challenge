import torch.utils.data as data
from PIL import Image
import numpy as np
from skimage.transform import SimilarityTransform


class FaceDataset(data.Dataset):
    """
    data  : data[0] = (image path, label)
    """

    def __init__(self, data, transform=None):
        self.samples = data
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, landmark heatmap, target score)
        """
        path, target = self.samples[index]
        target = np.float32(target)

        sample = Image.open(path)
        if sample.mode != "RGB":
            sample = sample.convert("RGB")

        if self.transform is not None:
            sample = self.transform(sample)

        return sample, target


import cv2
import pdb

class SmallFaceDataset(data.Dataset):
    """
    data  : data = {'id' : (image path), sfd, landmarks}
    """

    def __init__(self, data, transform=None, align=False):
        assert len(data) != 0
        self.samples = data
        self.transform = transform
        self.align = align

    def __len__(self):
        return len(self.samples)

    def align_face(self, img, src_img_size, landmark, origin):
        dst_size = 112
        new_dst_size = 224
        dst = np.array([
            [30.2946 + 8, 51.6963],
            [65.5318 + 8, 51.5014],
            [48.0252 + 8, 71.7366],
            [33.5493 + 8, 92.3655],
            [62.7299 + 8, 92.2041]], dtype=np.float32 )  * new_dst_size / dst_size
        p = src_img_size / new_dst_size
        dst = dst * p

        src = landmark - np.array(origin)
        # print("landmark2", src)
        #dst = np.transpose(landmark).reshape(1,5,2)
        #src = src.reshape(1,5,2)
        # print(src)
        # print(dst)
        # transmat = cv2.estimateRigidTransform(dst.astype(np.float32),
        #                                       src.astype(np.float32), False)
        # out = cv2.warpAffine(img, transmat, (dst_img_size, dst_img_size))
        tform = SimilarityTransform()
        tform.estimate(src, dst)
        M = tform.params[0:2, :]
        out = cv2.warpAffine(img, M, (src_img_size, src_img_size), borderValue=0.0)
        return out

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (img, landmark heatmap, target score)
        """

        path = self.samples[index]['_path']
        box = self.samples[index]['_doc']['sfd']
        label = self.samples[index]['_label']
        preds = np.array(self.samples[index]['_doc']['landmarks'])
        landmarks = np.array([
            preds[36:42, :2].mean(axis=0),
            preds[42:48, :2].mean(axis=0),
            preds[30, :2],
            preds[48, :2],
            preds[54, :2]
        ])


        img = Image.open(path)
        if img.mode != "RGB":
            img = img.convert("RGB")
        # pdb.set_trace()
        if self.align:
            # ratio = 224 / max(img.size)
            # landmarks *= ratio
            # img = np.array(img.resize((int(img.size[0] * ratio), int(img.size[1] * ratio)), Image.ANTIALIAS))
            sz = max(img.size)
            img = np.array(img)
            img = self.align_face(img, sz, landmarks, origin=[0, 0])
            img = Image.fromarray(img)
        else:

            y1, x1, y2, x2, s = box
            x2 = x1 + x2
            y2 = y1 + y2

            img = img.crop((x1, y1, x2, y2))

        # cv2.imwrite(f"/data/jongho/code/challenge/align_face/{self.samples[index]['_id']}.png", np.array(img))
        if self.transform is not None:
            img = self.transform(img)

        label = np.float32(label)

        return img, label

