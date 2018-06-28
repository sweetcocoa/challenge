import torch.utils.data as data
from PIL import Image
import numpy as np
import random
import torch
import cv2

RESIZE_THRESHOLD = 2400

class PatchLandmarkDataset(data.Dataset):
    """
    68개 랜드마크에 대한 패치를 돌려준다.
    """
    def __init__(self, data, transform=None, patch_size=(64, 64)):
        assert len(data) != 0
        self.samples = data
        self.transform = transform
        self.patch_size = patch_size

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    def __getitem__(self, index):
        _id = self.samples[index]['_id']
        path = self.samples[index]['_path']
        doc = self.samples[index]['_doc']
        landmarks = doc['landmarks']
        box = doc['sfd']

        q = Image.open(path)
        if q.mode != "RGB":
            q = q.convert("RGB")

        facesize = max(box[2], box[3])
        if facesize > 1000:
            ratio = 1000 / facesize
            landmarks = np.array(landmarks) * ratio
            box = np.array(box) * ratio
            q = q.resize((int(q.size[0] * ratio), int(q.size[1] * ratio)))

        raw_patch_size = self.patch_size

        images = []
        for landmark in landmarks:
            center = landmark[:2]

            q_image = self.get_crop_from_center(q, center, raw_patch_size)

            if self.transform is not None:
                q_image = self.transform(q_image)

            images.append(q_image)

        images = torch.stack(images)

        return images


    def get_crop_from_center(self, img, center, size):
        """
        :param img:
        :param center: ( top, left ) - A point on image array
        :return:
        """
        half_w = size[1] // 2
        half_h = size[0] // 2
        left = int(center[1])
        top = int(center[0])

        output_img = img.crop((left - half_w,
                               top - half_h,
                               left + half_w,
                               top + half_h,
        ))

        return output_img




class EmbeddingDataset(data.Dataset):
    """
    mongo의 embed vector를 돌려준다.
    embed : 랜드마크 기준 (68개 x 256)
    embed256 : 그리드 기준 (64개 x 256)
    """
    def __init__(self, data, transform=None, patch_size=(224,224), embed='grid'):
        assert len(data) != 0
        self.samples = data
        self.transform = transform
        self.patch_size = patch_size
        if embed == "grid":
            self.embed = "embed256"
        elif embed == "landmark":
            self.embed = "embed"
        else:
            raise ValueError("No such embed method " + embed )

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
        doc = self.samples[index]['_doc']
        label = self.samples[index]['_label']
        embed = doc[self.embed] # 68 x 64

        images = torch.tensor(embed).float()
        if self.embed == "embed256":
            images = images.permute((1, 0)).view(256, 8, 8)

        label = torch.tensor(label).float()

        return images, label


class TripletGridPatchDataset(data.Dataset):
    """
    얼굴 안에서 뽑은 패치 3개(p,q,n)를 돌려준다. 얼굴 크기는 224로 고정.
    """

    def __init__(self, data, transform=None, adj=False):
        assert len(data) != 0
        self.samples = data
        self.transform = transform
        self.patch_size = (64, 64)
        self.face_size = 224
        self.adj = adj

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

        negative_index = random.randint(0, len(self.samples) - 1)
        while negative_index == index:
            negative_index = random.randint(0, len(self.samples) - 1)

        query_path = self.samples[index]['_path']
        negative_path = self.samples[negative_index]['_path']
        box = self.samples[index]['_doc']['sfd']
        negative_box = self.samples[negative_index]['_doc']['sfd']
        raw_patch_size = self.patch_size

        q = Image.open(query_path)
        if q.mode != "RGB":
            q = q.convert("RGB")

        n = Image.open(negative_path)
        if n.mode != "RGB":
            n = n.convert("RGB")

        # Resize Image Keeping aspect ratio
        # with respect to its face bounding box's size.

        p_height = box[2]
        p_width = box[3]
        p_ratio_hw = [self.face_size / p_height, self.face_size / p_width]

        n_height = negative_box[2]
        n_width = negative_box[3]
        n_ratio_hw = [self.face_size / n_height, self.face_size / n_width]

        y1, x1, y2, x2, s = box
        x2 = x1 + x2
        y2 = y1 + y2

        ny1, nx1, ny2, nx2, ns = negative_box
        nx2 = nx1 + nx2
        ny2 = ny1 + ny2

        # x, y
        q_size = (int(q.size[0] * p_ratio_hw[1]), int(q.size[1] * p_ratio_hw[0]))
        q = q.resize(q_size)

        # x, y
        n_size = (int(n.size[0] * n_ratio_hw[1]), int(n.size[1] * n_ratio_hw[0]))
        n = n.resize(n_size)

        x1 *= p_ratio_hw[1]
        y1 *= p_ratio_hw[0]
        x2 *= p_ratio_hw[1]
        y2 *= p_ratio_hw[0]

        nx1 *= n_ratio_hw[1]
        ny1 *= n_ratio_hw[0]
        nx2 *= n_ratio_hw[1]
        ny2 *= n_ratio_hw[0]

        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        nx1, nx2, ny1, ny2 = int(nx1), int(nx2), int(ny1), int(ny2)

        q_center = (random.randrange(x1, x2), random.randrange(y1, y2))
        if self.adj:
            p_center = (np.clip(random.randrange(q_center[0] - 32, q_center[0] + 32), 0, q.size[0]),
                        np.clip(random.randrange(q_center[1] - 32, q_center[1] + 32), 0, q.size[1]))
        else:
            p_center = (random.randrange(x1, x2),
                        random.randrange(y1, y2))

        n_center = (random.randrange(nx1, nx2), random.randrange(ny1, ny2))

        q_image = self.get_crop_from_center(q, q_center, raw_patch_size)
        p_image = self.get_crop_from_center(q, p_center, raw_patch_size)
        n_image = self.get_crop_from_center(n, n_center, raw_patch_size)

        if self.transform is not None:
            q_image = self.transform(q_image)
            p_image = self.transform(p_image)
            n_image = self.transform(n_image)

        # pdb.set_trace()
        # q.save(f"/data/jongho/code/challenge/grid_patches/qbig_{self.samples[index]['_id']}.png")
        # n.save(f"/data/jongho/code/challenge/grid_patches/nbig_{self.samples[negative_index]['_id']}.png")
        #
        # cvq = cv2.rectangle(np.array(q), (x1, y1), (x2, y2), (0, 255, 0), 10)
        # cvn = cv2.rectangle(np.array(n), (nx1, ny1), (nx2, ny2), (0, 255, 0), 10)
        #
        # cv2.imwrite(f"/data/jongho/code/challenge/grid_patches/qbig_sfd{self.samples[index]['_id']}.png", cvq)
        # cv2.imwrite(f"/data/jongho/code/challenge/grid_patches/nbig_sfd{self.samples[negative_index]['_id']}.png", cvn)
        #
        # self.savecnt += 1

        return q_image, p_image, n_image

    def get_crop_from_center(self, img, center, size):
        """
        :param img:
        :param center: ( x, y ) - A point on image array
        :param size: (width, height)
        :return:
        """
        half_w = size[0] // 2
        half_h = size[1] // 2

        left = int(center[0])
        top = int(center[1])

        output_img = img.crop((left - half_w,
                               top - half_h,
                               left + half_w,
                               top + half_h,
                               ))

        return output_img

class TripletLandmarkPatchDataset(data.Dataset):
    """
    랜드마크에서 뽑은 패치 3개(p,q,n)를 돌려준다. 얼굴 크기는 224로 고정.
    """

    def __init__(self, data, transform=None):
        assert len(data) != 0
        self.samples = data
        self.transform = transform
        self.patch_size = (64, 64)
        self.face_size = 224

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

        negative_index = random.randint(0, len(self.samples) - 1)
        while negative_index == index:
            negative_index = random.randint(0, len(self.samples) - 1)

        query_path = self.samples[index]['_path']
        negative_path = self.samples[negative_index]['_path']
        landmarks = self.samples[index]['_doc']['landmarks']
        negative_landmarks = self.samples[negative_index]['_doc']['landmarks']
        box = self.samples[index]['_doc']['sfd']
        negative_box = self.samples[negative_index]['_doc']['sfd']
        raw_patch_size = self.patch_size

        q = Image.open(query_path)
        if q.mode != "RGB":
            q = q.convert("RGB")

        n = Image.open(negative_path)
        if n.mode != "RGB":
            n = n.convert("RGB")

        # Resize Image Keeping aspect ratio
        # with respect to its face bounding box's size.

        p_height = box[2]
        p_width = box[3]
        p_ratio_hw = [self.face_size / p_height, self.face_size / p_width]

        n_height = negative_box[2]
        n_width = negative_box[3]
        n_ratio_hw = [self.face_size / n_height, self.face_size / n_width]

        y1, x1, y2, x2, s = box
        x2 = x1 + x2
        y2 = y1 + y2

        ny1, nx1, ny2, nx2, ns = negative_box
        nx2 = nx1 + nx2
        ny2 = ny1 + ny2

        # x, y
        q_size = (int(q.size[0] * p_ratio_hw[1]), int(q.size[1] * p_ratio_hw[0]))
        q = q.resize(q_size)

        # x, y
        n_size = (int(n.size[0] * n_ratio_hw[1]), int(n.size[1] * n_ratio_hw[0]))
        n = n.resize(n_size)

        x1 *= p_ratio_hw[1]
        y1 *= p_ratio_hw[0]
        x2 *= p_ratio_hw[1]
        y2 *= p_ratio_hw[0]

        nx1 *= n_ratio_hw[1]
        ny1 *= n_ratio_hw[0]
        nx2 *= n_ratio_hw[1]
        ny2 *= n_ratio_hw[0]

        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        nx1, nx2, ny1, ny2 = int(nx1), int(nx2), int(ny1), int(ny2)

        q_landmark_index = random.randint(0, 67)
        p_landmark_index = random.randint(0, 67)
        n_landmark_index = random.randint(0, 67)
        while p_landmark_index == q_landmark_index:
            p_landmark_index = random.randint(0, 67)

        q_center = landmarks[q_landmark_index][:2]
        q_center = [q_center[0] * p_ratio_hw[1], q_center[1] * p_ratio_hw[0]]
        p_center = landmarks[p_landmark_index][:2]
        p_center = [p_center[0] * p_ratio_hw[1], p_center[1] * p_ratio_hw[0]]
        n_center = landmarks[n_landmark_index][:2]
        n_center = [n_center[0] * n_ratio_hw[1], n_center[1] * n_ratio_hw[0]]


        q_image = self.get_crop_from_center(q, q_center, raw_patch_size)
        p_image = self.get_crop_from_center(q, p_center, raw_patch_size)
        n_image = self.get_crop_from_center(n, n_center, raw_patch_size)

        if self.transform is not None:
            q_image = self.transform(q_image)
            p_image = self.transform(p_image)
            n_image = self.transform(n_image)

        # pdb.set_trace()
        # q.save(f"/data/jongho/code/challenge/grid_patches/qbig_{self.samples[index]['_id']}.png")
        # n.save(f"/data/jongho/code/challenge/grid_patches/nbig_{self.samples[negative_index]['_id']}.png")
        #
        # cvq = cv2.rectangle(np.array(q), (x1, y1), (x2, y2), (0, 255, 0), 10)
        # cvn = cv2.rectangle(np.array(n), (nx1, ny1), (nx2, ny2), (0, 255, 0), 10)
        #
        # cv2.imwrite(f"/data/jongho/code/challenge/grid_patches/qbig_sfd{self.samples[index]['_id']}.png", cvq)
        # cv2.imwrite(f"/data/jongho/code/challenge/grid_patches/nbig_sfd{self.samples[negative_index]['_id']}.png", cvn)
        #
        # self.savecnt += 1

        return q_image, p_image, n_image

    def get_crop_from_center(self, img, center, size):
        """
        :param img:
        :param center: ( x, y ) - A point on image array
        :param size: (width, height)
        :return:
        """
        half_w = size[0] // 2
        half_h = size[1] // 2

        left = int(center[0])
        top = int(center[1])

        output_img = img.crop((left - half_w,
                               top - half_h,
                               left + half_w,
                               top + half_h,
                               ))

        return output_img


class StrideGridPatchDataset(data.Dataset):
    """
    얼굴에서 64개의 그리드 패치 이미지를 돌려준다
    """

    def __init__(self, data, transform=None):
        assert len(data) != 0
        self.samples = data
        self.transform = transform
        self.patch_size = (64, 64)
        self.face_size = 224

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

        _id = self.samples[index]['_id']
        path = self.samples[index]['_path']
        doc = self.samples[index]['_doc']
        box = doc['sfd']
        raw_patch_size = self.patch_size

        q = Image.open(path)
        if q.mode != "RGB":
            q = q.convert("RGB")

        # Resize Image Keeping aspect ratio
        # with respect to its face bounding box's size.

        p_height = box[2]
        p_width = box[3]
        p_ratio_hw = [self.face_size / p_height, self.face_size / p_width]

        y1, x1, y2, x2, s = box
        x2 = x1 + x2
        y2 = y1 + y2

        # x, y
        q_size = (int(q.size[0] * p_ratio_hw[1]), int(q.size[1] * p_ratio_hw[0]))
        q = q.resize(q_size)

        x1 *= p_ratio_hw[1]
        y1 *= p_ratio_hw[0]
        x2 *= p_ratio_hw[1]
        y2 *= p_ratio_hw[0]

        x1, x2, y1, y2 = int(x1), int(x2), int(y1), int(y2)
        images = []


        for _y in range(y1, y1+self.face_size + 1, self.patch_size[1] // 2):
            for _x in range(x1, x1+self.face_size + 1, self.patch_size[0] // 2):
                center = [_x, _y]
                q_image = self.get_crop_from_center(q, center, raw_patch_size)
                # cv2.imwrite(f"/data/jongho/code/challenge/emb_patches/{self.samples[index]['_id']}_{_x}_{_y}.png", np.array(q_image))
                if self.transform is not None:
                    q_image = self.transform(q_image)
                images.append(q_image)

        images = torch.stack(images)
        return images

    def get_crop_from_center(self, img, center, size):
        """
        :param img:
        :param center: ( x, y ) - A point on image array
        :param size: (width, height)
        :return:
        """
        half_w = size[0] // 2
        half_h = size[1] // 2

        left = int(center[0])
        top = int(center[1])

        output_img = img.crop((left - half_w,
                               top - half_h,
                               left + half_w,
                               top + half_h,
                               ))

        return output_img
