import sys
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
#import face_alignment
import tinder
import img_proc

log = tinder.config.setup(parse_args=True)

import os
import logging
from pymongo import MongoClient
import tqdm
import json



import random
import cv2
import numpy as np

json.encoder.FLOAT_REPR = lambda f: ('%.3f' % f)

log = logging.getLogger()
log.setLevel(logging.INFO)

if 'seed' in os.environ.keys():
    random.seed(os.environ['seed'])
else:
    random.seed(2311)

if 'num_generate' not in os.environ.keys():
    os.environ['num_generate'] = '500'

if 'blur' not in os.environ.keys():
    os.environ['blur'] = 'no'

if 'mode' not in os.environ.keys():
    os.environ['mode'] = 'valid'


source_dir = f"/data/jongho/challenge/{os.environ['mode']}/swap_faces/"
swap_dir   = f"/data/jongho/challenge/{os.environ['mode']}_new/{os.environ['blur']}/swap_results/"
mask_dir   = f"/data/jongho/challenge/{os.environ['mode']}_new/{os.environ['blur']}/swap_masks/"

# source_dir = f"/data/jongho/challenge/train/swap_faces/"
# swap_dir   = f"/data/jongho/blur_feather/swap_results/"
# mask_dir   = f"/data/jongho/blur_feather/swap_masks/"


face_paths = os.listdir(source_dir)

print('# of faces (total): ', len(face_paths))

face_ids = [os.path.splitext(path)[0] for path in face_paths]
# import pdb
# pdb.set_trace()

# prepare mongo
client = MongoClient()
db = client['o2']
mongo_face = db['face']

docs = map(lambda _id: mongo_face.find_one(filter={'_id': _id}), face_ids)
docs = list(filter(lambda doc: (doc is not None) and ('landmarks' in doc), docs))

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11

FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 60))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 36))
JAW_POINTS = list(range(0, 17))

LEFT_HALF_POINTS = list(range(8, 17))
LEFT_NOSE = list(range(33, 36))
LEFT_MOUTH = list(range(51, 58))
LEFT_LIP = list(range(62, 67))
NOSE_CENTER = list(range(27, 31))

LEFT_TEMPLE = list(range(0, 2))
RIGHT_TEMPLE = list(range(15, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + JAW_POINTS)

# Points from the second image to overlay on the first. The convex hull of each
# element will be overlaid.

OVERLAY_METHODS = [
    # 얼굴
    [LEFT_BROW_POINTS + RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS + JAW_POINTS],
    # 왼쪽얼굴
    [LEFT_BROW_POINTS + LEFT_HALF_POINTS + NOSE_CENTER +LEFT_LIP],
    # 양쪽눈 + 관자놀이
    [LEFT_TEMPLE + LEFT_EYE_POINTS + RIGHT_EYE_POINTS + RIGHT_TEMPLE],
    # 왼눈
    [LEFT_EYE_POINTS],
    # 눈 + 코 + 입
    [LEFT_BROW_POINTS + RIGHT_BROW_POINTS + LEFT_EYE_POINTS + RIGHT_EYE_POINTS, NOSE_POINTS + MOUTH_POINTS],
    # 코 + 입
    [NOSE_POINTS+MOUTH_POINTS],
    # 코
    [NOSE_POINTS],
    # 입
    [MOUTH_POINTS],
    # 오른눈
    [RIGHT_EYE_POINTS],
    # 양쪽눈
    [LEFT_EYE_POINTS + RIGHT_EYE_POINTS],
]

OVERLAY_METHODS = [(str(i), OVERLAY_METHODS[i]) for i in range(len(OVERLAY_METHODS))]
OVERLAY_METHOD, OVERLAY_POINTS = random.choice(OVERLAY_METHODS)

# Amount of blur to use during colour correction, as a fraction of the
# pupillary distance.
COLOUR_CORRECT_BLUR_FRAC = 0.6


def draw_convex_hull(im, points, color):
    points = cv2.convexHull(points)
    cv2.fillConvexPoly(im, points, color=color)


def get_face_mask(im, landmarks):
    im = np.zeros(im.shape[:2], dtype=np.float64)

    for group in OVERLAY_POINTS:
        draw_convex_hull(im,
                         landmarks[group],
                         color=1)

    im = np.array([im, im, im]).transpose((1, 2, 0))

    im = (cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0) > 0) * 1.0
    im = cv2.GaussianBlur(im, (FEATHER_AMOUNT, FEATHER_AMOUNT), 0)

    return im


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:

        sum ||s*R*p1,i + T - p2,i||^2

    is minimized.

    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])



def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im


def correct_colours(im1, im2, landmarks1):
    blur_amount = COLOUR_CORRECT_BLUR_FRAC * np.linalg.norm(
        np.mean(landmarks1[LEFT_EYE_POINTS], axis=0) -
        np.mean(landmarks1[RIGHT_EYE_POINTS], axis=0))
    blur_amount = int(blur_amount)
    if blur_amount % 2 == 0:
        blur_amount += 1

    im1_blur = cv2.GaussianBlur(im1, (blur_amount, blur_amount), 0)
    im2_blur = cv2.GaussianBlur(im2, (blur_amount, blur_amount), 0)
    # Avoid divide-by-zero errors.
    im2_blur += (128 * (im2_blur <= 1.0)).astype(im2_blur.dtype)

    return (im2.astype(np.float64) * im1_blur.astype(np.float64) /
            im2_blur.astype(np.float64))


im1_doc = docs[0]
im2_doc = docs[1]
im1_id, im2_id = face_ids[0], face_ids[1]
y_angle = abs(im1_doc['rotation'][1] - im2_doc['rotation'][1])
swap_path = f"{swap_dir}{OVERLAY_METHOD}X{im1_id}X{im2_id}.png"
mask_path = f"{mask_dir}{OVERLAY_METHOD}X{im1_id}X{im2_id}.png"

for n in tqdm.tqdm(range(int(os.environ['num_generate']))):
    while (im1_id == im2_id) or os.path.exists(swap_path) or y_angle > 0.2:
        im1_doc = random.choice(docs)
        im2_doc = random.choice(docs)
        im1_id = im1_doc['_id']
        im2_id = im2_doc['_id']
        COLOUR_CORRECT_BLUR_FRAC = 0.55 + 0.1 * random.random()
        OVERLAY_METHOD, OVERLAY_POINTS = random.choice(OVERLAY_METHODS)
        swap_path = f"{swap_dir}{OVERLAY_METHOD}X{im1_id}X{im2_id}.png"
        mask_path = f"{mask_dir}{OVERLAY_METHOD}X{im1_id}X{im2_id}.png"
        y_angle = abs(im1_doc['rotation'][1] - im2_doc['rotation'][1])

    im1_path = source_dir + im1_id + ".png"
    im2_path = source_dir + im2_id + ".png"

    im1 = cv2.imread(im1_path, cv2.IMREAD_COLOR)
    im2 = cv2.imread(im2_path, cv2.IMREAD_COLOR)

    landmarks1 = np.matrix(im1_doc['landmarks'])[:, :2].astype(np.int)
    landmarks2 = np.matrix(im2_doc['landmarks'])[:, :2].astype(np.int)

    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])

    if os.environ['blur'] == 'feather':
        FEATHER_AMOUNT = max(im1.shape) // 20
        if FEATHER_AMOUNT % 2 == 0:
            FEATHER_AMOUNT += 1
    else:
        FEATHER_AMOUNT = 7

    mask = get_face_mask(im2, landmarks2)
    warped_mask = warp_im(mask, M, im1.shape)
    combined_mask = np.max([get_face_mask(im1, landmarks1), warped_mask], axis=0)

    warped_im2 = warp_im(im2, M, im1.shape)

    if os.environ['blur'] == 'no':
        output_im = im1 * (1.0 - combined_mask) + warped_im2 * combined_mask
    elif os.environ['blur'] == 'gaussian':
        warped_corrected_im2 = correct_colours(im1, warped_im2, landmarks1)
        output_im = im1 * (1.0 - combined_mask) + warped_corrected_im2 * combined_mask
    elif os.environ['blur'] == 'polygon':
        patch_img = img_proc.colorTransfer(im1, warped_im2, combined_mask)
        output_im = img_proc.blendImages(patch_img, im1, combined_mask, featherAmount=0.1 + 0.1 * random.random())
    elif os.environ['blur'] == 'feather':
        output_im = im1 * (1.0 - combined_mask) + warped_im2 * combined_mask
    else:
        raise ValueError("no such blur mode")

    cv2.imwrite(swap_path, output_im)
    cv2.imwrite(mask_path, combined_mask * 255)


