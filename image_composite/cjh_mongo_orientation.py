import os
from pymongo import MongoClient
import tqdm

import random
import numpy as np

import cjh_o2util as utils

random.seed(0)

source_dir = "/data/jongho/o2/data/faces/"

face_paths = os.listdir(source_dir)

print('# of faces (total): ', len(face_paths))

face_ids = [os.path.splitext(path)[0] for path in face_paths]

# prepare mongo
client = MongoClient()
db = client['o2']
mongo_face = db['face']

docs = map(lambda _id: mongo_face.find_one(filter={'_id': _id}), face_ids)
docs = list(filter(lambda doc: (doc is not None) and ('landmarks' in doc), docs))
paths = [source_dir + doc['_id'] + ".png" for doc in docs]

fov = 41

for doc in tqdm.tqdm(docs):
    ldmk = doc['landmarks']
    im_id = doc['_id']
    im_path = source_dir + im_id + ".png"
    im_width = doc['width']
    im_height = doc['height']
    ldmk = np.array(ldmk)
    preds = np.ascontiguousarray(ldmk[:, 0:2])
    sz = (im_height, im_width)
    feature_point, t_68, r_68 = utils.get_tr(preds, fov, sz[0], sz[1], False, 1)
    rt_68, rr_68 = utils.convert_for_render(fov, sz[0], sz[1],
                                            feature_point, t_68, r_68)

    mongo_face.update_one({'_id': im_id}, {'$set': {'position': rt_68, 'rotation':rr_68}}, upsert=True)
