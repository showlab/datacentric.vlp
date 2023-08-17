# select one sample of N similar samples
# padding text with
import numpy as np
import pandas as pd
import os
import cv2
import json
from fast_cluster import Kmeans
from sklearn.metrics.pairwise import cosine_similarity
import gc


# step1: load all training samples

img_feats = []
indexs = []

# select sample from train success 269
source_json = "metadata/cc3m/train_success_align_269.json"
annotation = json.load(open(source_json,'r'))

chunk_count = 0
feats = None
feats_dir = 'ExtractFeature/cc3m_codebook_feats'
num_subsets = len(os.listdir(feats_dir))
for subset in os.listdir(feats_dir):
    chunk_count += 1
    # if chunk_count > 1:
    #     break
    if chunk_count % 10 == 0:
        print("{} subsets loaded".format(chunk_count))
    feats = np.load(os.path.join(feats_dir, subset))
    img_feats.extend(feats['arr_0'])
    indexs.extend(feats['arr_1']) 
    del feats
    gc.collect()

# img_feats, text_feats, indexs, img_path, texts = feats['arr_0'], feats['arr_1'], feats['arr_2'], feats['arr_3'], feats['arr_4']
gallery = np.squeeze(img_feats)



print(gallery.shape)
# # step2: cluster 3M samples into 1000 clusters, each contain 30,00 samples
FastKmeans = Kmeans(1000)
# step2: cluster 3M samples into 100,000 clusters, each contain 30 samples
# FastKmeans = Kmeans(100000)

pseudoLabel = FastKmeans.cluster(gallery)
print([len(FastKmeans.images_lists[i]) for i in range(len(FastKmeans.images_lists))])


success_data = []
# training
out_json = "metadata/cc3m/train_full_codebook_select_1_of_4_1K_clusters.json"


N = 5
# step2: preserve only 1/N% data
for cluster in range(len(FastKmeans.images_lists)):
    # print(index)
    # find k nearest feats and gradually reduce the pooling 
    # step1: find 3 most similar samples
    # sub_gallery = FastKmeans.images_lists[cluster]
    sub_gallery = gallery[FastKmeans.images_lists[cluster]]
    gen_sample_num = len(sub_gallery)
    print("{} samples in cluser {}".format(gen_sample_num, cluster))
    for i in range(0, gen_sample_num, N):
        # step2: concat 3 captions together
        new_caption = annotation[FastKmeans.images_lists[cluster][i]]['caption']
        # /images/validation/0/910057478.jpg > validation/0/910057478.jpg
        new_img_path = annotation[FastKmeans.images_lists[cluster][i]]['image']
        # print(new_img_path, new_caption)
        success_data.append({'image': new_img_path, 'caption': new_caption})
        if i % 100 ==0:
            print("{}/{} finished in cluster {}/{}".format(i, gen_sample_num, cluster, len(FastKmeans.images_lists)))


# load the half itm scores file

ann = json.load(open('metadata/cc3m/cc3m_training_success_half_itm.json', 'r'))

img_paths = dict()
for i in range(len(ann)):
    img_paths[ann[i]['image']] = 0

# find the joint part
success_data_preserved = []
for i in range(len(success_data)):
    if success_data[i]['image'] in img_paths.keys():
        success_data_preserved.append(success_data[i])
    if i % 1000 == 0:
        print("{}/{} finished".format(i, len(success_data)))

print("{} samples preserved".format(len(success_data_preserved)))

#  665966 samples preserved, train_full_codebook_select_5_10_half_itm_1K_clusters.json

with open(out_json, 'w') as outfile:
    json.dump(success_data_preserved, outfile)