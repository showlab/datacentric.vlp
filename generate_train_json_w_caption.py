import zlib
import pandas as pd
import os
import json
import cv2
from csv import reader
from PIL import Image


# generated sample

# src_json_2 = "Code/BLIP/metadata/cc12m/train_success.json"
# src_json_1 = "Code/BLIP/metadata/cc12m/train_predict.json"
# src_tsv_1 = "Code/BLIP/metadata/cc12m/original_train.tsv"
# out_json = "Code/BLIP/metadata/cc12m/cc12m_train_success_w_generated_caption_behind.json"

src_json_2 = "Code/BLIP/metadata/laion40m/train_success.json"
src_json_1 = "Code/BLIP/metadata/laion40m/train_predict.json"
src_tsv_1 = "Code/BLIP/metadata/laion40m/original_train.tsv"
out_json = "Code/BLIP/metadata/laion40mm/laion40m_train_success_w_generated_caption_behind.json"

# src_json_2 = "Code/BLIP/metadata/yfcc15m/train_success.json"
# src_json_1 = "Code/BLIP/metadata/yfcc15m/train_predict.json"
# src_tsv_1 = "Code/BLIP/metadata/yfcc15m/original_train.tsv"
# out_json = "Code/BLIP/metadata/yfcc15m/yfcc15m_train_success_w_generated_caption_behind.json"


# step1: load generated caption data for cc12m
success_data = []
success_count = 0

with open(src_tsv_1, 'r') as read_obj:
    csv_reader = reader(read_obj, delimiter='\t')
    meta_data = list(csv_reader)
print("{} original sample".format(len(meta_data)))
# 8868095 original sample
ann = json.load(open(src_json_1, 'r'))
print("{} generated caption sample".format(len(ann)))
# 8777643 generated sample
for i in range(len(ann)):
    # print(ann[i])
    index = int(ann[i]['image_id'])
    # print(meta_data[index])
    try:
        meta_data[index][0] = meta_data[index][0] + '; ' +  ann[i]['caption']
        # meta_data[index][0] = ann[i]['caption']
    except Exception as e:
        print(index)
        print(e)

for i in range(len(meta_data)):
    meta_data[i][1] = meta_data[i][1].split('_')[-1]


# become to a key val pair

caption_key_dict = {}
for i in range(len(meta_data)):
    caption_key_dict[meta_data[i][1]] = meta_data[i][0]


# step2: padding generated caption to source data

ann2 = json.load(open(src_json_2, 'r'))

for i in range(len(ann2)):
    sample = ann2[i]
    unique_key = sample['image'].split('/')[-1].split('.')[0]
    # success downloaded 
    if unique_key in caption_key_dict.keys():
        success_data.append({'image': sample['image'], 'caption': caption_key_dict[unique_key]})
        success_count += 1
    else:
        success_data.append({'image': sample['image'], 'caption': sample['caption']})
    if i % 1000 == 0:
        print("{}/{} finished".format(i, len(meta_data)))

# step3: save to 
# 8344580/10375529 success loaded caption
# success: 8254334/10247372 success loaded caption
# 37528731/40045982 success loaded caption
print("{}/{} success loaded caption".format(success_count, len(ann2)))

with open(out_json, 'w') as outfile:
    json.dump(success_data, outfile)