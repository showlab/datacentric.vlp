import json
import os
import random

from torch.utils.data import Dataset

from PIL import Image
import numpy as np
from PIL import ImageFile
from PIL.Image import blend as blend
ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None

from data.utils import pre_caption
import os,glob

class pretrain_dataset(Dataset):
    def __init__(self, ann_file, laion_path, transform, img_mixup=False): 
        # self.img_root = "/CC3M/images" # "CC3M/images" # docker
        # self.img_root = "/CC3M/images"
        self.img_root = "/CC3M/images"
        # self.img_root = "/dataset/CC3M/images"
        self.ann_pretrain = []
        for f in ann_file:
            print('loading '+f)
            ann = json.load(open(f,'r'))
            self.ann_pretrain += ann
        
        self.laion_path = laion_path
        if self.laion_path:
            self.laion_files = glob.glob(os.path.join(laion_path,'*.json'))

            print('loading '+self.laion_files[0])
            with open(self.laion_files[0],'r') as f:
                self.ann_laion = json.load(f)  

            self.annotation = self.ann_pretrain + self.ann_laion
        else:
            self.annotation = self.ann_pretrain
            
        self.transform = transform
        self.img_mixup = img_mixup


    def reload_laion(self, epoch):
        n = epoch%len(self.laion_files)
        print('loading '+self.laion_files[n])
        with open(self.laion_files[n],'r') as f:
            self.ann_laion = json.load(f)      
        
        self.annotation = self.ann_pretrain + self.ann_laion    
        
    
    def __len__(self):
        return len(self.annotation)
    
    def __getitem__(self, index):    
        
        ann = self.annotation[index]   
        try:
            image = Image.open(os.path.join(self.img_root, ann['image'])).convert('RGB')   
        except FileNotFoundError:
            # print("choose another sample!")
            return self.__getitem__(random.randint(1, self.__len__()-1))
        except:
            return self.__getitem__(random.randint(1, self.__len__()-1))
        image = self.transform(image)
        caption = pre_caption(ann['caption'],30)
        return image, caption