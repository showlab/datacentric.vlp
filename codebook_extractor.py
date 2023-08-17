import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from models.blip_pretrain_vq_compress import blip_pretrain
import utils
from utils import warmup_lr_schedule, step_lr_schedule
from data import create_dataset, create_sampler, create_loader
import math

def extract_codebook(model, data_loader, device, config):
    model.train()  
    
    data_loader.sampler.set_epoch(0)

    sample_per_npz = 1000

    # if os.path.exists('ExtractFeature/laion40m_feats/codebook_subset_{}.npz'.format(i//sample_per_npz)):
    #     continue
    codebook_feats = []
    indexs = []

    for i, (image, caption, index) in enumerate(data_loader):
        
        image = image.to(device,non_blocking=True)
  
        alpha = config['alpha']*min(1,(0*len(data_loader)+i)/(2*len(data_loader))) 

        perplexity, min_encodings, min_encoding_indices = model(image, caption, alpha=alpha, return_codebook=True)  

        min_encoding_indices = min_encoding_indices.view(image.size(0), -1)

        codebook_feats.extend(min_encoding_indices.cpu().detach().numpy())

        indexs.extend(index.cpu().detach().numpy())
        # print(min_encoding_indices.size(), index)
        if (i % sample_per_npz == 0 and i > 0):
            print("{}/{} finished".format(i, len(data_loader)))
            np.savez('../ExtractFeature/laion40m_codebook_feats_val/codebook_subset_{}.npz'.format(i//sample_per_npz), np.array(codebook_feats), np.array(indexs))
            # print(indexs)
            # print(len(codebook_feats[0])) # 591
            codebook_feats = []
            indexs = []
        if i == len(data_loader) - 1:
            np.savez('../ExtractFeature/laion40m_codebook_feats_val/codebook_subset_{}.npz'.format(i//sample_per_npz+1), np.array(codebook_feats), np.array(indexs))
            codebook_feats = []
            indexs = []
        # if i > 1:
        #     break
    return True


def main(args, config):
    utils.init_distributed_mode(args)    
    
    device = torch.device(args.gpu)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True
    cudnn.deterministic = True
    #### Dataset #### 
    print("Creating dataset")
    datasets = [create_dataset('pretrain', config, min_scale=0.2)]
    print('number of training samples: %d'%len(datasets[0]))

    num_tasks = utils.get_world_size()
    global_rank = utils.get_rank()            
    samplers = create_sampler(datasets, [False], num_tasks, global_rank) # do not shuffle         

    data_loader = create_loader(datasets,samplers,batch_size=[config['batch_size']], num_workers=[args.num_workers], is_trains=[False], collate_fns=[None])[0]      

    print("="*50)
    print("time now is: ")
    print(time.strftime('%Y/%m/%d %H:%M:%S'))
    print("="*50)
    #### Model #### 
    print("Creating model")
    model = blip_pretrain(image_size=config['image_size'], vit=config['vit'], vit_grad_ckpt=config['vit_grad_ckpt'], 
                            vit_ckpt_layer=config['vit_ckpt_layer'], queue_size=config['queue_size'])

    # model = model.to(device)   
    model = model.cuda()

    optimizer = torch.optim.AdamW(params=model.parameters(), lr=config['init_lr'], weight_decay=config['weight_decay'])
    
    start_epoch = 0
    if args.checkpoint:    
        checkpoint = torch.load(args.checkpoint, map_location='cpu') 
        state_dict = checkpoint['model']    
        model.load_state_dict(state_dict)    
        
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch']+1                
        print('resume checkpoint from %s'%args.checkpoint)    
    
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module    
        
    print("Start extracting")
    start_time = time.time()      
    extract_codebook(model, data_loader, device, config) 
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Codebook extracting time {}'.format(total_time_str)) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/pretrain.yaml')
    parser.add_argument('--output_dir', default='output/Pretrain')  
    parser.add_argument('--checkpoint', default='')    
    parser.add_argument('--evaluate', action='store_true')    
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument("--num_workers", default=12, type=int, help="""Number of data loading workers per GPU.""", )
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=True, type=bool)
    parser.add_argument(
        "--local_rank",
        default=0,
        type=int,
        help="""local rank for distrbuted training.""",
    )
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))    
    
    # print("job beginning!")

    main(args, config)