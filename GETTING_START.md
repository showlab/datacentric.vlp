## 1. Prepare Dataset

### CC3M
Step1: First download train/val/test annotation files include URL from (google-research-datasets)[https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md].

Step2: We provided our script for downloading and split CC3M into subsplit in [cc3m_download.py](https://huggingface.co/sail/PTP/blob/main/download_cc3m.py). It's better to use our cript for downloading as the filename maybe different with different preprocess.

### LAION40M
Follow [img2dataset](https://github.com/rom1504/img2dataset/blob/main/dataset_examples/laion400m.md).
What's different is download image resoluation of 512x512. And only download first 1/10 metadata, it requires 4.04TB.



## 1. Download VIT Model
Download VIT pretrained model and place into dir.

```
mkdir pretrained_models && cd pretrained_models;
wget -c https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth;
```


## 2. Pre-train on CC3M/YFCC/LAION40M

We give a example for BLIP model as below:

```
python -m torch.distributed.launch --nnodes=4 --nproc_per_node=8 pretrain_vq_compress.py \
--config ./configs/codebook_cc3m.yaml --output_dir output/cc3m_experiment

```