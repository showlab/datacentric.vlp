python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 pretrain_pure_caption.py \
--config ./configs/pretrain_pure_caption.yaml --output_dir output/Pretrain_pure_caption