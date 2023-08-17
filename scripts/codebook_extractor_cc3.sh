python -m torch.distributed.launch --nproc_per_node=8 --master_port=1234 codebook_extractor.py \
--config ./configs/codebook.yaml --output_dir output/Codebook_extractor \
--checkpoint pretrained_models/vq_compress_v1_last.pth