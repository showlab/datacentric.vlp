## Install PyTorch


```bash
conda create -n ptp python==3.8
conda activate ptp
# CUDA 10.1
conda install pytorch==1.9.1 torchvision==0.10.1 torchaudio==0.9.1 cudatoolkit=10.1 -c pytorch
pip install -r requirements.txt
```