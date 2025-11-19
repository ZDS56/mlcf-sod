# MLCF-SOD: Multi-level feature enhancement and cross-layer fusion for small object ship detection


Our config file is at [configs/MY](configs/MY)

## Requirement

Please follow [docs/en/get_started.md](docs/en/get_started.md) and install the [mmdetection](https://github.com/open-mmlab/mmdetection) toolbox. 

a. Install [Pytorch 1.10.1](https://pytorch.org/)

b. Install [MMDetection](https://mmdetection.readthedocs.io/en/latest/) toolbox, required mmdet >= 2.7.0, mmcv-full >= 1.4.2. 

- Our project utilizes mmdet == 2.24.1, mmcv-full == 1.5.1

c. Install albumentations and other packages.

```shell
pip install nltk
pip install -r requirements/albu.txt
```

d. Install our Sparse Convolution Implementation

```shell
cd ./Sparse_conv
python setup.py install
cd ..
```



### 2. Training

```shell
% training on a single GPU
python tools/train.py /path/to/config-file --work-dir /path/to/work-dir

% training on multi GPUs
bash tools/dist_train.sh /path/to/config-file num-gpus --work-dir /path/to/work-dir
```



### 3. Test

```shell
python tools/test.py /path/to/config-file /path/to/work-dir/latest.pth --eval bbox
```








