# MLCF-SOD: Multi-level feature enhancement and cross-layer fusion for small object ship detection
The repo is the official implementation of CEASC.


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

#### Checkpoints: 
We provide the following checkpoints:
- RetinaNet baseline, corresponding to [baseline_retinanet_res18_visdrone](https://github.com/Cuogeihong/CEASC/blob/main/configs/UAV/baseline_retinanet_res18_visdrone.py): [Google Drive](https://drive.google.com/drive/folders/1Ws5UQri07GGZo_PUyGjFBQA5cI3pjn2K?usp=sharing)
- RetinaNet CEASC, corresponding to [dynamic_retinanet_res18_visdrone](https://github.com/Cuogeihong/CEASC/blob/main/configs/UAV/dynamic_retinanet_res18_visdrone.py): [Google Drive](https://drive.google.com/drive/folders/1Gu0D5XULRkMEGNTGGKNj7X6-WiZs2a34?usp=sharing)


### 3. Test

```shell
python tools/test.py /path/to/config-file /path/to/work-dir/latest.pth --eval bbox
```

## Citation

If you find our paper or this project helps your research, please kindly consider citing our paper in your publication.

```
@misc{ceasc,
      title={Adaptive Sparse Convolutional Networks with Global Context Enhancement for Faster Object Detection on Drone Images}, 
      author={Bowei Du and Yecheng Huang and Jiaxin Chen and Di Huang},
      year={2023},
      eprint={2303.14488},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```






