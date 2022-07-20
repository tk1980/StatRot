# Rotation Regularization Without Rotation

The Pytorch implementation for the ECCV2022 paper of "[Rotation Regularization Without Rotation](https://staff.aist.go.jp/takumi.kobayashi/publication/2022/ECCV2022.pdf)" by [Takumi Kobayashi](https://staff.aist.go.jp/takumi.kobayashi/).

### Citation

If you find our project useful in your research, please cite it as follows:

```
@inproceedings{kobayashi2022eccv,
  title={Rotation Regularization Without Rotation},
  author={Takumi Kobayashi},
  booktitle={Proceedings of European Conference on Computer Vision (ECCV)},
  year={2022}
}
```

## Usage

### Training
For example, the ResNet-10 is trained from scratch with our regularization on ImageNet-LT dataset by
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_train.py --dataset imagenetlt  --data ./datasets/imagenetlt --arch ResNet10Feature  --epochs 180 --out-dir ./results/imagenetlt/ResNet10Feature/train_1st_stage --workers 12 

CUDA_VISIBLE_DEVICES=0,1 python main_train.py --dataset imagenetlt  --data ./datasets/imagenetlt --arch ResNet10Feature  --epochs 30 --out-dir ./results/imagenetlt/ResNet10Feature/train_2nd_stage --workers 12 --first-model-file ./results/imagenetlt/ResNet10Feature/train_1st_stage/model_best.pth.tar
```

Note that the ImageNet-LT dataset must be downloaded at `./datasets/imagenetlt/` before the training and here we follow the imbalance-aware 2-stage training procedure presented in [1].

You can also apply our regularization to the framework of logit adjustment [2] by
```bash
CUDA_VISIBLE_DEVICES=0,1 python main_train.py --dataset imagenetlt  --data ./datasets/imagenetlt --arch ResNet10Feature --logit-adjust  --epochs 180 --out-dir ./results/imagenetlt/ResNet10Feature/logit_adjust --workers 12 
```

## Results

#### ImageNet

| Method  | ImageNet-LT |
|---|---|
| 2-stage [1] | 56.40   |
| logit-adjust [2]| 55.40   | 


## References

[1] Bingyi Kang, Saining Xie, Marcus Rohrbach, Zhicheng Yan, Albert Gordo, Jiashi Feng and Yannis Kalantidis. "DECOUPLING REPRESENTATION AND CLASSIFIER FOR LONG-TAILED RECOGNITION." In ICLR, 2020.

[2] Aditya Krishna Menon, Sadeep Jayasumana, Ankit Singh Rawat, Himanshu Jain, Andreas Veit and Sanjiv Kumar. "LONG-TAIL LEARNING VIA LOGIT ADJUSTMENT." In ICLR, 2021.



## Contact
takumi.kobayashi (At) aist.go.jp


## Acknowledgement
The class-wise sampler `utils/ClassAwareSampler.py` is from the [Classifier-Balancing](https://github.com/facebookresearch/classifier-balancing).