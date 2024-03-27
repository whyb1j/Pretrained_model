### 任务描述

预训练模型图像分类

### 数据来源

数据集：[mini-ImageNet](https://pan.baidu.com/s/1bQTtrkEgWfs_iaVRwxPF3Q#list/path=%2Fsharelink2168611027-758932909271168%2Fmini-imagenet&parentPath=%2Fsharelink2168611027-758932909271168)   密码：33e7   （仅使用了 test.csv ）

JSON文件：[标签映射](https://github.com/WZMIAOMIAO/deep-learning-for-image-processing/blob/master/pytorch_classification/mini_imagenet/imagenet_class_index.json)

### 模型在测试集上的准确率

| 模型名称  | 准确率 |
| --------- | ------ |
| AlexNet   | 75.73% |
| VGG19     | 85.29% |
| ResNet152 | 89.83% |

### 文件目录说明

```markdown
filetree 
├── /images
├── /results/
|  ├── /Alexnet
|  ├── /VGG19
|  ├── /Resnet152
├── /sub_images
├── imagenet_class_index.json
├── evaluate.py
├── test.py
├── model.py
├── README.md
├── test.csv
```

### 用法

#### 模型评估
从images文件夹中加载数据
```shell
python evaluate.py --root 'your dir' --batch_size 16 --num_workers 4 --device 'cuda:0'
```
输出结果为三个模型的准确率
#### 图像预测
从sub_images文件夹中加载数据(自己从数据集中挑选)
```shell
python test.py 
```
输出结果保存在results文件夹中
#### 模型打印

```shell
python model.py
```


