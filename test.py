import torch
from torchvision import models
from torchvision.io import read_image
import os
import argparse
import matplotlib.pyplot as plt

MODEL_DICT=['Alexnet','VGG19','Resnet152']
ROOT='D:\\allcodes\\智能信息网络\\实验1'

def load_model(name='VGG19'):
    """加载预训练模型
    :param name: 模型名称
    :return model: 模型
    :return category_names: 类别名称列表 
    :return preprocess: 图像预处理
    """
    if name=='Alexnet':
        model = models.alexnet
        weights=models.AlexNet_Weights.DEFAULT
    elif name == 'VGG19':
        model = models.vgg19
        weights=models.VGG19_Weights.DEFAULT
    elif name=='Resnet152' :
        model = models.resnet152
        weights = models.ResNet152_Weights.DEFAULT

    model=model(weights=weights.value)
    preprocess=weights.transforms()
    category_names=weights.meta["categories"]
    return model,category_names,preprocess

def save_image(image_path, title, model_name):
    """将张量转换成图像，并给图像加上标题，保存在指定路径
    :param image_path: 图像读取路径
    :param title: 图像标题
    :param model_name: 指定图像保存路径
    """
    img=plt.imread(image_path)
    plt.imshow(img)
    plt.title(title)
    plt.axis('off')
    filepath = f'{args.root}\\results\\{model_name}\\{title}.png'
    # 创建保存路径
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    plt.close()
 

def test(model, category_names, preprocess,model_name):
    """从指定路径中读取少量的图片，用于测试模型性能
    :param model: 预训练模型
    :param category_names: 所有预测类别的列表
    :param model_name: 选用的模型名称
    """
    imgs_path=f'{args.root}\\sub_images'
    img_paths=os.listdir(imgs_path)
    for img_path in img_paths:
        # print(img_path)
        img_path = f'{imgs_path}//{img_path}'
        img = read_image(img_path)
        img = preprocess(img).unsqueeze(0)
        #评估模式
        model.eval()
        #禁用梯度计算
        with torch.no_grad():
            prediction = model(img).squeeze(0).softmax(0)
        class_id = prediction.argmax().item()
        score = prediction[class_id].item()
        title = f"{category_names[class_id]} ({score*100:.2f}%)"
        save_image(img_path, title, model_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--model_dict", default=MODEL_DICT)
    args = parser.parse_args()
    print(args.model_dict)
    for name in args.model_dict:
        model,category_names,preprocess=load_model(name)
        test(model,category_names,preprocess,name)
