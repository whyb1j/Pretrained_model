from torchvision.io import read_image
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
import torch
import json
import csv
from PIL import Image
from tqdm import tqdm
import argparse

MODEL_DICT=['Alexnet','VGG19','Resnet152']
ROOT='D:\\allcodes\\智能信息网络\\实验1'

class mini_imagenet(Dataset):
    """
    重构Dateset类
    """
    def __init__(self,root,transform=None) :
        """初始化
        :param root: 根目录
        :param transform: 规定预处理方式 
        """
        #根目录
        self.root=root    
        #定义预处理方式
        if transform==None :
            self.transform = transforms.Compose([
                transforms.Resize(256),                     # 调整图像大小为256x256
                transforms.CenterCrop(224),                 # 中心裁剪为224x224
                transforms.ToTensor(),                      # 转换为Tensor
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化
            ])
        else :
            self.transform=transform
        self.get_map()
        #读取csv文件
        csv_path=f'{self.root}\\test.csv'
        with open(csv_path, 'r') as file:
            csv_reader = csv.reader(file)
            filename=[]
            class_id=[]
            next(csv_reader)
            for row in csv_reader :
                    filename.append(row[0])
                    class_id.append(row[1])
       # print(len(filename))      
        self.image_path_list=[]
        self.index_list=[]
        for i in range(len(filename)):
            image_path=f'{self.root}\\images\\{filename[i]}'
            self.image_path_list.append(image_path)           
            self.index_list.append(self.class2index[class_id[i]])
        #print(len(self.image_path_list))
    
    def __len__(self):
        """计算图像数量
        :return: 图像的数量
        """
        return len(self.image_path_list)
    
    def __getitem__(self, num):
        """返回数据集中对应索引位置的图像和其对应的类别索引
        :param num:索引位置
        :return: 图像和类别索引
        """
        image_path = self.image_path_list[num]
        img = Image.open(image_path)
        index = self.index_list[num]
        img = self.transform(img)
        return img, index
    
    def get_map(self) :
        """
        读取映射关系
        """
        class2index_path=f'{self.root}\\imagenet_class_index.json'
        with open(class2index_path,'r') as f :
            map=json.load(f)
        self.class2index={}
        self.index2class={}
        for index,(class_id,label) in enumerate(map.values()) :
            self.class2index[class_id] = int(index)
            self.index2class[index] = class_id

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

def evaluate(model,data_loader,model_name):
    """评估模型的准确率
    :param model:预训练模型
    :param data_loader: 数据加载器
    :param model_name: 模型名称
    """
    device=args.device
    #转移到同一个设备上
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad(): 
        for images, labels in tqdm(data_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    print(f"Accuracy of {model_name} on the test images: {accuracy * 100:.2f}%")
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # 添加命令行参数
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--model_dict", default=MODEL_DICT)
    parser.add_argument("--batch_size", default=16,type=int)
    parser.add_argument("--num_workers", default=2,type=int)  
    parser.add_argument("--device", default='cuda:0')  
    args = parser.parse_args()
    #加载数据集
    dataset = mini_imagenet(args.root)
    dataloader = DataLoader(dataset, batch_size=args.batch_size,
                            shuffle=False,num_workers=args.num_workers)
    # print(args.model_dict)
    for name in args.model_dict:
        model, category_names, preprocess = load_model(name)
        evaluate(model, dataloader, name)

