import torch
import torch.nn as nn
import torchvision.models as models
from torchinfo import summary
def Vgg19() :
    vgg19 = models.vgg19()
    total_neurons = 0
    input_neurons=224*224*3
    n_c=1
    n_l=1
    layer_n=0
    input_tensor = torch.randn(1, 3, 224, 224)
    total_neurons+=input_neurons
    print("input:",input_neurons,"个神经元")
    for layer in vgg19.children():
        if isinstance(layer, nn.AdaptiveAvgPool2d):
                input_tensor = layer(input_tensor)
                input_tensor=input_tensor.view(input_tensor.size(0), -1)
                layer_n+=1
                continue
        for sub_layer in layer.children() :
            input_tensor = sub_layer(input_tensor)
            layer_n+=1
            if isinstance(sub_layer, nn.Conv2d) :
                currenct_num_neurons = torch.prod(torch.tensor(input_tensor.shape[1:])).item()
                num_neurons=currenct_num_neurons
                total_neurons += num_neurons
                # print(input_tensor.shape)
                print(f"{sub_layer.__class__.__name__}_{layer_n}: {currenct_num_neurons} 个神经元")
                n_c+=1
                continue     
            if isinstance(sub_layer, nn.Linear) :
                currenct_num_neurons = torch.prod(torch.tensor(input_tensor.shape[1:])).item()
                num_neurons=currenct_num_neurons
                total_neurons += num_neurons
                print(f"{sub_layer.__class__.__name__}_{layer_n}: {currenct_num_neurons} 个神经元")
                n_l+=1
    print(f"共有{n_c}个卷积层")
    print(f"共有{n_l}个全连接层")
    print(f"总神经元数: {total_neurons}")
def to_list(layer):
  """计算所有子层
  :return: 所有子层
  """
  return list(layer.children())
def calculate(layer,input_tensor): 
     """单独计算特殊结构的神经元个数
     :return: 神经元个数
     """
     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear) :
        currenct_num_neurons = torch.prod(torch.tensor(input_tensor.shape[1:])).item()
        return currenct_num_neurons
     else:
        return 0     
def Resnet152() :
    i=0
    resnet152 = models.resnet152()
    total_neurons = 0 
    input_neurons=224*224*3
    total_neurons_downsample=256*56*56+512*28*28+1024*14*14+2048*7*7
    input_tensor = torch.randn(1, 3, 224, 224)
    for layer in resnet152.children():
        if isinstance(layer,nn.Linear):
            input_tensor=input_tensor.view(input_tensor.size(0), -1)
        if to_list(layer)==[]:      
            input_tensor=layer(input_tensor)
            x=calculate(layer,input_tensor)
            total_neurons+=x
            continue
        for sub_layer in layer.children():
            for ssub_layer in sub_layer.children():
                if to_list(ssub_layer)==[] :
                    input_tensor=ssub_layer(input_tensor)
                    x=calculate(ssub_layer,input_tensor)
                    total_neurons+=x
    print(f'卷积层神经元个数：{total_neurons-1000}')
    print(f'输入层神经元个数：{input_neurons}')
    print(f'下采样神经元个数：{total_neurons_downsample}')
    print(f'全连接层神经元个数：{1000}')
    total_neurons+=(total_neurons_downsample+input_neurons)
    print(f'总神经元个数：{total_neurons}')
def print_model(name):
    if name=='Alexnet':
        model = models.alexnet()
    elif name == 'VGG19':
        model = models.vgg19()
    elif name=='Resnet152' :
        model = models.resnet152()
    else:
        print("无效的模型名称。请选择'Alexnet'、'VGG19'或'Resnet152'之一。")
        return
    summary(model, (1,3, 224, 224))
if __name__ == '__main__' :
    MODEL_DICT=['Alexnet','VGG19','Resnet152']
    print_model(MODEL_DICT[0])
    print("-------VGG19-------")
    Vgg19()
    print("-------Resnet152-------")
    Resnet152()
