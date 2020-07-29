from IPython.core.interactiveshell import InteractiveShell # 표를 이쁘게 만들어주는 기능
import seaborn as sns # 데이터 분포를 시각화해주는 라이브러리
# PyTorch
# torchvision : 영상 분야를 위한 패키지, ImageNet, CIFAR10, MNIST와 같은 데이터셋을 위한 데이터 로더와 데이터 변환기 등이 포함되어 있다.
from torchvision import transforms, datasets, models
import torch

# optim : 가중치를 갱신할 Optimizer가 정의된 패키지. SGD + momentum, RMSProp, Adam등과 같은 알고리즘이 정의되어 있다.
# cuda : CUDA 텐서 유형에 대한 지원을 추가하는 패키지이다. CPU텐서와 동일한 기능을 구현하지만 GPU를 사용하여 계산한다.
from torch import optim, cuda

# DataLoader : 학습 데이터를 읽어오는 용도로 사용되는 패키지.
# sampler : 데이터 세트에서 샘플을 추출하는 용도로 사용하는 패키지
from torch.utils.data import DataLoader, sampler
import torch.nn as nn

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Data science tools
import numpy as np
import pandas as pd # Pandas : Data science를 위한 패키지이다.
import os

# Image manipulations
from PIL import Image
# Useful for examining network
from torchsummary import summary
# Timing utility
from timeit import default_timer as timer

# Visualizations
# import matplotlib
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt # matplotlib를 쓸때 seaborn이 있는것과 없는것이 생긴게 다르다.
plt.rcParams['font.size'] = 14

# Printing out all outputs

InteractiveShell.ast_node_interactivity = 'all'

import datetime
import sys

# 어떤 모델을 학습할 것인지
model_choice = "resnet18" # 이거만 바꾸면 된다.

# 몇 epoch 학습할 것인지
training_epoch = 100

# 배치 사이즈 조절
batch_size = 128

selected_opti = 'sgd'

Early_stop = True

def get_date():
    now = datetime.datetime.now()
    nowDatetime = now.strftime('%Y%m%d%H%M%S')
    return  nowDatetime

def save_result_to_txt(model_name):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, model_name + '_Results/')
    save_txt = "result_txt_" + get_date() + ".txt"

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)


    sys.stdout = open(results_dir + save_txt, 'w')


def setting_save_folder(save_file_name, model_name):
    script_dir = os.path.dirname(__file__)
    results_dir = os.path.join(script_dir, model_name + '_Results/')
    save_file = save_file_name  +"_bts" +str(batch_size) + "_ep" + str(training_epoch) + "_" + get_date()

    if not os.path.isdir(results_dir):
        os.makedirs(results_dir)

    return results_dir, save_file


def get_pretrained_model_last_layer_change(model_name, n_classes):

    """
    :param model_name: 불러올 pre trained 모델 이름
    :param n_classes: 분류할 클래스 갯수
    :return: 불러온 모델의 분류기 부분만 수정한 구조를 리턴한다.
    """

    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

            # 모델 구조를 보면 알겠지만 classifier 부분은 6개의 레이어로 이루어져있다.
            # 여기에서 6번째 레이어의 in_features를 꺼내서 n_inputs에 담는 코드이다.
        # n_inputs = model.classifier[6].in_features
        # n_inputs = model.avgpool.in_features

        print("\t- 변경 전 레이어")
        # print(model.classifier[6])
        print(model.classifier)

        # Add on classifier
        # 모델의 classifier 부분의 6번째 레이어에 새로운 레이어를 넣는 부분이다.
        # Linear 레이어와 Softmax 레이어가 들어간다.
        # Linear 레이어는 Fully-Connected Layer와 동일한 역할을 한다.
        # model.classifier[6] = nn.Sequential(
        #    nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=9216, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=4096, out_features=n_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        print("\t- 변경 후 레이어")
        #print(model.classifier[6])
        print(model.classifier)

    # VGG
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.classifier[0].in_features

        print("\t- 변경 전 레이어")
        # print(model.classifier[6])
        print(model.classifier)

        # Add on classifier
        # 모델의 classifier 부분의 6번째 레이어에 새로운 레이어를 넣는 부분이다.
        # Linear 레이어와 Softmax 레이어가 들어간다.
        # Linear 레이어는 Fully-Connected Layer와 동일한 역할을 한다.
        # model.classifier[6] = nn.Sequential(
        #    nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        print("\t- 변경 후 레이어")
        # print(model.classifier[6])
        print(model.classifier)

    elif model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.classifier[0].in_features

        print("\t- 변경 전 레이어")
        # print(model.classifier[6])
        print(model.classifier)


        # Add on classifier
        # 모델의 classifier 부분의 6번째 레이어에 새로운 레이어를 넣는 부분이다.
        # Linear 레이어와 Softmax 레이어가 들어간다.
        # Linear 레이어는 Fully-Connected Layer와 동일한 역할을 한다.
        # model.classifier[6] = nn.Sequential(
        #    nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        print("\t- 변경 후 레이어")
        # print(model.classifier[6])
        print(model.classifier)

    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.classifier[0].in_features

        print("\t- 변경 전 레이어")
        # print(model.classifier[6])
        print(model.classifier)

        # Add on classifier
        # 모델의 classifier 부분의 6번째 레이어에 새로운 레이어를 넣는 부분이다.
        # Linear 레이어와 Softmax 레이어가 들어간다.
        # Linear 레이어는 Fully-Connected Layer와 동일한 역할을 한다.
        # model.classifier[6] = nn.Sequential(
        #    nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        print("\t- 변경 후 레이어")
        # print(model.classifier[6])
        print(model.classifier)

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.classifier[0].in_features

        print("\t- 변경 전 레이어")
        # print(model.classifier[6])
        print(model.classifier)

        # Add on classifier
        # 모델의 classifier 부분의 6번째 레이어에 새로운 레이어를 넣는 부분이다.
        # Linear 레이어와 Softmax 레이어가 들어간다.
        # Linear 레이어는 Fully-Connected Layer와 동일한 역할을 한다.
        # model.classifier[6] = nn.Sequential(
        #    nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))
        model.classifier = nn.Sequential(
            nn.Linear(in_features=n_inputs, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5, inplace=False),
            nn.Linear(in_features=4096, out_features=n_classes, bias=True),
            nn.LogSoftmax(dim=1)
        )

        print("\t- 변경 후 레이어")
        # print(model.classifier[6])
        print(model.classifier)

    elif model_name == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    elif model_name == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)

    # ResNet
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        # ResNet 50의 경우 분류기 부분이 (fc): Linear(in_features=2048, out_features=1000, bias=True) 형식으로 되어있다.
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        print("\t- 변경 전 레이어")
        print(model.fc)

        n_inputs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    # Inception
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    # DenseNet
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier.in_features

        print("\t- 변경 전 레이어")
        print(model.classifier)

        # Add on classifier
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True),
            nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.classifier)

    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier.in_features

        print("\t- 변경 전 레이어")
        print(model.classifier)

        # Add on classifier
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True),
            nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.classifier)

    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier.in_features

        print("\t- 변경 전 레이어")
        print(model.classifier)

        # Add on classifier
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True),
            nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.classifier)

    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)


        for param in model.parameters():
            param.requires_grad = False
        n_inputs = model.classifier.in_features

        print("\t- 변경 전 레이어")
        print(model.classifier)

        # Add on classifier
        model.classifier = nn.Sequential(
            nn.Linear(n_inputs, n_classes), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.classifier)
    # MobileNet V2
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        # Freeze early layers
        for param in model.parameters():
            param.requires_grad = False

        print("\t- 변경 전 레이어")
        print(model.classifier[1])

        n_inputs = model.classifier[1].in_features

        # Add on classifier
        model.classifier[1] = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True),
            nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.classifier[1])

    # ResNeXt
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    elif model_name == 'resnext101':
        model = models.resnext101_32x8d(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    # ShuffleNet
    elif model_name == 'shufflenet_v2_05':
        model = models.shufflenet_v2_x0_5(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)

    elif model_name == 'shufflenet_v2_10':
        model = models.shufflenet_v2_x1_0(pretrained=True)

        for param in model.parameters():
            param.requires_grad = False

        n_inputs = model.fc.in_features

        print("\t- 변경 전 레이어")
        print(model.fc)

        model.fc = nn.Sequential(
            nn.Linear(n_inputs, n_classes, bias=True), nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.fc)
    elif model_name == 'shufflenet_v2_15':
        model = models.shufflenet_v2_x1_5(pretrained=True)
    elif model_name == 'shufflenet_v2_20':
        model = models.shufflenet_v2_x2_0(pretrained=True)

    # SqueezeNet
    elif model_name == 'squeezenet1.0':
        model = models.squeezenet1_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        print("\t- 변경 전 레이어")
        print(model.classifier)

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 30, kernel_size=(1,1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.classifier)

    elif model_name == 'squeezenet1.1':
        model = models.squeezenet1_1(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False

        print("\t- 변경 전 레이어")
        print(model.classifier)

        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=False),
            nn.Conv2d(512, 30, kernel_size=(1,1), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.LogSoftmax(dim=1))

        print("\t- 변경 후 레이어")
        print(model.classifier)

    # MNASNet
    elif model_name == 'mnasnet05':
        model = models.mnasnet0_5(pretrained=True)
    elif model_name == 'mnasnet075':
        model = models.mnasnet0_75(pretrained=True)
    elif model_name == 'mnasnet10':
        model = models.mnasnet1_0(pretrained=True)
    elif model_name == 'mnasnet13':
        model = models.mnasnet1_3(pretrained=True)

    # WideResNet
    elif model_name == 'wideresnet50':
        model = models.wide_resnet50_2(pretrained=True)
    elif model_name == 'wideresnet101':
        model = models.wide_resnet101_2(pretrained=True)



    model = model.to('cuda')

    return model



def init_dataset():
    # ## 데이터셋 경로 / GPU 학습 가능 여부 확인
    #
    # - 불러올 데이터셋의 경로를 지정한다.
    # - train, validation, test 로 나눠져 있으므로, 각각의 경로를 지정한다.
    # - 학습된 모델을 저장할 이름을 지정한다.
    # - 배치크기를 지정한다.
    # - GPU에서 학습이 가능한지 확인한다.

    # Location of data
    datadir = '/home/kunde/DeepCNN/ingredient_data_TR7_VA2_TE1/'  # 데이터셋 경로
    traindir = datadir + 'train/'
    validdir = datadir + 'valid/'
    testdir = datadir + 'test/'

    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # 썩을 왜 이렇게 해놨어?
                transforms.RandomResizedCrop(size=256, scale=(0.08, 1.0)), # 0.08 ~ 1.0 이 값이 디폴트값
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
        # Validation does not use augmentation
        'val':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        # Test does not use augmentation
        'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }


    # Datasets from each folder
    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'val':
            datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
        'test':
            datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }

    return datadir, traindir, validdir, testdir, image_transforms, data, dataloaders

def init_cv_dataset(total_K, current_k_fold):
    # ## 데이터셋 경로 / GPU 학습 가능 여부 확인
    #
    # - 불러올 데이터셋의 경로를 지정한다.
    # - train, validation, test 로 나눠져 있으므로, 각각의 경로를 지정한다.
    # - 학습된 모델을 저장할 이름을 지정한다.
    # - 배치크기를 지정한다.
    # - GPU에서 학습이 가능한지 확인한다.

    # Location of data
    datadir = '/home/kunde/DeepCNN/ingredient_data_TR9_TE1/'+ str(total_K) +'_fold_cross_validation_dataset/'  # 데이터셋 경로
    traindir = datadir + 'K_'+ str(current_k_fold) + '/train/'
    validdir = datadir + 'K_'+ str(current_k_fold) + '/valid/'
    testdir = datadir + 'test/'

    print('\n----------------------------------------------------------------')
    print(f'{current_k_fold} fold 데이터셋 세팅')
    print(f'데이터셋 경로 : {datadir}')
    print(f'학습 데이터셋 경로 : {traindir}')
    print(f'검증 데이터셋 경로 : {validdir}')
    print(f'테스트 데이터셋 경로 : {testdir}')
    print('----------------------------------------------------------------\n')

    image_transforms = {
        # Train uses data augmentation
        'train':
            transforms.Compose([
                # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # 썩을 왜 이렇게 해놨어?
                transforms.RandomResizedCrop(size=256, scale=(0.08, 1.0)), # 0.08 ~ 1.0 이 값이 디폴트값
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(),
                transforms.RandomHorizontalFlip(),
                transforms.CenterCrop(size=224),  # Image net standards
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225])  # Imagenet standards
            ]),
        # Validation does not use augmentation
        'val':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
        # Test does not use augmentation
        'test':
            transforms.Compose([
                transforms.Resize(size=256),
                transforms.CenterCrop(size=224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ]),
    }


    # Datasets from each folder
    data = {
        'train':
            datasets.ImageFolder(root=traindir, transform=image_transforms['train']),
        'val':
            datasets.ImageFolder(root=validdir, transform=image_transforms['val']),
        'test':
            datasets.ImageFolder(root=testdir, transform=image_transforms['test'])
    }

    # Dataloader iterators
    dataloaders = {
        'train': DataLoader(data['train'], batch_size=batch_size, shuffle=True),
        'val': DataLoader(data['val'], batch_size=batch_size, shuffle=True),
        'test': DataLoader(data['test'], batch_size=batch_size, shuffle=True)
    }

    return datadir, traindir, validdir, testdir, image_transforms, data, dataloaders


def category_dataframe(traindir, validdir, testdir):
    # Empty lists
    categories = []
    img_categories = []
    n_train = []
    n_valid = []
    n_test = []
    hs = []
    ws = []

    # os.listdir(path) : path에 존재하는 파일, 서브폴더 목록을 가져온다.

    # Iterate through each category
    for d in os.listdir(traindir):  # train 데이터의 경로를 탐색한다. os.listdir을 사용하면 train 폴더 내의 폴더들을 순차적으로 탐색한다.
        categories.append(d)  # categories라는 리스트에 추가해준다. 폴더명을 카테고리 이름으로 해놨으므로 카테고리명이 저장된다.

        # Number of each image
        train_imgs = os.listdir(traindir + d)
        valid_imgs = os.listdir(validdir + d)
        test_imgs = os.listdir(testdir + d)
        n_train.append(len(train_imgs))
        n_valid.append(len(valid_imgs))
        n_test.append(len(test_imgs))

        # Find stats for train images
        for i in train_imgs:
            img_categories.append(d)
            img = Image.open(traindir + d + '/' + i)  # 이미지 열기
            img_array = np.array(img)
            # Shape
            hs.append(img_array.shape[0])
            ws.append(img_array.shape[1])

    # Dataframe of categories
    # Pandas 라이브러리를 이용한 부분. Dataframe은 테이블 형식의 데이터를 다룰때 사용한다. 컬럼, 로우(데이터), 인덱스로 이루어져있다.
    cat_df = pd.DataFrame({'category': categories,
                           'n_train': n_train,
                           'n_valid': n_valid,
                           'n_test': n_test}).sort_values('category')


    image_df = pd.DataFrame({
        'category': img_categories,
        'height': hs,
        'width': ws
    })

    return cat_df, image_df


def train(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          save_file_name,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=1,
          early_stop=True):
    """

    :param model: 학습할 모델을 입력받는다.
    :param criterion: 학습에 사용할 손실함수를 입력받는다
    :param optimizer: 학습에 사용할 최적화함수를 입력받는다
    :param train_loader: 학습에 사용할 training dataset을 입력받는다(dataloader 형식)
    :param valid_loader: 학습에 사용할 vaildation dataset을 입력받는다(dataloader 형식)
    :param save_file_name: 최적의 모델을 저장하기 위한 이름을 입력받는다.
    :param max_epochs_stop: 몇 만큼 vaild loss 값의 감소가 없다면 학습을 중단할지 설정한다.
    :param n_epochs: 최대 학습 epoch값을 입력받는다
    :param print_every: 몇 epoch마다 학습 상황을 출력할지 입력받는다
    :param early_stop: 조기 중단을 할지 말지 결정한다.
    :return:
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    train_on_gpu = cuda.is_available() # GPU를 사용할 수 있는지 없는지 판단한다.

    # Early stopping intialization
    epochs_no_improve = 0  # epoch을 진행해도 valid_loss의 감소가 없으면 1씩 올라간다.
    valid_loss_min = np.Inf  # np.Inf : 무한대

    valid_max_acc = 0  # ???
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:  # model이 아직 학습되지 않았다면 model.epochs라는 변수가 없을 것이다. 그래서 에러가 나기 때문에 except문이 실행된다.
        print(f'\n\n이미 {model.epochs} epochs 만큼 학습된 모델입니다. 추가학습을 시작합니다.\n')
    except:
        model.epochs = 0
        print('\n\n----------------------------------------------------------------')
        print('학습 시작')
        print(f'총 {n_epochs} epochs 학습할 예정입니다.')
        print('----------------------------------------------------------------\n')

    overall_start = timer()  # 학습에 들어가기전의 시간을 기록한다.

    # Main loop
    for epoch in range(n_epochs):  # 입력받은 Epochs 만큼 반복한다.

        # keep track of training and validation loss each epoch
        # train_loss와 vaild_loss, train_acc와 vaild_acc를 기록할 변수를 만든다.
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()  # 학습모드로 설정한다.
        start = timer()  # epochs의 시작 시간을 기록한다.

        # Training loop
        # data : 학습에 사용될 이미지 데이터, target : 이미지에 라벨링된 데이터(여기에서는 폴더명)
        for ii, (data, target) in enumerate(train_loader):

            # print('\r', f'\ntest : {data.size}, {target.size}', end='')
            # Tensors to gpu
            if train_on_gpu:  # GPU에서 트레이닝이 되는지 여부를 담은 변수
                data, target = data.cuda(), target.cuda()  # .cuda()메소드를 사용해서 GPU에서 연산이 가능하도록 바꿔준다.

            # Clear gradients
            optimizer.zero_grad()

            # Predicted outputs are log probabilities
            output = model(
                data)  # 여기에서 모델은 학습에 사용되는 VGG나 AlexNet과 같은 구조를 말한다. 이 모델은 함수로써 쓰이며 input값으로 데이터를 넣으면 output이 나온다.
            # 카테고리별 확률값이 저장되어 나올 것이다. softmax이므로 확률을 모두 더하면 1이 나온다.

            # Loss and backpropagation of gradients
            loss = criterion(output, target)  # loss 값 업데이트

            # 역전파 단계 : 모델의 매개변수에 대한 손실의 변화도를 계산한다.
            loss.backward()

            # 이 함수를 호출하면 매개변수가 갱신된다.
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            # loss는 (1,)형태의 Tensor이며, loss.item()은 loss의 스칼라 값이다.
            # 여기에서 data.size(0)는 배치사이즈를 말한다.
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1) # output에 저장된 확률값중 가장 높은 값을 가진 인덱스를 리턴한다..
            correct_tensor = pred.eq(target.data.view_as(pred)) #

            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print('\r',
                f'Epoch: {epoch}\t학습진행률 : {100 * (ii + 1) / len(train_loader):.2f}%' \
                + f'\t 현재 Epoch에서 걸린 시간 : {timer() - start:.2f}s' \
                + f'\t Train_Loss : {train_loss / len(train_loader.dataset):.4f}' \
                + f'\t Train_Acc : {100 * (train_acc / len(train_loader.dataset)):.2f}%',
                end='')  # end='\r' : 해당 줄의 처음으로 와서 다시 출력한다.

        # After training loops ends, start validation ===============================================
        else:  # 트레이닝 루프가 끝나면 실행되는 곳이다.
            model.epochs += 1  # 트레이닝 루프 한번을 반복했기 때문에 epoch을 1 올려준다.

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()  # 평가모드로 설정한다. pytorch에는 train(), eval() 두가지 모드밖에 없다. eval()모드에서는 드랍아웃이 작동하지 않는다.
                start_eval = timer()
                print('')
                # Validation loop
                for ii, (data, target) in enumerate(valid_loader):
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    # 평가시엔 역전파는 수행하지 않는다.
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)

                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)
                    print('\r',
                        f'\t\t\t평가진행률 : {100 * (ii + 1) / len(valid_loader):.2f}%' \
                        + f'\t 현재 Epoch에서 걸린 시간 : {timer() - start_eval:.2f}s' \
                        + f'\t Vaild_Loss : {valid_loss / len(valid_loader.dataset):.4f}' \
                        + f'\t Vaild_Acc : {100 * (valid_acc / len(valid_loader.dataset)):.2f}%',
                        end='')  # end='\r' : 해당 줄의 처음으로 와서 다시 출력한다.

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    # print(
                    #     f'\n\t\t\tTraining Loss: {train_loss:.4f} \t\t Validation Loss: {valid_loss:.4f}'
                    # )
                    # print(
                    #     f'\t\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    # )
                    print(
                        f'\n\t\t\t현재 Epochs에서 Train과 Vaild에 걸린 시간 : {timer() - start:.2f}s\n'
                    )

                # Save the model if validation loss decreases
                # 예를 들어보자. 초기 valid_loss_min이 무한대값이다. 당연히 epoch 0에선 이 값보다 작을수밖에 없다.
                # 따라서 valid_loss_min 값이 epoch 0에서의 valid_loss값으로 바뀐다.
                # epoch 1부터 valid_loss가 이전 epoch보다 작아지지 않는다면 epochs_no_improve 값이 증가한다.
                # 만약 작아지지 않는 상태가 max_epochs_stop 값보다 커지게 되면 중지한다.
                # 그 이유는 학습이 계속 진행되더라도 loss 값이 더 이상 작아지지 않으므로, 수렴했다고 볼 수 있기 때문이다.

                if valid_loss < valid_loss_min:
                    # Save model
                    # torch.save(model.state_dict(), save_file_name)  # 이때 저장되는 모델은 최적의 epochs를 가진 모델이다.
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1

                    # Trigger early stopping
                    if early_stop == True:  # Early_stop 옵션이 있는 경우에만 진행한다.
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stop! {max_epochs_stop} epochs 동안 valid loss 값이 감소하지 않았습니다.\n' \
                                + f'현재까지 진행한 총 epochs : {epoch}\t 최상의 epochs : {best_epoch} (loss: {valid_loss_min:.4f} and acc: {100 * valid_acc:.4f}%)'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'\n[ 총 학습시간 : {total_time:.2f}s, Epoch당 평균 학습 시간 : {total_time / (epoch + 1):.2f}s ]')

                            # Load the best state dict
                            # model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            model.optimizer = optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'
                                ])
                            return model, history

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start

    print('----------------------------------------------------------------')
    print('학습 결과')
    print(
        f'\n최상의 epoch 수: {best_epoch} (loss: {valid_loss_min:.4f}, acc: {100 * valid_acc:.4f}%)'
    )
    print(f'[ 총 학습시간 : {total_time:.2f}s, Epoch당 평균 학습 시간 : {total_time / (epoch + 1):.2f}s ]')
    print('----------------------------------------------------------------\n')

    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history


def train_cv(model,
          criterion,
          optimizer,
          train_loader,
          valid_loader,
          max_epochs_stop=3,
          n_epochs=20,
          print_every=1,
          early_stop=True):
    """

    :param model: 학습할 모델을 입력받는다.
    :param criterion: 학습에 사용할 손실함수를 입력받는다
    :param optimizer: 학습에 사용할 최적화함수를 입력받는다
    :param train_loader: 학습에 사용할 training dataset을 입력받는다(dataloader 형식)
    :param valid_loader: 학습에 사용할 vaildation dataset을 입력받는다(dataloader 형식)
    :param save_file_name: 최적의 모델을 저장하기 위한 이름을 입력받는다.
    :param max_epochs_stop: 몇 만큼 vaild loss 값의 감소가 없다면 학습을 중단할지 설정한다.
    :param n_epochs: 최대 학습 epoch값을 입력받는다
    :param print_every: 몇 epoch마다 학습 상황을 출력할지 입력받는다
    :param early_stop: 조기 중단을 할지 말지 결정한다.
    :return:
        model (PyTorch model): trained cnn with best weights
        history (DataFrame): history of train and validation loss and accuracy
    """

    train_on_gpu = cuda.is_available() # GPU를 사용할 수 있는지 없는지 판단한다.

    # Early stopping intialization
    epochs_no_improve = 0  # epoch을 진행해도 valid_loss의 감소가 없으면 1씩 올라간다.
    valid_loss_min = np.Inf  # np.Inf : 무한대

    valid_max_acc = 0  # ???
    history = []

    # Number of epochs already trained (if using loaded in model weights)
    try:  # model이 아직 학습되지 않았다면 model.epochs라는 변수가 없을 것이다. 그래서 에러가 나기 때문에 except문이 실행된다.
        print(f'\n\n이미 {model.epochs} epochs 만큼 학습된 모델입니다. 추가학습을 시작합니다.\n')
    except:
        model.epochs = 0
        print('\n\n----------------------------------------------------------------')
        print('학습 시작')
        print(f'총 {n_epochs} epochs 학습할 예정입니다.')
        print('----------------------------------------------------------------\n')

    overall_start = timer()  # 학습에 들어가기전의 시간을 기록한다.

    # Main loop
    for epoch in range(n_epochs):  # 입력받은 Epochs 만큼 반복한다.

        # keep track of training and validation loss each epoch
        # train_loss와 vaild_loss, train_acc와 vaild_acc를 기록할 변수를 만든다.
        train_loss = 0.0
        valid_loss = 0.0

        train_acc = 0
        valid_acc = 0

        # Set to training
        model.train()  # 학습모드로 설정한다.
        start = timer()  # epochs의 시작 시간을 기록한다.

        # Training loop
        # data : 학습에 사용될 이미지 데이터, target : 이미지에 라벨링된 데이터(여기에서는 폴더명)
        for ii, (data, target) in enumerate(train_loader):

            # print('\r', f'\ntest : {data.size}, {target.size}', end='')
            # Tensors to gpu
            if train_on_gpu:  # GPU에서 트레이닝이 되는지 여부를 담은 변수
                data, target = data.cuda(), target.cuda()  # .cuda()메소드를 사용해서 GPU에서 연산이 가능하도록 바꿔준다.

            # Clear gradients
            optimizer.zero_grad()

            # Predicted outputs are log probabilities
            output = model(
                data)  # 여기에서 모델은 학습에 사용되는 VGG나 AlexNet과 같은 구조를 말한다. 이 모델은 함수로써 쓰이며 input값으로 데이터를 넣으면 output이 나온다.
            # 카테고리별 확률값이 저장되어 나올 것이다. softmax이므로 확률을 모두 더하면 1이 나온다.

            # Loss and backpropagation of gradients
            loss = criterion(output, target)  # loss 값 업데이트

            # 역전파 단계 : 모델의 매개변수에 대한 손실의 변화도를 계산한다.
            loss.backward()

            # 이 함수를 호출하면 매개변수가 갱신된다.
            optimizer.step()

            # Track train loss by multiplying average loss by number of examples in batch
            # loss는 (1,)형태의 Tensor이며, loss.item()은 loss의 스칼라 값이다.
            # 여기에서 data.size(0)는 배치사이즈를 말한다.
            train_loss += loss.item() * data.size(0)

            # Calculate accuracy by finding max log probability
            _, pred = torch.max(output, dim=1) # output에 저장된 확률값중 가장 높은 값을 가진 인덱스를 리턴한다..
            correct_tensor = pred.eq(target.data.view_as(pred)) #

            # Need to convert correct tensor from int to float to average
            accuracy = torch.mean(correct_tensor.type(torch.FloatTensor))

            # Multiply average accuracy times the number of examples in batch
            train_acc += accuracy.item() * data.size(0)

            # Track training progress
            print('\r',
                f'Epoch: {epoch}\t학습진행률 : {100 * (ii + 1) / len(train_loader):.2f}%' \
                + f'\t 현재 Epoch에서 걸린 시간 : {timer() - start:.2f}s' \
                + f'\t Train_Loss : {train_loss / len(train_loader.dataset):.4f}' \
                + f'\t Train_Acc : {100 * (train_acc / len(train_loader.dataset)):.2f}%',
                end='')  # end='\r' : 해당 줄의 처음으로 와서 다시 출력한다.

        # After training loops ends, start validation ===============================================
        else:  # 트레이닝 루프가 끝나면 실행되는 곳이다.
            model.epochs += 1  # 트레이닝 루프 한번을 반복했기 때문에 epoch을 1 올려준다.

            # Don't need to keep track of gradients
            with torch.no_grad():
                # Set to evaluation mode
                model.eval()  # 평가모드로 설정한다. pytorch에는 train(), eval() 두가지 모드밖에 없다. eval()모드에서는 드랍아웃이 작동하지 않는다.
                start_eval = timer()
                print('')
                # Validation loop
                for ii, (data, target) in enumerate(valid_loader):
                    # Tensors to gpu
                    if train_on_gpu:
                        data, target = data.cuda(), target.cuda()

                    # Forward pass
                    # 평가시엔 역전파는 수행하지 않는다.
                    output = model(data)

                    # Validation loss
                    loss = criterion(output, target)

                    # Multiply average loss times the number of examples in batch
                    valid_loss += loss.item() * data.size(0)

                    # Calculate validation accuracy
                    _, pred = torch.max(output, dim=1)
                    correct_tensor = pred.eq(target.data.view_as(pred))
                    accuracy = torch.mean(
                        correct_tensor.type(torch.FloatTensor))
                    # Multiply average accuracy times the number of examples
                    valid_acc += accuracy.item() * data.size(0)
                    print('\r',
                        f'\t\t\t평가진행률 : {100 * (ii + 1) / len(valid_loader):.2f}%' \
                        + f'\t 현재 Epoch에서 걸린 시간 : {timer() - start_eval:.2f}s' \
                        + f'\t Vaild_Loss : {valid_loss / len(valid_loader.dataset):.4f}' \
                        + f'\t Vaild_Acc : {100 * (valid_acc / len(valid_loader.dataset)):.2f}%',
                        end='')  # end='\r' : 해당 줄의 처음으로 와서 다시 출력한다.

                # Calculate average losses
                train_loss = train_loss / len(train_loader.dataset)
                valid_loss = valid_loss / len(valid_loader.dataset)

                # Calculate average accuracy
                train_acc = train_acc / len(train_loader.dataset)
                valid_acc = valid_acc / len(valid_loader.dataset)

                history.append([train_loss, valid_loss, train_acc, valid_acc])

                # Print training and validation results
                if (epoch + 1) % print_every == 0:
                    # print(
                    #     f'\n\t\t\tTraining Loss: {train_loss:.4f} \t\t Validation Loss: {valid_loss:.4f}'
                    # )
                    # print(
                    #     f'\t\t\tTraining Accuracy: {100 * train_acc:.2f}%\t Validation Accuracy: {100 * valid_acc:.2f}%'
                    # )
                    print(
                        f'\n\t\t\t현재 Epochs에서 Train과 Vaild에 걸린 시간 : {timer() - start:.2f}s\n'
                    )

                # Save the model if validation loss decreases
                # 예를 들어보자. 초기 valid_loss_min이 무한대값이다. 당연히 epoch 0에선 이 값보다 작을수밖에 없다.
                # 따라서 valid_loss_min 값이 epoch 0에서의 valid_loss값으로 바뀐다.
                # epoch 1부터 valid_loss가 이전 epoch보다 작아지지 않는다면 epochs_no_improve 값이 증가한다.
                # 만약 작아지지 않는 상태가 max_epochs_stop 값보다 커지게 되면 중지한다.
                # 그 이유는 학습이 계속 진행되더라도 loss 값이 더 이상 작아지지 않으므로, 수렴했다고 볼 수 있기 때문이다.

                if valid_loss < valid_loss_min:
                    # Save model
                    # torch.save(model.state_dict(), save_file_name)  # 이때 저장되는 모델은 최적의 epochs를 가진 모델이다.
                    # Track improvement
                    epochs_no_improve = 0
                    valid_loss_min = valid_loss
                    valid_best_acc = valid_acc
                    best_epoch = epoch

                # Otherwise increment count of epochs with no improvement
                else:
                    epochs_no_improve += 1

                    # Trigger early stopping
                    if early_stop == True:  # Early_stop 옵션이 있는 경우에만 진행한다.
                        if epochs_no_improve >= max_epochs_stop:
                            print(
                                f'\nEarly Stop! {max_epochs_stop} epochs 동안 valid loss 값이 감소하지 않았습니다.\n' \
                                + f'현재까지 진행한 총 epochs : {epoch}\t 최상의 epochs : {best_epoch} (loss: {valid_loss_min:.2f} and acc: {100 * valid_acc:.2f}%)'
                            )
                            total_time = timer() - overall_start
                            print(
                                f'\n[ 총 학습시간 : {total_time:.2f}s, Epoch당 평균 학습 시간 : {total_time / (epoch + 1):.2f}s ]')

                            # Load the best state dict
                            # model.load_state_dict(torch.load(save_file_name))
                            # Attach the optimizer
                            model.optimizer = optimizer

                            # Format history
                            history = pd.DataFrame(
                                history,
                                columns=[
                                    'train_loss', 'valid_loss', 'train_acc',
                                    'valid_acc'
                                ])
                            return model, history, total_time, best_epoch, epoch

    # Attach the optimizer
    model.optimizer = optimizer
    # Record overall time and print out stats
    total_time = timer() - overall_start

    print('----------------------------------------------------------------')
    print('학습 결과')
    print(
        f'\n최상의 epoch 수: {best_epoch} (loss: {valid_loss_min:.2f}, acc: {100 * valid_acc:.2f}%)'
    )
    print(f'[ 총 학습시간 : {total_time:.2f}s, Epoch당 평균 학습 시간 : {total_time / (epoch + 1):.2f}s ]')
    print('----------------------------------------------------------------\n')

    # Format history
    history = pd.DataFrame(
        history,
        columns=['train_loss', 'valid_loss', 'train_acc', 'valid_acc'])
    return model, history, total_time, best_epoch, epoch

def get_loss_function():
    return nn.NLLLoss()


def get_optimizer(parameter):
    # Adam의 Default Learning Rate = 1e-3 = 0.001
    if selected_opti == 'adam':
        return optim.Adam(parameter)
    elif selected_opti == 'sgd':
        return optim.SGD(parameter, lr=0.01, momentum=0.9)



def print_model_architecture(model_name):
    """
    모델 구조를 출력해주는 함수이다.
    만약 모델이 다운로드가 안되어 있다면 다운받는다.
    :param model_name: 출력할 모델 명을 입력받는다.
    :return:
    """

    # AlexNet
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)

    # VGG
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
    elif model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
    elif model_name == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
    elif model_name == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)

    # ResNet
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)

    # Inception
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)

    # DenseNet
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)

    # MobileNet V2
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)

    # ResNeXt
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
    elif model_name == 'resnext101':
        model = models.resnext101_32x8d(pretrained=True)

    # ShuffleNet
    elif model_name == 'shufflenet_v2_05':
        model = models.shufflenet_v2_x0_5(pretrained=True)
    elif model_name == 'shufflenet_v2_10':
        model = models.shufflenet_v2_x1_0(pretrained=True)
    elif model_name == 'shufflenet_v2_15':
        model = models.shufflenet_v2_x1_5(pretrained=True)
    elif model_name == 'shufflenet_v2_20':
        model = models.shufflenet_v2_x2_0(pretrained=True)

    # SqueezeNet
    elif model_name == 'squeezenet1.0':
        model = models.squeezenet1_0(pretrained=True)
    elif model_name == 'squeezenet1.1':
        model = models.squeezenet1_1(pretrained=True)

    # MNASNet
    elif model_name == 'mnasnet05':
        model = models.mnasnet0_5(pretrained=True)
    elif model_name == 'mnasnet075':
        model = models.mnasnet0_75(pretrained=True)
    elif model_name == 'mnasnet10':
        model = models.mnasnet1_0(pretrained=True)
    elif model_name == 'mnasnet13':
        model = models.mnasnet1_3(pretrained=True)

    # WideResNet
    elif model_name == 'wideresnet50':
        model = models.wide_resnet50_2(pretrained=True)
    elif model_name == 'wideresnet101':
        model = models.wide_resnet101_2(pretrained=True)

    print(model)



def save_checkpoint(model, path, model_name):
    """Save a PyTorch model checkpoint

    Params
    --------
        model (PyTorch model): model to save
        path (str): location to save model. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # model_name = path.split('-')[0]
    # assert (model_name in ['vgg16', 'resnet50']), "Path must have the correct model name"

    # Basic details
    checkpoint = {
        'class_to_idx': model.class_to_idx,
        'idx_to_class': model.idx_to_class,
        'epochs': model.epochs,
    }

        # AlexNet
    if model_name == 'alexnet':
        # Check to see if model was parallelized
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

        # VGG
    elif model_name == 'vgg11':
        # Check to see if model was parallelized
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'vgg13':
        # Check to see if model was parallelized
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'vgg16':
        # Check to see if model was parallelized
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'vgg19':
        # Check to see if model was parallelized
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'vgg11_bn':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'vgg13_bn':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'vgg16_bn':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'vgg19_bn':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

        # ResNet
    elif model_name == 'resnet18':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'resnet34':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'resnet50':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'resnet101':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'resnet152':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()

        # Inception
    elif model_name == 'googlenet':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'inception_v3':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()

        # DenseNet
    elif model_name == 'densenet121':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'densenet161':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'densenet169':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

    elif model_name == 'densenet201':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

        # MobileNet V2
    elif model_name == 'mobilenet_v2':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()

        # ResNeXt
    elif model_name == 'resnext50':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'resnext101':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    #
        # ShuffleNet
    elif model_name == 'shufflenet_v2_05':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'shufflenet_v2_10':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'shufflenet_v2_15':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'shufflenet_v2_20':
        checkpoint['fc'] = model.fc
        checkpoint['state_dict'] = model.state_dict()
    #
         # SqueezeNet
    elif model_name == 'squeezenet1.0':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()
    elif model_name == 'squeezenet1.1':
        checkpoint['classifier'] = model.classifier
        checkpoint['state_dict'] = model.state_dict()
    #
    #     # MNASNet
    # elif model_name == 'mnasnet05':
    # elif model_name == 'mnasnet075':
    # elif model_name == 'mnasnet10':
    # elif model_name == 'mnasnet13':
    #
    #     # WideResNet
    # elif model_name == 'wideresnet50':
    # elif model_name == 'wideresnet101':

    # Add the optimizer
    checkpoint['optimizer'] = model.optimizer
    checkpoint['optimizer_state_dict'] = model.optimizer.state_dict()

    # Save the data to the path
    torch.save(checkpoint, path)
    print('학습된 모델이 저장되었습니다.')


def load_checkpoint(path, inference_type='gpu'):
    """Load a PyTorch model checkpoint

    Params
    --------
        path (str): saved model checkpoint. Must start with `model_name-` and end in '.pth'

    Returns
    --------
        None, save the `model` to `path`

    """

    # Whether to train on a gpu
    train_on_gpu = cuda.is_available()  # GPU를 사용할 수 있는지 없는지 판단한다.
    print(f'Train on gpu: {train_on_gpu}')
    print(f'Inference Type: {inference_type}')



    # Get the model name
    model_name = path.split('-')[0]
    model_name = model_name.split('/')[-1]
    print(f'불러온 모델 : {model_name}')

    assert (model_name in ['alexnet',
                           'vgg11','vgg13','vgg16','vgg19','vgg13',
                           'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                           'googlenet', 'inception_v3',
                           'densenet121', 'densenet161', 'densenet169', 'densenet201',
                           'mobilenet_v2', 'resnext50', 'resnext101',
                           'shufflenet_v2_05', 'shufflenet_v2_10', 'shufflenet_v2_15', 'shufflenet_v2_20',
                           'squeezenet1.0', 'squeezenet1.1',
                           'mnasnet05', 'mnasnet075', 'mnasnet10', 'mnasnet13',
                           'wideresnet50', 'wideresnet101']), "Path must have the correct model name"

    # Load in checkpoint
    load_start = timer()
    if inference_type == 'gpu':
        checkpoint = torch.load(path)
    elif inference_type == 'cpu':
        # GPU로 저장된 모델을 전부 CPU로 동작하도록 불러온다.
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
    load_end = timer() - load_start
    print(f'## Torch.load end tiem : {load_end*1000.0:.2f}ms')


    load_state_start = timer()
    # AlexNet
    if model_name == 'alexnet':
        model = models.alexnet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    # VGG
    elif model_name == 'vgg11':
        model = models.vgg11(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'vgg19':
        model = models.vgg19(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'vgg11_bn':
        model = models.vgg11_bn(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'vgg13_bn':
        model = models.vgg13_bn(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'vgg16_bn':
        model = models.vgg16_bn(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    elif model_name == 'vgg19_bn':
        model = models.vgg19_bn(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    # ResNet
    elif model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    elif model_name == 'resnet152':
        model = models.resnet152(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']


    # Inception
    elif model_name == 'googlenet':
        model = models.googlenet(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']
    elif model_name == 'inception_v3':
        model = models.inception_v3(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # DenseNet
    elif model_name == 'densenet121':
        model = models.densenet121(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
    elif model_name == 'densenet161':
        model = models.densenet161(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
    elif model_name == 'densenet169':
        model = models.densenet169(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
    elif model_name == 'densenet201':
        model = models.densenet201(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    # MobileNet V2
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    # ResNeXt
    elif model_name == 'resnext50':
        model = models.resnext50_32x4d(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']
    elif model_name == 'resnext101':
        model = models.resnext101_32x8d(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # ShuffleNet
    elif model_name == 'shufflenet_v2_05':
        model = models.shufflenet_v2_x0_5(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']
    elif model_name == 'shufflenet_v2_10':
        model = models.shufflenet_v2_x1_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']
    elif model_name == 'shufflenet_v2_15':
        model = models.shufflenet_v2_x1_5(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']
    elif model_name == 'shufflenet_v2_20':
        model = models.shufflenet_v2_x2_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = checkpoint['fc']

    # SqueezeNet
    elif model_name == 'squeezenet1.0':
        model = models.squeezenet1_0(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']
    elif model_name == 'squeezenet1.1':
        model = models.squeezenet1_1(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.classifier = checkpoint['classifier']

    # MNASNet
    elif model_name == 'mnasnet05':
        model = models.mnasnet0_5(pretrained=True)
    elif model_name == 'mnasnet075':
        model = models.mnasnet0_75(pretrained=True)
    elif model_name == 'mnasnet10':
        model = models.mnasnet1_0(pretrained=True)
    elif model_name == 'mnasnet13':
        model = models.mnasnet1_3(pretrained=True)

    # WideResNet
    elif model_name == 'wideresnet50':
        model = models.wide_resnet50_2(pretrained=True)
    elif model_name == 'wideresnet101':
        model = models.wide_resnet101_2(pretrained=True)

    load_state_end = timer() - load_state_start
    print(f'## Load Model State : {load_state_end*1000.0:.2f}ms')

    flag1 = timer()
    # Load in the state dict
    model.load_state_dict(checkpoint['state_dict'])
    flag2 = timer() - flag1
    total_params = sum(p.numel() for p in model.parameters())
    # print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{total_trainable_params:,} total gradient parameters.')

    if train_on_gpu and inference_type == 'gpu':
        model = model.to('cuda')
        print("GPU 에서 동작합니다!!!")

    flag3 = timer() - flag2 - flag1
    # Model basics
    model.class_to_idx = checkpoint['class_to_idx']
    model.idx_to_class = checkpoint['idx_to_class']
    model.epochs = checkpoint['epochs']
    flag4 = timer() - flag3 - flag2 - flag1
    # Optimizer
    optimizer = checkpoint['optimizer']
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    flag5 = timer() - flag4 - flag3 - flag2 -flag1

    print(f'## Flag 2 : {flag2*1000.0:.2f}ms, 3 : {flag3*1000.0:.2f}ms, 4: {flag4*1000.0:.2f}ms, 5: {flag5*1000.0:.2f}ms')

    return model, optimizer




def save_distribution_of_images(category_dataframe, model_name):
    results_dir, sample_file_name = setting_save_folder("distribution_of_images", model_name)

    category_dataframe.set_index('category')['n_train'].plot.bar(
        color='r', figsize=(18, 12))
    plt.xticks(rotation=80)
    plt.ylabel('Count')
    plt.title('Training Images by Category')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)
    plt.savefig(results_dir + sample_file_name)

    print('\n----------------------------------------------------------------')
    print('이미지 분포 그래프가 저장되었습니다.')
    print(f'저장 경로 : {results_dir}')
    print(f'분포 그래프 파일 명 : {sample_file_name}')
    print('----------------------------------------------------------------\n')
    plt.close('all')

def save_number_of_trainig_image_top1_top5(results, model_name, etc=''):
    results_dir, sample_file_name_top1 = setting_save_folder(etc + "number_of_image_top1", model_name)
    results_dir, sample_file_name_top5 = setting_save_folder(etc + "number_of_image_top5", model_name)

    # Plot using seaborn
    sns.lmplot(
        y='top1', x='n_train', data=results, height=8) # height=8만 있으면 800x800짜리 이미지가 만들어진다.
    plt.xlabel('images')
    plt.ylabel('Accuracy (%)')
    plt.title('Top 1 Accuracy vs Number of Training Images')
    plt.ylim(-5, 105) # y축 그래프의 범위 지정, -5 ~ 105까지로 설정되어 있다.
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(results_dir + sample_file_name_top1)

    sns.lmplot(
        y='top3', x='n_train', data=results, height=8)
    plt.xlabel('images')
    plt.ylabel('Accuracy (%)')
    plt.title('Top 3 Accuracy vs Number of Training Images')
    plt.ylim(-5, 105)
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(results_dir + sample_file_name_top5)

    print('\n----------------------------------------------------------------')
    print("학습 이미지 수에 따른 Top1, Top3 정확도 그래프가 저장되었습니다.")
    print(f'저장 경로 : {results_dir}')
    print(f'Top1 파일 명 : {sample_file_name_top1}')
    print(f'Top5 파일 명 : {sample_file_name_top5}')
    print('----------------------------------------------------------------\n')
    plt.close('all')

def save_train_valid_loss(history, model_name, etc=''):
    results_dir, sample_file_name_loss = setting_save_folder(etc + "loss", model_name)
    results_dir, sample_file_name_acc = setting_save_folder(etc + "acc", model_name)

    plt.figure(figsize=(8, 6))
    for c in ['train_loss', 'valid_loss']:
        plt.plot(
            history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Negative Log Likelihood')
    plt.title('Training and Validation Losses')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(results_dir + sample_file_name_loss)

    plt.figure(figsize=(8, 6))
    for c in ['train_acc', 'valid_acc']:
        plt.plot(
            100 * history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(results_dir + sample_file_name_acc)

    print('\n----------------------------------------------------------------')
    print('Training, Validation의 Loss와 Accuracy 그래프가 저장되었습니다.')
    print(f'저장 경로 : {results_dir}')
    print(f'Loss 파일 명 : {sample_file_name_loss}')
    print(f'Acc 파일 명 : {sample_file_name_acc}')
    print('----------------------------------------------------------------\n')
    plt.close('all')

def imshow_tensor(image, ax=None, title=None):
    """Imshow for Tensor."""

    if ax is None:
        fig, ax = plt.subplots()

    # Set the color channel as the third dimension
    image = image.numpy().transpose((1, 2, 0))

    # Reverse the preprocessing steps
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean

    # Clip the image pixel values
    image = np.clip(image, 0, 1)

    ax.imshow(image)
    plt.axis('off')

    return ax, image


def process_image(image_path):
    """Process an image path into a PyTorch tensor"""

    image = Image.open(image_path) # 이미지 경로에 있는 이미지를 불러온다.
    print(f'## Image Info : {image}')
    # Resize
    img = image.resize((256, 256)) # 256 크기로 리사이즈 한다. Numpy에서 제공하는 resize 함수이다.

    # Center crop
    width = 256
    height = 256
    new_width = 224
    new_height = 224

    left = (width - new_width) / 2 # (256 - 224)/2 = 16
    top = (height - new_height) / 2 # 16
    right = (width + new_width) / 2 # (256 + 224) / 2 = 240
    bottom = (height + new_height) / 2 # 240

    img = img.crop((left, top, right, bottom)) # 이미지를 크롭한다. 이미지 크롭 방법은 가운데에서 224 크기로 이미지를 자르는 것이다.

    # Convert to numpy, transpose color dimension and normalize
    img = np.array(img).transpose((2, 0, 1)) / 256

    # Standardization
    means = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    stds = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

    img = img - means
    img = img / stds

    img_tensor = torch.Tensor(img)

    return img_tensor


def predict(image_path, model, topk=5, inference_type = 'gpu'):
    """Make a prediction for an image using a trained model

    Params
    --------
        image_path (str): filename of the image
        model (PyTorch model): trained model for inference
        topk (int): number of top predictions to return

    Returns

    """
    real_class = image_path.split('/')[-2] # 이미지 경로에서 폴더이름을 빼낸다. 폴더 이름이 카테고리 명이다.

    # Convert to pytorch tensor
    img_process_start = timer()
    img_tensor = process_image(image_path)
    img_process_end = timer() - img_process_start
    print(f'## Image Process Time : {img_process_end*1000.0:.2f}ms')
    # Resize

    if inference_type == 'gpu':
        img_tensor = img_tensor.view(1, 3, 224, 224).cuda()
    elif inference_type == 'cpu':
        img_tensor = img_tensor.view(1, 3, 224, 224)

    # Set to evaluation
    with torch.no_grad():
        model.eval()
        # Model outputs log probabilities
        in_model_start = timer()
        out = model(img_tensor)
        in_model_end = timer() - in_model_start
        print(f'## Input image to Model : {in_model_end*1000.0:.2f}ms')
        ps = torch.exp(out)

        # Find the topk predictions
        topk, topclass = ps.topk(topk, dim=1)

        # Extract the actual classes and probabilities
        top_classes = [
            model.idx_to_class[class_] for class_ in topclass.cpu().numpy()[0]
        ]
        top_p = topk.cpu().numpy()[0]
        print(f'## Image Process + Model Inference Time : {(img_process_end + in_model_end)*1000.0:.2f}ms')
        return img_tensor.cpu().squeeze(), top_p, top_classes, real_class



def display_prediction(image_path, model, topk, model_name, etc=''):
    """Display image and preditions from model"""
    results_dir, random_predict = setting_save_folder(etc + "random_predict", model_name)

    start_inference = timer() # 이미지 추론 시작 시간

    # Get predictions
    img, ps, classes, y_obs = predict(image_path, model, topk)
    # Convert results to dataframe for plotting
    result = pd.DataFrame({'p': ps}, index=classes) # 추론 결과

    inference_time = timer() - start_inference # 이미지 추론 끝나는 시간

    # Show the image
    plt.figure(figsize=(16, 5))
    ax = plt.subplot(1, 2, 1)
    ax, img = imshow_tensor(img, ax=ax)

    # Set title to be the actual class
    ax.set_title(y_obs, size=20)

    ax = plt.subplot(1, 2, 2)
    # Plot a bar plot of predictions
    result.sort_values('p')['p'].plot.barh(color='blue', edgecolor='k', ax=ax)
    plt.xlabel('Predicted Probability')
    plt.tight_layout()
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    plt.savefig(results_dir + random_predict)

    print('\n----------------------------------------------------------------')
    print('Ground_Truth : ' + y_obs)
    print(result)
    print(f'추론에 걸린 시간 : {inference_time*1000.0:.2f}ms')
    print('Test 이미지에 대한 Top5 예측 결과가 저장되었습니다.')
    print(f'저장 경로 : {results_dir}')
    print(f'결과 파일 명: {random_predict}')
    print('----------------------------------------------------------------\n')
    plt.close('all')
    return inference_time


def img_prediction(image_path, model, topk, gt, inference_type = 'gpu'):
    start_inference = timer() # 추론시간 측정을 위한 시작 시간 입력
    # Get predictions
    img, ps, classes, y_obs = predict(image_path, model, topk, inference_type)
    # Convert results to dataframe for plotting
    result = pd.DataFrame({'p': ps}, index=classes)

    inference_time = timer() - start_inference # 추론이 끝난 시간 - 추론 시작 시간 = 추론에 걸린 시간

    print('\n----------------------------------------------------------------')
    print('Ground_Truth : '+ gt)
    print(result)
    print(f"Top1 정확도 : {(result['p'][0])*100:.2f}%")
    print(f'추론에 걸린 시간 : {inference_time*1000.0:.2f}ms')
    print('----------------------------------------------------------------\n')

    return inference_time




def accuracy(output, target, topk=(1, )):
    """Compute the topk accuracy(s)"""
    output = output.to('cuda')
    target = target.to('cuda')
    # print(f'output : {output}\ntarget: {target}')

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        # print(f'maxk: {maxk}\nbatchsize : {batch_size}')
        # Find the predicted classes and transpose
        _, pred = output.topk(k=maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()

        # print(f'pred : {pred}')
        # Determine predictions equal to the targets
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        # print(f'corret : {correct}')
        res = []

        # For each k, find the percentage of correct
        for k in topk:
            # print(f'k : {k}\ntopk: {topk}')
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            # print(f'correct_k : {correct_k}')
            res.append(correct_k.mul_(100.0 / batch_size).item())
            # print(f'for in res : {res}')
        return res

def evaluate(model, test_loader, criterion, n_classes, topk=(1, 3)):
    """Measure the performance of a trained PyTorch model

    Params
    --------
        model (PyTorch model): trained cnn for inference
        test_loader (PyTorch DataLoader): test dataloader
        topk (tuple of ints): accuracy to measure

    Returns
    --------
        results (DataFrame): results for each category

    """

    classes = []
    losses = []
    # Hold accuracy results
    acc_results = np.zeros((len(test_loader.dataset), len(topk)))
    i = 0

    model.eval()
    with torch.no_grad():

        # Testing loop
        for data, targets in test_loader:


            data, targets = data.to('cuda'), targets.to('cuda')

            # Raw model output
            out = model(data)
            # Iterate through each example
            for pred, true in zip(out, targets):
                # Find topk accuracy
                acc_results[i, :] = accuracy(
                    pred.unsqueeze(0), true.unsqueeze(0), topk)
                classes.append(model.idx_to_class[true.item()])
                # Calculate the loss
                loss = criterion(pred.view(1, n_classes), true.view(1))
                losses.append(loss.item())
                # print(f'acc_result : {acc_results}')
                i += 1

    # Send results to a dataframe and calculate average across classes
    results = pd.DataFrame(acc_results, columns=[f'top{i}' for i in topk])
    # print(f'result : {results}')
    # print(f'result top1 : {results["top1"].mean()}, top5 : {results["top5"].mean()}')
    results['class'] = classes
    results['loss'] = losses
    results = results.groupby(classes).mean()

    return results.reset_index().rename(columns={'index': 'class'})



def training_result(results):
    # Weighted column of test images
    results['weighted'] = results['n_test'] / results['n_test'].sum()

    # Create weighted accuracies
    for i in (1, 3):
        results[f'weighted_top{i}'] = results['weighted'] * results[f'top{i}']

    # Find final accuracy accounting for frequencies
    top1_weighted = results['weighted_top1'].sum()
    top5_weighted = results['weighted_top3'].sum()
    loss_weighted = (results['weighted'] * results['loss']).sum()

    print('\n----------------------------------------------------------------')
    print(f'Final test cross entropy per image = {loss_weighted:.4f}.')
    print(f'Final test top 1 weighted accuracy = {top1_weighted:.2f}%')
    print(f'Final test top 3 weighted accuracy = {top5_weighted:.2f}%')
    print('----------------------------------------------------------------\n')







