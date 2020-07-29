import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch
import sys

dataset = dset.ImageFolder(root="img/",
                           transform=transforms.Compose([
                                # transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)), # 썩을 왜 이렇게 해놨어?
                                transforms.RandomResizedCrop(size=256, scale=(0.08, 1.0)), # 0.08 ~ 1.0 이 값이 디폴트값
                                transforms.RandomRotation(degrees=15),
                                transforms.ColorJitter(),
                                transforms.RandomHorizontalFlip(),
                                transforms.CenterCrop(size=224),  # Image net standards
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406],
                                                     [0.229, 0.224, 0.225])  # Imagenet standards
                            ]))

dataloader = torch.utils.data.DataLoader(dataset,
                                         batch_size=2,
                                         shuffle=True,
                                         )
for i, data in enumerate(dataloader):
    print(data[0].size())  # input image
    print(data[1])         # class label
