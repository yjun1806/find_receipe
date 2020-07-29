import os

import numpy as np
import torch.nn as nn
from torch import cuda
from torchsummary import summary
import time
import train_util
import sys




# 학습할 모델 입력
# print("학습 가능한 모델 리스트\n - 'alexnet'")
# print("'vgg11','vgg13','vgg16','vgg19','vgg13'")
# print("'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn'")
# print("'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',")
# print("'googlenet', 'inception_v3',")
# print("'densenet121', 'densenet161', 'densenet169', 'densenet201',")
# print("'mobilenet_v2', 'resnext50', 'resnext101',")
# print("'shufflenet_v2_05', 'shufflenet_v2_10', 'shufflenet_v2_15', 'shufflenet_v2_20',")
# print("'squeezenet1.0', 'squeezenet1.1',")
# print("'mnasnet05', 'mnasnet075', 'mnasnet10', 'mnasnet13',")
# print("'wideresnet50', 'wideresnet101'")

model_choice = train_util.model_choice

assert (model_choice in ['alexnet',
                           'vgg11','vgg13','vgg16','vgg19','vgg13',
                           'vgg11_bn', 'vgg13_bn', 'vgg16_bn', 'vgg19_bn',
                           'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
                           'googlenet', 'inception_v3',
                           'densenet121', 'densenet161', 'densenet169', 'densenet201',
                           'mobilenet_v2', 'resnext50', 'resnext101',
                           'shufflenet_v2_05', 'shufflenet_v2_10', 'shufflenet_v2_15', 'shufflenet_v2_20',
                           'squeezenet1.0', 'squeezenet1.1',
                           'mnasnet05', 'mnasnet075', 'mnasnet10', 'mnasnet13',
                           'wideresnet50', 'wideresnet101']), "정확한 모델명을 입력해주세요."

# 학습 내용 파일로 저장
# train_util.save_result_to_txt(model_choice)

# 몇 epoch 학습할 것인지
training_epoch = train_util.training_epoch

# 배치 사이즈 조절
batch_size = train_util.batch_size

# 데이터 경로 지정부분
datadir, traindir, validdir, testdir, image_transforms, data, dataloaders = train_util.init_dataset()
cat_df , image_df = train_util.category_dataframe(traindir, validdir, testdir)

# 학습된 데이터 저장시 이름을 정하는 부분
save_file_name = './ModelSave/' + model_choice + '-transfer.pt'
checkpoint_path = './ModelSave/' + model_choice + '-transfer_bts' +str(batch_size) + "_ep" + str(training_epoch) + "_" +train_util.get_date()+'.pth'


# 이미지 갯수 분포 그래프 이미지로 저장
# train_util.save_distribution_of_images(cat_df, model_choice)


# Whether to train on a gpu
train_on_gpu = cuda.is_available() # GPU를 사용할 수 있는지 없는지 판단한다.
if train_on_gpu:
    print('학습 모드 : GPU\n')
else:
    print('학습 모드 : CPU\n')


cat_df.sort_values('n_train', ascending=False, inplace=True)
# print(f'{cat_df.head()}\n')
# print(f'{cat_df.tail()}\n')


# iter() : 전달된 데이터의 반복자를 꺼내 반환한다.
# trainiter = iter(dataloaders['train'])
# next() : 반복자를 입력받아 그 반복자가 다음에 출력해야할 요소를 반환한다.
# features, labels = next(trainiter) # 1개만 꺼내기위해 넣은 코드인듯
# print(f'{features.shape} , {labels.shape}') # 그냥 단순히 어떤 데이터가 어떤 형태로 들어있는지 알려주기 위한 코드인듯.

n_classes = len(cat_df)

print('\n----------------------------------------------------------------')
print(f'\t- 학습할 모델 이름 : {model_choice}')
print(f'\t- 총 학습할 Epoch 수 : {training_epoch}\n\t- 배치크기 : {batch_size}')
print(f'\t- 이미지 카테고리 수 : {n_classes}')
print(f'\t- 훈련 이미지 수 : {len(data["train"].imgs)} 장')
print(f'\t- 검증 이미지 수 : {len(data["val"].imgs)} 장')
print(f'\t- 시험 이미지 수 : {len(data["test"].imgs)} 장')
print(f'\t- 총이미지 수 : {len(data["train"].imgs) + len(data["val"].imgs) + len(data["test"].imgs)} 장')
print(f'\t- 학습에 사용된 GPU 종류 : {cuda.get_device_name(0)}')
print(f'\t- GPU 학습 가능 여부 : {cuda.is_available()}')
print('\t- 모델 구조')

# 모델 구조 출력
train_util.print_model_architecture(model_choice)

# Pretrained_model 호출 및 다운로드
model = train_util.get_pretrained_model_last_layer_change(model_choice, n_classes=n_classes)

# Loss 함수 설정
criterion = train_util.get_loss_function()

# 최적화 함수 설정
optimizer = train_util.get_optimizer(model.parameters())

print(f'\t- Loss 함수 : {criterion}\n\t- Optimizer 종류: {optimizer}\n')
print("\t- 전체 카테고리 당 이미지 수")
print(cat_df)
print('----------------------------------------------------------------\n')

# 학습할 모델 요약
summary(model, input_size=(3, 224, 224), batch_size=batch_size, device='cuda')


model.class_to_idx = data['train'].class_to_idx
model.idx_to_class = {
    idx: class_
    for class_, idx in model.class_to_idx.items()
}


for p in optimizer.param_groups[0]['params']:
    if p.requires_grad:
        print(p.shape) # 최적화해야할 파라미터 그룹 출력

cuda.empty_cache() # GPU 캐시 초기화

model, history = train_util.train(
    model, # 사용할 모델
    criterion, # 사용할 Loss 함수
    optimizer, # 사용할 Optimizer함수
    dataloaders['train'], # train 데이터셋
    dataloaders['val'], # validation 데이터셋
    save_file_name=save_file_name, # 저장할 이름
    max_epochs_stop=10, # 몇 epoch 동안 vaild loss의 감소가 없으면 학습을 중단할 것인지
    n_epochs=training_epoch, # 최대 몇 epochs 학습할것인지
    print_every=1, # 몇 epoch마다 출력할 것인지
    early_stop=train_util.Early_stop) # Early_stop을 할것인지


# Loss, Acc 그래프 저장 함수
train_util.save_train_valid_loss(history, model_choice)

# 모델 저장 함수
train_util.save_checkpoint(model, path=checkpoint_path, model_name=model_choice)

# 랜덤하게 이미지를 한장 뽑아내는 함수
np.random.seed = 100
def random_test_image():
    """Pick a random test image from the test directory"""
    c = np.random.choice(cat_df['category'])
    root = testdir + c + '/'
    img_path = root + np.random.choice(os.listdir(root))
    return img_path

avg_inference_time = 0
# 랜덤하게 이미지를 선택한 후 Top5 예측치를 확인하는 함수
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)
time.sleep(1.5)
avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice)

print(f'총 10회 추론시간 평균 : {avg_inference_time*1000.0/10:.2f}ms')

# testiter = iter(dataloaders['test'])
# print(f'testiter : {testiter}')
# # Get a batch of testing images and labels
# features, targets = next(testiter)
# print(f'features : {features}, target : {targets}')
#
# acc_res = train_util.accuracy(model(features.to('cuda')), targets, topk=(1, 5))
# print(f'acc_res : {acc_res}')
# print(f'Top1 정확도 : {acc_res[0]:.2f}%')
# print(f'Top5 정확도 : {acc_res[1]:.2f}%')

# criterion = nn.NLLLoss()
# Evaluate the model on all the training data

results = train_util.evaluate(model, dataloaders['test'], criterion, n_classes)
results = results.merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])

results.sort_values('top1', ascending=False, inplace=True)
print("\nTop1 정확도가 가장 높은 카테고리 5개")
print(results.head())

print("\nTop1 정확도가 가장 낮은 카테고리 5개")
print(results.tail())

results.sort_values('top5', ascending=False, inplace=True)
print("\nTop5 정확도가 가장 높은 카테고리 5개")
print(results.head())

print("\nTop5 정확도가 가장 낮은 카테고리 5개")
print(results.tail())

print("\n전체 카테고리별 결과")
print(results)
print(f'\n\tTop1 정확도 : {results["top1"].mean():.4f}%\n\tTop5 정확도 : {results["top5"].mean():.4f}%')

train_util.save_number_of_trainig_image_top1_top5(results, model_choice)
# print(results)

# 마지막 결과
train_util.training_result(results)




sys.exit(0)