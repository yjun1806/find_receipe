import os

import numpy as np
import torch.nn as nn
from torch import cuda
from torchsummary import summary
import time
import train_util
import sys


# Cross_validation 전용 파일

K_fold = 5

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

# Whether to train on a gpu
train_on_gpu = cuda.is_available()  # GPU를 사용할 수 있는지 없는지 판단한다.
if train_on_gpu:
    print('학습 모드 : GPU\n')
else:
    print('학습 모드 : CPU\n')

print('\n----------------------------------------------------------------')
print(f'\t- 학습할 모델 이름 : {model_choice}')
print(f'\t- 학습할 Epoch 수 : {training_epoch}\n\t- 배치크기 : {batch_size}')
print(f'\t- Cross Validation K : {K_fold}')
print(f'\t- 학습에 사용된 GPU 종류 : {cuda.get_device_name(0)}')
print(f'\t- GPU 학습 가능 여부 : {cuda.is_available()}')
print('\t- 모델 구조')
# 모델 구조 출력
train_util.print_model_architecture(model_choice)
print('----------------------------------------------------------------\n')


results = []
train_results = []
total_avg_inference_time = 0.0

for k in range(K_fold):
    # 데이터 경로 지정부분
    datadir, traindir, validdir, testdir, image_transforms, data, dataloaders = train_util.init_cv_dataset(K_fold, k)
    cat_df , image_df = train_util.category_dataframe(traindir, validdir, testdir)

    cat_df.sort_values('n_train', ascending=False, inplace=True)

    n_classes = len(cat_df)

    print('\n----------------------------------------------------------------')
    print(f'\t- 현재 K_fold : {k}')
    print(f'\t- 이미지 카테고리 수 : {n_classes}')
    print(f'\t- 훈련 이미지 수 : {len(data["train"].imgs)} 장')
    print(f'\t- 검증 이미지 수 : {len(data["val"].imgs)} 장')
    print(f'\t- 시험 이미지 수 : {len(data["test"].imgs)} 장')
    print(f'\t- 총이미지 수 : {len(data["train"].imgs) + len(data["val"].imgs) + len(data["test"].imgs)} 장')


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

    model, history, total_time, best_epochs, epochs = train_util.train_cv(
        model, # 사용할 모델
        criterion, # 사용할 Loss 함수
        optimizer, # 사용할 Optimizer함수
        dataloaders['train'], # train 데이터셋
        dataloaders['val'], # validation 데이터셋
        max_epochs_stop=10, # 몇 epoch 동안 vaild loss의 감소가 없으면 학습을 중단할 것인지
        n_epochs=training_epoch, # 최대 몇 epochs 학습할것인지
        print_every=1, # 몇 epoch마다 출력할 것인지
        early_stop=train_util.Early_stop) # Early_stop을 할것인지

    tmp = {'total_time': total_time, 'best_epochs': best_epochs, 'epochs': epochs}
    train_results.append(tmp)

    # Loss, Acc 그래프 저장 함수
    train_util.save_train_valid_loss(history, model_choice, etc = '[K_' + str(K_fold) + '_'+str(k)+'_fold] ')

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
    for a in range(10):
        avg_inference_time += train_util.display_prediction(random_test_image(), model, topk=5, model_name=model_choice, etc = '[K_' + str(K_fold) + '_'+str(k)+'_fold] ')
        time.sleep(1.1)

    total_avg_inference_time += avg_inference_time

    print(f'총 10회 추론시간 평균 : {avg_inference_time*1000.0/10:.2f}ms')


    results.append(train_util.evaluate(model, dataloaders['test'], criterion, n_classes))
    results[k] = results[k].merge(cat_df, left_on='class', right_on='category').drop(columns=['category'])

    results[k].sort_values('top1', ascending=False, inplace=True)
    print("\nTop1 정확도가 가장 높은 카테고리 5개")
    print(results[k].head())

    print("\nTop1 정확도가 가장 낮은 카테고리 5개")
    print(results[k].tail())

    results[k].sort_values('top5', ascending=False, inplace=True)
    print("\nTop5 정확도가 가장 높은 카테고리 5개")
    print(results[k].head())

    print("\nTop5 정확도가 가장 낮은 카테고리 5개")
    print(results[k].tail())

    print("\n전체 카테고리별 결과")
    print(results[k])
    print(f'\n\tTop1 정확도 : {results[k]["top1"].mean():.4f}%\n\tTop5 정확도 : {results[k]["top5"].mean():.4f}%')

    train_util.save_number_of_trainig_image_top1_top5(results[k], model_choice, etc = '[K_' + str(K_fold) + '_'+str(k)+'_fold] ')

    # 마지막 결과
    train_util.training_result(results[k])

    print(f'{k}번째 fold 학습 완료')


print('전체 Fold 학습 완료')
top1_mean = 0.0
top5_mean = 0.0
best_epochs_mean = 0.0
epochs_mean = 0.0
fold_total_time = 0.0

for i in range(K_fold):
    top1_mean += results[i]["top1"].mean()
    top5_mean += results[i]["top5"].mean()
    best_epochs_mean += train_results[i]["best_epochs"]
    epochs_mean += (train_results[i]["epochs"] +1)
    fold_total_time += train_results[i]["total_time"]
    print(f'{i}_fold - Top1 정확도 : {results[i]["top1"].mean():.4f}%\tTop5 정확도 : {results[i]["top5"].mean():.4f}%')
    print(f'{i}_fold - Best Epochs : {train_results[i]["best_epochs"]}\t진행한 Epochs : {train_results[i]["epochs"]}')
    print(f'{i}_fold - Total_Train_time : {train_results[i]["total_time"]:.2f}s\tEpochs 당 평균 학습 시간 : {train_results[i]["total_time"] / (train_results[i]["epochs"] +1):.2f}s')

print('\n----------------------------------------------------------------')
print(f'{K_fold}_Fold 평균 Top1 정확도 : {top1_mean/K_fold:.4f}%\t평균 Top5 정확도 : {top5_mean/K_fold:.4f}%')
print(f'추론시간 K x 10회 평균 : {total_avg_inference_time*1000.0/(10*K_fold):.2f}ms')
print(f'평균 Best Epochs 평균 : {best_epochs_mean/K_fold:.2f}')
print(f'평균 Epochs : {epochs_mean/K_fold:.2f}')
print(f'전체 학습 시간 : {fold_total_time:.2f}s')
print(f'Epochs 당 평균 학습 시간 : {fold_total_time/epochs_mean:.2f}s')
print('----------------------------------------------------------------\n')
sys.exit(0)

