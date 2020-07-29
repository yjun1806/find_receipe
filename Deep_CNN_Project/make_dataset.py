import os
import shutil
import sys


K = 5


file_dir = os.path.dirname(__file__)
dataset_dir = file_dir + '/ingredient_data_TR9_TE1/'
train_set_dir = dataset_dir + 'train/'
test_set_dir = dataset_dir + 'test'
k_fold_dir_name = str(K) + '_fold_cross_validation_dataset'

categories = []

# k_fold_cross_validation_dataset 이라는 폴더가 없는 경우 폴더를 생성해준다.
if not os.path.isdir(dataset_dir + k_fold_dir_name):
    os.makedirs(dataset_dir + k_fold_dir_name) # 폴더 생성 부분
    shutil.copytree(test_set_dir, dataset_dir + k_fold_dir_name + '/test') # 이용할 test dataset을 copy한다.
else:
    print('이미 생성되어 있습니다.')
    sys.exit(0)

# _fold_cross_validation_dataset 폴더 내부에 각각의 fold 데이터셋을 넣기위한 폴더를 만들어준다.
for i in range(K):
    k_fold_dir = 'K_' + str(i)
    if not os.path.isdir(dataset_dir + k_fold_dir_name + '/' + k_fold_dir):
        os.makedirs(dataset_dir + k_fold_dir_name + '/' + k_fold_dir)
        os.makedirs(dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/train')
        os.makedirs(dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/valid')

# 최종적으로 생성되는 디렉토리 트리는 다음과 같다. (K=5일때 예시)
# ├── 5_fold_cross_validation_dataset
# │   ├── K_0
# │   │   ├── train
# │             ├── categories
# │   │   └── valid
# │             ├── categories
# │   ├── K_1
# │   │   ├── train
# │             ├── categories
# │   │   └── valid
# │             ├── categories
# │   ├── K_2
# │   │   ├── train
# │             ├── categories
# │   │   └── valid
# │             ├── categories
# │   ├── K_3
# │   │   ├── train
# │             ├── categories
# │   │   └── valid
# │             ├── categories
# │   ├── K_4
# │   │   ├── train
# │             ├── categories
# │   │   └── valid
# │             ├── categories
# │   └── train
# │       ├── categories





for d in os.listdir(train_set_dir): # train 디렉토리의 하위 디렉토리 이름을 추출할 수 있다.
    train_imags = os.listdir(train_set_dir + d) # train의 하위 디렉토리 하나를 선택
    # print(f'{d} : {len(train_imags)}, {train_imags}')

    for k in range(K):
        k_fold_dir = 'K_' + str(k)
        # /k_fold_cross_validation_dataset/K_k/train/ 디렉토리 안에 카테고리 디렉토리를 생성해준다.
        if not os.path.isdir(dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/train/' + d):
            os.makedirs(dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/train/' + d)

        # /k_fold_cross_validation_dataset/K_k/valid/ 디렉토리 안에 카테고리 디렉토리를 생성해준다.
        if not os.path.isdir(dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/valid/' + d):
            os.makedirs(dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/valid/' + d)


        div_k = int(len(train_imags)/K)
        print(f'{d} : div_k = {div_k}, train_imags : {len(train_imags)}')
        ori_path = train_set_dir + d
        copy_train_path = dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/train/' + d
        copy_valid_path = dataset_dir + k_fold_dir_name + '/' + k_fold_dir + '/valid/' + d

        if k == 0:
            for i in range(div_k*k, div_k*(k+1)):
                shutil.copyfile(ori_path + '/' + train_imags[i], copy_valid_path + '/'+ train_imags[i])

            for i in range(div_k*(k+1), len(train_imags)):
                shutil.copyfile(ori_path + '/' + train_imags[i], copy_train_path + '/'+ train_imags[i])

            # print(f'{d} - valid index : {div_k*k} ~ {div_k*(k+1)}, train index : {div_k*(k+1) +1} ~ {len(train_imags)-1}, whold : {len(train_imags)}')
        elif k == K-1:
            for i in range(0, div_k *k):
                shutil.copyfile(ori_path + '/' + train_imags[i], copy_train_path + '/' + train_imags[i])

            for i in range(div_k*k , len(train_imags)):
                shutil.copyfile(ori_path + '/' + train_imags[i], copy_valid_path + '/' + train_imags[i])

            # print(f'{d} - train index : {0} ~ {div_k*k}, valid index : {div_k*k + 1} ~ {len(train_imags)-1}, whold : {len(train_imags)}')
        elif 0 < k < K-1:
            for i in range(0, div_k *k):
                shutil.copyfile(ori_path + '/' + train_imags[i], copy_train_path + '/' + train_imags[i])

            for i in range(div_k*k, div_k*(k+1)):
                shutil.copyfile(ori_path + '/' + train_imags[i], copy_valid_path + '/' + train_imags[i])

            for i in range(div_k*(k+1), len(train_imags)):
                shutil.copyfile(ori_path + '/' + train_imags[i], copy_train_path + '/' + train_imags[i])

        print(f'{d}, k : {k} - train set : {len(os.listdir(copy_train_path))}, valid set : {len(os.listdir(copy_valid_path))}')
            # print(f'{d} - train index : {0} ~ {div_k*k}, valid index : {div_k*k + 1} ~ {div_k*(k+1)}, train index : {div_k*(k+1) +1} ~ {len(train_imags)-1}, whold : {len(train_imags)}')


print(f'Cross Validation Set 생성 완료')
sys.exit(0)