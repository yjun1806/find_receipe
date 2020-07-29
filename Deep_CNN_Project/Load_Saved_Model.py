import train_util
import sys
from timeit import default_timer as timer


path = '/home/kunde/DeepCNN/ModelSave/vgg19-transfer_bts128_ep100_20190825005705.pth'
inference_type = 'cpu'
model_load_start = timer()
model, optimizer = train_util.load_checkpoint(path, inference_type)
model_load_end = timer() - model_load_start
print(f'Total Model load time : {model_load_end*1000.0:.2f}ms') # 저장된 모델을 불러오는데 걸리는 시간
print('\n\n---------------- [ Model Load End / Inference Start ] -----------------------------\n\n')
ground_truth = ['eggplant', 'eggplant','eggplant', 'paprika', 'daikon', 'greenpumpkin', 'eggplant', 'cucumber', 'spinach', 'potato', 'carrot']

total_inference_time = 0.0
for x in range(1, 12):
    predict_img_path = f'/home/kunde/DeepCNN/real_test/test{x}.jpg'
    total_inference_time += train_util.img_prediction(predict_img_path, model, topk=3, gt=ground_truth[x-1], inference_type = inference_type)

print(f'평균 추론 시간 : {total_inference_time/11*1000.0:.2f}ms')
print(f'저장된 모델 불러오기 + 전체 걸린 시간 : {(total_inference_time + model_load_end)*1000.0:.2f}ms')

sys.exit(0)