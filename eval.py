"""
File: train_gender_classifier.py
Author: Octavio Arriaga
Email: arriaga.camargo@gmail.com
Github: https://github.com/oarriaga
Description: Train gender classification model
"""
import os

import paddle
from paddle.io import DataLoader
from tqdm import tqdm
from load_data import DataManager, split_imdb_data
from model import SimpleCNN, MiniXception
from dataset import IMDBDataset

# parameters
gpu_id = 2
batch_size = 256
num_workers = 4
validation_split = .2
do_random_crop = False
num_classes = 2
dataset_name = 'imdb'
input_shape = (64, 64, 1)
grayscale = input_shape[2] == 1

images_path = 'dataset/imdb_crop/'
model_path = 'trained_models/gender_models/MiniXception/best.pdparams'
# data loader
data_loader = DataManager(dataset_name,dataset_path=os.path.join(images_path,'imdb.mat'))
ground_truth_data = data_loader.get_data()
train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)

paddle.set_device(f'gpu:{gpu_id}')
eval_dataset = IMDBDataset(ground_truth_data, batch_size,
                          input_shape[:2],
                          val_keys, None,
                          path_prefix=images_path,
                          vertical_flip_probability=0,
                          grayscale=grayscale,
                          do_random_crop=do_random_crop, mode='val')

eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

print(f'Number of validation samples: {len(eval_dataset)}, iter: {len(eval_loader)}', )

# model parameters/compilation
model = MiniXception(input_shape[-1], num_classes)
model.set_state_dict(paddle.load(model_path))
metric_eval = paddle.metric.Accuracy()


model.eval()
with paddle.no_grad():
    for image, label in tqdm(eval_loader()):
        out = model(image)
        correct = metric_eval.compute(out, label)
        metric_eval.update(correct)
acc = metric_eval.accumulate()
print(acc)
