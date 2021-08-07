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
import paddle.nn.functional as F
import numpy as np
from tqdm import tqdm
from load_data import DataManager, split_imdb_data
from model import SimpleCNN, MiniXception
from dataset import IMDBDataset
from log import get_logger

# parameters
gpu_id = 0
batch_size = 32
num_workers = 8
num_epochs = 1000
print_iter = 10
validation_split = .2
do_random_crop = False
num_classes = 2
dataset_name = 'imdb'
input_shape = (64, 64, 3)
grayscale = input_shape[2] == 1

images_path = 'dataset/imdb_crop/'
log_file_path = 'trained_models/gender_models/SimpleCNN/gender_training.log'
trained_models_path = 'trained_models/gender_models/SimpleCNN/'
os.makedirs(trained_models_path, exist_ok=True)
logger = get_logger('root', log_file_path)
# data loader
data_loader = DataManager(dataset_name,dataset_path=os.path.join(images_path,'imdb.mat'))
ground_truth_data = data_loader.get_data()
train_keys, val_keys = split_imdb_data(ground_truth_data, validation_split)

paddle.set_device(f'gpu:{gpu_id}')

train_dataset = IMDBDataset(ground_truth_data, batch_size,
                            input_shape[:2],
                            train_keys, None,
                            path_prefix=images_path,
                            vertical_flip_probability=0,
                            grayscale=grayscale,
                            do_random_crop=do_random_crop, mode='train')

eval_dataset = IMDBDataset(ground_truth_data, batch_size,
                          input_shape[:2],
                          val_keys, None,
                          path_prefix=images_path,
                          vertical_flip_probability=0,
                          grayscale=grayscale,
                          do_random_crop=do_random_crop, mode='val')

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

logger.info(f'train with paddle version: {paddle.__version__}, gpu: {gpu_id}')
logger.info(f'Number of training samples: {len(train_dataset)}, iter: {len(train_loader)}', )
logger.info(f'Number of validation samples: {len(eval_dataset)}, iter: {len(eval_loader)}', )

# model parameters/compilation
model = SimpleCNN(input_shape[-1], num_classes)

# scheduler = paddle.optimizer.lr.ReduceOnPlateau(learning_rate=0.001, mode='max', factor=0.1, patience=50, verbose=True)
scheduler = paddle.optimizer.lr.StepDecay(learning_rate=0.001, step_size=150, gamma=0.1, verbose=True)
opt = paddle.optimizer.Adam(learning_rate=scheduler, parameters=model.parameters())
metric_eval = paddle.metric.Accuracy()
metric_train = paddle.metric.Accuracy()
best_metric = {'epoch': 0, 'acc': 0}

for e in range(num_epochs):
    model.train()
    for i, (image, label) in enumerate(train_loader()):
        out = model(image)
        loss = F.cross_entropy(out, label)
        avg_loss = paddle.mean(loss)
        avg_loss.backward()
        opt.step()
        opt.clear_grad()

        correct = metric_train.compute(out, label)
        metric_train.update(correct)
        if i % print_iter == 0:
            logger.info(f"Epoch [{e}/ {num_epochs}] batch [{i}/{len(train_loader)}]: loss = {np.mean(loss.numpy()):.4f}, acc = {metric_train.accumulate():.4f}, lr = {opt.get_lr()}")
    logger.info(f"Epoch {e} train acc = {metric_train.accumulate():.4f}, lr = {opt.get_lr()}")
    # eval
    model.eval()
    with paddle.no_grad():
        for image, label in tqdm(eval_loader()):
            out = model(image)
            correct = metric_eval.compute(out, label)
            metric_eval.update(correct)
    acc = metric_eval.accumulate()
    metric_eval.reset()
    scheduler.step()

    if acc > best_metric['acc']:
        model_path = os.path.join(trained_models_path,'best.pdparams')
        paddle.save(model.state_dict(), model_path)
        best_metric['acc'] = acc
        best_metric['epoch'] = e
    model_path = os.path.join(trained_models_path,'latest.pdparams')
    paddle.save(model.state_dict(), model_path)

    logger.info(f"Epoch {e} eval acc = {acc:.4f}, bect_acc = {best_metric['acc']:.4f}, best epoch = {best_metric['epoch']}")
