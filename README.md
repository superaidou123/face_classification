# face_classification by Paddle

|  model   | acc  | acc(offcial repo) | acc(paper) |
|  ----  | ----  | ----  |----  |
| MiniXception  | 95.58% ([MiniXception/best_model](trained_models/gender_models/MiniXception/best.pdparams))| 95% ([gender_mini_XCEPTION.21-0.95.hdf5](https://github.com/oarriaga/face_classification/blob/master/trained_models/gender_models/gender_mini_XCEPTION.21-0.95.hdf5)) |95% |
| SimpleCNN  | 95.76% ([SimpleCNN/best_model](trained_models/gender_models/SimpleCNN/best.pdparams)) | 94.88% ([simple_CNN.81-0.96.hdf5](https://github.com/oarriaga/face_classification/blob/master/trained_models/gender_models/simple_CNN.81-0.96.hdf5)) |94.88% |

## dataset 
download imdb dataset (gender classification) from [link](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) and unzip it to dataset folder
```bash
tar -xf imdb_crop.tar
```

## train

set `images_path` in line 33 of [train.py](train.py) and use follow script to start training
```python
python3 train.py
```

## eval

set `images_path` and 'model_path' in line 28 and 29 of [eval.py](eval.py) and use follow script to start eval
```python
python3 eval.py
```

## demo

set `gender_model_path` in line 64 of [demo.py](demo.py) and use follow script to start eval
```python
python3 demo.py
```

sample

![demo img](images/predicted_test_image.png)