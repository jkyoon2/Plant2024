
import os
import pandas as pd
import numpy as np
import imageio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from config import CONFIG
from tqdm import tqdm
tqdm.pandas()  # Add this line to enable the progress_apply method

def load_data():
    train_pickle_path = 'train.pkl'
    test_pickle_path = 'test.pkl'

    if os.path.exists(train_pickle_path) and os.path.exists(test_pickle_path):
        train = pd.read_pickle(train_pickle_path)
        test = pd.read_pickle(test_pickle_path)
    else:
        train = pd.read_csv(CONFIG.TRAIN_CSV_PATH)
        #이상치 제거 추가
        for column in CONFIG.TARGET_COLUMNS:
            lower_quantile = train[column].quantile(CONFIG.Lower_Quantile)
            upper_quantile = train[column].quantile(CONFIG.Upper_Quantile)
            train = train[(train[column] >= lower_quantile) & (train[column] <= upper_quantile)]
        train['file_path'] = train['id'].apply(lambda s: get_file_path(s, CONFIG.TRAIN_IMAGE_PATH))
        train['jpeg_bytes'] = train['file_path'].progress_apply(lambda fp: open(fp, 'rb').read())
        train.to_pickle('train.pkl')
        
        test = pd.read_csv(CONFIG.TEST_CSV_PATH)
        test['file_path'] = test['id'].apply(lambda s: get_file_path(s, CONFIG.TEST_IMAGE_PATH))
        test['jpeg_bytes'] = test['file_path'].progress_apply(lambda fp: open(fp, 'rb').read())
        test.to_pickle('test.pkl')
    return train, test

def get_file_path(image_id, primary_path):
    file_path = f'{primary_path}/{image_id}.jpeg'
    if not os.path.exists(file_path):
        file_path = f'/root/.dataset/train_images/{image_id}.jpeg' #원본 이미지 디렉토리
    return file_path

class PlantDataset(Dataset):
    def __init__(self, X_jpeg_bytes, y, transforms=None):
        self.X_jpeg_bytes = X_jpeg_bytes
        self.y = y
        self.transforms = transforms

    def __len__(self):
        return len(self.X_jpeg_bytes)

    def __getitem__(self, index):
        X_sample = self.transforms(
            image=imageio.imread(self.X_jpeg_bytes[index]),
        )['image']
        y_sample = self.y[index]
        return X_sample, y_sample
