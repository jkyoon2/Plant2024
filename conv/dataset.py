import os
import pandas as pd
import numpy as np
import imageio.v3 as imageio
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from config import CONFIG
from tqdm import tqdm
tqdm.pandas()

def load_data():
    train_pickle_path = 'train.pkl'
    test_pickle_path = 'test.pkl'

    def read_image(fp, backup_path):
        try:
            return open(fp, 'rb').read()
        except FileNotFoundError:
            return open(backup_path, 'rb').read()

    if os.path.exists(train_pickle_path) and os.path.exists(test_pickle_path):
        train = pd.read_pickle(train_pickle_path)
        test = pd.read_pickle(test_pickle_path)
    else:
        train = pd.read_csv(CONFIG.TRAIN_CSV_PATH)
        train['file_path'] = train['id'].apply(lambda s: f'{CONFIG.TRAIN_IMAGE_PATH}/{s}.jpeg')
        train['backup_file_path'] = train['id'].apply(lambda s: f'/root/data/train_images/{s}.jpeg')
        tqdm.pandas()
        train['jpeg_bytes'] = train.apply(lambda row: read_image(row['file_path'], row['backup_file_path']), axis=1)
        train.drop(columns=['backup_file_path'], inplace=True)
        train.to_pickle('train.pkl')
        
        test = pd.read_csv(CONFIG.TEST_CSV_PATH)
        test['file_path'] = test['id'].apply(lambda s: f'{CONFIG.TEST_IMAGE_PATH}/{s}.jpeg')
        test['backup_file_path'] = test['id'].apply(lambda s: f'/root/data/test_images/{s}.jpeg')
        tqdm.pandas()
        test['jpeg_bytes'] = test.apply(lambda row: read_image(row['file_path'], row['backup_file_path']), axis=1)
        test.drop(columns=['backup_file_path'], inplace=True)
        test.to_pickle('test.pkl')
    
    return train, test

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
