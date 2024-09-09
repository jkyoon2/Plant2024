import logging
import torch
import pandas as pd
from tqdm import tqdm  # Change this import
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from model import Model
from dataset import load_data, PlantDataset
from config import CONFIG
from logging_config import setup_logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np

def evaluate():
    # Initialize logging
    setup_logging()
    logging.info("Evaluation session started.")

    # Load data from pickle files
    _, test = load_data()

    # Define transformations
    MEAN = [0.485, 0.456, 0.406]  # Convert numpy array to list
    STD = [0.229, 0.224, 0.225]  # Convert numpy array to list

    TEST_TRANSFORMS = A.Compose([
        A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
        A.ToFloat(),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
        ToTensorV2(),
    ])

    # Prepare test data
    test_dataset = PlantDataset(
        test['jpeg_bytes'].values,
        test['id'].values,
        TEST_TRANSFORMS,
    )

    # Define model
    model = Model().to('cuda')
    model.load_state_dict(torch.load(CONFIG.MODEL_NAME))
    model.eval()

    # Initialize scaler and fit with training data (assumes train data is loaded here)
    train, _ = load_data()
    y_train = np.zeros_like(train[CONFIG.TARGET_COLUMNS], dtype=np.float32)
    EPSILON = 1e-6  # Small constant to avoid log10(0)

    for target_idx, target in enumerate(CONFIG.TARGET_COLUMNS):
        v = train[target].values
        if target in CONFIG.LOG_FEATURES:
            v = np.where(v <= 0, EPSILON, v)  # Replace non-positive values with EPSILON
            v = np.log10(v)
        y_train[:, target_idx] = v

    SCALER = StandardScaler()
    SCALER.fit(y_train)

    # Evaluation and submission
    submission_rows = []
    with torch.no_grad():
        for X_sample_test, test_id in tqdm(test_dataset):
            y_pred = model(X_sample_test.unsqueeze(0).to('cuda')).detach().cpu().numpy()
            y_pred = SCALER.inverse_transform(y_pred).squeeze()
            row = {'id': test_id}
            for k, v in zip(CONFIG.TARGET_COLUMNS, y_pred):
                if k in CONFIG.LOG_FEATURES:
                    row[k.replace('_mean', '')] = 10 ** v
                else:
                    row[k.replace('_mean', '')] = v
            submission_rows.append(row)

    submission_df = pd.DataFrame(submission_rows)
    submission_df.to_csv('submission.csv', index=False)
    logging.info("Evaluation complete. Submission file created.")
