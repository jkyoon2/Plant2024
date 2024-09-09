import logging
import torch
import pandas as pd
from tqdm import tqdm  # Change this import
from torch.utils.data import DataLoader
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
    model = Model().to('cuda')  # Ensure the model has 5 output classes
    model.load_state_dict(torch.load(CONFIG.MODEL_NAME))
    model.eval()

    # Define label to index mapping
    lab2idx = {0: 0, 2: 1, 3: 2, 4: 3, 6: 4}
    idx2lab = {v: k for k, v in lab2idx.items()}

    # Load the test CSV file
    test_df = pd.read_csv(CONFIG.TEST_CSV_PATH)

    # Evaluation and submission
    submission_rows = []
    with torch.no_grad():
        for X_sample_test, test_id in tqdm(test_dataset):
            y_pred = model(X_sample_test.unsqueeze(0).to('cuda')).detach().cpu()
            y_pred_softmax = torch.softmax(y_pred, dim=1).numpy().squeeze()
            predicted_index = np.argmax(y_pred_softmax)
            predicted_label = idx2lab[predicted_index]
            prediction_score = y_pred_softmax[predicted_index]

            row = {
                'id': test_id,
                'prediction_label': predicted_label,
                'prediction_score': prediction_score
            }
            submission_rows.append(row)

    submission_df = pd.DataFrame(submission_rows)
    # Merge the new predictions with the original test DataFrame
    final_df = test_df.merge(submission_df, on='id', how='left')
    final_df.to_csv('test_species_pred.csv', index=False)
    logging.info("Evaluation complete. Submission file created.")