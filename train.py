import logging
import torch
from tqdm.notebook import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from model import Model
from dataset import load_data, PlantDataset
from config import CONFIG
from logging_config import setup_logging
import albumentations as A
from albumentations.pytorch import ToTensorV2
import psutil
from torch import nn
import torchmetrics
import gc

def train():
    # Initialize logging
    setup_logging()
    logging.info("Training session started.")

    # Load data from pickle files
    train, test = load_data()

    # Define transformations
    MEAN = [0.485, 0.456, 0.406]  # Convert numpy array to list
    STD = [0.229, 0.224, 0.225]  # Convert numpy array to list

    TRAIN_TRANSFORMS = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomSizedCrop(
            [448, 512],
            CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE, w2h_ratio=1.0, p=0.75),
        A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
        A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.25),
        A.ImageCompression(quality_lower=85, quality_upper=100, p=0.25),
        A.ToFloat(),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
        ToTensorV2(),
    ])

    TEST_TRANSFORMS = A.Compose([
        A.Resize(CONFIG.IMAGE_SIZE, CONFIG.IMAGE_SIZE),
        A.ToFloat(),
        A.Normalize(mean=MEAN, std=STD, max_pixel_value=1),
        ToTensorV2(),
    ])

    # Prepare training data
    y_train = np.zeros_like(train[CONFIG.TARGET_COLUMNS], dtype=np.float32)
    EPSILON = 1e-6  # Small constant to avoid log10(0)

    for target_idx, target in enumerate(CONFIG.TARGET_COLUMNS):
        v = train[target].values
        if target in CONFIG.LOG_FEATURES:
            v = np.where(v <= 0, EPSILON, v)  # Replace non-positive values with EPSILON
            v = np.log10(v)
        y_train[:, target_idx] = v

    SCALER = StandardScaler()
    y_train = SCALER.fit_transform(y_train)

    train_dataset = PlantDataset(
        train['jpeg_bytes'].values,
        y_train,
        TRAIN_TRANSFORMS,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=CONFIG.BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        num_workers=psutil.cpu_count(),
    )

    # Prepare test data
    test_dataset = PlantDataset(
        test['jpeg_bytes'].values,
        test['id'].values,
        TEST_TRANSFORMS,
    )

    
    # 메모리 초기화
    gc.collect()
    torch.cuda.empty_cache()

    # Define model, optimizer, and scheduler
    model = Model().to('cuda')
    optimizer = AdamW(model.parameters(), lr=CONFIG.LR_MAX, weight_decay=CONFIG.WEIGHT_DECAY)
    lr_scheduler = OneCycleLR(optimizer, max_lr=CONFIG.LR_MAX, total_steps=CONFIG.N_STEPS, pct_start=0.1, anneal_strategy='cos', div_factor=1e1, final_div_factor=1e1)

    # Initialize metrics
    mae_metric = torchmetrics.MeanAbsoluteError().to('cuda')
    r2_metric = torchmetrics.R2Score(num_outputs=CONFIG.N_TARGETS).to('cuda')

    # Training loop
    for epoch in range(CONFIG.N_EPOCHS):
        model.train()
        mae_metric.reset()
        r2_metric.reset()
        
        for step, (X_batch, y_true) in enumerate(train_dataloader):
            X_batch = X_batch.to('cuda')
            y_true = y_true.to('cuda')
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = nn.SmoothL1Loss()(y_pred, y_true)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            
            # Update metrics
            mae_metric(y_pred, y_true)
            r2_metric(y_pred, y_true)

            if (step + 1) % 100 == 0:
                logging.info(f'EPOCH {epoch + 1}: Step {step + 1} Loss: {loss.item()} MAE: {mae_metric.compute().item()} R²: {r2_metric.compute().item()}')

        # 에포크가 끝난 후 메모리 해제
        gc.collect()
        torch.cuda.empty_cache()

    # Save the model
    torch.save(model.state_dict(), CONFIG.MODEL_NAME)
    logging.info("Training complete!")
