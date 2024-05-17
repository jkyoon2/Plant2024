import os

class Config():
    MODEL_NAME = 'swin_22k_epoch10.pth'
    TRAIN_CSV_PATH = '/root/.dataset/train.csv'
    TEST_CSV_PATH = '/root/.dataset/test.csv'
    TRAIN_IMAGE_PATH = '/root/.dataset/train_images'
    TEST_IMAGE_PATH = '/root/.dataset/test_images'
    IMAGE_SIZE = 384
    BACKBONE = 'swin_large_patch4_window12_384.ms_in22k_ft_in1k'
    TARGET_COLUMNS = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    N_TARGETS = len(TARGET_COLUMNS)
    BATCH_SIZE = 4
    LR_MAX = 1e-4
    WEIGHT_DECAY = 0.01
    N_EPOCHS = 10
    TRAIN_MODEL = True
    IS_INTERACTIVE = os.environ.get('KAGGLE_KERNEL_RUN_TYPE') == 'Interactive'
    LOG_FEATURES = ['X4_mean', 'X11_mean', 'X18_mean', 'X50_mean', 'X26_mean', 'X3112_mean']
    N_TRAIN_SAMPLES = 55489  # train.csv 파일의 sample 개수
    N_STEPS_PER_EPOCH = (N_TRAIN_SAMPLES // BATCH_SIZE)
    N_STEPS = N_STEPS_PER_EPOCH * N_EPOCHS + 1
    
CONFIG = Config()
