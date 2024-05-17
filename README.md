# Kaggle Plant Traits 2024

This project aims to predict plant traits using deep learning models. The code is designed to run on a server environment, using PyTorch and various other libraries. The project includes data loading, model training, evaluation, and logging functionalities.

## Directory Structure

Swin_22k/
├── config.py \
├── dataset.py
├── evaluate.py
├── logging_config.py
├── main.py
├── model.py
├── requirements.txt
└── train.py


## Setup

### Prerequisites

Ensure you have the following installed:
- Python 3.10 or higher
- Virtualenv (recommended)

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/MinJeonging/Plant2024.git
    cd Plant2024/.Swin_22k
    ```

2. **Create a virtual environment and activate it:**
    ```sh
    python -m venv env
    source env/bin/activate  # On Windows, use `env\Scripts\activate`
    ```

3. **Install the required packages:**
    ```sh
    pip install -r requirements.txt
    ```

### Dataset Preparation

Ensure that the dataset is placed in the correct directory structure. If you have `train.pkl` and `test.pkl` already created, ensure they are in the `.swin` directory.

If you need to generate `train.pkl` and `test.pkl`:
1. Place your CSV files and images in the respective paths specified in `dataset.py`.

### Running the Project

1. **Train the model:**
    ```sh
    python main.py
    ```

This will start the training process, which includes logging the loss, MAE, and R² metrics.

### Evaluating the Model

The `main.py` script includes both training and evaluation steps. To only evaluate the model, ensure the `evaluate` function is called as required.

### Configuration

The configuration for the model, training parameters, and other settings are defined in `config.py`.

### Logging

Logging is set up using the `logging_config.py` file. Logs are printed to the console and can be redirected to a file if necessary.

### Notes

- Ensure you have sufficient GPU resources to run the training process.
- Modify the `config.py` file to adjust parameters like batch size, learning rate, and others to fit your dataset and computational resources.

## License

This project is licensed under the Kaggle - see the [LICENSE](https://www.kaggle.com/competitions/planttraits2024) file for details.

### Detailed File Descriptions

#### `config.py`
Defines configuration parameters for the project including image size, model backbone, target columns, and training settings.

#### `dataset.py`
Handles data loading and preprocessing. It includes a function to load data from CSV or pickle files and a dataset class for PyTorch.

#### `train.py`
Contains the training loop, model initialization, data preparation, and logging of metrics such as loss, MAE, and R².

#### `evaluate.py`
Evaluates the trained model on the test dataset and creates a submission file. It uses the same preprocessing and transformations as the training script.

#### `logging_config.py`
Sets up logging configuration for the project to log messages to the console or a file.

#### `model.py`
Defines the model architecture using `timm` to load a pre-trained backbone and modify it for the specific task.

#### `main.py`
The entry point for the project. It calls the training and evaluation functions to execute the complete pipeline.

#### `requirements.txt`
Lists all the dependencies required to run the project. This includes PyTorch, Albumentations, Timm, Torchmetrics, and others.

---

Make sure to replace `yourusername` with your actual GitHub username in the clone URL.

### Adding the `requirements.txt` file
Ensure your `requirements.txt` includes all the necessary dependencies:

```plaintext
numpy
pandas
matplotlib
imageio
torch
torchvision
timm
albumentations
torchmetrics
tqdm
scikit-learn
psutil
opencv-python-headless==4.5.5.62

