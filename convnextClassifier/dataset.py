from datasets import load_dataset
from transformers import AutoImageProcessor

def load_data(data_dir):
    dataset = load_dataset("imagefolder", data_dir=data_dir, split='train')
    return dataset

def push_dataset_to_hub(dataset, repo_name):
    dataset.push_to_hub(repo_name)

def get_labels(dataset):
    labels = dataset.features["label"].names
    id2label = {k: v for k, v in enumerate(labels)}
    label2id = {v: k for k, v in enumerate(labels)}
    return id2label, label2id

def get_image_processor(model_name):
    image_processor = AutoImageProcessor.from_pretrained(model_name)
    return image_processor