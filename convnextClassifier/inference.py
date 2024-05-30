import torch
from transformers import pipeline, AutoModelForImageClassification, AutoImageProcessor
from PIL import Image
import torch.nn.functional as F

def load_model_and_processor(repo_name):
    try:
        # Attempt to load the image processor from the specified repository
        image_processor = AutoImageProcessor.from_pretrained(repo_name)
    except OSError:
        # If the repository does not contain the processor config, use the processor from a specific model
        print(f"Warning: The repository '{repo_name}' does not contain a 'preprocessor_config.json' file. Using the processor from 'facebook/convnext-large-224-22k-1k' instead.")
        image_processor = AutoImageProcessor.from_pretrained("facebook/convnext-large-224-22k-1k")
    
    # Load the model from the specified repository
    model = AutoModelForImageClassification.from_pretrained(repo_name)
    return model, image_processor

def predict(model, image_processor, image_path):
    image = Image.open(image_path).convert("RGB")
    pixel_values = image_processor(image, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(pixel_values)
        logits = outputs.logits

    # 소프트맥스를 사용하여 확률 계산
    probabilities = F.softmax(logits, dim=-1)

    # 예측된 클래스 인덱스와 해당 확률 추출
    predicted_class_idx = logits.argmax(-1).item()
    predicted_class_prob = probabilities[0, predicted_class_idx].item()

    return model.config.id2label[predicted_class_idx], predicted_class_prob

def create_pipeline(repo_name):
    return pipeline(model=repo_name)