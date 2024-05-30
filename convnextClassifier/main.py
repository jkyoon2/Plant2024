from dataset import load_data, push_dataset_to_hub, get_labels, get_image_processor
from transforms import get_transforms
from train import create_dataloader, train_model
from inference import load_model_and_processor, predict, create_pipeline
from datasets import load_dataset
from huggingface_hub import HfApi
import os

def dataset_exists_on_hub(repo_name: str) -> bool:
    api = HfApi()
    try:
        api.dataset_info(repo_name)
        return True
    except Exception:
        return False

def main():
    data_repo_name = "Juliekyungyoon/plant-kaggle"
    model_repo_name = 'Juliekyungyoon/plant-kaggle-conv'
    data_dir = "/root/plant_img_data"
    
    if dataset_exists_on_hub(data_repo_name):
        print(f"Dataset {data_repo_name} exists on the hub. Loading from hub.")
        dataset = load_dataset(data_repo_name, split='train')
    else:
        print(f"Dataset {data_repo_name} does not exist on the hub. Loading from local and pushing to hub.")
        dataset = load_data(data_dir)
        push_dataset_to_hub(dataset, data_repo_name)
    
    # 레이블 가져오기
    id2label, label2id = get_labels(dataset)
    
    # 이미지 프로세서 가져오기
    image_processor = get_image_processor("facebook/convnext-large-224-22k-1k")
    
    # 데이터 변환 설정
    transform = get_transforms(image_processor)
    
    # 데이터로더 생성
    dataloader = create_dataloader(dataset, transform)
    
    # 모델 학습
    model = train_model("facebook/convnext-large-224-22k-1k", dataloader, id2label, label2id)
    
    # 모델과 프로세서 로드
    model, image_processor = load_model_and_processor(model_repo_name)
    
    # 예측 수행
    image_path = "/root/test.png"  # 예시 이미지 경로
    predicted_label = predict(model, image_processor, image_path)
    print("Predicted label:", predicted_label)
    
    # 파이프라인 생성 및 예측
    pipe = create_pipeline(model_repo_name)
    prediction = pipe(image_path)
    print("Pipeline prediction:", prediction)

if __name__ == "__main__":
    main()