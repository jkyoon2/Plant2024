import os
from PIL import Image
import numpy as np
from lang_sam import LangSAM

def segment_images(folder_path, output_folder, text_prompt):
    # LangSAM 모델 초기화
    model = LangSAM()

    # 폴더 내 이미지 파일들의 리스트 생성
    image_files = os.listdir(folder_path)

    # 출력 폴더가 존재하지 않으면 생성
    os.makedirs(output_folder, exist_ok=True)

    # 선택된 이미지 파일들에 대해 반복
    for filename in image_files:
        try:
            # 이미지 파일 경로 생성
            image_path = os.path.join(folder_path, filename)
            
            # 이미지 불러오기 및 알파 채널 제거
            image_pil = Image.open(image_path).convert("RGB")
            image_np = np.array(image_pil)
            
            # 객체 분할 예측
            masks, boxes, phrases, logits = model.predict(image_pil, text_prompt)
            
            # 배경을 흰색으로 설정
            result_image_np = np.ones_like(image_np, dtype=np.uint8) * 255  # 흰색 배경

            # 마스크 적용된 부분만 원본 이미지 유지
            for mask in masks:
                mask_np = np.array(mask)
                result_image_np[mask_np] = image_np[mask_np]

            # 결과 이미지를 PIL 이미지로 변환
            result_image = Image.fromarray(result_image_np, 'RGB')

            # JPEG 형식으로 저장
            output_path = os.path.join(output_folder, filename)
            result_image.save(output_path, format='JPEG')

        except Exception as e:
            print(f"Error processing {filename} in {folder_path}: {e}")

# 폴더와 프롬프트 설정
folders_and_prompts = {
    "0": "isolate the main plant or the plant things from the background",
    "1": "isolate the main plant or fruit from the background",
    "2": "isolate the main plant or flower or branch from the background",
    "3": "isolate the main plant or flower or fruit from the background",
    "4": "isolate the only plant with out sand from the background",
    "5": "isolate the main plant or the plant things from the background",
    "6": "isolate the plant things and flowers from the background",
    "8": "isolate the sand plants from the background",
    "9": "isolate the dry plants and flower from the background",
    "10": "isolate the plants or flower from the background"
}

# 기본 경로 설정
base_input_path = "/root/sorted_images"
base_output_path = "/root/segmented_sort"

# 모든 폴더에 대해 세그멘테이션 수행 및 하나의 폴더에 저장
for folder, prompt in folders_and_prompts.items():
    input_path = os.path.join(base_input_path, folder)
    output_path = os.path.join(base_output_path, folder)
    segment_images(input_path, output_path, prompt)
