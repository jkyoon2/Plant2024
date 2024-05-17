import os
from train import train
from evaluate import evaluate
from config import CONFIG

def main():
    
    if not os.path.exists(CONFIG.MODEL_NAME):
        train()
    else:
        print(f"Model found at {CONFIG.MODEL_NAME}, skipping training.")
    
    evaluate()

if __name__ == "__main__":
    main()
