import os
from train import train
from evaluate import evaluate
from config import CONFIG
from tendo import singleton
import psutil
import os

def is_already_running(script_name):
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        if proc.info['cmdline'] and script_name in proc.info['cmdline'] and proc.info['pid'] != os.getpid():
            return True
    return False

if is_already_running('main.py'):
    print("Script is already running. Exiting.")
    exit(0)


def main():
    
    if not os.path.exists(CONFIG.MODEL_NAME):
        train()
    else:
        print(f"Model found at {CONFIG.MODEL_NAME}, skipping training.")
    
    evaluate()

if __name__ == "__main__":
    main()

