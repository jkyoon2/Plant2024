import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForImageClassification
from tqdm import tqdm

def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}

def create_dataloader(dataset, transform, batch_size=4):
    processed_dataset = dataset.with_transform(transform)
    dataloader = DataLoader(processed_dataset, collate_fn=collate_fn, batch_size=batch_size, shuffle=True)
    return dataloader

def train_model(model_name, dataloader, id2label, label2id, num_epochs=1, lr=5e-5):
    model = AutoModelForImageClassification.from_pretrained(model_name, id2label=id2label, label2id=label2id, ignore_mismatched_sizes=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        print("Epoch:", epoch)
        correct = 0
        total = 0
        for idx, batch in enumerate(tqdm(dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            outputs = model(pixel_values=batch["pixel_values"], labels=batch["labels"])
            loss, logits = outputs.loss, outputs.logits
            loss.backward()
            optimizer.step()

            total += batch["labels"].shape[0]
            predicted = logits.argmax(-1)
            correct += (predicted == batch["labels"]).sum().item()
            accuracy = correct / total

            if idx % 100 == 0:
                print(f"Loss after {idx} steps:", loss.item())
                print(f"Accuracy after {idx} steps:", accuracy)

    return model