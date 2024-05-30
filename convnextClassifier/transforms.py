from torchvision.transforms import (
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

def get_transforms(image_processor):
    image_size = image_processor.size["shortest_edge"]
    image_mean = image_processor.image_mean
    image_std = image_processor.image_std

    transform = Compose(
        [
            Resize((image_size, image_size)),
            RandomResizedCrop(image_size),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize(mean=image_mean, std=image_std)
        ]
    )

    def train_transforms(examples):
        examples["pixel_values"] = [transform(image.convert("RGB")) for image in examples["image"]]
        return examples

    return train_transforms