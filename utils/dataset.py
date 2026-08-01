from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader

IMAGE_SIZE = 224

BATCH_SIZE = 16


transform_train = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomRotation(15),

    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
    ),

    transforms.RandomApply([
        transforms.GaussianBlur(3)
    ], p=0.3),

    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
])


def get_dataloaders(dataset_path):

    train_dataset = datasets.ImageFolder(
        f"{dataset_path}/train",
        transform=transform_train
    )

    val_dataset = datasets.ImageFolder(
        f"{dataset_path}/val",
        transform=transform_test
    )

    test_dataset = datasets.ImageFolder(
        f"{dataset_path}/test",
        transform=transform_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    return train_loader, val_loader, test_loader