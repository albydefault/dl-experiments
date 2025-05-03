from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, CelebA
from torch.utils.data import DataLoader

DATASET_DIR = "/home/alazar/desktop/datasets"
DATALOADER_REGISTRY = {}

def register_dataloader(name):
    """
    A decorator to register a dataloader in the DATALOADERS dictionary.
    """
    def decorator(dataloader_class):
        DATALOADER_REGISTRY[name] = dataloader_class
        return dataloader_class
    return decorator

### CIFAR10 and MNIST datasets






### Register the dataloaders
@register_dataloader("cifar10")
def make_cifar10_loader(batch_size=128):
    cifar10_dataset = CIFAR10(
        root=DATASET_DIR,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    return DataLoader(
        cifar10_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


@register_dataloader("mnist")
def make_mnist_loader(batch_size=128):
    mnist_dataset = MNIST(
        root=DATASET_DIR,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((32, 32)),
            transforms.Normalize((0.5,), (0.5,)),
        ])
    )
    return DataLoader(
        mnist_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

@register_dataloader("celeba")
def make_celeba_loader(batch_size=128):
    celeba_dataset = CelebA(
        root=DATASET_DIR,
        split='train',
        download=True,
        transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    return DataLoader(
        celeba_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )