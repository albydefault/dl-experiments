import os
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, CelebA, Flowers102
from torch.utils.data import DataLoader

def make_data_loader(
    dataset_name: str,
    batch_size: int = 128,
    resize: tuple[int, int] = (32, 32),
    num_workers: int = 4,
    shuffle: bool = True,
    download: bool = True,
    root: str | os.PathLike = None,
) -> tuple[DataLoader, DataLoader]:
    """
    General data loader factory function.
    
    Args:
        dataset_name: Name of the dataset ('mnist', 'cifar10', 'celeba', 'flowers102')
        batch_size: Batch size for data loaders
        resize: Target image size as (height, width)
        train_test_split: Not used currently (datasets have predefined splits)
        num_workers: Number of worker processes for data loading
        shuffle: Whether to shuffle training data
        download: Whether to download dataset if not present
        root: Root directory for dataset storage
        
    Returns:
        Tuple of (train_loader, test_loader)
    """
    dataset_name = dataset_name.lower()
    if root is None:
        root = os.environ.get("DATASET_DIR", "./data")

    if dataset_name == 'mnist':
        return make_mnist_loader(batch_size=batch_size, resize=resize, num_workers=num_workers, shuffle=shuffle, download=download, root=root)
    elif dataset_name == 'cifar10':
        return make_cifar10_loader(batch_size=batch_size, resize=resize, num_workers=num_workers, shuffle=shuffle, download=download, root=root)
    elif dataset_name == 'celeba':
        return make_celeba_loader(batch_size=batch_size, resize=resize, num_workers=num_workers, shuffle=shuffle, download=download, root=root)
    elif dataset_name in ['flowers102', 'flowers']:
        return make_flowers_loader(batch_size=batch_size, resize=resize, num_workers=num_workers, shuffle=shuffle, download=download, root=root)
    else:
        available = ['mnist', 'cifar10', 'celeba', 'flowers102']
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {available}")

def make_mnist_loader(batch_size=128, resize=(32, 32), num_workers=4, shuffle=True, download=True, root: str | os.PathLike = "./data"):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    mnist_train_set = MNIST(
        root=root,
        train=True,
        download=download,
        transform=transform
    )

    mnist_test_set = MNIST(
        root=root,
        train=False,
        download=download,
        transform=transform
    )
    train_loader = DataLoader(
        mnist_train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        mnist_test_set,
        batch_size=batch_size,
        shuffle=False,  # Test set should never be shuffled
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def make_cifar10_loader(batch_size=128, resize=(32, 32), num_workers=4, shuffle=True, download=True, root: str | os.PathLike = "./data"):
    # Build transform list - more explicit and cleaner than the previous approach
    transform_list = []
    
    # Only add resize if different from native CIFAR-10 size
    if resize != (32, 32):
        transform_list.append(transforms.Resize(resize))
    
    # Add augmentations and normalization
    transform_list.extend([
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    transform = transforms.Compose(transform_list)

    cifar10_train_set = CIFAR10(
        root=root,
        train=True,
        download=download,
        transform=transform
    )

    cifar10_test_set = CIFAR10(
        root=root,
        train=False,
        download=download,
        transform=transform
    )
    
    train_loader = DataLoader(
        cifar10_train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        cifar10_test_set,
        batch_size=batch_size,
        shuffle=False,  # Test set should never be shuffled
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


def make_celeba_loader(batch_size=128, resize=(32, 32), num_workers=4, shuffle=True, download=True):
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # CelebA images are 178x218, crop to square
        transforms.Resize(resize),   # Then resize to target size
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    celeba_train_set = CelebA(
        root=DATASET_DIR,
        split='train',
        download=download,
        transform=transform
    )

    celeba_test_set = CelebA(
        root=DATASET_DIR,
        split='test',
        download=download,
        transform=transform
    )
    
    train_loader = DataLoader(
        celeba_train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        celeba_test_set,
        batch_size=batch_size,
        shuffle=False,  # Test set should never be shuffled
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader

def make_flowers_loader(batch_size=128, resize=(32, 32), num_workers=4, shuffle=True, download=True):
    transform = transforms.Compose([
        transforms.Resize(resize),  # Flowers102 images vary in size, always resize
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    flowers_train_set = Flowers102(
        root=DATASET_DIR,
        split='train',
        download=download,
        transform=transform
    )

    flowers_test_set = Flowers102(
        root=DATASET_DIR,
        split='test',
        download=download,
        transform=transform
    )
    
    train_loader = DataLoader(
        flowers_train_set,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    test_loader = DataLoader(
        flowers_test_set,
        batch_size=batch_size,
        shuffle=False,  # Test set should never be shuffled
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, test_loader
def make_celeba_loader(batch_size=64, resize=(64, 64), num_workers=4, shuffle=True, download=True, root: str | os.PathLike = "./data"):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize[0]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = CelebA(root=root, split='train', download=download, transform=transform)
    test_set = CelebA(root=root, split='test', download=download, transform=transform)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader

def make_flowers_loader(batch_size=64, resize=(64, 64), num_workers=4, shuffle=True, download=True, root: str | os.PathLike = "./data"):
    transform = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(resize[0]),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    train_set = Flowers102(root=root, split='train', download=download, transform=transform)
    test_set = Flowers102(root=root, split='test', download=download, transform=transform)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
