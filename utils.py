import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

def load_image(path: str, target_size: tuple[int, int]=None, tensor: bool=True) -> Image or torch.Tensor:
    image = Image.open(path).convert("RGB")
    if target_size:
        image = image.resize(target_size)
    if tensor:
        return ToTensor()(image).clamp(0, 1)
    else:
        return image


def save_image(tensor: torch.Tensor, path: str) -> None:
    image = ToPILImage()(tensor.squeeze(0).clamp(0, 1))
    image.save(path)


def add_noise(img: torch.Tensor, noise_factor: float=0.1) -> torch.Tensor:
    noisy_img = img + noise_factor * torch.randn_like(img)
    return noisy_img.clamp(0, 1)


def plot_image(tensor: torch.Tensor, title: str=None) -> None:
    image = ToPILImage()(tensor.squeeze(0))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()
