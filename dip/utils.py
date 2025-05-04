import torch
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image
from PIL.Image import Image as PILImage
from typing import Tuple, Union


def load_image(path: str, target_size: Tuple[int, int] = None, tensor: bool = True) -> Union[PILImage, torch.Tensor]:
    """
    Loads an image from a specified path, processes it as RGB, and optionally resizes.

    :param path: Path to the image file to be loaded.
    :param target_size: Tuple specifying the desired image size (width, height). If
        None, the image is not resized.
    :param tensor: If True, the processed image is converted to a PyTorch Tensor with normalized values
        in the range [0, 1]. If False, the function returns the image as a PIL Image object.

    :return: The processed image as a PIL Image or PyTorch Tensor.
    """
    image = Image.open(path).convert("RGB")
    if target_size:
        image = image.resize(target_size)
    if tensor:
        return ToTensor()(image).clamp(0, 1)
    else:
        return image


def save_image(tensor: torch.Tensor, path: str) -> None:
    """
    Save a tensor as an image to a specified path on disk.

    :param tensor: A PyTorch tensor representing the image to save. Formatted as (channels, height, width).
    :param path: A string specifying the file path where the image should be saved and filename (with extension).

    :return: None
    """
    image = ToPILImage()(tensor.squeeze(0).clamp(0, 1))
    image.save(path)


def add_noise(img: torch.Tensor, noise_factor: float = 0.1) -> torch.Tensor:
    """
    Adds random noise to an input image tensor by adding Gaussian noise and clamps
    the resulting values between 0 and 1.

    :param img: The input image represented as a torch.Tensor.
    :param noise_factor: A float that scales the magnitude of Gaussian noise added
        to the input image. Default is 0.1.

    :return: A new torch.Tensor with added Gaussian noise, normalized to the range [0, 1].
    """
    noisy_img = img + noise_factor * torch.randn_like(img)
    return noisy_img.clamp(0, 1).detach()


def plot_image(tensor: torch.Tensor, title: str = None) -> None:
    """
    Plots a single image tensor using Matplotlib.

    :param tensor: PyTorch tensor representing the image to be plotted.
    :param title: Optional title to be displayed above the image. Defaults to None.

    :return: None; displays the image.
    """
    image = ToPILImage()(tensor.squeeze(0))
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def plot_images(tensors: list, titles: list = None) -> None:
    """
    Displays a sequence of tensors as images in a single row with optional titles.

    :param tensors: A list of images (as tensors) to be displayed.
    :param titles: An optional list of strings containing titles for each
        image. Default is None.

    :return: None; displays the images.
    """
    num_images = len(tensors)
    plt.figure(figsize=(5 * num_images, 5))
    for i, tensor in enumerate(tensors):
        image = ToPILImage()(tensor.squeeze())
        plt.subplot(1, num_images, i + 1)
        plt.imshow(image)
        if titles and i < len(titles):
            plt.title(titles[i])
        plt.axis('off')
    plt.show()
