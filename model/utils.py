import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.optim as optim


def loader(size):
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor()
    ])


unloader = transforms.ToPILImage()


def image_loader(image_name: str, size: int = 512) -> torch.Tensor:
    """
    Loads and preprocesses an image from the given file name.

    Args:
        image_name (str): The file name of the image.
        size (int): Image size.

    Returns:
        torch.Tensor: The preprocessed image as a tensor.
    """
    image = Image.open(image_name)
    image = image.resize((512, 512))
    image = loader(size)(image).unsqueeze(0)

    return image.to('cpu', torch.float)


def gram_matrix(inp: torch.Tensor) -> torch.Tensor:
    """
    Computes the Gram matrix of the input tensor.

    Args:
        inp (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The Gram matrix.
    """
    a, b, c, d = inp.size()
    features = inp.view(a * b, c * d)
    G = torch.mm(features, features.t())

    return G.div(a * b * c * d)


def get_input_optimizer(input_img: torch.Tensor) -> optim.Optimizer:
    """
    Returns an optimizer for the input tensor.

    Args:
        input_img (torch.Tensor): The input tensor.

    Returns:
        optim.Optimizer: The optimizer.
    """
    optimizer = optim.LBFGS([input_img])

    return optimizer
