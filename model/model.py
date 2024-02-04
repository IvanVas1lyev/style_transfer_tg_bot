import math
from typing import Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random

import model.constants as constants
import model.utils as utils


class Normalization(nn.Module):
    def __init__(self, mean, std):
        """
        Normalization module that normalizes input tensors.

        Args:
            mean (torch.Tensor): The mean tensor for normalization.
            std (torch.Tensor): The standard deviation tensor for normalization.
        """
        super(Normalization, self).__init__()

        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the normalization module.

        Args:
            img (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The normalized image tensor.
        """
        return (img - self.mean) / self.std


class ContentLoss(nn.Module):
    def __init__(self, target):
        """
        Content loss module that computes the mean squared error loss between the input and target.

        Args:
            target (torch.Tensor): The target tensor.
        """
        super(ContentLoss, self).__init__()
        self.target = target.detach()
        self.loss = None

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the content loss module.

        Args:
            inp (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor.
        """
        self.loss = F.mse_loss(inp, self.target)

        return inp


class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        """
        Style loss module that computes the mean squared error loss between the input and target feature's gram matrices.

        Args:
            target_feature (torch.Tensor): The target feature tensor.
        """
        super(StyleLoss, self).__init__()
        self.target = utils.gram_matrix(target_feature).detach()

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the style loss module.

        Args:
            inp (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The input tensor.
        """
        G = utils.gram_matrix(inp)
        self.loss = F.mse_loss(G, self.target)

        return inp


def get_style_model_and_losses(
        cnn: nn.Module,
        normalization_mean: torch.Tensor,
        normalization_std: torch.Tensor,
        style_img: torch.Tensor,
        content_img: torch.Tensor,
        content_layers: list = constants.content_layers_default,
        style_layers: list = constants.style_layers_default
) -> tuple:
    """
    Create a style transfer model and compute the style and content losses.

    Args:
        cnn (nn.Module): The pretrained CNN model for feature extraction.
        normalization_mean (torch.Tensor): The mean tensor for normalization.
        normalization_std (torch.Tensor): The standard deviation tensor for normalization.
        style_img (torch.Tensor): The style image tensor.
        content_img (torch.Tensor): The content image tensor.
        content_layers (list, optional): The list of layer names to be considered for content loss.
            Defaults to constants.content_layers_default.
        style_layers (list, optional): The list of layer names to be considered for style loss.
            Defaults to constants.style_layers_default.

    Returns:
        tuple: A tuple containing the style transfer model, style losses, and content losses.
    """
    normalization = Normalization(normalization_mean, normalization_std)
    content_losses = []
    style_losses = []
    model = nn.Sequential(normalization)

    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]

    return model, style_losses, content_losses


async def run_style_transfer(
        cnn: torch.nn.Module,
        normalization_mean: torch.Tensor,
        normalization_std: torch.Tensor,
        content_img: torch.Tensor,
        style_img: torch.Tensor,
        input_img: torch.Tensor,
        num_steps: int = 300,
        style_weight: int = 1000000,
        content_weight: int = 1,
        log: Callable = print
) -> torch.Tensor:
    """
    Run the style transfer process.

    Args:
        cnn (torch.nn.Module): The CNN model to use.
        normalization_mean (torch.Tensor): The mean values for normalization.
        normalization_std (torch.Tensor): The standard deviation values for normalization.
        content_img (torch.Tensor): The content image.
        style_img (torch.Tensor): The style image.
        input_img (torch.Tensor): The input image to be transformed.
        num_steps (int): The number of optimization steps (default: 300).
        style_weight (int): The weight of the style loss (default: 1000000).
        content_weight (int): The weight of the content loss (default: 1).
        log (callable): The function to log the progress (default: print).

    Returns:
        torch.Tensor: The transformed input image.
    """
    model, style_losses, content_losses = get_style_model_and_losses(
        cnn, normalization_mean, normalization_std, style_img, content_img
    )

    input_img.requires_grad_(True)
    model.eval()
    model.requires_grad_(False)

    optimizer = utils.get_input_optimizer(input_img)
    run = [0]
    message = constants.log_msgs[0]

    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)

            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0

            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss

            style_score *= style_weight
            content_score *= content_weight

            loss = style_score + content_score
            loss.backward()
            run[0] += 1

            return style_score + content_score

        optimizer.step(closure)

        if run[0] % 40 == 0:
            await log(f'{math.floor(run[0] * 10 / num_steps) * "⬛"}'
                      f'{(10 - math.floor(run[0] * 10 / num_steps)) * "⬜"}'
                      f'\n\n{message}')

            message = random.choice(constants.log_msgs)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return input_img


async def run_model(
        style_img_path: str,
        content_img_path: str,
        output_file_path: str,
        log_func: Callable,
        size: int = 512
) -> None:
    """
    Run the style transfer model and save the output image.

    Args:
        style_img_path (str): The path to the style image file.
        content_img_path (str): The path to the content image file.
        output_file_path (str): The path to save the output image.
        log_func (callable): The function to log the progress.
        size (int): Image size.
    """
    style_img = utils.image_loader(style_img_path, size)
    content_img = utils.image_loader(content_img_path, size)

    input_img = content_img.clone()

    output = await run_style_transfer(
        constants.vgg19,
        constants.cnn_normalization_mean,
        constants.cnn_normalization_std,
        content_img,
        style_img,
        input_img,
        num_steps=300,
        log=log_func
    )

    image = output.cpu().clone()
    image = image.squeeze(0)
    image = utils.unloader(image)

    fig = plt.figure(frameon=False, figsize=(512 / 96, 512 / 96), dpi=96)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(image, aspect='auto')

    fig.savefig(output_file_path)
