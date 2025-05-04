import torch.nn as nn
from torch import Tensor
from typing import Union, List


def down_layer(in_channels: int, out_channels: int, k: int) -> nn.Sequential:
    """
    Constructs a down-sampling layer using convolution, batch normalization, and LeakyReLU activation operations.
    The layer reduces spatial dimensions and increases the number of channels (feature maps).

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels.
    :param k: Kernel size for the convolutional layers.

    :return: A Sequential layer encapsulating the down-sampling operation.
    """
    layer = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=2, padding=k // 2),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=1, padding=k // 2),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1)
    )
    return layer


def up_layer(in_channels: int, out_channels: int, k: int, upsampling: str = 'bilinear') -> nn.Sequential:
    """
    Constructs an upsampling layer using transpose convolution, batch normalization, bilinear interpolation or
    nearest neighbor upsampling, and LeakyReLU activation operations.
    The layer increased spatial dimensions and decreases the number of channels (feature maps).

    :param in_channels: The number of input channels.
    :param out_channels: The number of output channels.
    :param k: Kernel size for the transposed convolution operation.
    :param upsampling: Defines the upsampling method. Accepted values are 'bilinear' or
        'nearest'. Defaults to 'bilinear'.

    :return: A Sequential layer encapsulating the up-sampling operation.
    """
    if upsampling == 'bilinear':
        upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
    elif upsampling == 'nearest':
        upsample_layer = nn.Upsample(scale_factor=2, mode='nearest')
    else:
        raise ValueError("Invalid upsampling method")

    layer = nn.Sequential(
        nn.BatchNorm2d(in_channels),
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=k, stride=1, padding=k // 2),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1),
        nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=1 // 2),
        nn.BatchNorm2d(out_channels),
        nn.LeakyReLU(negative_slope=0.1),
        upsample_layer
    )
    return layer


def skip_layer(in_channels: int, out_channels: int, k: int) -> Union[nn.Sequential, None]:
    """
    Creates a skip connection layer with a convolutional block, batch normalization, and LeakyReLU activation.
    If `out_channels` is 0, the function returns None (there is no skip connection).
    Kernal size k is usually 1, to preserve spatial dimensions.

    :param in_channels: Number of input channels.
    :param out_channels: Number of output channels; if set to 0, no layer is created.
    :param k: Kernel size for the convolutional layer

    :return: A Sequential layer encapsulating the skip-connection operation.
    """
    if out_channels != 0:
        layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding=k // 2),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=0.1)
        )
        return layer
    else:
        return None


class DIPNet(nn.Module):
    """
    Defines a deep image prior network for image reconstruction, which consists
    of an UNet-like encoder-decoder architecture with skip connections. The network supports
    different upsampling methods and generates three-channel output images.

    :ivar n: Number of layers in the encoder/decoder.
    :ivar c: Number of input channels in the input tensor.
    :ivar encoder: Module list comprising the downsampling (encoder) layers.
    :ivar decoder: Module list comprising the upsampling (decoder) layers.
    :ivar skip_connections: Module list comprising the layers skip connection layers.
    :ivar final_layer: Convolutional layer to produce the three-channel output image.
    """

    def __init__(self, nu: List[int], ku: List[int], nd: List[int], kd: List[int], ns: List[int], ks: List[int],
                 d_in: tuple, upsampling: str = 'bilinear') -> None:
        """
        Initializes the DIPNet class with encoder, decoder, and skip connection modules.

        :param nu: List of integers specifying the output channel dimensions used in the up-sampling (decoder) layers.
        :param ku: List of integers specifying the kernel sizes used by the up-sampling (decoder) layers.
        :param nd: List of integers specifying the output channel dimensions used in the down-sampling (encoder) layers.
        :param kd: List of integers specifying the kernel sizes used by the down-sampling (encoder) layers.
        :param ns: List of integers specifying the output channel dimensions used in the skip connection layers.
            Set a value of 0 for layers without skip connections.
        :param ks: List of integers specifying the kernel sizes used by the skip connection layers.
            Should usually be 1, to preserve spatial dimensions.
        :param d_in: A tuple representing the dimensions of the input data; formatted as (channels, height, width).
        :param upsampling: A string specifying the up-sampling method, either 'bilinear' or 'nearest'.
            Defaults to 'bilinear'.
        """
        super(DIPNet, self).__init__()

        # Confirm there are an equal number of encoder, decoder, and skip connection layers
        assert len(nu) == len(ku) == len(ns) == len(ks) == len(nd) == len(kd)
        self.n = len(nu)  # Depth of the network
        self.c = d_in[1]  # Channels in the input tensor

        encoder_layers = []
        decoder_layers = []
        skip_layers = []

        # Initialize 'top' encoder and skip connections layers separately from 'top' decoder
        encoder_layers.append(down_layer(self.c, nd[0], kd[0]))
        skip_layers.append(skip_layer(nd[0], ns[0], ks[0]))

        # Construct 'middle' layers
        for i in range(1, self.n):
            encoder_layers.append(down_layer(nd[i - 1], nd[i], kd[i]))
            skip_layers.append(skip_layer(nd[i], ns[i], ks[i]))
        for i in range(self.n - 1):
            decoder_layers.append(up_layer(nd[i + 1], nd[i], ku[i]))

        # Initialize 'top' decode layer separately
        decoder_layers.append(up_layer(nu[self.n - 1], nu[self.n - 1], ku[self.n - 1]))

        # Correct order
        decoder_layers.reverse()
        skip_layers.reverse()

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.skip_connections = nn.ModuleList(skip_layers)
        self.final_layer = nn.Conv2d(nu[0], 3, kernel_size=1, stride=1)

    def forward(self, x: Tensor) -> Tensor:
        """
            Processes the input tensor through the network.

            :param x: Input tensor to be passed through the network.
            :return: Output tensor after processing through the network.
            """

        # Run through encoder and store state after each layer for use in skip connections
        skip_outs = [None]
        for layer in self.encoder:
            x = layer(x)
            skip_outs.append(x)
        skip_outs = skip_outs[:-1]

        # Run through decoder, adding skip connection layer output (if present) after each decoder layer
        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            if self.skip_connections[idx] and skip_outs[-(idx + 1)] is not None:
                skip = self.skip_connections[idx](skip_outs[-(idx + 1)])

                # Repeat skip connection until channel number matches decoder output channel number
                repeat_factor = x.shape[1] // skip.shape[1]
                skip = skip.repeat(1, repeat_factor, 1, 1)

                # Sum decoder output and skip connection output
                x += skip

        x = self.final_layer(x)

        return x
