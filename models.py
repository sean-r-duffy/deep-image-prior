import torch
import torch.nn as nn
import torch.optim as optim


unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=3, init_features=32, pretrained=False)


class DIPNet(nn.Module):
    def __init__(self, nu, ku, nd, kd, ns, ks, d_in, upsampling='bilinear'):
        super(DIPNet, self).__init__()
        assert len(nu) == len(ku) == len(ns) == len(ks) == len(nd) == len(kd)
        self.n = len(nu)
        self.c = d_in[1]

        def down_layer(in_channels, out_channels, k):
            layer = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=2, padding=k // 2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1),
                nn.Conv2d(out_channels, out_channels, kernel_size=k, stride=1, padding=k // 2),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(negative_slope=0.1)
            )
            return layer

        def up_layer(in_channels, out_channels, k):
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

        def skip_layer(in_channels, out_channels, k):
            if out_channels != 0:
                layer = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=k, stride=1, padding=k // 2),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(negative_slope=0.1)
                )
                return layer
            else:
                return None

        encoder_layers = []
        decoder_layers = []
        skip_layers = []
        encoder_layers.append(down_layer(self.c, nd[0], kd[0]))
        skip_layers.append(skip_layer(nd[0], ns[0], ks[0]))
        for i in range(1, self.n):
            encoder_layers.append(down_layer(nd[i-1], nd[i], kd[i]))
            skip_layers.append(skip_layer(nd[i], ns[i], ks[i]))
        for i in range(self.n-1):
            decoder_layers.append(up_layer(nd[i+1], nd[i], ku[i]))
        decoder_layers.append(up_layer(nu[self.n-1], nu[self.n-1], ku[self.n-1]))
        decoder_layers.reverse()
        skip_layers.reverse()

        self.encoder = nn.ModuleList(encoder_layers)
        self.decoder = nn.ModuleList(decoder_layers)
        self.skip_connections = nn.ModuleList(skip_layers)
        self.final_layer = nn.Conv2d(nu[0], 3, kernel_size=1, stride=1)

    def forward(self, x):
        skip_outs = [None]
        for layer in self.encoder:
            x = layer(x)
            skip_outs.append(x)
        skip_outs = skip_outs[:-1]

        for idx, layer in enumerate(self.decoder):
            x = layer(x)
            if self.skip_connections[idx] and skip_outs[-(idx + 1)] is not None:
                skip = self.skip_connections[idx](skip_outs[-(idx + 1)])
                repeat_factor = x.shape[1] // skip.shape[1]
                skip = skip.repeat(1, repeat_factor, 1, 1)
                x += skip

        x = self.final_layer(x)

        return x