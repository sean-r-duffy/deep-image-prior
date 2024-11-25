import torch
import torch.nn as nn
import torch.optim as optim


unet = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=3, init_features=32, pretrained=False)


class DeepImagePriorNet(nn.Module):
    def __init__(self, nu, ku, ns, ks, upsampling='bilinear'):
        super(DeepImagePriorNet, self).__init__()
        
        self.encoder = nn.ModuleList([
            nn.Conv2d(n_in, n_out, kernel_size=k, stride=2, padding=k // 2)
            for n_in, n_out, k in zip([32] + nu[:-1], nu, ku)
        ])

        self.decoder = nn.ModuleList([
            nn.ConvTranspose2d(n_in, n_out, kernel_size=k, stride=2, padding=k // 2, output_padding=1)
            for n_in, n_out, k in zip(nu[::-1], nu[::-1][1:] + [32], ku[::-1])
        ])
        
        self.skip_connections = nn.ModuleList([
            nn.Conv2d(n_in, n_out, kernel_size=k, stride=1, padding=k // 2)
            for n_in, n_out, k in zip([32] + ns[:-1], ns, ks)
        ])

        self.upsampling = upsampling

    def forward(self, x):
        skip_outs = []

        for layer in self.encoder:
            x = nn.LeakyReLU(negative_slope=0.1)(layer(x))
            skip_outs.append(x)

        for idx, layer in enumerate(self.decoder):
            if self.upsampling == 'bilinear':
                x = nn.functional.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            elif self.upsampling == 'nearest':
                x = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
            x = nn.LeakyReLU(negative_slope=0.1)(layer(x))
            if idx < len(skip_outs):
                x += self.skip_connections[idx](skip_outs[-(idx + 1)])
        
        return x