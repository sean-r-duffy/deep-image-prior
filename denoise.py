import torch
import os
from argparse import ArgumentParser
from tqdm import tqdm
from torch import nn, optim
from dip.models import DIPNet
from dip.utils import load_image, save_image


# Training function
def train(image_path, save_interval, output_dir, device):
    # Load image
    base_filename = os.path.splitext(os.path.basename(image_path))[0]
    image = load_image(image_path, tensor=True).to(device)
    
    # Hyperparameters
    w = image.shape[-1]
    h = image.shape[-2]
    z_shape, u_range = (1, 3, w, h), (0, 0.1)
    nu = nd = [128, 128, 128, 128, 128]
    ku = kd = [3, 3, 3, 3, 3]
    ns = [0, 0, 0, 4, 0]
    ks = [1, 1, 1, 1, 1]
    sigma_p = 1 / 30
    epochs = 10000
    lr = 0.01
    upsampling = 'bilinear'
    loss_function = nn.MSELoss()    

    # Initialize model and optimizer
    model = DIPNet(nd=nd, kd=kd, nu=nu, ku=ku, ns=ns, ks=ks, d_in=z_shape, upsampling=upsampling).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Initialize random noise input tensor
    z = torch.rand(z_shape).uniform_(u_range[0], u_range[1]).to(device)

    losses = []
    for epoch in tqdm(range(epochs)):
        # Apply random perturbation at each iteration
        perturbed_z = z + (torch.randn_like(z) * sigma_p).clamp(0, 1)

        # Forward pass, loss calculation, and backprop of loss
        optimizer.zero_grad()
        output = model(perturbed_z)
        loss = loss_function(output, image)
        loss.backward()
        losses.append(loss.item())
        optimizer.step()

        # Print loss, model output
        if epoch % save_interval == 0 or epoch == epochs - 1:
            save_image(output, f"{base_filename}_denoising_iter_{epoch}.png")

    save_image(model(z), f"{output_dir}/{base_filename}_denoising_final.png")


if __name__ == "__main__":
    # Parse command line arguments
    parser = ArgumentParser()
    parser.add_argument("image", type=str, help="Path to input file", required=True)
    parser.add_argument("--iterations", type=int, default=5000, help="Number of training iterations")
    parser.add_argument("--save_interval", type=int, default=50, help="Write the model output every n iterations")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory to save output images")
    parser.add_argument("--use_gpu", type=bool, default=True, help="If True and GPU is available, use it for training")
    args = parser.parse_args()

    # Use GPU if available
    if args.use_gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    print(f'Using: {device}')

    # Set random seeds
    seed = 41
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    train(args.image, args.save_interval, args.output_dir, device)
