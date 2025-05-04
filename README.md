# Deep Image Prior Denoising

[Deep Image Prior](https://arxiv.org/abs/1711.10925) by Ulyanov et al. showed that the structure of CNNs themselves 
could serve as image priors. 
Their modified UNet proved to be very effective for image reconstruction. 
Without prior training, the network is able to perform various reconstruction tasks, include denoising.
Here, we look at this denoising task.
Not only are we able to recreate a network effective at denoising, but also confirm the results seen by the authors 
when comparing different image structures. 
The network learns natural images more easily than random noise, supporting the claims by the authors that 
"the structure of a generator network is sufficient to capture a great deal of low-level image statistics 
prior to any learning".  

Our findings are shown in `experiment.ipynb`.

## Usage instructions

Clone the environment by installing the packages in `environment.yml` or running
```bash
conda env create -f environment.yml
conda activate deep-image-prior
```  

To denoise an image run
```bash
python denoise.py <path/to/noisy/image>
```  

Optional arguments
- `--iterations`: Number of training iterations (default: 8000)
- `--save_interval`: Write model output every n iterations (default: 1000)
- `--output_dir`: Directory to save output images (default: "outputs")
- `--use_gpu`: If True and GPU is available, use it for training (default: True)

Additional Notes
- The network will eventually overfit the image, reconstructing the noise. This varies by image. 
For best results, early stopping and monitoring are required. This entails adjusting `--iterations` and `--save_interval`
as appropriate.
- A GPU is reccomended. Both `denoise.py` and `experiment.ipynb` will make use of a GPU if available.
