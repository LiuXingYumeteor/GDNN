import torch
import torch.nn.functional as F

def psnr_loss(target, input, max_pixel=1.0):
    """
    Calculate the PSNR (Peak Signal-to-Noise Ratio) loss between a target and an input image.

    Parameters:
    - target: Tensor of the target image.
    - input: Tensor of the input (reconstructed) image.
    - max_pixel: The maximum pixel value in the image (default is 1.0 for images normalized to [0, 1]).

    Returns:
    - PSNR loss.
    """
    mse_loss = F.mse_loss(input, target)
    if mse_loss == 0:
        return torch.tensor(float('inf'), device=target.device)  # Ensure tensor is on the same device as input/target
    max_pixel_tensor = torch.tensor(max_pixel, device=target.device, dtype=target.dtype)  # Convert max_pixel to tensor
    psnr = 20 * torch.log10(max_pixel_tensor) - 10 * torch.log10(mse_loss)
    return -psnr  # Return negative PSNR because higher PSNR is better, but we minimize loss

