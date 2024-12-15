import numpy as np
import torch

def amplitude_mutate(amp1, amp2, beta=0.001, ratio=1.0):
    amp1_ = np.fft.fftshift(amp1, axes=(0, 1))
    amp2_ = np.fft.fftshift(amp2, axes=(0, 1))

    amp1_ = amp1_.squeeze(0)
    amp2_ = amp2_.squeeze(0)
    amp1_ = amp1_.transpose(1, 2, 0)
    amp2_ = amp2_.transpose(1, 2, 0)

    # print(amp1_.shape)

    h, w, c = amp1_.shape
    h_crop = int(h * beta)
    w_crop = int(w * beta)
    h_start = h // 2 - h_crop // 2
    w_start = w // 2 - w_crop // 2

    amp1_c = np.copy(amp1_)
    amp2_c = np.copy(amp2_)
    amp1_[h_start:h_start + h_crop, w_start:w_start + w_crop] = \
        ratio * amp2_c[h_start:h_start + h_crop, w_start:w_start + w_crop] \
        + (1 - ratio) * amp1_c[h_start:h_start + h_crop, w_start:w_start + w_crop]

    amp1_ = np.fft.ifftshift(amp1_, axes=(0, 1))


    amp1_ = amp1_.transpose(2, 0, 1)
    amp1_ = torch.tensor(amp1_, dtype=torch.float32)

    amp1_ = amp1_.unsqueeze(0)
    return amp1_


def fourier_transform(image_np):
    '''perform 2D-fft of the input image and return amplitude and phase component'''
    # image.shape: H, W, C
    if image_np.requires_grad:
        image_np = image_np.detach()

        # Convert to NumPy array
    image_np = image_np.cpu()
    image_np = image_np.numpy()

    fft_image_np = np.fft.fft2(image_np, axes=(0, 1))
    # extract amplitude and phase of both ffts
    amp_np, pha_np = np.abs(fft_image_np), np.angle(fft_image_np)
    return amp_np, pha_np

def fourier_image_perturbation(image1_np, image2_np, beta=0.001, ratio=0.0):
    '''perturb image via mutating the amplitude component'''
    amp1_np, pha1_np = fourier_transform(image1_np)
    amp2_np, pha2_np = fourier_transform(image2_np)
    # mutate the amplitude part of source with target
    amp_src_ = amplitude_mutate(amp1_np, amp2_np, beta=beta, ratio=ratio)

    # mutated fft of source
    fft_image1_ = amp_src_ * np.exp(1j * pha1_np)

    # get the mutated image
    image12 = np.fft.ifft2(fft_image1_, axes=(0, 1))
    image12 = np.real(image12)
    image12 = np.uint8(np.clip(image12, 0, 255))

    return image12

# --ratio 1.0 --beta_opt 0.006
# image = fourier_image_perturbation(image, selected_td_image, beta=0.006 , ratio=1.0)
