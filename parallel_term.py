from PIL import Image

import cupy as cp
import cupyx
from cupyx.scipy.signal import convolve2d

import numpy as np
import operator
import matplotlib.pyplot as plt

# Naive: CPU Sequential Implementation of Box Blur
def box_blur(filename, kernel_size):
    if kernel_size % 2 == 0 or kernel_size < 1:
        raise ValueError("Kernel size must be an odd number and greater than 0")

    img = Image.open(filename).convert('RGB')
    new_img = img.copy()
    width, height = img.size

    offset = kernel_size // 2
    normalization_factor = kernel_size ** 2

    for x in range(offset, width - offset):
        for y in range(offset, height - offset):
            sum_pixels = (0, 0, 0)
            for kx in range(-offset, offset + 1):
                for ky in range(-offset, offset + 1):
                    pixel = img.getpixel((x + kx, y + ky))
                    sum_pixels = tuple(map(operator.add, sum_pixels, pixel))

            avg_pixel = tuple(map(lambda x: x // normalization_factor, sum_pixels))
            new_img.putpixel((x, y), avg_pixel)

    return new_img

# CUDA Implementation of Box Blur using scipy convolve 2d
def box_blur_cupy(filename, k_size):
  # Open Image, separate out RGB
  img = Image.open(filename)
  img

  width, height = img.size
  size = img.size

  r_channel, g_channel, b_channel = img.split()

  # Convert to numpy arrays
  r_array = np.array(r_channel)
  g_array = np.array(g_channel)
  b_array = np.array(b_channel)

  # Define a simple 3x3 kernel for box blur
  kernel_size = k_size
  kernel = np.full((kernel_size, kernel_size), 1/(kernel_size**2))

  # Convert numpy arrays to cupy arrays
  r_cp = cp.array(r_array)
  g_cp = cp.array(g_array)
  b_cp = cp.array(b_array)
  kernel_cp = cp.array(kernel)

  # 2D Convolution Function
  c_r = convolve2d(r_cp, kernel_cp, mode='same')
  c_g = convolve2d(g_cp, kernel_cp, mode='same')
  c_b = convolve2d(b_cp, kernel_cp, mode='same')

  # Turn cupy arrays into an image
  r_convolved = cp.asnumpy(c_r)
  g_convolved = cp.asnumpy(c_g)
  b_convolved = cp.asnumpy(c_b)

  # Normalize and convert to uint8 if necessary
  r_convolved = np.clip(r_convolved, 0, 255).astype('uint8')
  g_convolved = np.clip(g_convolved, 0, 255).astype('uint8')
  b_convolved = np.clip(b_convolved, 0, 255).astype('uint8')

  # Merge channels back into an image
  img_convolved = Image.merge("RGB", (Image.fromarray(r_convolved),
                                      Image.fromarray(g_convolved),
                                      Image.fromarray(b_convolved)))

  # Display the image
  return img_convolved

# box_blur('/content/dog.jpg', 15)

box_blur_cupy('./testImages/audi.jpg', 15).save("./output.jpg")

