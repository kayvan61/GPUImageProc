from PIL import Image

import cupy as cp
import cupyx
from cupyx.scipy.signal import convolve2d as c2d

import operator
import numpy as np
import pandas as pd
from scipy.signal import convolve2d
from concurrent.futures import ThreadPoolExecutor

import time
import psutil
import matplotlib.pyplot as plt

# Open the RGB image

def box_blur_grayscale_cupy(image, k_size, ker):
  # Open Image, separate out RGB
  img = image

  width, height = img.size
  size = img.size

  # Define a simple 3x3 kernel for box blur
  kernel_size = k_size
  kernel = ker

  # Convert to numpy arrays
  img_arr = np.array(img)

  # Convert numpy arrays to cupy arrays
  img_cp = cp.array(img_arr)
  kernel_cp = cp.array(kernel)

  # 2D Convolution Function
  c_r = c2d(img_cp, kernel_cp, mode='same')

  # Turn cupy arrays into an image
  img_convolved = cp.asnumpy(c_r)

  # Normalize and convert to uint8 if necessary
  img_convolved = np.clip(img_convolved, 0, 255).astype('uint8')

  # Display the image
  return img_convolved


def sobel(image):
  kernelx = [[-1, 0 , 1],
           [-2, 0, 2],
           [-1, 0, 1]]
  kernely = [[-1, -2 , -1],
            [0, 0, 0],
            [1, 2, 1]]

  image1 = box_blur_grayscale_cupy(image, 3, kernelx)
  image2 = box_blur_grayscale_cupy(image, 3, kernely)


# Convert the images to NumPy arrays
  array1 = np.array(image1)
  array2 = np.array(image2)

  # Add the pixel values
  result_array = np.clip(array1 + array2, 0, 255).astype('uint8')

  # Convert the NumPy array back to a Pillow image
  result_image = Image.fromarray(result_array)

  return result_image

def gradient_thresholding(gradient_image, threshold):
    # Create a binary image based on the threshold
    gradient_image = np.array(gradient_image)
    edges_binary = (gradient_image > threshold).astype(np.uint8) * 255

    # Convert NumPy array to a Pillow image for visualization or further processing
    edges_pil = Image.fromarray(edges_binary)

    return edges_pil


def dual_thresholding(gradient_image, lower_threshold, upper_threshold):
    # Convert Pillow Image to NumPy array if necessary
    gradient_image = np.array(gradient_image)

    # Create a binary image based on the dual thresholds
    edges_binary = np.zeros_like(gradient_image, dtype=np.uint8)
    edges_binary[gradient_image > upper_threshold] = 255  # Strong edges
    edges_binary[(gradient_image >= lower_threshold) & (gradient_image <= upper_threshold)] = 128  # Potential edges

    # Convert NumPy array to a Pillow image for visualization or further processing
    edges_pil = Image.fromarray(edges_binary)

    return edges_pil


rgb_image = Image.open("./audi.jpg")  # Replace with the path to your RGB image

# Convert the RGB image to grayscale
grayscale_image = rgb_image.convert("L")

sobel_image = sobel(grayscale_image)

thresholding_image = gradient_thresholding(sobel_image, 100)

dual_thresholded_image = dual_thresholding(thresholding_image, 75, 200)

dual_thresholded_image

