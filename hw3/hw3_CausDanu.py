import numpy as np
from skimage import filters
from skimage.util import random_noise
import imageio
import matplotlib.pyplot as plt

im = imageio.imread('sample.jpg')
salt_pepper_noisy_im = random_noise(im, mode="s&p", salt_vs_pepper=0.05)

fig = plt.figure("Noisy Image")
plt.imshow(salt_pepper_noisy_im)

noisy_image = np.asarray(salt_pepper_noisy_im)

# Median Filter:
def median_filter(image, kernel_size):
    image_red = image[:,:,0]
    image_green = image[:,:,1]
    image_blue = image[:,:,2]

    median_filter = np.ones(shape=(kernel_size, kernel_size))
    median_filtered_red = filters.rank.median(image_red, selem=median_filter)
    median_filtered_green = filters.rank.median(image_green, selem=median_filter)
    median_filtered_blue = filters.rank.median(image_blue, selem=median_filter)

    median_filtered = np.zeros(shape=(480,640,3), dtype=np.uint8)
    median_filtered[:,:,0] = median_filtered_red
    median_filtered[:,:,1] = median_filtered_green
    median_filtered[:,:,2] = median_filtered_blue
    return median_filtered

median_filtered = median_filter(noisy_image, 9)

fig = plt.figure("Median Filtered")
plt.imshow(median_filtered)

# Box Filter:
def box_filter(image, kernel_size):
    image_red = image[:, :, 0]
    image_green = image[:, :, 1]
    image_blue = image[:, :, 2]

    mean_filter = 1/(kernel_size*kernel_size) * np.ones(shape=(kernel_size, kernel_size))
    box_filtered_red = filters.rank.mean(image_red, selem=mean_filter)
    box_filtered_green = filters.rank.mean(image_green, selem=mean_filter)
    box_filtered_blue = filters.rank.mean(image_blue, selem=mean_filter)

    box_filtered = np.zeros(shape=(480, 640, 3), dtype=np.uint8)
    box_filtered[:,:,0] = box_filtered_red
    box_filtered[:,:,1] = box_filtered_green
    box_filtered[:,:,2] = box_filtered_blue
    return box_filtered

box_filtered = box_filter(noisy_image, 9)
fig = plt.figure("Box Filtered")
plt.imshow(box_filtered)

# Gaussian Filter
gaussian_filtered = filters.gaussian(noisy_image, sigma=2, multichannel=True)
fig = plt.figure("Gaussian Filtered")
plt.imshow(gaussian_filtered)


# Manual Median Filter:
image_red = noisy_image[:, :, 0]
image_green = noisy_image[:, :, 1]
image_blue = noisy_image[:, :, 2]

manual_median = np.ones(shape=(480, 640, 3))
manual_median_red = np.zeros(shape=(480, 640))
manual_median_green = np.zeros(shape=(480, 640))
manual_median_blue = np.zeros(shape=(480, 640))

for row in range(480):
    for col in range(640):
        manual_median_red[row, col] = np.median(image_red[row:row+9, col:col+9])
        manual_median_green[row, col] = np.median(image_green[row:row + 9, col:col + 9])
        manual_median_blue[row, col] = np.median(image_blue[row:row + 9, col:col + 9])

manual_median[:,:,0] = manual_median_red
manual_median[:,:,1] = manual_median_green
manual_median[:,:,2] = manual_median_blue

fig = plt.figure("Manual Median Filter")
plt.imshow(manual_median)

# Manual Box Filter:
manual_box = np.ones(shape=(480, 640, 3))
manual_box_red = np.zeros(shape=(480, 640))
manual_box_green = np.zeros(shape=(480, 640))
manual_box_blue = np.zeros(shape=(480, 640))

for row in range(480):
    for col in range(640):
        manual_box_red[row, col] = np.mean(image_red[row:row+9, col:col+9])
        manual_box_green[row, col] = np.mean(image_green[row:row + 9, col:col + 9])
        manual_box_blue[row, col] = np.mean(image_blue[row:row + 9, col:col + 9])

manual_box[:,:,0] = manual_box_red
manual_box[:,:,1] = manual_box_green
manual_box[:,:,2] = manual_box_blue

fig = plt.figure("Manual Box Filter")
plt.imshow(manual_box)

# DoG filter:
gaussian_filtered_sigma_2 = filters.gaussian(noisy_image, sigma=2, multichannel=True)
gaussian_filtered_sigma_5 = filters.gaussian(noisy_image, sigma=5, multichannel=True)
dog = np.subtract(gaussian_filtered_sigma_2, gaussian_filtered_sigma_5).clip(0, 1)
fig = plt.figure("DoG")
plt.imshow(dog)

# DoB filter:
box_filtered19 = box_filter(noisy_image, 19)
box_filtered43 = box_filter(noisy_image, 43)

dob = np.subtract(box_filtered19, box_filtered43)
fig = plt.figure("DoB")
plt.imshow(dob)

plt.show()