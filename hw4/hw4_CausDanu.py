import imageio
import numpy as np
import matplotlib.pyplot as plt
from skimage import filters
from skimage import transform

im = imageio.imread("test-image.jpg")
im_float = np.asarray(im, dtype=np.float64)
im_mean = np.mean(im_float, axis=2)

# Laplacian Pyramid Layer 1:
gaussian11 = filters.gaussian(im_mean, sigma=3, multichannel=False)
layer11 = transform.rescale(gaussian11, 0.5, anti_aliasing=True, multichannel=False)
gaussian12 = filters.gaussian(im_mean, sigma=9, multichannel=False)
layer12 = transform.rescale(gaussian12, 0.5, anti_aliasing=True, multichannel=False)
laplacian11 = layer11 - layer12
laplacian12 = layer12 - layer11

# Laplacian Pyramid Layer 2:
gaussian21 = filters.gaussian(gaussian11, sigma=3, multichannel=False)
layer21 = transform.rescale(gaussian21, 0.5, anti_aliasing=True, multichannel=False)
gaussian22 = filters.gaussian(gaussian12, sigma=9, multichannel=False)
layer22 = transform.rescale(gaussian22, 0.5, anti_aliasing=True, multichannel=False)
laplacian21 = layer21 - layer22
laplacian22 = layer22 - layer21

# Laplacian Pyramid Layer 3:
gaussian31 = filters.gaussian(gaussian21, sigma=3, multichannel=False)
layer31 = transform.rescale(gaussian31, 0.5, anti_aliasing=True, multichannel=False)
gaussian32 = filters.gaussian(gaussian22, sigma=9, multichannel=False)
layer32 = transform.rescale(gaussian32, 0.5, anti_aliasing=True, multichannel=False)
laplacian31 = layer31 - layer32
laplacian32 = layer32 - layer31

# Laplacian Pyramid Layer 4:
gaussian41 = filters.gaussian(gaussian31, sigma=3, multichannel=False)
layer41 = transform.rescale(gaussian41, 0.5, anti_aliasing=True, multichannel=False)
gaussian42 = filters.gaussian(gaussian32, sigma=9, multichannel=False)
layer42 = transform.rescale(gaussian42, 0.5, anti_aliasing=True, multichannel=False)
laplacian41 = layer41 - layer42
laplacian42 = layer42 - layer41

fig = plt.figure("On Off: ", figsize=(5, 8))
fig.add_subplot(511)
plt.imshow(im_mean, cmap="gray")
fig.add_subplot(512)
plt.imshow(laplacian11, cmap="gray")
fig.add_subplot(513)
plt.imshow(laplacian21, cmap="gray")
fig.add_subplot(514)
plt.imshow(laplacian31, cmap="gray")
fig.add_subplot(515)
plt.imshow(laplacian41, cmap="gray")


fig2 = plt.figure("Off On: ", figsize=(5, 8))
fig2.add_subplot(511)
plt.imshow(im_mean, cmap="gray")
fig2.add_subplot(512)
plt.imshow(laplacian12, cmap="gray")
fig2.add_subplot(513)
plt.imshow(laplacian22, cmap="gray")
fig2.add_subplot(514)
plt.imshow(laplacian32, cmap="gray")
fig2.add_subplot(515)
plt.imshow(laplacian42, cmap="gray")

plt.show()