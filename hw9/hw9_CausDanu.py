import matplotlib.pyplot as plt
import imageio
import numpy as np
from skimage.transform import AffineTransform, warp
from skimage.color import rgb2gray
from skimage.feature import (match_descriptors, ORB, plot_matches)

im = imageio.imread("Elbphilharmonie.jpg")
im = rgb2gray(im)

im1 = np.fliplr(im)

tform = AffineTransform(scale=(1.5, 1.5), rotation=-20*np.pi/180, translation=(-300, -300))
im2 = warp(im, tform, output_shape=im.shape)
print(im2)

# Plot the 3 images
fig = plt.figure("All three")
plt.subplot(311)
plt.imshow(im, cmap="gray")
plt.subplot(312)
plt.imshow(im1, cmap="gray")
plt.subplot(313)
plt.imshow(im2, cmap="gray")

plt.show()
#####################################################################################################

# Plot the ORB descriptor points/correspondences

descriptor_extractor = ORB(n_keypoints=100)

descriptor_extractor.detect_and_extract(im)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(im1)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(im2)
keypoints3 = descriptor_extractor.keypoints
descriptors3 = descriptor_extractor.descriptors

matches01 = match_descriptors(descriptors1, descriptors2, cross_check=True)
matches02 = match_descriptors(descriptors1, descriptors3, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)

plt.gray()

plot_matches(ax[0], im, im1, keypoints1, keypoints2, matches01)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Flipped Image 1")

plot_matches(ax[1], im, im2, keypoints1, keypoints3, matches02)
ax[1].axis('off')
ax[1].set_title("Original Image vs. Affine Transformed Image 2")

plt.show()

