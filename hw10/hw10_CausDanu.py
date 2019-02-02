import matplotlib.pyplot as plt
import imageio
import numpy as np
from scipy.spatial import distance
from skimage.transform import AffineTransform, warp
from skimage.color import rgb2gray
from skimage.feature import hog
from skimage import exposure
from skimage.exposure import adjust_gamma

im = imageio.imread("Elbphilharmonie.jpg")
im = rgb2gray(im)

im1 = np.fliplr(im)

tform = AffineTransform(scale=(1.5, 1.5), rotation=-20*np.pi/180, translation=(-300, -300))
im2 = warp(im, tform, output_shape=im.shape)

#################################################################################################

v, hog_im = hog(im, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm = 'L1', visualize=True)
v1, hog_im1 = hog(im1, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm = 'L1', visualize=True)
v2, hog_im2 = hog(im2, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm = 'L1', visualize=True)

print("im size = ", im.shape )
print("features size = {}".format(v.shape))
print("features number of non-zero elements = ", np.count_nonzero(v))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax1.imshow(im, cmap=plt.cm.gray)
ax1.set_title('Input image')

# Rescale histogram for better display
hog_im_rescaled = exposure.rescale_intensity(hog_im, in_range=(0, 10))

ax2.imshow(hog_im_rescaled, cmap=plt.cm.gray)
ax2.set_title('Histogram of Oriented Gradients for im')
#################################################################################################
fig1, (ax11, ax12) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax11.imshow(im1, cmap=plt.cm.gray)
ax11.set_title('Input image')

# Rescale histogram for better display
hog_im1_rescaled = exposure.rescale_intensity(hog_im1, in_range=(0, 10))

ax12.imshow(hog_im1_rescaled, cmap=plt.cm.gray)
ax12.set_title('Histogram of Oriented Gradients for im1')
#################################################################################################
fig2, (ax21, ax22) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax21.imshow(im2, cmap=plt.cm.gray)
ax21.set_title('Input image')

# Rescale histogram for better display
hog_im2_rescaled = exposure.rescale_intensity(hog_im2, in_range=(0, 10))

ax22.imshow(hog_im2_rescaled, cmap=plt.cm.gray)
ax22.set_title('Histogram of Oriented Gradients for im2')
#################################################################################################

dist_v_to_v1 =  distance.euclidean(v, v1)
dist_v_to_v2 =  distance.euclidean(v, v2)
dist_v1_to_v2 = distance.euclidean(v1, v2)
print(dist_v_to_v1) # this is the smallest distance
print(dist_v_to_v2) # close to the distance between v1 and v2
print(dist_v1_to_v2) # this is the greatest distance

#################################################################################################

im3 = adjust_gamma(im, 2)
v3, hog_im3 = hog(im3, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), block_norm = 'L1', visualize=True)


fig3, (ax31, ax32) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

ax31.imshow(im3, cmap=plt.cm.gray)
ax31.set_title('Input image')

# Rescale histogram for better display
hog_im3_rescaled = exposure.rescale_intensity(hog_im3, in_range=(0, 10))

ax32.imshow(hog_im3_rescaled, cmap=plt.cm.gray)
ax32.set_title('Histogram of Oriented Gradients for im3')
plt.show()

dist_v_to_v3 =  distance.euclidean(v, v3)
print(dist_v_to_v3) # very low distance compared to the previous ones (6 times smaller than the previous minimum between v and v1)
#################################################################################################

'''
Feature vectors have a size of 9600 because :
* Image size is 480 by 640
* We have 16 by 16 block size because there is one cell per block and each cell has 16*16 pixels.
* 480/16=30 pixels per block along width and 640/16=40 pixels per block along height.
* Hence a total of 30*40=1200 pixels per block
* For each block we compute a histogram of 8 orientations: 1200*8=9600

The smallest difference is between v and v1 if we consider the following descriptors: v, v1 and v2

The distance between v and v3 is 1.4581778988512681, which is approximately 6 times smaller than the previous minimum between v and v1
'''