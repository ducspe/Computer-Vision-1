import imageio
from skimage.color import rgb2gray
from skimage import feature
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from skimage.transform import hough_line, hough_line_peaks

im = imageio.imread('airport.tif')
im = rgb2gray(im)

edges = feature.canny(im, sigma=2, low_threshold=115)

h, theta, d = hough_line(edges)
print("h: ", h.shape)

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
ax = axes.ravel()

ax[0].imshow(edges, cmap=cm.gray)
ax[0].set_title('Edge image')
ax[0].set_axis_off()

ax[1].imshow(np.log(1 + h),
             extent=[np.rad2deg(theta[-1]), np.rad2deg(theta[0]), d[-1], d[0]],
             cmap=cm.gray, aspect=1/1.5)
ax[1].set_title('Hough transform')
ax[1].set_xlabel('Angles (degrees)')
ax[1].set_ylabel('Distance (pixels)')
ax[1].axis('image')

ax[2].imshow(im, cmap=cm.gray)

index = 0
for _, angle, dist in zip(*hough_line_peaks(h, theta, d)):
    y0 = (dist - 0 * np.cos(angle)) / np.sin(angle)
    y1 = (dist - im.shape[1] * np.cos(angle)) / np.sin(angle)
    ax[2].plot((0, im.shape[1]), (y0, y1), '-r')
    index+=1
    if index > 2:
        break

ax[2].set_xlim((0, im.shape[1]))
ax[2].set_ylim((im.shape[0], 0))
ax[2].set_axis_off()
ax[2].set_title('Detected lines')

plt.tight_layout()
plt.show()

