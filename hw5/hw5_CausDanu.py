from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import match_template

template = io.imread("coco264316clock.jpg", as_gray=True)
hor_template = np.rot90(template)
im = io.imread("coco264316.jpg", as_gray=True)
hor_im = np.rot90(im)
result1 = match_template(im, template)
ij1 = np.unravel_index(np.argmax(result1), result1.shape)
x1, y1 = ij1[::-1]
result2 = match_template(im, hor_template)
result3 = match_template(hor_im, template)
result4 = match_template(hor_im, hor_template)
ij4 = np.unravel_index(np.argmax(result4), result4.shape)
x4, y4 = ij4[::-1]

fig = plt.figure("Template/Image Not Flipped", figsize=(10,10))
plt.subplot(131)
plt.imshow(template, cmap="gray")
ax2 = plt.subplot(132)
plt.imshow(im, cmap="gray")
plt.subplot(133)
htemp, wtemp = template.shape
rect = plt.Rectangle((x1, y1), wtemp, htemp, edgecolor='r', facecolor='none')
ax2.add_patch(rect)
plt.imshow(result1, cmap="gray")

fig = plt.figure("Template Flipped Horizontally", figsize=(10,10))
plt.subplot(131)
plt.imshow(hor_template, cmap="gray")
plt.subplot(132)
plt.imshow(im, cmap="gray")
plt.subplot(133)
plt.imshow(result2, cmap="gray")

fig = plt.figure("Image Flipped Horizontally", figsize=(10,10))
plt.subplot(131)
plt.imshow(template, cmap="gray")
plt.subplot(132)
plt.imshow(hor_im, cmap="gray")
plt.subplot(133)
plt.imshow(result3, cmap="gray")

fig = plt.figure("Image and Template Flipped", figsize=(10,10))
plt.subplot(131)
plt.imshow(hor_template, cmap="gray")
ax2 = plt.subplot(132)
plt.imshow(hor_im, cmap="gray")
plt.subplot(133)
htemp, wtemp = template.shape
rect = plt.Rectangle((x4, y4), wtemp, htemp, edgecolor='r', facecolor='none')
ax2.add_patch(rect)
plt.imshow(result4, cmap="gray")

plt.show()


