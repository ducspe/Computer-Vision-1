import imageio
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
import numpy as np

im = imageio.imread("MaruTaro.jpg")
im = rgb2gray(im)

M = im.shape[0]
N = im.shape[1]
P = 2*M
Q = 2*N

im_padded = np.pad(im, ((0, P-M), (0, Q-N)), mode='constant', constant_values=(0,0)) # Band-limit the signal by padding
im_fft = np.fft.fftshift(np.fft.fft2(im_padded))
mag = np.abs(im_fft)

################# Ideal Lowpass ###########################################################
ideal_lowpass_filter = np.zeros(shape=(P, Q))
offset = 150
radius = offset

for i in range(P//2-offset, P//2+offset):
    for j in range(Q//2-offset, Q//2+offset):
        if (i-P//2)**2 + (j-Q//2)**2 < radius**2: # without this if we will have a square shape and not a circle...
            ideal_lowpass_filter[i, j]=1.0


F_ideal = im_fft * ideal_lowpass_filter
ideal_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_ideal)))


################# Gaussian Lowpass ###########################################################
D_zero_squared=150.0**2
glpf = np.zeros(shape=(P,Q))
offset2 = 500
radius2 = offset2
for i in range(P//2-offset2, P//2+offset2):
    for j in range(Q//2-offset2, Q//2+offset2):
        D_squared = (i-P//2)**2 + (j-Q//2)**2
        if D_squared < radius2**2:
            glpf[i,j] = np.exp(-D_squared/(2*D_zero_squared))

F_gaussian = im_fft * glpf
gaussian_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_gaussian)))

##################################################################################
# Ideal Highpass
high_ideal = np.ones(shape=ideal_lowpass_filter.shape, dtype=np.float64) - ideal_lowpass_filter
F_high_ideal = im_fft * high_ideal
ideal_high_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_high_ideal)))

##################################################################################
# Gaussian Highpass
high_gaussian = np.ones(shape=ideal_lowpass_filter.shape, dtype=np.float64) - glpf
F_high_gaussian = im_fft * high_gaussian
gaussian_high_filtered = np.real(np.fft.ifft2(np.fft.ifftshift(F_high_gaussian)))

##################################################################################
# Plot everything:
fig1 = plt.figure("Original Image In Grayscale")
plt.imshow(im, cmap="gray")
#plt.show()

fig2 = plt.figure("Ideal Lowpass Filter")
plt.imshow(ideal_lowpass_filter, cmap="gray")
#plt.show()

fig3 = plt.figure("FFT image magnitude spectrum")
plt.imshow(np.log(mag), cmap="gray")
#plt.show()

fig4 = plt.figure("Ideal Lowpass Filtered Image")
plt.imshow(ideal_filtered, cmap="gray")
#plt.show()

fig5 = plt.figure("Gaussian Lowpass Filter")
plt.imshow(glpf, cmap="gray")
#plt.show()

fig6 = plt.figure("Gaussian Lowpass Filtered Image")
plt.imshow(gaussian_filtered, cmap="gray")
#plt.show()

fig7 = plt.figure("Ideal Highpass Filter")
plt.imshow(high_ideal,cmap="gray")
#plt.show()

fig8 = plt.figure("Ideal Highpass Filtered Image")
plt.imshow(ideal_high_filtered,cmap="gray")
#plt.show()

fig9 = plt.figure("Gaussian Highpass Filter")
plt.imshow(high_gaussian,cmap="gray")
#plt.show()

fig10 = plt.figure("Gaussian Highpass Filtered Image")
plt.imshow(gaussian_high_filtered,cmap="gray")
plt.show()