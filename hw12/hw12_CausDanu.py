import numpy as np
from skimage import io
from skimage.segmentation import slic
import matplotlib.pyplot as plt

img = io.imread("0001_rgb.png")
ground = io.imread("0001_label.png")

print(img.shape)
print(ground)

ground_set = set(ground.flatten())
number_of_segments = len(ground_set)

print(number_of_segments)  # To see how many true (/ground) labels are there in total

segments = slic(img, n_segments=number_of_segments, compactness=50)  # By visualizing the map I felt compactness = 50 is a good number
print(segments.shape)
print(segments)


fig = plt.figure("Segmentation Fig")

ax1 = fig.add_subplot(211)
ax1.imshow(img)
ax2 = fig.add_subplot(212)
ax2.imshow(segments)
plt.show()


intersection_counter = 0
ground_truth_counter = 0
underseg_error = {}  # Undersegmentation error dictionary where values are the errors and keys are the ground truth segments
for ground_segment in ground_set:
    for row in range(len(ground)):
        for col in range(len(ground[row])):
            if segments[row][col] == ground[row][col]:  # increment counter when pixel labels overlap/intersect
                intersection_counter += 1
            if ground_segment == ground[row][col]:  # count the total pixels of a particular ground segment
                ground_truth_counter += 1

    underseg_error[ground_segment] = (intersection_counter-ground_truth_counter)/ground_truth_counter  # formula from previous homework

print(underseg_error)

# The undersegmentation error becomes more negative if we increase the number of superpixels because we will have
# more unique labels than in the actual ground truth set and therefore these new labels are actually not correct
# and hence the error becomes larger (in absolute terms).
