import numpy as np
import pickle
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

################################################   Part 1   #####################################################################################

train = unpickle("./cifar-10-batches-py/data_batch_1")


nrofimages=30

countauto = 0
autolist = []
for i in range(10000):
    label0 = train[b'labels'][i]
    if label0 == 1:
        countauto += 1
        autolist.append(train[b'data'][i])
    if countauto == nrofimages:
        break

countdeer = 0
deerlist = []
for i in range(10000):
    label0 = train[b'labels'][i]
    if label0 == 4:
        countdeer += 1
        deerlist.append(train[b'data'][i])
    if countdeer == nrofimages:
        break


countship = 0
shiplist = []
for i in range(10000):
    label0 = train[b'labels'][i]
    if label0 == 8:
        countship += 1
        shiplist.append(train[b'data'][i])
    if countship == nrofimages:
        break

all_selected_images = autolist + deerlist + shiplist

for im in all_selected_images:
    im0 = im.reshape(3,32,32).transpose([1, 2, 0])
    plt.imshow(im0)
    plt.show()


grayscale = np.zeros(shape=(nrofimages*3,1024)) # "*3" because we've appended the 3 lists: automobile, deer and ship
grayscaleclipped = np.zeros(shape=(nrofimages*3,1024))
grayscalefinal = np.zeros(shape=(nrofimages*3,1024))
for imindex in range(nrofimages*3):

    for pixel in range(1024):
        grayscale[imindex][pixel]=((float(all_selected_images[imindex][pixel]) + float(all_selected_images[imindex][pixel+1024]) + float(all_selected_images[imindex][pixel+2048]))/3) # cast to float to avoid overflow

    np.clip(grayscale, 0, 255, out=grayscaleclipped)
    grayscalefinal = grayscale.astype('uint8')

    im = grayscalefinal[imindex].reshape((32,32))
    plt.imshow(im)
    plt.show()
    plt.imshow(all_selected_images[imindex].reshape(3,32,32).transpose([1,2,0]))
    plt.show()

    plt.hist(np.histogram(grayscalefinal[imindex].ravel(), bins=20))
    plt.show()



################################################   Part 2   #####################################################################################

test = unpickle("./cifar-10-batches-py/test_batch")
test_im = test[b'data'][:10]

# for im in test_im:
#     plt.imshow(im.reshape(3,32,32).transpose([1,2,0]))
#     plt.show()

# Store the grayscale version of all n train images in a file so that we do not waste computation time on this each time we run the program:

# n = 10000
# trainimagegrayscale = np.zeros(shape=(n, 1024))
# trainimagegrayscaleclipped = np.zeros(shape=(n, 1024))
# for trainindex, trainimage in enumerate(train[b'data'][:n]):
#     print(trainindex)
#     for pixel in range(1024):
#         trainimagegrayscale[trainindex][pixel] = ((float(trainimage[pixel]) + float(trainimage[pixel+1024]) + float(trainimage[pixel+2048]))/3)
#     np.clip(trainimagegrayscale, 0, 255, out=trainimagegrayscaleclipped)
#     trainimagegrayscalefinal = trainimagegrayscaleclipped.astype('uint8')
#
#
#
# with open('trainimagegrayscale.pkl', 'wb') as f:
#     pickle.dump(trainimagegrayscalefinal, f)

# End of storage code

# Load the grayscale train images from file that I stored previously for convenience:
with open('trainimagegrayscale.pkl', 'rb') as f:
    trainimagegrayscalefinal = pickle.load(f)

# Extract test data and convert to grayscale:
grayscale_testim = np.zeros(shape=(10, 1024))

for pixel in range(1024):
    for imindex in range(10):
        grayscale_testim[imindex][pixel] = (float(test_im[imindex][pixel]) + float(test_im[imindex][pixel + 1024]) + float(test_im[imindex][pixel+2048]))/3

np.clip(grayscale_testim, 0, 255, out=grayscale_testim)
grayscale_testim = grayscale_testim.astype('uint8')
# End of test data extraction

# Compare the histograms:
hitcount = 0 # to count the number of times prediction is correct
for testindex in range(10):
    distancelist = []
    finalgrayim=grayscale_testim[testindex].reshape((32,32))
    truelabel = test[b'labels'][testindex]
    print('True label: ', truelabel)

    prevdist = np.linalg.norm( np.histogram(grayscale_testim[testindex].ravel(), bins=256)[0] - np.histogram(trainimagegrayscalefinal[0].ravel(), bins=256)[0] )
    indexofmin = train[b'labels'][0]
    for trainindex, trainimage in enumerate(trainimagegrayscalefinal):
        dist = np.linalg.norm( np.histogram(grayscale_testim[testindex].ravel(), bins=256)[0] - np.histogram(trainimagegrayscalefinal[trainindex].ravel(), bins=256)[0] )
        if dist < prevdist:
            prevdist = dist
            indexofmin = train[b'labels'][trainindex]

    print("Predicted label " + str(indexofmin))

    if truelabel == indexofmin:
        hitcount += 1

accuracy = hitcount/10

print("Accuracy: ", accuracy)

