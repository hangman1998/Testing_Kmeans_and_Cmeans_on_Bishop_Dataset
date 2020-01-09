import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
from sklearn import decomposition

# parameters:
# number of points in the dataset:
n = 500
# frame height and width:
height = 150
width = 150


# creating our synthetic dataset:
meaw = plt.imread('Dataset/base images/meaw.jfif')
doggy = plt.imread('Dataset/base images/doggy.jfif')
puppy = plt.imread('Dataset/base images/puppy.jfif')
plt.suptitle('base images:')
plt.subplot('131')
plt.imshow(puppy)
plt.title('puppy')

plt.subplot('132')
plt.imshow(doggy)
plt.title('doggy')

plt.subplot('133')
plt.imshow(meaw)
plt.title('meaw')
plt.show()

categories = np.concatenate([np.expand_dims(meaw, axis=0),
                             np.expand_dims(doggy, axis=0),
                             np.expand_dims(puppy, axis=0)
                             ], axis=0)
latent = np.concatenate([
    np.expand_dims(np.random.randint(low=0, high=height - 70, size=n), axis=1),
    np.expand_dims(np.random.randint(low=0, high=width - 70, size=n), axis=1),
    np.expand_dims(np.random.randint(low=0, high=360, size=n), axis=1),
    np.expand_dims(np.random.randint(low=0, high=3, size=n), axis=1)
], axis=1)

dataset = list()
for i in range(n):
    x = np.zeros(shape=(height, width, 3), dtype=np.int)
    x_pos = latent[i, 0]
    y_pos = latent[i, 1]
    theta = latent[i, 2]
    category = latent[i, 3]
    rotated = ndimage.rotate(categories[category, :, :, :], angle=theta, output=np.int, order=1)
    x[x_pos:x_pos + rotated.shape[0], y_pos:y_pos + rotated.shape[1], :] = rotated
    dataset.append(x)
dataset = np.asarray(dataset)


# showing some samples of the data set:
random_indexes = np.random.randint(0, n, 9)
plt.suptitle('9 random samples of the newly created dataset:')
for i in range(9):
    plt.subplot('33{}'.format(i))
    plt.title('x = {} y = {}  theta = {}'.format(latent[random_indexes[i], 0], latent[random_indexes[i], 1], latent[random_indexes[i], 2]))
    plt.imshow(dataset[random_indexes[i]])
    plt.gca().set_axis_off()
plt.show()


# showing latent variables in a 3D Plot:
ax = Axes3D(plt.gcf())
ax.scatter(latent[:, 0], latent[:, 1], latent[:, 2], c=latent[:, 3], cmap=plt.cm.Set1)
ax.set_title("Latent Variables")
ax.set_xlabel("x position")
ax.set_ylabel("y position")
ax.set_zlabel("rotation angle")
plt.show()


# showing 3D version of dataset:
pca_model = decomposition.PCA(3)
dataset_dim_reduced = pca_model.fit_transform(dataset.reshape(n, height * width * 3))
ax = Axes3D(plt.gcf())
ax.scatter(dataset_dim_reduced[:, 0], dataset_dim_reduced[:, 1], dataset_dim_reduced[:, 2], c=latent[:, 3], cmap=plt.cm.Set1)
ax.set_title("PCA Dimensions")
ax.set_xlabel("First axis")
ax.w_xaxis.set_ticklabels([])
ax.set_ylabel("Second axis")
ax.w_yaxis.set_ticklabels([])
ax.set_zlabel("Third axis")
ax.w_zaxis.set_ticklabels([])
plt.show()

# saving the dataset at last:
np.save('Dataset/dataset.npy', dataset)
np.save('Dataset/latent.npy', latent)

