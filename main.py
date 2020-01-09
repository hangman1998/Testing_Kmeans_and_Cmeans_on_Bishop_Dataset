"""
Date : 9/1/2020
Author : Yousef Javaherian
Details : in this script im going to test K-means and C-means Algorithms on a synthetic dataset.
"""
import numpy as np


def k_means(data, number_of_clusters, stopping_criteria=0,
            initialization_method='random_choose_from_data', initial_centers=None):
    assert len(data.shape) == 2
    assert stopping_criteria >= 0
    assert number_of_clusters > 1
    assert initialization_method in {'random_choose_from_data', 'random', 'custom'}
    # inferring number of data points and dimension of data points from size of data
    n = data.shape[0]
    d = data.shape[1]
    k = number_of_clusters
    # initializing the centers:
    if initialization_method == 'random_choose_from_data':
        assert initial_centers is None
        z_init = data[np.random.randint(0, n, size=k), :]
    elif initialization_method == 'random':
        assert initial_centers is None
        z_init = np.random.uniform(low=data.min(axis=0), high=data.max(axis=0), size=(k, d))
    else:
        assert initial_centers.shape == (k, d)
        z_init = initial_centers
    z = z_init.copy()
    # initializing loop controllers:
    counter = 0
    loss = np.inf
    # running:
    while True:
        diff = np.repeat(np.expand_dims(data, axis=1), repeats=k, axis=1) - np.repeat(np.expand_dims(z, axis=0),
                                                                                      repeats=n, axis=0)
        distances = np.linalg.norm(diff, axis=2)
        new_loss = np.mean(np.min(distances, axis=1), axis=0)
        for i in range(k):
            cluster_matrix = np.argmin(distances, axis=1)
            slice_data = data[cluster_matrix == i, :]
            if slice_data.shape[0] != 0:
                z[i, :] = np.mean(slice_data, axis=0)
        counter += 1
        if loss - new_loss <= stopping_criteria:
            loss = new_loss
            break
        else:
            loss = new_loss

    return {'cluster_matrix': cluster_matrix, 'centers': z, 'initial_centers': z_init,  'achieved loss': loss, 'iterations': counter}


def fcm(x: np.array, number_of_clusters: int, m: int):
    x = np.array(x)
    assert len(x.shape) == 2
    d = x.shape[0]
    n = x.shape[1]
    c = np.random.uniform(low=np.min(x, axis=1), high=np.max(x, axis=1), size=(number_of_clusters, d)).transpose()
    w = np.random.uniform(0, 1, size=(n, number_of_clusters))
    number_of_iterations = 0
    while True:
        number_of_iterations += 1
        c_new = np.matmul(x, np.power(w, m)) / (np.matmul(np.ones(shape=(1, n)), np.power(w, m)))
        if np.linalg.norm(c - c_new) < 10 ** -6:
            break
        else:
            c = c_new
        x_rep = np.repeat(np.expand_dims(np.transpose(x), 2), number_of_clusters, 2)
        c_rep = np.repeat(np.expand_dims(c, 0), n, 0)
        distances = np.power(np.linalg.norm(x_rep - c_rep, axis=1), 2 / (m - 1))
        w = 1 / distances
        w = w / np.repeat(np.expand_dims(np.sum(w, axis=1), 1), number_of_clusters, 1)
    return {'weights': w, 'centers': c, 'number of iterations': number_of_iterations}


if __name__ == "__main__":
    dataset = np.load('Dataset/dataset.npy')
    N = dataset.shape[0]
    dataset = dataset.reshape((N, -1))
    latent = np.load('Dataset/latent.npy')

    res_fcm = fcm(np.transpose(dataset), 3, 2)
    print('fcm finished !!!')
    res_k_means = k_means(dataset, 3)
    print('kmeans finished !!!')
    np.save('res_k_means', res_k_means)
    np.save('res_fcm', res_fcm)
