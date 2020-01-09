import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.cluster import adjusted_mutual_info_score, contingency_matrix

# return {'cluster_matrix': cluster_matrix, 'centers': z, 'initial_centers': z_init, 'achieved loss': loss, 'iterations': counter}
# return {'weights': w, 'centers': c, 'number of iterations': number_of_iterations}


if __name__ == "__main__":
    # loading dataset & results:
    dataset = np.load('Dataset/dataset.npy')
    latent = np.load('Dataset/latent.npy')
    res_fcm = np.load('res_fcm.npy', allow_pickle=True)
    res_k_means = np.load('res_k_means.npy', allow_pickle=True)

    predicted_target_fcm = np.argmax(res_fcm.item().get('weights'), 1)
    predicted_target_k_means = res_k_means.item().get('cluster_matrix')
    true_target = latent[:, 3]
    # saving into a pandas data frame :
    df = pd.DataFrame(data=np.concatenate([true_target[:, np.newaxis], predicted_target_k_means[:, np.newaxis],
                                           predicted_target_fcm[:, np.newaxis]], axis=1),
                      columns=['truth', 'k means', 'fcm'])
    df.to_csv('result.csv')

    print('Contingency Matrix between True & FCM:\n',
          contingency_matrix(true_target, predicted_target_fcm))
    print('Contingency Matrix between True & KMeans:\n',
          contingency_matrix(true_target, predicted_target_k_means))
    print('Contingency Matrix between KMeans & FCM:\n',
          contingency_matrix(predicted_target_k_means, predicted_target_fcm))

    print('adjusted mutual information score of FCM & KMeans:\n',
          adjusted_mutual_info_score(predicted_target_fcm,  predicted_target_k_means))
    print('adjusted mutual information score of True & FCM:\n',
          adjusted_mutual_info_score(true_target,predicted_target_fcm))
    print('adjusted mutual information score of True & KMeans:\n',
          adjusted_mutual_info_score(predicted_target_k_means, true_target))

    kmean_centers = res_k_means.item().get('centers')
    kmean_centers = kmean_centers.reshape((-1, 150, 150, 3))
    plt.suptitle('KMeans Centers:')
    plt.subplot('131')
    plt.imshow(kmean_centers[0, :, :, :])
    plt.subplot('132')
    plt.imshow(kmean_centers[1, :, :, :])
    plt.subplot('133')
    plt.imshow(kmean_centers[2, :, :, :])
    plt.show()

    fcm_centers = res_fcm.item().get('centers')
    fcm_centers = np.transpose(fcm_centers)
    fcm_centers = fcm_centers.reshape((3, 150, 150, 3))
    plt.suptitle('FCM Centers:')
    plt.subplot('131')
    plt.imshow(fcm_centers[0, :, :, :])
    plt.subplot('132')
    plt.imshow(fcm_centers[1, :, :, :])
    plt.subplot('133')
    plt.imshow(fcm_centers[2, :, :, :])
    plt.show()
