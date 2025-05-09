import numpy as np
import torch

from util import mv_dataset, read_mymat, build_ad_dataset, process_data, get_validation_set, mv_tabular_collate
from scipy.spatial.distance import cdist
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
reg_param = 1e-3

def get_samples(x, y, sn, train_index, test_index, n_sample, k, if_mean=False, ):
    """
    Retrieve the set of the k nearest samples with missing data on the training dataset.
    :param x: dataset: view_num * (dataset_num, dim,)
    :param y: label: (dataset_num,)
    :param sn: missing index matrix: (dataset_num, view_num,)
    :param train_index: (train_num,)
    :param test_index: (test_num,)
    :param n_sample: sampling frequency
    :param k: number of neighbors
    :return:
    """
    view_num = len(x)
    data_num = x[0].shape[0]

    print("calculate distance")
    dist_all_set = [cdist(x[i], x[i], 'euclidean') for i in range(view_num)]
    dismiss_view_index = [[np.array([]) for __ in range(view_num)] for _ in range(view_num)]
    for i in range(view_num - 1):
        for j in range(i + 1, view_num):
            sn_temp = sn[:, [i, j]]
            sn_temp_sum = np.sum(sn_temp, axis=1)
            sn_temp_sum[test_index] = 0  # Mask the test set sample
            dismiss_view_index[i][j] = dismiss_view_index[j][i] = np.where(sn_temp_sum == 2)[
                0]  # Sample index that exists in both views i and j

    print("Fill the missing views in the training set")
    sn_train = sn[train_index]
    x_train = [x[v][train_index] for v in range(view_num)]
    y_train = y[train_index]

    # Step 1: Process complete samples
    x_train_dismiss_index = np.where(np.sum(sn_train, axis=1) == view_num)[0]
    x_complete = [x_train[_][x_train_dismiss_index] for _ in range(view_num)]
    y_complete = y_train[x_train_dismiss_index]
    sn_complete = sn_train[x_train_dismiss_index]

    # Step 2: Process incomplete samples
    x_train_miss_index = np.where(np.sum(sn_train, axis=1) < view_num)[0]
    x_incomplete = [np.repeat(x_train[_][x_train_miss_index], n_sample, axis=0) for _ in range(view_num)]
    y_incomplete = np.repeat(y_train[x_train_miss_index], n_sample, axis=0)
    sn_incomplete = np.repeat(sn_train[x_train_miss_index], n_sample, axis=0)

    for i, train_idx in enumerate(x_train_miss_index):
        y_i = y_train[train_idx]
        miss_view_index = np.nonzero(sn_train[train_idx] == 0)[0]
        dismiss_view_index_temp = np.nonzero(sn_train[train_idx] != 0)[0]

        for v in miss_view_index:
            neighbors_index_temp = np.array([], dtype=np.int_)
            for vv in dismiss_view_index_temp:
                dismiss_view_set = dismiss_view_index[v][vv]  # Find samples with both views present
                dist_temp = np.full(data_num, np.inf)
                dist_temp[dismiss_view_set] = dist_all_set[vv][train_idx, dismiss_view_set]
                nearest_index_temp = np.argpartition(dist_temp, k)[:k]
                neighbors_index_temp = np.unique(np.concatenate((neighbors_index_temp, nearest_index_temp)))

            x_neighbors_temp = x[v][neighbors_index_temp]
            mean = np.mean(x_neighbors_temp, axis=0)
            cov = np.cov(x_neighbors_temp, rowvar=0) + np.eye(len(mean)) * reg_param
            rng = np.random.default_rng()
            L = np.linalg.cholesky(cov)
            samples_v = rng.normal(size=(n_sample, len(mean))) @ L.T + mean

            x_incomplete[v][i * n_sample:(i + 1) * n_sample] = samples_v

    x_train = [np.concatenate((x_complete[_], x_incomplete[_]), axis=0) for _ in range(view_num)]
    y_train = np.concatenate((y_complete, y_incomplete), axis=0)
    Sn_train = np.concatenate((sn_complete, sn_incomplete), axis=0)

    print("Fill the missing views in the test set")
    sn_test = sn[test_index]
    x_test_dissmiss_index = np.where(np.sum(sn_test, axis=1) == view_num)[0]
    if if_mean:
        # impute missing views with the mean of the multiple sampling points
        x_test = [x[_][test_index][x_test_dissmiss_index] for _ in range(view_num)]
        y_test = y[test_index][x_test_dissmiss_index]
    else:
        # impute missing views with all of the multiple sampling points
        x_test = [np.repeat(x[_][test_index][x_test_dissmiss_index], 10, axis=0) for _ in
                  range(view_num)]
        y_test = np.repeat(y[test_index][x_test_dissmiss_index], 10, axis=0)
    miss_count = 0;
    for i in test_index.flat:
        if if_mean:
            x_i = [np.expand_dims(x[_][i], axis=0) for _ in range(view_num)]
            y_i = np.expand_dims(y[i], axis=0)
        else:
            x_i = [np.repeat(np.expand_dims(x[_][i], axis=0), 10, axis=0) for _ in range(view_num)]
            y_i = np.repeat(np.expand_dims(y[i], axis=0), 10, axis=0)

        sn_temp = sn[i]
        x_miss_view_index = np.nonzero(sn_temp == 0)[0]
        x_dismiss_view_index = np.nonzero(sn_temp)[0]
        # incomplete samples
        if x_miss_view_index.shape[0] != 0:
            miss_count += 1
            for j in x_miss_view_index.flat:
                # obtain neighbor index of x_i on the j_th view
                neighbors_index_temp = np.array([], dtype=np.int_)
                for jj in x_dismiss_view_index.flat:
                    dismiss_view_index_temp = dismiss_view_index[j][jj]  # sample whose j_th and jj_th views exist
                    dist_temp = np.full(data_num, np.inf)
                    dist_temp[dismiss_view_index_temp] = dist_all_set[jj][i, dismiss_view_index_temp]
                    nearest_index_temp = np.argpartition(dist_temp, k)[:k]
                    neighbors_index_temp = np.unique(
                        np.concatenate((neighbors_index_temp, nearest_index_temp), ))  # concatenate all existing views

                x_neighbors_temp = x[j][neighbors_index_temp]
                mean = np.mean(x_neighbors_temp, axis=0)
                cov = np.cov(x_neighbors_temp, rowvar=0)
                rng = np.random.default_rng()
                cov = cov + np.eye(len(cov)) * reg_param
                L = np.linalg.cholesky(cov)
                x_samples_temp = rng.normal(size=(10, len(cov))) @ L.T + mean
                x_i[j] = x_samples_temp

            x_test = [np.concatenate((x_test[_], x_i[_]), axis=0) for _ in range(view_num)]
            y_test = np.concatenate((y_test, y_i), axis=0)
    x_test = process_data(x_test, view_num)
    return x_train, y_train, x_test, y_test, Sn_train


