import torch
import numpy as np
import sys
from scipy import spatial
import scipy.linalg as sli
import torch.nn.functional as F
import torch.nn as nn
from torch import optim

# referred from https://github.com/penghu-cs/SDML/blob/master/utils_PyTorch.py


def to_tensor(x):
    x = torch.tensor(x)
    if torch.cuda.is_available():
        x = x.cuda()
    return x


def to_data(x):
    if torch.cuda.is_available():
        x = x.cpu()
    return x.numpy()


def multi_test(data, data_labels, MAP=None, metric='cosine'):
    n_view = len(data)
    res = np.zeros([n_view, n_view])
    if MAP is None:
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                else:
                    from sklearn.neighbors import KNeighborsClassifier
                    neigh = KNeighborsClassifier(n_neighbors=1, metric=metric)
                    neigh.fit(data[j], data_labels[j])
                    la = neigh.predict(data[i])
                    res[i, j] = np.sum((la == data_labels[i].reshape([-1])).astype(int)) / float(la.shape[0])
    else:
        if MAP == -1:
            res = [np.zeros([n_view, n_view]), np.zeros([n_view, n_view])]
        for i in range(n_view):
            for j in range(n_view):
                if i == j:
                    continue
                else:
                    if len(data_labels[j].shape) == 1:
                        tmp = fx_calc_map_label(data[j], data_labels[j], data[i], data_labels[i], -1, metric=metric)
                    else:
                        Ks = [50, 0] if MAP == -1 else [MAP]
                        tmp = []
                        for k in Ks:
                            tmp.append(fx_calc_map_multilabel_k(data[j], data_labels[j], data[i], data_labels[i], k=k, metric=metric))
                    if type(tmp) is list:
                        for _i in range(len(tmp)):
                            res[_i][i, j] = tmp[_i]
                    else:
                        res[i, j] = tmp
    return res


def fx_calc_map_label(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = spatial.distance.cdist(test, train, metric)

    ord = dist.argsort(1)

    # numcases = dist.shape[1]
    numcases = train_labels.shape[0]
    if k == 0:
        k = numcases
    if k == -1:
        ks = [50, numcases]
    else:
        ks = [k]

    def calMAP(_k):
        _res = []
        for i in range(len(test_label)):
            order = ord[i]
            p = 0.0
            r = 0.0
            for j in range(_k):
                if(len(train_labels)>order[j]):
                    if test_label[i] == train_labels[order[j]]:
                        r += 1
                        p += (r / (j + 1))  # precision
            if r > 0:
                _res += [p / r]  # AP
            else:
                _res += [0]
        return np.mean(_res)  # MAP

    res = []
    for k in ks:
        res.append(calMAP(k))
    return res


def fx_calc_map_multilabel_k(train, train_labels, test, test_label, k=0, metric='cosine'):
    dist = spatial.distance.cdist(test, train, metric)
    ord = dist.argsort()
    numcases = dist.shape[0]
    if k == 0:
        k = numcases
    res = []
    sum = np.zeros(test_label.shape[1])
    cnt = np.zeros(test_label.shape[1])
    for i in range(numcases):
        order = ord[i].reshape(-1)[0: k]

        tmp_label = (np.dot(train_labels[order], test_label[i]) > 0)
        if tmp_label.sum() > 0:  # 如果正样本数目大于0
            prec = tmp_label.cumsum() / np.arange(1.0, 1 + tmp_label.shape[0])  # precision
            total_pos = float(tmp_label.sum())  # R
            if total_pos > 0:
                ap = np.dot(tmp_label, prec) / total_pos
                res += [ap]
    return np.mean(res)


def predict(model, data, batch_size=32, isLong=False):
    batch_count = int(np.ceil(data[0].shape[0] / float(batch_size)))
    results = []
    imgs_pre = []
    txts_pre = []
    with torch.no_grad():
        for i in range(batch_count):
            t_imgs = to_tensor(data[0][i * batch_size: (i + 1) * batch_size])
            t_imgs = t_imgs.long() if isLong else t_imgs
            t_txts = to_tensor(data[1][i * batch_size: (i + 1) * batch_size])
            t_txts = t_txts.long() if isLong else t_txts
            view1_feature, view2_feature = model(t_imgs, t_txts)[0:2]
            imgs_pre.append(to_data(view1_feature))
            txts_pre.append(to_data(view2_feature))
    return np.concatenate(imgs_pre), np.concatenate(txts_pre)


