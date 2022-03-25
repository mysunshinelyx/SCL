import numpy as np
import json
from scipy.io import loadmat
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

# parts of code referred from https://github.com/penghu-cs/SDML/blob/master/data_loader.py


def ind2vec(ind, N=None):
    ind = np.asarray(ind)
    if N is None:
        N = ind.max() + 1
    return np.arange(N) == np.repeat(ind, N, axis=1)


def load_deep_features(data_name, semi_set=False, sup_rate=0):
    if data_name == 'wikipedia':
        path = './datasets/wikipedia/'
        MAP = -1
        img_train = loadmat(path + "train_img.mat")['train_img']
        img_test = loadmat(path + "test_img.mat")['test_img']
        text_train = loadmat(path + "train_txt.mat")['train_txt']
        text_test = loadmat(path + "test_txt.mat")['test_txt']
        label_train = loadmat(path + "train_img_lab.mat")['train_img_lab']
        label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']

        label_train = ind2vec(label_train).astype(int)
        label_test = ind2vec(label_test).astype(int)

        size = img_train.shape[0]
        sup_size = int(size * sup_rate)
        train_data = [img_train, text_train]
        train_sup_data = [img_train[:sup_size], text_train[:sup_size]]
        valid_data = [img_test[:231], text_test[:231]]
        test_data = [img_test[231:], text_test[231:]]

        train_labels = [label_train, label_train]
        train_sup_labels = [label_train[:sup_size], label_train[:sup_size]]
        valid_labels = [label_test[:231], label_test[:231]]
        test_labels = [label_test[231:], label_test[231:]]

    elif data_name == 'nus_wide':
        path = './datasets/nus_wide/'
        MAP = -1
        img_train = loadmat(path + "train_img.mat")['train_img']
        img_test = loadmat(path + "test_img.mat")['test_img']
        text_train = loadmat(path + "train_txt.mat")['train_txt']
        text_test = loadmat(path + "test_txt.mat")['test_txt']
        label_train = loadmat(path + "train_img_lab.mat")['train_img_lab']
        label_test = loadmat(path + "test_img_lab.mat")['test_img_lab']

        label_train = ind2vec(label_train).astype(int)
        label_test = ind2vec(label_test).astype(int)

        size = img_train.shape[0]
        sup_size = int(size * sup_rate)
        train_data = [img_train, text_train]
        train_sup_data = [img_train[:sup_size], text_train[:sup_size]]

        valid_data = [img_test[:1000], text_test[:1000]]
        test_data = [img_test[1000:], text_test[1000:]]

        train_labels = [label_train, label_train]
        train_sup_labels = [label_train[:sup_size], label_train[:sup_size]]
        valid_labels = [label_test[:1000], label_test[:1000]]
        test_labels = [label_test[1000:], label_test[1000:]]

    elif data_name == 'mscoco':
        MAP = -1
        path = './datasets/mscoco/'
        train_img_feats = np.load(path+'train_img_feats.npy').astype('float32')
        test_img_feats = np.load(path+'val_img_feats.npy').astype('float32')
        train_text_feats = np.load(path+'train_txt_feats.npy').astype('float32')
        test_text_feats = np.load(path+'val_txt_feats.npy').astype('float32')

        size = train_img_feats.shape[0]
        sup_size = int(sup_rate * size)

        with open(path+"val_img_id_to_label.json", "r") as f:
            val_labels = json.load(f)

        with open(path+"train_img_id_to_label.json", "r") as f:
            train_labels = json.load(f)

        test_imgs_labels = np.array(list(val_labels.values()))
        train_imgs_labels = np.array(list(train_labels.values()))

        train_data = [train_img_feats, train_text_feats]
        train_sup_data = [train_img_feats[:sup_size], train_text_feats[:sup_size]]
        test_data = [test_img_feats[:5000], test_text_feats[:5000]]
        valid_data = [test_img_feats[5000:10000], test_text_feats[5000:10000]]

        train_labels = [train_imgs_labels, train_imgs_labels]
        train_sup_labels = [train_imgs_labels[:sup_size], train_imgs_labels[:sup_size]]
        test_labels = [test_imgs_labels[:5000], test_imgs_labels[:5000]]
        valid_labels = [test_imgs_labels[5000:10000], test_imgs_labels[5000:10000]]

    if semi_set:
        return train_data, train_sup_data, train_sup_labels, valid_data, valid_labels, test_data, test_labels, MAP
    else:
        return train_data, train_labels, valid_data, valid_labels, test_data, test_labels, MAP


class LabeledDataSet(Dataset):
    def __init__(self, images, texts, labels):
        self.images = images
        self.texts = texts
        self.labels = labels

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        label = self.labels[index]
        return img, text, label

    def __len__(self):
        count = len(self.images)
        assert len(
            self.images) == len(self.labels)
        return count


class UnLabeledDataSet(Dataset):
    def __init__(self, images, texts):
        self.images = images
        self.texts = texts

    def __getitem__(self, index):
        img = self.images[index]
        text = self.texts[index]
        return img, text

    def __len__(self):
        count = len(self.images)
        return count


def get_data(dataname, batch_size, semi_set, sup_rate):
    train_data, train_labeled_data, train_labels, valid_data, valid_labels, test_data, test_labels, MAP = load_deep_features(dataname, semi_set, sup_rate)
    labeled_dataset = LabeledDataSet(train_labeled_data[0], train_labeled_data[1], train_labels[0])
    unlabeled_dataset = UnLabeledDataSet(train_data[0], train_data[1])
    labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    unlabeled_dataloader = DataLoader(unlabeled_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    return labeled_dataloader, unlabeled_dataloader, valid_data, valid_labels, test_data, test_labels, MAP


