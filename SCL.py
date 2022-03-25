from torch import optim
import utils
import os
import dataloader
from torch.autograd import Variable
import scipy.io as sio
import copy
import nni
from nt_xent import NTXentLoss
import numpy as np
import torch
from mi import MiLoss_Aug
from model import SCL
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiments')


# parts of code referred from
# https://github.com/penghu-cs/SDML/blob/master/SDML.py

class Solver(object):
    def __init__(self, config):
        wv_matrix = None
        self.datasets = config.datasets
        self.output_shape = config.output_shape
        data = dataloader.load_deep_features(config.datasets)
        self.datasets = config.datasets
        (self.train_data, self.train_labels, self.valid_data, self.valid_labels,
         self.test_data, self.test_labels, self.MAP) = data

        # number of modalities
        self.n_view = len(self.train_data)  # n_view=2

        for v in range(self.n_view):
            if min(self.train_labels[v].shape) == 1:  # (2173, 1)->(2173)
                self.train_labels[v] = self.train_labels[v].reshape([-1])
            if min(self.valid_labels[v].shape) == 1:
                self.valid_labels[v] = self.valid_labels[v].reshape([-1])
            if min(self.test_labels[v].shape) == 1:
                self.test_labels[v] = self.test_labels[v].reshape([-1])

        # num_classes
        if len(self.train_labels[0].shape) == 1:
            self.classes = np.unique(np.concatenate(self.train_labels).reshape(
                [-1]))
            self.classes = self.classes[self.classes >= 0]
            self.num_classes = len(self.classes)
        else:
            self.num_classes = self.train_labels[0].shape[1]

        if self.output_shape == -1:
            self.output_shape = self.num_classes

        self.input_shape = [self.train_data[v].shape[1] for v in
                            range(self.n_view)]

        self.lr = config.lr
        self.lr_1 = config.lr_1
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.batch_size = config.batch_size
        self.alpha = config.alpha
        self.beta = config.beta
        self.gamma = config.gamma
        self.delta = config.delta
        self.theta = config.theta

        self.epochs = config.epochs
        self.sample_interval = config.sample_interval

        self.just_valid = config.just_valid
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.temperature = config.temperature

        if self.batch_size < 0:
            self.batch_size = 100 if self.num_classes < 100 else 500

        self.model = SCL(img_input_dim=self.input_shape[0],
                         text_input_dim=self.input_shape[1],
                         common_emb_dim=self.output_shape,
                         class_num=self.num_classes)
        self.miloss = MiLoss_Aug(self.input_shape[0], self.input_shape[1],
                             self.output_shape)
        self.criterion = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean()
        self.ntxent_loss = NTXentLoss(device=self.device,
                                      batch_size=self.batch_size,
                                      temperature=self.temperature,
                                      use_cosine_similarity=True)

    def to_var(self, x):
        """Converts numpy to variable."""
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x)  # torch.autograd.Variable

    def to_data(self, x):
        """Converts variable to numpy."""
        try:
            if torch.cuda.is_available():
                x = x.cpu()
            return x.data.numpy()
        except Exception as e:
            return x

    def to_one_hot(self, x):
        if len(x.shape) == 1 or x.shape[1] == 1:
            one_hot = (self.classes.reshape([1, -1]) == x.reshape(
                [-1, 1])).astype('float32')
            labels = one_hot
            y = self.to_var(torch.as_tensor(labels))
        else:
            y = self.to_var(torch.as_tensor(x.astype('float32')))
        return y

    def view_result(self, _acc):
        res = ''
        if not isinstance(_acc, list):
            res += ((' - mean: %.5f' % (np.sum(_acc) / (
                    self.n_view * (self.n_view - 1)))) + ' - detail:')
            for _i in range(self.n_view):
                for _j in range(self.n_view):
                    if _i != _j:
                        res += ('%.5f' % _acc[_i, _j]) + ','
        else:
            R = [50, 'ALL']
            for _k in range(len(_acc)):
                res += (' R = ' + str(R[_k]) + ': ')
                res += ((' - mean: %.5f' % (np.sum(_acc[_k]) / (
                        self.n_view * (self.n_view - 1)))) + ' - detail:')
                for _i in range(self.n_view):
                    for _j in range(self.n_view):
                        if _i != _j:
                            res += ('%.5f' % _acc[_k][_i, _j]) + ','
        return res

    def train(self):
        if not self.just_valid:
            self.train_self_supervised()

        self.model.load_state_dict(torch.load('features/' + self.datasets +
               '/checkpoint.model'))
        self.model.cuda()
        self.model.eval()
        # test
        test_fea = utils.predict(self.model, [self.test_data[0], self.test_data[1]])
        test_lab = self.test_labels
        test_results = utils.multi_test(test_fea, test_lab, self.MAP)
        avg_acc = (test_results[1][0, 1] + test_results[1][1, 0]) / 2.
        nni.report_final_result(avg_acc)
        print("test resutls:" + self.view_result(
            test_results))
        sio.savemat('features/' + self.datasets + '/test_fea.mat',
                    {'test_fea': test_fea,
                     'test_lab': self.test_labels})

    def calc_loss(self, imgs, txts, view1_feature, view2_feature):
        ins_loss = self.criterion(view1_feature, view2_feature)
        nt_loss = self.ntxent_loss(view1_feature, view2_feature)
        mi_loss = (self.miloss(imgs, view2_feature) + self.miloss(txts, view1_feature)) \
                + (self.miloss(imgs, view1_feature) + self.miloss(txts, view2_feature))

        nt_loss = nt_loss * self.alpha
        mi_loss = mi_loss * self.beta
        ins_loss = ins_loss * self.gamma
        emb_loss = nt_loss + ins_loss + mi_loss

        return emb_loss

    def train_self_supervised(self):
        seed = 0
        import numpy as np
        np.random.seed(seed)
        import random as rn
        rn.seed(seed)
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if torch.cuda.is_available():
            self.model.cuda()
            self.miloss.cuda()

        emb_train_op = optim.Adam([{"params": self.model.img_net.parameters()},
                                   {"params": self.model.text_net.parameters()}
                                   ], lr=self.lr, betas=(self.beta1, self.beta2))

        mi_train_op = optim.Adam(self.miloss.parameters(), lr=self.lr_1)

        best_valid_acc = 0
        best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.train()
        self.miloss.train()

        for epoch in range(self.epochs):
            self.model.train()
            self.miloss.train()
            rand_idx = np.arange(self.train_data[0].shape[0])
            np.random.shuffle(rand_idx)
            batch_count = int(self.train_data[0].shape[0] / float(
                self.batch_size))

            running_loss = 0.0

            print(("Epoch %d/%d") % (epoch + 1, self.epochs))

            for batch_idx in range(batch_count):

                emb_train_op.zero_grad()
                mi_train_op.zero_grad()

                idx = rand_idx[batch_idx *
                               self.batch_size:(batch_idx + 1) * self.batch_size]
                # train_labels = self.to_one_hot(self.train_labels[0][idx])
                train_imgs = self.to_var(torch.as_tensor(self.train_data[0][idx]))
                train_txts = self.to_var(torch.as_tensor(self.train_data[1][idx]))

                view1_feature, view2_feature, \
                    view1_label_predict, view2_label_predict = self.model(
                        train_imgs, train_txts)[:4]
                emb_loss = self.calc_loss(
                    train_imgs, train_txts, view1_feature, view2_feature)

                emb_loss.backward()
                emb_train_op.step()
                mi_train_op.step()

                running_loss += emb_loss.item()

            epoch_loss = running_loss / batch_count
            print("train_loss:{:.4f}".format(epoch_loss))

            # writer tensorboard
            writer.add_scalar('train_loss', epoch_loss, epoch * batch_count)
            writer.add_scalar('lr', emb_train_op.state_dict()['param_groups'][0]['lr'],
                              epoch * batch_count)

            # valid
            if (epoch + 1) % self.sample_interval == 0:
                self.model.eval()
                self.miloss.eval()

                valid_fea = utils.predict(self.model, [self.valid_data[0], self.valid_data[1]])
                valid_labels = self.valid_labels
                valid_results = utils.multi_test(valid_fea, valid_labels, self.MAP)

                v_avg_acc = (valid_results[1][0, 1] +
                             valid_results[1][1, 0]) / 2.

                print('valid_results:', self.view_result(valid_results))
                writer.add_scalar('t_avg_acc', v_avg_acc, epoch * batch_count)
                nni.report_intermediate_result(v_avg_acc)
                if v_avg_acc > best_valid_acc:
                    best_valid_acc = v_avg_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())

        # save model
        if not os.path.exists('features/' + self.datasets):
            os.makedirs('features/' + self.datasets)
        torch.save(best_model_wts, 'features/' + self.datasets +
                   '/checkpoint.model')

