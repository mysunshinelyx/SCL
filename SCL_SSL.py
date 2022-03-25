import torch
import numpy as np
from torch import optim
import utils
import dataloader
from torch.autograd import Variable
import scipy.io as sio
import copy
import nni
import os
from mi import MiLoss_Aug
from model import SCL
from nt_xent import NTXentLoss
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiments')


# parts of code referred from
# https://github.com/penghu-cs/SDML/blob/master/SDML.py

class Solver(object):
    def __init__(self, config):
        wv_matrix = None
        self.datasets = config.datasets
        self.batch_size = config.batch_size
        self.output_shape = config.output_shape
        self.semi_set = config.semi_set
        self.sup_rate = config.sup_rate

        data = dataloader.get_data(self.datasets, self.batch_size, self.semi_set, self.sup_rate)
        (self.labeled_dataloader, self.unlabeled_dataloader, self.valid_data,
         self.valid_labels, self.test_data, self.test_labels, self.MAP) = data

        # number of modalities
        self.n_view = len(self.valid_data)  # n_view=2

        # num_classes
        if len(self.valid_labels[0].shape) == 1:
            self.classes = np.unique(np.concatenate(self.valid_labels).reshape(
                [-1]))
            self.classes = self.classes[self.classes >= 0]
            self.num_classes = len(self.classes)
        else:
            self.num_classes = self.valid_labels[0].shape[1]

        if self.output_shape == -1:
            self.output_shape = self.num_classes

        self.input_shape = [self.valid_data[v].shape[1] for v in
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
        self.view_id = config.view_id

        self.epochs = config.epochs
        self.sample_interval = config.sample_interval

        self.just_valid = config.just_valid
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.temperature = config.temperature

        if self.batch_size < 0:
            self.batch_size = 100 if self.num_classes < 100 else 500

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
        print("Loading model..")
        self.model.load_state_dict(
            torch.load('features/' + self.datasets + '/checkpoint.model'))
        print("Load Finished!")
        self.model.cuda()
        self.model.eval()
        # test
        test_fea = utils.predict(self.model,
                                 [self.test_data[0], self.test_data[1]])
        test_lab = self.test_labels
        test_results = utils.multi_test(test_fea, test_lab, self.MAP)
        avg_acc = (test_results[1][0, 1] + test_results[1][1, 0]) / 2.
        nni.report_final_result(avg_acc)
        print("test resutls:" + self.view_result(test_results))
        sio.savemat('features/' + self.datasets + '/test_fea.mat',
                    {'test_fea': test_fea, 'test_lab': self.test_labels})

    def calc_loss(self, imgs, txts, view1_feature, view2_feature):
        ins_loss = self.criterion(view1_feature, view2_feature)
        self.ntxent_loss = NTXentLoss(device=self.device, batch_size=view1_feature.shape[0],
                                      temperature=self.temperature,
                                      use_cosine_similarity=True)
        nt_loss = self.ntxent_loss(view1_feature, view2_feature)
        mi_loss = (self.miloss(imgs, view2_feature) +
                   self.miloss(txts, view1_feature) +
                   self.miloss(imgs, view1_feature) +
                   self.miloss(txts, view2_feature))

        nt_loss = nt_loss * self.alpha
        mi_loss = mi_loss * self.beta
        ins_loss = ins_loss * self.gamma

        emb_loss = nt_loss + mi_loss + ins_loss
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

        self.model = SCL(img_input_dim=self.input_shape[0],
                     text_input_dim=self.input_shape[1],
                     common_emb_dim=self.output_shape,
                     class_num=self.num_classes)
        self.miloss = MiLoss_Aug(self.input_shape[0], self.input_shape[1],
                             self.output_shape)

        if torch.cuda.is_available():
            self.model.cuda()
            self.miloss.cuda()

        emb_train_op = optim.Adam([{"params": self.model.img_net.parameters()},
                                   {"params": self.model.text_net.parameters()},
                                   {"params": self.model.label_classifier.parameters()}
                                   ], lr=self.lr, betas=(self.beta1, self.beta2))

        mi_train_op = optim.Adam(self.miloss.parameters(), lr=self.lr_1)

        self.criterion = lambda x, y: (((x - y) ** 2).sum(1).sqrt()).mean()
        self.ntxent_loss = NTXentLoss(device=self.device, batch_size=self.batch_size,
                                      temperature=self.temperature,
                                      use_cosine_similarity=True)

        best_valid_acc = 0
        best_test_fea = []
        best_model_wts = copy.deepcopy(self.model.state_dict())

        self.model.train()
        labeled_train_iter = iter(self.labeled_dataloader)
        unlabeled_train_iter = iter(self.unlabeled_dataloader)
        for epoch in range(self.epochs):
            self.model.train()
            try:
                train_labeled_data = labeled_train_iter.next()
            except:
                labeled_train_iter = iter(self.labeled_dataloader)
                train_labeled_data = labeled_train_iter.next()

            try:
                train_unlabeled_data = unlabeled_train_iter.next()
            except:
                unlabeled_train_iter = iter(self.unlabeled_dataloader)
                train_unlabeled_data = unlabeled_train_iter.next()

            running_loss = 0.0

            print(("Epoch %d/%d") % (epoch + 1, self.epochs))

            emb_train_op.zero_grad()
            mi_train_op.zero_grad()

            # labeled data
            train_labels = torch.as_tensor((train_labeled_data[-1])).float().cuda()
            train_labeled_imgs = self.to_var(torch.as_tensor(train_labeled_data[0]))
            train_labeled_txts = self.to_var(torch.as_tensor(train_labeled_data[1]))

            # unlabeled data
            train_unlabeled_imgs = self.to_var(torch.as_tensor(train_unlabeled_data[0]))
            train_unlabeled_txts = self.to_var(torch.as_tensor(train_unlabeled_data[1]))
            view1_feature, view2_feature, \
                view1_label_predict, view2_label_predict = self.model(
                    train_labeled_imgs, train_labeled_txts)[:4]
            # labeled loss
            self.batch_size = view1_feature.shape[0]
            # one-hot to label
            # index = torch.argmax(train_labels, 1)
            # label_single = index.float().view(self.batch_size, 1)
            # one-hot CE loss
            log_prob_1 = torch.nn.functional.log_softmax(view1_label_predict,
                                                         dim=1)
            log_prob_2 = torch.nn.functional.log_softmax(view2_label_predict,
                                                         dim=1)
            label_loss = -torch.sum(
                log_prob_1 * train_labels) / self.batch_size - torch.sum(
                log_prob_2 * train_labels) / self.batch_size
            label_loss = label_loss * self.theta
            # contrastive loss
            from nt_xent import ContrastiveLossWithClass
            ntxent_class = ContrastiveLossWithClass(device=self.device, batch_size=view1_feature.shape[0],
                                      temperature=self.temperature,
                                      use_cosine_similarity=True)
            contra_loss = ntxent_class(view1_feature, view2_feature, train_labels)
            contra_loss = self.delta * contra_loss
            # unlabeled loss
            view1_feature, view2_feature = self.model(
                train_unlabeled_imgs, train_unlabeled_txts)[:2]
            # self.batch_size = view1_feature.shape[0]
            unlabeled_loss = self.calc_loss(train_unlabeled_imgs, train_unlabeled_txts,
                                            view1_feature, view2_feature)
            emb_loss = label_loss + contra_loss + unlabeled_loss
            # emb_loss = label_loss
            emb_loss.backward()
            emb_train_op.step()
            mi_train_op.step()

            running_loss += emb_loss.item()

            print("train_loss:{:.4f}".format(running_loss))

            # test
            if (epoch + 1) % self.sample_interval == 0:
                self.model.eval()
                valid_labels = self.valid_labels
                valid_fea = utils.predict(self.model, [self.valid_data[0], self.valid_data[1]])
                valid_results = utils.multi_test(valid_fea, valid_labels, self.MAP)

                v_avg_acc = (valid_results[1][0, 1]
                             + valid_results[1][1, 0]) / 2.
                print('valid_results:', self.view_result(valid_results))
                nni.report_intermediate_result(v_avg_acc)
                if v_avg_acc > best_valid_acc:
                    best_valid_acc = v_avg_acc
                    best_model_wts = copy.deepcopy(self.model.state_dict())
                    # best_test_fea = test_fea

        # save model
        if not os.path.exists('features/' + self.datasets):
            os.makedirs('features/' + self.datasets)
        torch.save(best_model_wts, 'features/' + self.datasets +
                   '/checkpoint.model')
