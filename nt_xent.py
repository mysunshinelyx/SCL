import torch
import numpy as np


class NTXentLoss(torch.nn.Module):

    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(NTXentLoss, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device
        self.softmax = torch.nn.Softmax(dim=-1)
        self.mask_samples_from_same_repr = self._get_correlated_mask().type(torch.bool)
        self.similarity_function = self._get_similarity_function(use_cosine_similarity)
        self.label_similarity_function = self._get_similarity_function(use_cosine_similarity=False)
        self.criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _get_similarity_function(self, use_cosine_similarity):
        if use_cosine_similarity:
            self._cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
            return self._cosine_simililarity
        else:
            return self._dot_simililarity

    def _get_correlated_mask(self):
        diag = np.eye(2 * self.batch_size)
        l1 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=-self.batch_size)
        l2 = np.eye((2 * self.batch_size), 2 * self.batch_size, k=self.batch_size)
        mask = torch.from_numpy((diag + l1 + l2))
        mask = (1 - mask).type(torch.bool)
        return mask.to(self.device)

    @staticmethod
    def _dot_simililarity(x, y):
        v = torch.tensordot(x.unsqueeze(1), y.T.unsqueeze(0), dims=2)
        # x shape: (N, 1, C)
        # y shape: (1, C, 2N)
        # v shape: (N, 2N)
        return v

    def _cosine_simililarity(self, x, y):
        # x shape: (N, 1, C)
        # y shape: (1, 2N, C)
        # v shape: (N, 2N)
        v = self._cosine_similarity(x.unsqueeze(1), y.unsqueeze(0))
        return v

    def forward(self, zis, zjs):
        representations = torch.cat([zjs, zis], dim=0)

        similarity_matrix = self.similarity_function(representations, representations)

        # filter out the scores from the positive samples
        l_pos = torch.diag(similarity_matrix, self.batch_size)
        r_pos = torch.diag(similarity_matrix, -self.batch_size)
        positives = torch.cat([l_pos, r_pos]).view(2 * self.batch_size, 1)

        negatives = similarity_matrix[self.mask_samples_from_same_repr].view(2 * self.batch_size, -1)  # 除去正样本和自身

        logits = torch.cat((positives, negatives), dim=1)
        logits /= self.temperature
        # labels = torch.zeros(2 * self.batch_size).to(self.device).long()
        # loss = self.criterion(logits, labels)
        exp = torch.exp(logits)
        sum_exp = torch.sum(exp, dim=1)
        log_softmax = -torch.log(exp[:, 0] / (sum_exp - exp[:, 0]))
        loss = log_softmax.sum()
        return loss / (2 * self.batch_size)


class ContrastiveLossWithClass(NTXentLoss):
    def __init__(self, device, batch_size, temperature, use_cosine_similarity):
        super(ContrastiveLossWithClass, self).__init__(device, batch_size, temperature, use_cosine_similarity)

    def forward(self, zis, zjs, labels=None):
        sim11 = self.similarity_function(zis, zis)
        sim12 = self.similarity_function(zis, zjs)
        sim21 = self.similarity_function(zjs, zis)
        sim22 = self.similarity_function(zjs, zjs)

        # labels similarity
        sim_lab = self.label_similarity_function(labels, labels)
        # mask, sim_lab >= 1 set 1
        mask = torch.ge(sim_lab, 1).to(self.device)
        # positive samples
        pos_mask = mask * (torch.ones(self.batch_size, self.batch_size).cuda() - torch.eye(self.batch_size).cuda())  # remove the same instances
        pos_sim11 = pos_mask * sim11
        pos_sim12 = mask * sim12
        pos_sim21 = mask * sim21
        pos_sim22 = pos_mask * sim22
        # negative samples
        neg_sim11 = (~ mask) * sim11
        neg_sim12 = (~ mask) * sim12
        neg_sim21 = (~ mask) * sim21
        neg_sim22 = (~ mask) * sim22
        # for image, intra+inter
        img_pos_sim = torch.cat((pos_sim11, pos_sim12), dim=1) / self.temperature
        img_neg_sim = torch.cat((neg_sim11, neg_sim12), dim=1) / self.temperature
        img_loss = - torch.log(torch.sum(torch.exp(img_pos_sim), dim=1) / torch.sum(torch.exp(img_neg_sim), dim=1)).mean()

        # for text, intra+inter
        txt_pos_sim = torch.cat((pos_sim21, pos_sim22), dim=1) / self.temperature
        txt_neg_sim = torch.cat((neg_sim21, neg_sim22), dim=1) / self.temperature
        txt_loss = -torch.log(torch.sum(torch.exp(txt_pos_sim), dim=1) / torch.sum(torch.exp(txt_neg_sim), dim=1)).mean()

        # total loss
        return (img_loss + txt_loss) / 2
