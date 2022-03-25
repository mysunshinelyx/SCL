import torch
import torch.nn as nn
import torch.nn.functional as F


class SpN(nn.Module):
    def __init__(self, input_dim, mid_dim):
        super().__init__()
        self.l0 = nn.Linear(input_dim, mid_dim)

    def forward(self, x):
        return F.relu(self.l0(x))


class Discriminator(nn.Module):
    def __init__(self, mid_dim, out_dim, feature_dim):  # 1024, 512, 512
        super().__init__()
        self.l1 = nn.Linear(mid_dim, out_dim)

        self.l2 = nn.Linear(out_dim+feature_dim, out_dim)  # 512+512, 512
        self.l2 = nn.Linear(512+512, out_dim)
        self.l3 = nn.Linear(out_dim, out_dim)  # 512, 512
        self.l4 = nn.Linear(out_dim, 1)  # 512, 1

    def forward(self, x, y):
        h = F.relu(self.l1(x))
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l2(h))
        h = F.relu(self.l3(h))
        return self.l4(h)


class MiLoss(nn.Module):
    def __init__(self, img_input_dim, txt_input_dim, feature_dim):
        super().__init__()
        self.img_input_dim = img_input_dim
        self.txt_input_dim = txt_input_dim
        self.imgNN = SpN(input_dim=img_input_dim, mid_dim=1024)
        self.txtNN = SpN(input_dim=txt_input_dim, mid_dim=1024)
        self.d = Discriminator(1024, 512, feature_dim)

    def forward(self, x, y):
        x_prime = torch.cat((x[1:], x[0].unsqueeze(0)), dim=0)
        if x.shape[1] == self.img_input_dim:
            x = self.imgNN(x)
            x_prime = self.imgNN(x_prime)
        else:
            x = self.txtNN(x)
            x_prime = self.txtNN(x_prime)
        Ej = -F.softplus(-self.d(x, y)).mean()
        Em = F.softplus(self.d(x_prime, y)).mean()
        jsd_loss = (Em - Ej)

        return jsd_loss


class MiLoss_Aug(MiLoss):
    def __init__(self, img_input_dim, txt_input_dim, feature_dim):
        super(MiLoss_Aug, self).__init__(img_input_dim, txt_input_dim, feature_dim)

    def forward(self, x, y):
        if x.shape[1] == self.img_input_dim:
            x = self.imgNN(x)
        else:
            x = self.txtNN(x)

        # negative samples
        neg_list = []
        neg = x
        for i in range(x.shape[0] - 1):
            neg = torch.cat((neg[1:], neg[0].unsqueeze(0)), dim=0)
            neg_list.append(neg)
        x_neg = torch.cat(neg_list, dim=0)
        y_neg = y.repeat(x.shape[0] - 1, 1)

        # compute nce loss
        batch_size = x.shape[0]
        positive_samples = self.d(x, y)
        negative_samples = self.d(x_neg, y_neg).reshape(batch_size-1, batch_size).transpose(0, 1)
        preds = torch.cat([positive_samples, negative_samples], dim=1)
        labels = torch.zeros(batch_size).long().cuda()
        criterion = torch.nn.CrossEntropyLoss()
        nce_loss = criterion(preds, labels)
        return nce_loss / batch_size


