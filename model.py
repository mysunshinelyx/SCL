import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class SCL(nn.Module):
    def __init__(self, img_input_dim=4096, text_input_dim=5000,
                 common_emb_dim=1024, class_num=10, init_weight=True):
        super(SCL, self).__init__()
        self.img_net = nn.Sequential(
            nn.Linear(img_input_dim, common_emb_dim*2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.text_net = nn.Sequential(
            nn.Linear(text_input_dim, common_emb_dim*2),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.label_classifier = nn.Sequential(
            nn.Linear(common_emb_dim, class_num)
        )
        self.l1 = nn.Linear(common_emb_dim*2, common_emb_dim)

        self.drop_out = nn.Dropout(0.5)
        if init_weight:
            self._initialize_weights()

    def forward(self, img, text):
        view1_feature = self.img_net(img)
        view2_feature = self.text_net(text)
        view1_feature = self.drop_out(F.relu(self.l1(view1_feature)))
        view2_feature = self.drop_out(F.relu(self.l1(view2_feature)))
        # norm
        norm_view1_feature = torch.norm(view1_feature, dim=1, keepdim=True)
        norm_view2_feature = torch.norm(view2_feature, dim=1, keepdim=True)
        view1_feature = view1_feature / norm_view1_feature
        view2_feature = view2_feature / norm_view2_feature

        view1_label_predict = self.label_classifier(view1_feature)
        view2_label_predict = self.label_classifier(view2_feature)

        return \
            view1_feature, view2_feature, view1_label_predict, \
            view2_label_predict

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, 1)


