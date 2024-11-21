import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# class Attn_Net_Gated(nn.Module):
#     def __init__(self, L=1024, D=256, dropout=False, n_classes=1):
#         super(Attn_Net_Gated, self).__init__()
#         self.attention_a = [
#             nn.Linear(L, D),
#             nn.Tanh()]
#
#         self.attention_b = [nn.Linear(L, D),
#                             nn.Sigmoid()]
#         if dropout:
#             self.attention_a.append(nn.Dropout(0.25))
#             self.attention_b.append(nn.Dropout(0.25))
#
#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)
#
#         self.attention_c = nn.Linear(D, n_classes)
#
#     def forward(self, x):
#         a = self.attention_a(x)
#         b = self.attention_b(x)
#         A = a.mul(b)
#         A = self.attention_c(A)  # N x n_classes
#         return A, x

class Attn_Net_Gated(nn.Module):
    def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net_Gated, self).__init__()
        self.attention_a = [nn.Linear(L, D), nn.Tanh()]
        self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
        if dropout > 0:
            self.attention_a.append(nn.Dropout(dropout))
            self.attention_b.append(nn.Dropout(dropout))
        self.attention_a = nn.Sequential(*self.attention_a)
        self.attention_b = nn.Sequential(*self.attention_b)
        self.attention_c = nn.Linear(D, n_classes)

    def forward(self, x, only_A=False):
        a = self.attention_a(x)
        b = self.attention_b(x)
        A = a.mul(b)
        A = self.attention_c(A)  # N x n_classes
        if only_A:
            return A
        return A, x

# class CLAM_SB(nn.Module):
#     def __init__(self, gate=True, size_arg="small", dropout=False, k_sample=8, n_classes=3,
#                  instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
#         super().__init__()
#         self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
#         size = self.size_dict[size_arg]
#         fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
#         attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1) if gate else Attn_Net(
#             L=size[1], D=size[2], dropout=dropout, n_classes=1)
#         fc.append(attention_net)
#         self.attention_net = nn.Sequential(*fc)
#         self.classifiers = nn.Linear(size[1], n_classes)
#         instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
#         self.instance_classifiers = nn.ModuleList(instance_classifiers)
#         self.k_sample = k_sample
#         self.instance_loss_fn = instance_loss_fn
#         self.n_classes = n_classes
#         self.subtyping = subtyping
#
#     # instance-level evaluation for in-the-class attention branch
#     def inst_eval(self, A, h, classifier):
#         device = h.device
#         if len(A.shape) == 1:
#             A = A.view(1, -1)
#         k = min(self.k_sample, A.size(1))
#         top_p_ids = torch.topk(A, k)[1][-1]
#         top_p = torch.index_select(h, dim=0, index=top_p_ids)
#         top_n_ids = torch.topk(-A, k, dim=1)[1][-1]
#         top_n = torch.index_select(h, dim=0, index=top_n_ids)
#         p_targets = self.create_positive_targets(k, device)
#         n_targets = self.create_negative_targets(k, device)
#
#         all_targets = torch.cat([p_targets, n_targets], dim=0)
#         all_instances = torch.cat([top_p, top_n], dim=0)
#         logits = classifier(all_instances)
#         all_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
#         instance_loss = self.instance_loss_fn(logits, all_targets)
#         return instance_loss, all_preds, all_targets
#
#     # instance-level evaluation for out-of-the-class attention branch
#     def inst_eval_out(self, A, h, classifier):
#         k = min(self.k_sample, A.size(1))
#         device = h.device
#         if len(A.shape) == 1:
#             A = A.view(1, -1)
#         top_p_ids = torch.topk(A, k)[1][-1]
#         top_p = torch.index_select(h, dim=0, index=top_p_ids)
#         p_targets = self.create_negative_targets(k, device)
#         logits = classifier(top_p)
#         p_preds = torch.topk(logits, 1, dim=1)[1].squeeze(1)
#         instance_loss = self.instance_loss_fn(logits, p_targets)
#         return instance_loss, p_preds, p_targets
#
#     @staticmethod
#     def create_positive_targets(length, device):
#         return torch.full((length,), 1, device=device).long()
#
#     @staticmethod
#     def create_negative_targets(length, device):
#         return torch.full((length,), 0, device=device).long()
#
#     def forward(self, h, label=None, instance_eval=True, return_features=False, attention_only=False):
#
#
#         A, h = self.attention_net(h)  # NxK
#         A = A.view(-1, 1)  # 调整 A 形状为 [N, 1]
#         h = h.view(-1, h.size(-1))  # 调整 h 形状为 [N, F]
#
#         # 转置 A
#         A = torch.transpose(A, 0, 1)  # 使 A 的形状为 [1, N]
#
#         if attention_only:
#             return A
#
#         A_raw = A
#         A = F.softmax(A, dim=1)  # softmax over N
#
#         if instance_eval:
#             total_inst_loss = 0.0
#             all_preds = []
#             all_targets = []
#             inst_labels = F.one_hot(label, num_classes=self.n_classes).squeeze()  # binarize label
#             for i in range(len(self.instance_classifiers)):
#                 inst_label = inst_labels[i].item()
#                 classifier = self.instance_classifiers[i]
#                 if inst_label == 1:  # in-the-class:
#                     instance_loss, preds, targets = self.inst_eval(A, h, classifier)
#                     all_preds.extend(preds.cpu().numpy())
#                     all_targets.extend(targets.cpu().numpy())
#                 else:  # out-of-the-class
#                     if self.subtyping:
#                         instance_loss, preds, targets = self.inst_eval_out(A, h, classifier)
#                         all_preds.extend(preds.cpu().numpy())
#                         all_targets.extend(targets.cpu().numpy())
#                     else:
#                         continue
#                 total_inst_loss += instance_loss
#
#             if self.subtyping:
#                 total_inst_loss /= len(self.instance_classifiers)
#
#         M = torch.mm(A, h)
#         logits = self.classifiers(M)
#         Y_hat = torch.topk(logits, 1, dim=1)[1]
#         Y_prob = F.softmax(logits, dim=1)
#
#         results_dict = {}
#         if instance_eval:
#             results_dict = {'instance_loss': total_inst_loss, 'inst_labels': np.array(all_targets),
#                             'inst_preds': np.array(all_preds)}
#
#         if return_features:
#             results_dict.update({'features': M})
#
#         return logits





#origin without instance
class CLAM_SB(nn.Module):
    def __init__(self, gate=True, size_arg="small", dropout=0, k_sample=8, n_classes=3,
                 instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
        super().__init__()
        self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
        size = self.size_dict[size_arg]
        fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
        attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1) if gate else Attn_Net(
            L=size[1], D=size[2], dropout=dropout, n_classes=1)
        fc.append(attention_net)
        self.attention_net = nn.Sequential(*fc)
        self.classifiers = nn.Linear(size[1], n_classes)
        instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
        self.instance_classifiers = nn.ModuleList(instance_classifiers)
        self.k_sample = k_sample
        self.instance_loss_fn = instance_loss_fn
        self.n_classes = n_classes
        self.subtyping = subtyping


    def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
        A, h = self.attention_net(h)  # NxK

        # 调整 A 和 h 的形状
        A = A.view(-1, 1)  # 调整 A 形状为 [N, 1]
        h = h.view(-1, h.size(-1))  # 调整 h 形状为 [N, F]

        # 转置 A
        A = torch.transpose(A, 0, 1)  # 使 A 的形状为 [1, N]

        if attention_only:
            return A

        A_raw = A
        A = F.softmax(A, dim=1)  # softmax over N

        # 计算 M = A * h
        M = torch.mm(A, h)  # M 的形状应为 [1, F]

        logits = self.classifiers(M)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        Y_prob = F.softmax(logits, dim=1)

        results_dict = {}

        if return_features:
            results_dict.update({'features': M})

        return logits
#
#         # return logits, Y_prob, Y_hat, A_raw, results_dict


# with k
# class CLAM_SB(nn.Module):
#     def __init__(self, gate=True, size_arg="small", dropout=True, k_sample=8, n_classes=3,
#                  instance_loss_fn=nn.CrossEntropyLoss(), subtyping=False, embed_dim=1024):
#         super().__init__()
#         self.size_dict = {"small": [embed_dim, 512, 256], "big": [embed_dim, 512, 384]}
#         size = self.size_dict[size_arg]
#         fc = [nn.Linear(size[0], size[1]), nn.ReLU(), nn.Dropout(dropout)]
#         attention_net = Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1) if gate else Attn_Net(
#             L=size[1], D=size[2], dropout=dropout, n_classes=1)
#         fc.append(attention_net)
#         self.attention_net = nn.Sequential(*fc)
#         self.classifiers = nn.Linear(size[1], n_classes)
#         instance_classifiers = [nn.Linear(size[1], 2) for _ in range(n_classes)]
#         self.instance_classifiers = nn.ModuleList(instance_classifiers)
#         self.k_sample = k_sample
#         self.instance_loss_fn = instance_loss_fn
#         self.n_classes = n_classes
#         self.subtyping = subtyping
#
#     def forward(self, h, label=None, instance_eval=False, return_features=False, attention_only=False):
#         A, h = self.attention_net(h)  # NxK
#
#         # 调整 A 和 h 的形状
#         A = A.view(-1, 1)  # 调整 A 形状为 [N, 1]
#         h = h.view(-1, h.size(-1))  # 调整 h 形状为 [N, F]
#
#         # 使用 top-k 选择注意力最高的 k 个实例
#         k = min(self.k_sample, A.size(0))  # k_sample 表示选取的实例数量
#         A_topk, topk_indices = torch.topk(A, k, dim=0, largest=True, sorted=False)
#         h_topk = h[topk_indices.squeeze(1), :]
#
#         # 重新计算注意力权重，并进行矩阵乘法
#         A_topk = torch.transpose(A_topk, 0, 1)  # 使 A 的形状为 [1, k]
#         A_topk = F.softmax(A_topk, dim=1)  # softmax over k
#
#         # 计算 M = A * h_topk
#         M = torch.mm(A_topk, h_topk)  # M 的形状应为 [1, F]
#
#         logits = self.classifiers(M)
#         Y_hat = torch.topk(logits, 1, dim=1)[1]
#         Y_prob = F.softmax(logits, dim=1)
#
#         results_dict = {}
#
#         if return_features:
#             results_dict.update({'features': M})
#
#         return logits


def initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.zero_()

        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


class Attn_Net(nn.Module):
    def __init__(self, L=512, D=256, dropout=0.25, n_classes=1):
        super(Attn_Net, self).__init__()
        self.module = [nn.Linear(L, D), nn.Tanh()]
        if dropout > 0:
            self.module.append(nn.Dropout(dropout))
        self.module.append(nn.Linear(D, n_classes))
        self.module = nn.Sequential(*self.module)

    def forward(self, x):
        return self.module(x), x  # N x n_classes


# class Attn_Net_Gated(nn.Module):
#     def __init__(self, L=1024, D=256, dropout=0.25, n_classes=1):
#         super(Attn_Net_Gated, self).__init__()
#         self.attention_a = [nn.Linear(L, D), nn.Tanh()]
#         self.attention_b = [nn.Linear(L, D), nn.Sigmoid()]
#         if dropout > 0:
#             self.attention_a.append(nn.Dropout(dropout))
#             self.attention_b.append(nn.Dropout(dropout))
#         self.attention_a = nn.Sequential(*self.attention_a)
#         self.attention_b = nn.Sequential(*self.attention_b)
#         self.attention_c = nn.Linear(D, n_classes)
#
#     def forward(self, x, only_A=False):
#         a = self.attention_a(x)
#         b = self.attention_b(x)
#         A = a.mul(b)
#         A = self.attention_c(A)  # N x n_classes
#         if only_A:
#             return A
#         return A, x


class ABMIL(nn.Module):
    def __init__(self, n_classes=3, dropout=0):
        super(ABMIL, self).__init__()
        fc_size = [1000, 256]
        self.n_classes = n_classes
        self.path_attn_head = Attn_Net_Gated(L=fc_size[0], D=fc_size[1], dropout=dropout, n_classes=1)
        self.classifiers = nn.Linear(fc_size[0], n_classes)

    def forward(self, wsi_h):
        wsi_trans = wsi_h.squeeze(0).squeeze(0).squeeze(0)
        # Ensure wsi_trans is 2D [num_instances, feature_dim]

        # Attention Pooling
        path = self.path_attn_head(wsi_trans, only_A=True)
        ori_path = path.view(1, -1)


        path = F.softmax(ori_path, dim=1)

        M = torch.mm(path, wsi_trans)  # all instance

        attn = path.detach().cpu().numpy()

        # ---->predict (classifier head)
        logits = self.classifiers(M)

        # # Attention Pooling
        # path = self.path_attn_head(wsi_trans, only_A=True)
        # ori_path = path.view(1, -1)
        # path = F.softmax(ori_path, dim=1)
        # M = torch.mm(path, wsi_trans)  # all instance
        # attn = path.detach().cpu().numpy()
        # # ---->predict (cox head)
        # logits = self.classifiers(M)

        return logits, attn


class Attn_Net_Gated_MultiHead(nn.Module):
    def __init__(self, L=1024, D=256, num_heads=4, dropout=0.25, n_classes=1):
        super(Attn_Net_Gated_MultiHead, self).__init__()
        self.num_heads = num_heads
        self.attention_a = nn.ModuleList([nn.Linear(L, D) for _ in range(num_heads)])
        self.attention_b = nn.ModuleList([nn.Linear(L, D) for _ in range(num_heads)])
        self.attention_c = nn.ModuleList([nn.Linear(D, n_classes) for _ in range(num_heads)])
        self.temperature = nn.Parameter(torch.ones(1))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x, only_A=False):
        A = 0
        for i in range(self.num_heads):
            a = torch.tanh(self.attention_a[i](x))
            b = torch.sigmoid(self.attention_b[i](x))
            attention = a * b
            A += self.attention_c[i](attention)

        A = A / self.num_heads
        A = A / self.temperature  # Applying learnable temperature
        if only_A:
            return A
        return A, x


class ABMIL_MultiHead(nn.Module):
    def __init__(self, n_classes=3, dropout=0.25, num_heads=4):
        super(ABMIL_MultiHead, self).__init__()
        fc_size = [1000, 256]
        self.n_classes = n_classes
        self.path_attn_head = Attn_Net_Gated_MultiHead(L=fc_size[0], D=fc_size[1], num_heads=num_heads, dropout=dropout,
                                                       n_classes=1)
        self.classifiers = nn.Linear(fc_size[0], n_classes)
        self.layer_norm = nn.LayerNorm(fc_size[0])  # Adding layer normalization

    def forward(self, wsi_h):
        wsi_trans = wsi_h.squeeze(0).squeeze(0).squeeze(0)
        # Ensure wsi_trans is 2D [num_instances, feature_dim]

        # Attention Pooling
        path = self.path_attn_head(wsi_trans, only_A=True)
        ori_path = path.view(1, -1)
        path = F.softmax(ori_path, dim=1)

        M = torch.mm(path, wsi_trans)  # all instances
        M = self.layer_norm(M)  # Applying layer normalization

        attn = path.detach().cpu().numpy()

        # ---->predict (classifier head)
        logits = self.classifiers(M)

        return logits, attn


# class ABMIL(nn.Module):
#     def __init__(self, input_dim=1024, hidden_dim=512, output_dim=3):
#         super(ABMIL, self).__init__()
#
#         # Attention-based pooling
#         self.attention = nn.Sequential(
#             nn.Linear(input_dim, hidden_dim),
#             nn.Tanh(),
#             nn.Linear(hidden_dim, 1)
#         )
#
#         # Classifier
#         self.classifier = nn.Linear(input_dim, output_dim)
#
#     def forward(self, x):
#         # x shape: [batch_size, num_instances, input_dim]
#
#         # Apply attention mechanism
#         attention_scores = self.attention(x)  # [batch_size, num_instances, 1]
#         attention_weights = torch.softmax(attention_scores, dim=1)  # [batch_size, num_instances, 1]
#
#         # Aggregate instances using attention weights
#         aggregated_features = torch.sum(attention_weights * x, dim=1)  # [batch_size, input_dim]
#
#         # Classify aggregated features
#         logits = self.classifier(aggregated_features)  # [batch_size, output_dim]
#
#         return logits, attention_weights
