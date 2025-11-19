from packaging import version
import torch
from torch import nn



class PatchKDCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weight=None, epoch=None, clusters=None):
        # print(f"feat_q {feat_q.shape}")
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # print(f"clusters: {clusters.shape} {type(clusters)}") # 1, 256, 1
        # print(clusters)
        # positive logit
        # print(f"check {feat_q.view(batchSize, 1, -1).shape}")
        l_pos = torch.bmm(feat_q.view(batchSize, 1, -1), feat_k.view(batchSize, -1, 1)) #(256, 1, 256) * 256, 256, 1)
        l_pos = l_pos.view(batchSize, 1) # [256, 1]
        # print(f"l_pos: {l_pos.shape}")
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim) # (1, 256, 256)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim) # (1, 256, 256)

        # print(f"feat_q : {feat_q.shape}, feat_k : {feat_k.shape}")

        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))
        # print(f"l_neg_curbatch {l_neg_curbatch.shape}")

        # weighted by semantic relation
        if weight is not None:
            l_neg_curbatch *= weight

        if (self.opt.step_gamma) & (epoch > self.opt.step_gamma_epoch):
        # if len(clusters) > 0:
            kDCE_gamma = 1

            # print(clusters.view(batch_dim_for_bmm, -1, 1).shape)
            same_label = clusters.view(batch_dim_for_bmm, -1, 1) == clusters.view(batch_dim_for_bmm, 1, -1)
            # print(f"same_label: {same_label.shape}")
            # diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
            # l_neg_curbatch.masked_fill_(diagonal, -10.0)
            l_neg_curbatch[same_label] = -10.0
        else:
            diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]
            l_neg_curbatch.masked_fill_(diagonal, -10.0)

        l_neg = l_neg_curbatch.view(-1, npatches)

        # l_pos_km = kDCE_gamma
        # l_denom = (l_neg - l_pos)/self.opt.nce_T

        # v = -torch.log(torch.exp(l_pos/self.opt.nce_T)/torch.exp(l_denom))
        logits = (l_neg-l_pos)/self.opt.nce_T
        v = torch.logsumexp(logits, dim=1)
        loss_vec = torch.exp(v-v.detach())
        #
        # for monitoring
        out_dummy = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T
        CELoss_dummy = self.cross_entropy_loss(out_dummy, torch.zeros(out_dummy.size(0), dtype=torch.long, device=feat_q.device))

        loss = loss_vec.mean()-1+CELoss_dummy.detach()
        # loss = v.mean()
        return loss
