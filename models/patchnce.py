from packaging import version
import torch
from torch import nn


class PatchNCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, weight=None):
        num_patches = feat_q.shape[0] # 256
        dim = feat_q.shape[1] # 256
        feat_k = feat_k.detach()

        # print(f"feat_q {feat_q.shape}")

        # pos logit, dot product of vec at the same sample position
        l_pos = torch.bmm(
            feat_q.view(num_patches, 1, -1), feat_k.view(num_patches, -1, 1)) # (256, 1, 1)
        # print(f"l pos shape {l_pos.shape}")
        l_pos = l_pos.view(num_patches, 1) # (256, 1)
        # print(f"l pos shape2 {l_pos.shape}")

        # neg logit

        # Should the negatives from the other samples of a minibatch be utilized?
        # In CUT and FastCUT, we found that it's best to only include negatives
        # from the same image. Therefore, we set
        # --nce_includes_all_negatives_from_minibatch as False
        # However, for single-image translation, the minibatch consists of
        # crops from the "same" high-resolution image.
        # Therefore, we will include the negatives from the entire minibatch.
        if self.opt.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = self.opt.batch_size

        # print(f"batch_dim_for_bmm{batch_dim_for_bmm}")

        # reshape features to batch size
        feat_q = feat_q.view(batch_dim_for_bmm, -1, dim) # (1, 256, 256)
        feat_k = feat_k.view(batch_dim_for_bmm, -1, dim) # (1, 256, 256)

        # print(f"feat_q: {feat_q.shape}, feat_k: {feat_k.shape}")

        npatches = feat_q.size(1) # 256
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1)) # (1, 256, 256)
        # print(f"l neg : {l_neg_curbatch.shape}")
        # weighted by semantic relation
        if weight is not None:
            l_neg_curbatch *= weight

        # diagonal entries are similarity between same features, and hence meaningless.
        # just fill the diagonal with very small number, which is exp(-10) and almost zero
        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :] # (1, 256, 256)
        # print(f"diagonal : {diagonal.shape}")

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches) # (256, 256)
        # print(f"l neg : {l_neg.shape}")

        out = torch.cat((l_pos, l_neg), dim=1) / self.opt.nce_T # (256, 257)
        # print(f"out : {out.shape}")

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device)) #256
        # print(f"loss: {loss.shape}")
        return loss


class PatchDISTANCELoss(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.distance_loss = torch.nn.L1Loss()
        # self.cross_entropy_loss = torch.nn.CrossEntropyLoss(reduction='none')
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k):
        num_patches = feat_q.shape[0]
        # print(f"num patches: {num_patches}")
        N = num_patches * num_patches

        dim = feat_q.shape[1]
        feat_k = feat_k.detach()

        # q distance
        d_q = torch.bmm(feat_q.view(num_patches, -1, 1), feat_q.view(num_patches, 1, -1))

        # k distance
        d_k = torch.bmm(feat_k.view(num_patches, -1, 1), feat_k.view(num_patches, 1, -1))

        # print(f"d_q {d_q.shape}, d_k {d_k.shape}")
        loss = self.distance_loss(d_q, d_k)

        return loss