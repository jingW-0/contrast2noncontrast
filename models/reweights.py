from packaging import version
import torch
from torch import nn


class Normalize(nn.Module):

    def __init__(self, power=2):
        super(Normalize, self).__init__()
        self.power = power

    def forward(self, x):
        norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
        out = x.div(norm + 1e-7)
        return out

class ReweightSample(nn.Module):
    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool

    def forward(self, feat_q, feat_k, feat_g, epoch=None):
        '''
        :param feat_q: target
        :param feat_k: source
        :return: SRC loss, weights for hDCE
        '''
        # print(f"feat_q {feat_q.shape}")
        # 256 * 256
        batchSize = feat_q.shape[0]
        dim = feat_q.shape[1]
        feat_k = feat_k.detach()
        batch_dim_for_bmm = 1   # self.opt.batch_size
        feat_k = Normalize()(feat_k)
        feat_q = Normalize()(feat_q)
        feat_g = Normalize()(feat_g)

        ## SRC
        feat_q_v = feat_q.view(batch_dim_for_bmm, -1, dim) # (1, 256, 256)
        feat_k_v = feat_k.view(batch_dim_for_bmm, -1, dim)
        feat_g_v = feat_g.view(batch_dim_for_bmm, -1, dim)

        spatial_q = torch.bmm(feat_q_v, feat_q_v.transpose(2, 1)) # (1, 256, 256)
        spatial_k = torch.bmm(feat_k_v, feat_k_v.transpose(2, 1))
        spatial_g = torch.bmm(feat_g_v, feat_g_v.transpose(2, 1))

        spatial_kg = spatial_k / (spatial_g + 1e-7)
        #
        # print(f"min :{spatial_kg.min()}, max: {spatial_kg.max()}")

        # print(f"sk: {spatial_k}")
        # print(f"sg: {spatial_g}")
        # print(f"skg: {spatial_kg}")

        # weight_seed = spatial_k.clone().detach()
        weight_seed = spatial_kg.clone().detach()

        diagonal = torch.eye(self.opt.num_patches, device=feat_k_v.device, dtype=self.mask_dtype)[None, :, :] # (1, 256, 256)

        # print(f"spatial q: {spatial_q.shape}, diagnal: {diagonal.shape}")

        HDCE_gamma = 20
        # if self.opt.use_curriculum:
        #     HDCE_gamma = HDCE_gamma + (self.opt.HDCE_gamma_min - HDCE_gamma) * (epoch) / (self.opt.n_epochs + self.opt.n_epochs_decay)
        #     if (self.opt.step_gamma)&(epoch>self.opt.step_gamma_epoch):
        #         HDCE_gamma = 1
        #

        ## weights by semantic relation
        weight_seed.masked_fill_(diagonal, -10.0) # 1e-10 along diagonal
        # print(f"weight seed {weight_seed.shape}")
        weight_out = nn.Softmax(dim=2)(weight_seed.clone() / HDCE_gamma).detach()
        # print(f"weight out {weight_out.shape}")
        wmax_out, _ = torch.max(weight_out, dim=2, keepdim=True)
        # print(f"wmaxout {wmax_out.shape}")
        weight_out /= wmax_out


        return weight_out