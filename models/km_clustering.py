from packaging import version
import torch
from torch import nn
from .kmeans import KMeans, KMeans_cos


# import faiss
# from sklearn.cluster import KMeans
# from cuml.cluster import KMeans



class KMCluster(nn.Module):
    def __init__(self, opt, mode="eu"):
        super().__init__()
        self.opt = opt
        self.mask_dtype = torch.uint8 if version.parse(torch.__version__) < version.parse('1.2.0') else torch.bool
        self.mode = mode

    def forward(self, feat_g):
        batchSize = feat_g.shape[0]
        dim = feat_g.shape[1]
        print(f"km cluster, feat_g {feat_g.shape} {self.mode}")

        # print(f"feat_g: {feat_g.shape}")

        # if self.opt.nce_includes_all_negatives_from_minibatch:
        #     # reshape features as if they are all negatives of minibatch of size 1.
        #     batch_dim_for_bmm = 1
        # else:
        batch_dim_for_bmm = self.opt.batch_size

        # reshape features to batch size
        feat_g = feat_g.view(batch_dim_for_bmm, -1, dim)
        # print(f"feat g: {feat_g.shape}")
        npatches = feat_g.size(1)

        # feat_gc = feat_g.view(-1, dim).detach().cpu()
        feat_gc = feat_g.view(-1, dim).detach()
        # print(f"feat gc {feat_gc.shape}")
        # k = self.opt.num_cluster
        # kmeans = KMeans(n_clusters=k, random_state=0).fit(feat_gc)
        if self.mode == "eu":
            modelkm = KMeans(n_clusters=self.opt.num_cluster)
        elif self.mode == "cos":
            modelkm = KMeans_cos(n_clusters=self.opt.num_cluster)

        modelkm.fit(feat_gc)
        preds = modelkm.predict(feat_gc)
        # print(f"preds: {preds.shape}, {preds}")

        # print(f"kmeans {kmeans.labels_.shape}")
        # labels = kmeans.labels_.reshape(batch_dim_for_bmm, -1, 1)
        # labels = modelkm.labels_
        # print(f"labels {labels.shape}")
        # print(labels)

        # return torch.from_numpy(labels)
        return preds.detach()
