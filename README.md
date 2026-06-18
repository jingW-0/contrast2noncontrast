# Segmenting heart chambers from non‑contrast CT scans using contrastive unpaired image translation

This repository contains implementation of the first phase in "Promise and challenges of heart chamber segmentation from non-contrast CT scans using contrastive unpaired image translation: a feasibility study".

Codes adapted from [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) and [Hneg_SRC](https://github.com/jcy132/Hneg_SRC)

To start Visdom:

```bash
python -m visdom.server -p 8097
```

To see training options:

```bash
python train.py --help
```

To start training:

```bash
python train.py
```
