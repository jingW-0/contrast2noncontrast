# Segmenting heart chambers from non-contrast CT scans using contrastive unpaired image translation

This repository contains the implementation of the first phase in "Promise and challenges of heart chamber segmentation from non-contrast CT scans using contrastive unpaired image translation: a feasibility study."

The code is adapted from [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) and [Hneg_SRC](https://github.com/jcy132/Hneg_SRC).

## Environment setup

The provided Conda environment was exported from the working Windows environment used for this project. Its core stack is:

- Python 3.9.20
- PyTorch 1.12.0
- torchvision 0.13.0
- CUDA Toolkit 11.3.1

An NVIDIA GPU and a driver compatible with CUDA 11.3 are required by the default training configuration.

### 1. Download the repository

```bash
git clone https://github.com/jingW-0/contrast2noncontrast.git
cd contrast2noncontrast
```

### 2. Create the Conda environment

Install Anaconda or Miniconda, then run this command from the repository root:

```bash
conda env create -f environment.yml
```

Activate the environment:

```bash
conda activate cutdce
```

The environment installation includes all Conda and pip packages listed in `environment.yml`. The separate `requirements.txt` records the Python packages from the working environment; Conda setup is recommended because it also installs the matching CUDA Toolkit and compiled dependencies.

### 3. Verify the installation

```bash
python -c "import torch, torchvision; print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available()); print('CUDA runtime:', torch.version.cuda)"
```

The expected core versions are PyTorch `1.12.0`, torchvision `0.13.0`, and CUDA runtime `11.3`. The default configuration uses GPU 0, so `CUDA available` should print `True`.

## Training

Start Visdom in a separate terminal with the `cutdce` environment activated:

```bash
python -m visdom.server -p 8097
```

Open [http://localhost:8097](http://localhost:8097) in a browser.

View all training options:

```bash
python train.py --help
```

Start training with folder-based input:

```bash
python train.py
```

To load image paths from Excel manifests instead of scanning `--dataroot`:

```bash
python train.py --read_folder False --data_train_A filelist_A.xlsx --data_train_B filelist_B.xlsx
```

Each workbook must contain a column named `files`.
