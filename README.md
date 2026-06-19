# Segmenting heart chambers from non-contrast CT scans using contrastive unpaired image translation

This repository contains the implementation of the first phase in "Promise and challenges of heart chamber segmentation from non-contrast CT scans using contrastive unpaired image translation: a feasibility study."

The code is adapted from [contrastive-unpaired-translation](https://github.com/taesungp/contrastive-unpaired-translation) and [Hneg_SRC](https://github.com/jcy132/Hneg_SRC).

## Environment setup

The project uses Python 3.9, PyTorch 1.13.1, torchvision 0.14.1, and CUDA 11.7. An NVIDIA GPU with a driver compatible with CUDA 11.7 is required by the default training configuration.

### Conda (recommended)

From the repository root, create and activate the environment:

```bash
conda env create -f environment.yml
conda activate cutdce
```

### pip

Create a Python 3.9 virtual environment, activate it, and install the requirements:

```bash
python -m venv .venv
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Linux:

```bash
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Verify the installation

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

The default configuration uses GPU 0, so `CUDA available` should print `True`.

## Training

Start Visdom in a separate terminal:

```bash
python -m visdom.server -p 8097
```

Visdom downloads its browser assets over HTTPS the first time it starts. If it
fails with an SSL certificate or ASN.1 error, repair the certificate packages
inside the activated environment:

```bash
conda install --force-reinstall openssl ca-certificates certifi
```

On Windows PowerShell, also clear invalid certificate-file overrides before
retrying:

```powershell
Remove-Item Env:SSL_CERT_FILE -ErrorAction SilentlyContinue
Remove-Item Env:REQUESTS_CA_BUNDLE -ErrorAction SilentlyContinue
python -m visdom.server -p 8097
```

View all training options:

```bash
python train.py --help
```

Start training:

```bash
python train.py
```

To load image paths from the Excel manifests instead of scanning `--dataroot`:

```bash
python train.py --read_folder False --data_train_A filelist_A.xlsx --data_train_B filelist_B.xlsx
```

Each workbook must contain a column named `files`.
