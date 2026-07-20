# DSKRNet

This repository provides the paper-consistent implementation entry for **DSKRNet**.

## Main file

```text
nnunet_mednext/network_architecture/DSKRNet.py
```

The main model class is:

```python
from nnunet_mednext.network_architecture.DSKRNet import DSKRNet
```

## Naming update

The code keeps the same computational graph and only aligns module names and comments with the paper framework:

- SPA-related structural guidance path
- SDKR-related selective local correction path
- SCU-style conservative residual write-back path

The forward computation, layer order, tensor operations, and final segmentation logits are unchanged. Therefore, the Dice value should not change when the same weights and evaluation pipeline are used.

## Local upload steps

```bash
git init
git add .
git commit -m "Initial release of DSKRNet"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```
