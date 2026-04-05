# Equi-ViT: Rotational Equivariant Vision Transformer for Robust Histopathology Analysis
 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE) [![arXiv:2601.09130](https://img.shields.io/badge/arXiv-2601.09130-B31B1B.svg)](https://arxiv.org/abs/2601.09130)
 
 *Fuyao Chen, Yuexi Du, Elèonore V. Lieffrig, Nicha C. Dvornek, John A. Onofrey*
 
 *Yale University*

## News

- **Apr 2026**: Paper accepted by IEEE ISBI 2026!

## Abstract

> Vision Transformers (ViTs) have gained rapid adoption in computational pathology for their ability to model long-range dependencies through self-attention, addressing the limitations of convolutional neural networks that excel at local pattern capture but struggle with global contextual reasoning. Recent pathology-specific foundation models have further advanced performance by leveraging large-scale pretraining. However, standard ViTs remain inherently non-equivariant to transformations such as rotations and reflections, which are ubiquitous variations in histopathology imaging. To address this limitation, we propose Equi-ViT, which integrates an equivariant convolution kernel into the patch embedding stage of a ViT architecture, imparting built-in rotational equivariance to learned representations. Equi-ViT achieves superior rotation-consistent patch embeddings and stable classification performance across image orientations. Our results on a public colorectal cancer dataset demonstrate that incorporating equivariant patch embedding enhances data efficiency and robustness, suggesting that equivariant transformers could potentially serve as more generalizable backbones for the application of ViT in histopathology, such as digital pathology foundation models.

## Installation

1. Install base dependencies and Transformers from source:
```bash
pip install torch torchvision tensorboard numpy scikit-learn Pillow tqdm
pip install git+https://github.com/huggingface/transformers.git
```

2. Install GMR-Conv kernel following the [GMR-Conv GitHub page](https://github.com/XYPB/GMR-Conv.git).

3. Patch transformers with Equi-ViT:
```bash
python -c "import transformers, pathlib; p=pathlib.Path(transformers.__file__).resolve().parent/'models'/'vit'; print(p)"
cp modeling_vit_gmr.py "$(python -c \"import transformers, pathlib; print(pathlib.Path(transformers.__file__).resolve().parent/'models'/'vit')\")"/modeling_vit_gmr.py
```

4. Verify setup:
```bash
python -c "from transformers.models.vit.modeling_vit_gmr import ViTForImageClassificationGMR; print('✓ Ready!')"
```

## Datasets

Download the NCT-CRC-HE-100K dataset from [Here](https://zenodo.org/records/1214456).

## Usage (Training and Evaluation)

```bash
python main.py \
	--epochs 10 --batch_size 64 --vit_ckpt google/vit-base-patch16-224 \
	--random_init --out runs/out_dir \
	--patch_size 16 --image_size 224 \
    --kernel_size1 6 --kernel_size2 11 \
	--fix_rotate
```

## Reference

```bibtex
@ARTICLE{Chen2026-eq,
	title         = "Equi-ViT: Rotational Equivariant Vision Transformer for Robust Histopathology Analysis",
	author        = "Chen, Fuyao and Du, Yuexi and Lieffrig, El{\'e}onore V. and Dvornek, Nicha C and Onofrey, John A",
	month         =  jan,
	year          =  2026,
	archivePrefix = "arXiv",
	primaryClass  = "eess.IV",
	eprint        = "2601.09130"
}
```

## Acknowledgement

This work builds on the open-source ecosystems of [Hugging Face Transformers](https://github.com/huggingface/transformers) and [GMR-Conv](https://github.com/XYPB/GMR-Conv). We thank the maintainers and contributors for making these tools available to the community.
