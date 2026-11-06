# Training Vision Transformers to Focus: Emergence of Spatial Attention Head Specialization

Official Pytorch implementation of the paper:  
**Training Vision Transformers to Focus: Emergence of Spatial Attention Head Specialization**

[![NeurIPS 2026](https://img.shields.io/badge/NeurIPS%202026-Accepted-brightgreen)](https://neurips.cc/virtual/2026/poster/155703)
[![OpenReview](https://img.shields.io/badge/OpenReview-Paper-orange)](https://openreview.net/forum?id=3aWhzsaHua)
[![Project Page](https://img.shields.io/badge/Project%20Page-blue)](https://yuji-nozawa.github.io/ReFoCo)


This repository provides:

- Pretrained models trained using our method
- Training scripts for our method
- Evaluation scripts, including attention map visualization and feature extraction

## Installation
Clone the repository and install the dependencies from `requirements.txt`:
```bash
pip install -r requirements.txt
```

## Pretrained Models
Pretrained models are available on [Hugging Face](https://huggingface.co/yuji-nozawa/ReFoCo).

| Model | Download |
|---|---|
| ViT-B/16-Reg | [model_vit_focus_base.pth](https://huggingface.co/yuji-nozawa/ReFoCo/resolve/main/model_vit_focus_base.pth?download=true) |
| ViT-L/16-Reg | [model_vit_focus_large.pth](https://huggingface.co/yuji-nozawa/ReFoCo/resolve/main/model_vit_focus_large.pth?download=true) |
| ViT-g/16-Reg | [model_vit_focus_giant.pth](https://huggingface.co/yuji-nozawa/ReFoCo/resolve/main/model_vit_focus_giant.pth?download=true) |

All pretrained models we provide are teacher models. 
The scripts below are configured to use teacher models. To use a student model, specify `--model_type student`.

## Data preparation
This repository supports training and evaluation on ImageNet-1k.
The dataset should be organized in the same way as for DINOv2. Please refer to the [DINOv2 GitHub repository](https://github.com/facebookresearch/dinov2) for details.

## Training
### Training ViT-L/16-Reg. on ImageNet-1k
The following is an example command for training a model using our proposed method:
```bash
PYTHONPATH=./vit_focus python -m torch.distributed.launch \
    --nproc_per_node=4 ./vit_focus/dinov2/train/train.py \
    --config-file ./vit_focus/dinov2/configs/train/vitl16_train_reg4.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    train.dataset_path=ImageNet:split=TRAIN:root=<PATH/TO/DATASET/ROOT>:extra=<PATH/TO/DATASET/EXTRA>
```

The training code saves the weights of the teacher and the student in the `eval` folder every 12500 iterations for evaluation. In our paper, we use the teacher model for evaluation.

To change the model size, specify a different configuration file using `--config-file`.

## Evaluation
After preparing the pretrained model, the following evaluations can be performed.

To change the model configuration, modify `--arch` and specify a different configuration file using `--config-file`.

### Attention map visualization
- **Single image**

To visualize the attention maps for the provided example image, run:
```bash
PYTHONPATH=./vit_focus python -m torch.distributed.launch \
    --nproc_per_node=1 ./vit_focus/eval/visualize_attention_image.py \
    --image_path ./examples/four_dounts.png \
    --pretrained-weights <PATH/TO/WEIGHTS> \
    --arch v2largereg16 \
    --config-file ./vit_focus/dinov2/configs/eval/vitl16_eval_reg4.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR>
```
You can also replace `./examples/four_dounts.png` with your own image.

- **Dataset average**

To visualize the average of the attention maps across ImageNet-1k, run:
```bash
PYTHONPATH=./vit_focus python -m torch.distributed.launch \
    --nproc_per_node=1 ./vit_focus/eval/visualize_attention_average.py \
    --root_path <PATH/TO/DATASET/ROOT> --extra_path <PATH/TO/DATASET/EXTRA> \
    --pretrained-weights <PATH/TO/WEIGHTS> \
    --arch v2largereg16 \
    --config-file ./vit_focus/dinov2/configs/eval/vitl16_eval_reg4.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR>
```

### Feature extraction

This script extracts features from ImageNet-1k using the pretrained model. 

```bash
PYTHONPATH=./vit_focus python -m torch.distributed.launch \
    --nproc_per_node=1 ./vit_focus/eval/extract_phs_features.py \
    --root_path <PATH/TO/DATASET/ROOT> --extra_path <PATH/TO/DATASET/EXTRA> \
    --pretrained-weights <PATH/TO/WEIGHTS> \
    --arch v2largereg16 \
    --config-file ./vit_focus/dinov2/configs/eval/vitl16_eval_reg4.yaml \
    --output-dir <PATH/TO/OUTPUT/DIR> \
    --hs_flag False \
    --is_outside False
```
- `--hs_flag`: If True, PHS is applied during feature extraction.
- `--is_outside`: If False, the inside of the box is used as the visual prompt; if True, the outside is used. 
- By default, extracted features are not saved. To save them, use the `--save_features` option.
- By default, the validation set of ImageNet-1k is used. 
For a custom dataset, set `--dataset_type Custom` and specify the dataset directory with `--root_path`. `--extra_path` is not required.

**Note:** This script currently uses a placeholder box example.

## Citation
If you find this work useful, please cite:
```bibtex
@inproceedings{nozawa2026focus,
    title     = {Training Vision Transformers to Focus: Emergence of Spatial Attention Head Specialization},
    author    = {Nozawa, Yuji and Lin, Yu-Chieh and Ng, Youyang},
    booktitle = {The Fortieth Annual Conference on Neural Information Processing Systems},
    year      = {2026}
}
```

## License
Apache-2.0 License. See LICENSE file for details.

## Support
While we welcome feedback, we provide no formal support for this project. Maintenance (including bug fixes) is not guaranteed. Please understand that we may not be able to review or accept all Pull Requests. Thank you for your understanding.