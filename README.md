# AudioMosaic: TODO_SUBTITLE_HERE

<div align="center">
  <!-- <a href="https://arxiv.org/abs/TODO" target="_blank"><img src="https://img.shields.io/badge/arXiv-b5212f.svg?logo=arxiv" alt="arXiv"></a> -->
  <a href="https://huggingface.co/collections/hanxunh/audiomosaic" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20HuggingFace-AudioMosaic-blue.svg" alt="HuggingFace"></a>
  <a href="https://github.com/HanxunH/AudioMosaic/blob/main/LICENSE" target="_blank"><img alt="License" src="https://img.shields.io/badge/LICENSE-MIT-green"></a>
  <a><img alt="Made with Python" src="https://img.shields.io/badge/made_with-Python-blue"></a>
</div>

Code for ICML2026 paper [AudioMosaic: Contrastive Masked Audio Representation Learning](https://openreview.net/forum?id=OXJ7KqVOoT).

---

## AudioMosaic

AudioMosaic is a self-supervised audio foundation model that combines masked learning with contrastive training over mel-spectrogram patches. The model is pretrained on AudioSet-2M and transfers to a wide range of downstream audio tasks including AudioSet-20K/2M tagging, ESC-50 classification, Speech Commands, EnvSDD spoof detection, and audio-language reasoning via LTU-style instruction tuning.

---
## Installation

```shell
git clone https://github.com/HanxunH/AudioMosaic.git
cd AudioMosaic
pip3 install -r requirements.txt
```

The codebase requires PyTorch (>= 2.0), `mlconfig`, `timm`, `audiomentations`, `torchaudio`, and the standard scientific Python stack. For the LTU experiments, additional dependencies include the bundled `peft_custom/` and `modeling_ltu/transformers/`.

---
## Data preparation

All paths in the configs and dataset JSON files use `/PATH/TO/` as a placeholder. Replace this prefix with the location of your local datasets, for example:

```shell
sed -i 's|/PATH/TO/|/your/data/root/|g' configs/AudioMosaic/*.yaml dataset/**/*.json
```

Required datasets:
- **AudioSet** (balanced/unbalanced/eval) — JSONs in `dataset/audioset_util/`
- **ESC-50** — 5 splits, configs in `configs/AudioMosaic/finetune_esc_split{1..5}.yaml`
- **Speech Commands v1/v2** — JSONs in `dataset/spc_util/`
- **EnvSDD** — JSONs in `dataset/envsdd_util/`
- (Optional, LTU) **OpenAQA / Clotho / AudioCaps / VGGSound / FSD50K / TUT / DCASE** — see `main_ltu_inference.py`

---
## Pretraining

Single GPU:

```shell
python3 main_pretrain.py --exp_path experiments/AudioMosaic       \
                         --exp_config configs/AudioMosaic         \
                         --exp_name pretrain                      \
                         --seed 42
```

Distributed (SLURM):

```shell
srun python3 main_pretrain.py --ddp --dist_eval                   \
                              --exp_path experiments/AudioMosaic  \
                              --exp_config configs/AudioMosaic    \
                              --exp_name pretrain                 \
                              --seed 42
```

---
## Fine-tuning

After pretraining, the encoder checkpoint is at `experiments/AudioMosaic/pretrain/checkpoints/model_state_dict.pt`. Each fine-tuning task has its own config under `configs/AudioMosaic/`.

```shell
# AudioSet-20K
python3 main_finetune.py --exp_path experiments/AudioMosaic    \
                         --exp_config configs/AudioMosaic      \
                         --exp_name finetune_as20k             \
                         --seed 7

# ESC-50 (5 splits)
for split in 1 2 3 4 5; do
    python3 main_finetune.py --exp_path experiments/AudioMosaic    \
                             --exp_config configs/AudioMosaic      \
                             --exp_name finetune_esc_split${split} \
                             --seed 7
done

# Speech Commands v1 / v2
python3 main_finetune.py --exp_config configs/AudioMosaic --exp_path experiments/AudioMosaic --exp_name finetune_spc1 --seed 7
python3 main_finetune.py --exp_config configs/AudioMosaic --exp_path experiments/AudioMosaic --exp_name finetune_spc2 --seed 7

# EnvSDD spoof detection (ATA / TTA / AASIST variants)
python3 main_finetune.py --exp_config configs/AudioMosaic --exp_path experiments/AudioMosaic --exp_name finetune_envsdd      --seed 7
python3 main_finetune.py --exp_config configs/AudioMosaic --exp_path experiments/AudioMosaic --exp_name finetune_envsdd_ata  --seed 7
python3 main_finetune.py --exp_config configs/AudioMosaic --exp_path experiments/AudioMosaic --exp_name finetune_envsdd_tta  --seed 7
```

---
## Linear probing

Per-layer probes (layer 0 to 11), weighted sum, and attentive probe configs are provided under `configs/AudioMosaic/`.

```shell
# A single layer probe
python3 main_finetune.py --exp_config configs/AudioMosaic         \
                         --exp_path experiments/AudioMosaic       \
                         --exp_name linear_prob_as20k_layer11     \
                         --linear_probe --seed 7

# Weighted-sum / attentive probe
python3 main_finetune.py --exp_config configs/AudioMosaic --exp_path experiments/AudioMosaic --exp_name linear_prob_as20k_weighted_sum --linear_probe --seed 7
python3 main_finetune.py --exp_config configs/AudioMosaic --exp_path experiments/AudioMosaic --exp_name linear_prob_as20k_attentive    --linear_probe --seed 7
```

The `--linear_probe` flag is auto-detected from any `exp_name` containing `linear_prob`, but can also be passed explicitly.

---
## LTU instruction tuning

Place the LTU base model under `checkpoints/ltu_pretrained_mdls/`. The four-stage LTU recipe is:

```shell
for stage in stage1_proj_cla stage2_all_cla stage3_all_closed stage4_all_mix; do
    python3 main_ltu.py --exp_config configs/AudioMosaic     \
                        --exp_path experiments/AudioMosaic   \
                        --exp_name ${stage} --seed 42
done
```

Inference / evaluation:

```shell
python3 main_ltu_inference.py --base_model checkpoints/ltu_pretrained_mdls \
                              --model_path checkpoints/ltu_ori_paper.bin   \
                              --model_type AudioMosaic-LTU                 \
                              --eval_dataset clotho
```

---
## Pretrained checkpoints

All checkpoints are released on HuggingFace under the [**AudioMosaic collection**](https://huggingface.co/collections/hanxunh/audiomosaic). Each repo is **self-contained** — the model architecture is vendored alongside the weights, so no AudioMosaic install is needed to load it.

### Pretrained encoder
| Model | Description |
|---|---|
| [AudioMosaic-vit-b16-pretrained](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-pretrained) | ViT-B/16 encoder pretrained on AudioSet-2M with NT-Xent + masking |

### Audio tagging (AudioSet)
| Model | Metric |
|---|---|
| [AudioMosaic-vit-b16-finetune-as20k](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-as20k) | AS-20K mAP **42.53** |
| [AudioMosaic-vit-b16-finetune-as2m](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-as2m) | AS-2M mAP **50.19** |
| [AudioMosaic-vit-b16-linear-prob-as20k](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-linear-prob-as20k) | AS-20K mAP **29.40** (linear probe) |
| [AudioMosaic-vit-b16-linear-prob-as20k-attentive](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-linear-prob-as20k-attentive) | AS-20K mAP **33.55** (attentive probe) |

### ESC-50 (5-fold)
| Model | Acc |
|---|---|
| [AudioMosaic-vit-b16-finetune-esc-split1](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-esc-split1) | 97.25 |
| [AudioMosaic-vit-b16-finetune-esc-split2](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-esc-split2) | 98.75 |
| [AudioMosaic-vit-b16-finetune-esc-split3](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-esc-split3) | 97.00 |
| [AudioMosaic-vit-b16-finetune-esc-split4](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-esc-split4) | 98.50 |
| [AudioMosaic-vit-b16-finetune-esc-split5](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-esc-split5) | 95.75 |

### Speech Commands
| Model | Acc |
|---|---|
| [AudioMosaic-vit-b16-finetune-spc1](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-spc1) | v1 **99.03** |
| [AudioMosaic-vit-b16-finetune-spc2](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-spc2) | v2 **98.34** |

### EnvSDD (spoof detection, EER % on 4 test splits)
| Model | test01 | test02 | test03 | test04 |
|---|---|---|---|---|
| [AudioMosaic-vit-b16-finetune-envsdd-ata](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-envsdd-ata) | 0.00 | 0.00 | 0.30 | 0.30 |
| [AudioMosaic-vit-b16-finetune-envsdd-tta](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-finetune-envsdd-tta) | 0.00 | 0.06 | 0.37 | 4.50 |

### LTU (audio-language model)
| Model | Description |
|---|---|
| [AudioMosaic-vit-b16-ltu-stage4](https://huggingface.co/hanxunh/AudioMosaic-vit-b16-ltu-stage4) | End-to-end Llama-7B + AudioMosaic encoder + LoRA (stage 4, all mixed) |

Each repo provides a `load_model.py` and minimal usage example in its model card.

---
## Citation

```bibtex
@inproceedings{huang2026audiomosaic,
  title={AudioMosaic: Contrastive Masked Audio Representation Learning},
  author={Hanxun Huang, Qizhou Wang, Xingjun Ma, Cihang Xie, Christopher Leckie, Sarah Erfani},
  booktitle={ICML},
  year={2026}
}
```

---
## Part of the code is based on the following repos:

- AudioMAE: https://github.com/facebookresearch/AudioMAE
- BEATs: https://github.com/microsoft/unilm/tree/master/beats
- EAT: https://github.com/cwx-worst-one/EAT
- SSLAM: https://github.com/ta012/SSLAM
- LTU: https://github.com/YuanGongND/ltu
- timm: https://github.com/huggingface/pytorch-image-models
- mlconfig: https://github.com/narumiruna/mlconfig
