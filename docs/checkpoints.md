# Checkpoints

The checkpoints are on Hugging Face at [sidiangongyuan/GeoScaffold](https://huggingface.co/sidiangongyuan/GeoScaffold). From the repository root, this command downloads them to `checkpoints/urbaning/`:

```bash
hf download sidiangongyuan/GeoScaffold --local-dir . --include "checkpoints/*"
```

Each folder under `checkpoints/urbaning/` holds the final checkpoint of one run and the config of that run; pass the folder to `--model_dir` to evaluate it. The AP values are those of the results table: UrbanIng-V2X validation split (crossing1/2), in percent.

<!-- begin:checkpoints -->

| Method | Arm | Folder | File | Size (MB) | Evaluation flags | AP@0.3 / 0.5 / 0.7 |
|---|---|---|---|---|---|---|
| Ego-only | reference | `checkpoints/urbaning/ego_only/native/` | `net_epoch20.pth` | 25.3 | `--fusion_method nofusion --gt_config configs/urbaning/fcooper/native.yaml` | 22.55 / 16.12 / 5.41 |
| Early Fusion | reference | `checkpoints/urbaning/early_fusion/native/` | `net_epoch20.pth` | 25.3 | `--fusion_method early` | 23.76 / 17.35 / 6.46 |
| Late Fusion | reference | `checkpoints/urbaning/late_fusion/native/` | `net_epoch20.pth` | 25.3 | `--fusion_method late` | 27.25 / 19.40 / 6.88 |
| AttFuse | Native | `checkpoints/urbaning/attfuse/native/` | `net_epoch20.pth` | 27.9 | `--fusion_method intermediate` | 24.25 / 18.18 / 7.16 |
| AttFuse | +GeoScaffold | `checkpoints/urbaning/attfuse/geoscaffold/` | `net_epoch20.pth` | 31.6 | `--fusion_method intermediate` | 25.19 / 18.89 / 7.30 |
| F-Cooper | Native | `checkpoints/urbaning/fcooper/native/` | `net_epoch60.pth` | 27.9 | `--fusion_method intermediate` | 31.62 / 25.35 / 11.19 |
| F-Cooper | +GeoScaffold | `checkpoints/urbaning/fcooper/geoscaffold/` | `net_epoch60.pth` | 31.6 | `--fusion_method intermediate` | 33.12 / 27.43 / 12.57 |
| CoBEVT | Native | `checkpoints/urbaning/cobevt/native/` | `net_epoch20.pth` | 37.4 | `--fusion_method intermediate` | 28.27 / 22.45 / 8.76 |
| CoBEVT | +GeoScaffold | `checkpoints/urbaning/cobevt/geoscaffold/` | `net_epoch20.pth` | 41.1 | `--fusion_method intermediate` | 28.53 / 22.94 / 9.07 |
| Where2comm | Native | `checkpoints/urbaning/where2comm/native/` | `net_epoch60.pth` | 27.9 | `--fusion_method intermediate` | 30.21 / 24.89 / 11.13 |
| Where2comm | +GeoScaffold | `checkpoints/urbaning/where2comm/geoscaffold/` | `net_epoch60.pth` | 31.6 | `--fusion_method intermediate` | 30.90 / 26.16 / 11.78 |
| V2X-ViT | Native | `checkpoints/urbaning/v2xvit/native/` | `net_epoch30.pth` | 38.3 | `--fusion_method intermediate` | 33.86 / 26.38 / 11.16 |
| V2X-ViT | +GeoScaffold | `checkpoints/urbaning/v2xvit/geoscaffold/` | `net_epoch30.pth` | 42.0 | `--fusion_method intermediate` | 36.35 / 29.08 / 12.58 |
| V2VNet | Native | `checkpoints/urbaning/v2vnet/native/` | `net_epoch20.pth` | 52.9 | `--fusion_method intermediate` | 22.84 / 17.11 / 6.65 |
| V2VNet | +GeoScaffold | `checkpoints/urbaning/v2vnet/geoscaffold/` | `net_epoch20.pth` | 56.7 | `--fusion_method intermediate` | 23.48 / 17.95 / 7.07 |
| DiscoNet | Native | `checkpoints/urbaning/disconet/native/` | `net_epoch40.pth` | 81.1 | `--fusion_method intermediate` | 30.64 / 25.52 / 11.37 |
| DiscoNet | +GeoScaffold | `checkpoints/urbaning/disconet/geoscaffold/` | `net_epoch40.pth` | 84.9 | `--fusion_method intermediate` | 30.84 / 25.75 / 11.61 |
| CoAlign | Native | `checkpoints/urbaning/coalign/native/` | `net_epoch30.pth` | 49.4 | `--fusion_method intermediate` | 28.22 / 19.96 / 7.38 |
| CoAlign | +GeoScaffold | `checkpoints/urbaning/coalign/geoscaffold/` | `net_epoch30.pth` | 53.2 | `--fusion_method intermediate` | 28.27 / 20.10 / 7.45 |
| ERMVP | Native | `checkpoints/urbaning/ermvp/native/` | `net_epoch30.pth` | 32.0 | `--fusion_method intermediate` | 29.19 / 23.76 / 10.04 |
| ERMVP | +GeoScaffold | `checkpoints/urbaning/ermvp/geoscaffold/` | `net_epoch30.pth` | 35.8 | `--fusion_method intermediate` | 29.86 / 24.15 / 10.72 |
| HEAL | Native | `checkpoints/urbaning/heal/native/` | `net_epoch30.pth` | 20.9 | `--fusion_method intermediate` | 26.59 / 20.72 / 9.13 |
| HEAL | +GeoScaffold | `checkpoints/urbaning/heal/geoscaffold/` | `net_epoch30.pth` | 24.6 | `--fusion_method intermediate` | 28.12 / 22.53 / 9.41 |
| S-AdaFusion | Native | `checkpoints/urbaning/sadafusion/native/` | `net_epoch30.pth` | 27.9 | `--fusion_method intermediate` | 29.56 / 21.43 / 7.61 |
| S-AdaFusion | +GeoScaffold | `checkpoints/urbaning/sadafusion/geoscaffold/` | `net_epoch30.pth` | 31.6 | `--fusion_method intermediate` | 29.99 / 21.49 / 7.99 |

25 checkpoints, 964 MB in total. The CoAlign alignment files (`checkpoints/urbaning/coalign/alignment/`) add 139 MB.

<!-- end:checkpoints -->

`checkpoints/urbaning/coalign/alignment/` holds `train_all_agents.json` and `val_all_agents.json`, the stage-1 box-alignment results that CoAlign reads during training and evaluation. They record the split folders relative to the repository root (`datasets/UrbanIng-V2X-OpenCOOD-format/...`), so run CoAlign from the root with the dataset at that path.

`python tools/check_checkpoints.py` builds every model on the CPU and loads its checkpoint.
