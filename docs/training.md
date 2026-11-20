# Training

All runs use one GPU, batch size 4, and seed 1669408686, and start from the repository root. Pass the seed on the command line: `--seed` replaces the seed stored in the config, and its default value 0 draws a random seed.

## Protocol

Native and +GeoScaffold differ only in the GeoScaffold switch (`model.args.geometry_scaffold.enabled` and `loss.args.geometry_scaffold.weight`; AttFuse uses the equivalent `counterfactual_training` keys). Within a method, both arms share the data, seed, sample order, augmentation, optimizer settings, learning-rate schedule, and number of epochs. Each method keeps its own training horizon.

Most methods first train a shared Native prefix, and both arms resume from its last checkpoint. Checkpoints store model weights only, so each arm starts a new Adam optimizer, while the learning-rate schedule continues from the global epoch. The prefix configs of Where2comm, V2X-ViT, DiscoNet, and ERMVP keep the 20-epoch learning-rate schedule of the shared recipe, and their arm configs use the schedule of their longer horizon. Only the final checkpoint of each arm is evaluated.

<!-- begin:schedule -->

| Method | Shared Native prefix | Native and +GeoScaffold arms | GeoScaffold loss |
|---|---|---|---|
| AttFuse | `prefix.yaml`: epochs 1-11 | epochs 12-20 | epochs 12-20 |
| F-Cooper | none | epochs 1-60 | epochs 1-60 |
| CoBEVT | `prefix.yaml`: epochs 1-11 | epochs 12-20 | epochs 12-20 |
| Where2comm | `prefix.yaml`: epochs 1-11 | epochs 12-60 | epochs 12-60 |
| V2X-ViT | `prefix.yaml`: epochs 1-11 | epochs 12-30 | epochs 12-30 |
| V2VNet | `prefix.yaml`: epochs 1-11 | epochs 12-20 | epochs 12-20 |
| DiscoNet | `prefix.yaml`: epochs 1-11 | epochs 12-40 | epochs 12-40 |
| CoAlign | `prefix.yaml`: epochs 1-11; `prefix_stage2.yaml`: epochs 12-21 | epochs 22-30 | epochs 22-30 |
| ERMVP | `prefix.yaml`: epochs 1-11 | epochs 12-30 | epochs 12-30 |
| HEAL | `prefix.yaml`: epochs 1-11 | epochs 12-30 | epochs 12-30 |
| S-AdaFusion | `prefix.yaml`: epochs 1-11 | epochs 12-30 | epochs 12-30 |
| Ego-only (reference) | none | `native.yaml`: epochs 1-20 | none |
| Early Fusion (reference) | none | `native.yaml`: epochs 1-20 | none |
| Late Fusion (reference) | none | `native.yaml`: epochs 1-20 | none |

<!-- end:schedule -->

## Methods with a shared prefix

AttFuse, CoBEVT, Where2comm, V2X-ViT, V2VNet, DiscoNet, ERMVP, HEAL, and S-AdaFusion. Set `M` to the folder name in `configs/urbaning/`:

```bash
M=cobevt
python -m opencood.tools.train --hypes_yaml configs/urbaning/$M/prefix.yaml --seed 1669408686
PREFIX=$(ls -d opencood/logs/lidar/${M}_prefix_* | tail -n 1)
for ARM in native geoscaffold; do
  mkdir -p runs/$M/$ARM
  cp configs/urbaning/$M/$ARM.yaml runs/$M/$ARM/config.yaml
  cp $PREFIX/net_epoch11.pth runs/$M/$ARM/
  python -m opencood.tools.train --hypes_yaml runs/$M/$ARM/config.yaml \
      --model_dir runs/$M/$ARM --seed 1669408686
done
```

With `--model_dir`, the training script reads `config.yaml` from that folder and resumes from the newest `net_epoch*.pth` in it.

## CoAlign

The CoAlign prefix has two stages, and both arms branch at epoch 22:

```bash
python -m opencood.tools.train --hypes_yaml configs/urbaning/coalign/prefix.yaml --seed 1669408686
PREFIX=$(ls -d opencood/logs/lidar/coalign_prefix_* | tail -n 1)
mkdir -p runs/coalign/prefix_stage2
cp configs/urbaning/coalign/prefix_stage2.yaml runs/coalign/prefix_stage2/config.yaml
cp $PREFIX/net_epoch11.pth runs/coalign/prefix_stage2/
python -m opencood.tools.train --hypes_yaml runs/coalign/prefix_stage2/config.yaml \
    --model_dir runs/coalign/prefix_stage2 --seed 1669408686
for ARM in native geoscaffold; do
  mkdir -p runs/coalign/$ARM
  cp configs/urbaning/coalign/$ARM.yaml runs/coalign/$ARM/config.yaml
  cp runs/coalign/prefix_stage2/net_epoch21.pth runs/coalign/$ARM/
  python -m opencood.tools.train --hypes_yaml runs/coalign/$ARM/config.yaml \
      --model_dir runs/coalign/$ARM --seed 1669408686
done
```

CoAlign training and evaluation read the stage-1 box-alignment results in `checkpoints/urbaning/coalign/alignment/`, which come with the checkpoints. CoAlign's uncertainty-aware stage-1 detector produced them; this release ships the results without the stage-1 pipeline.

## F-Cooper and the reference detectors

F-Cooper trains both arms from scratch for 60 epochs, with GeoScaffold supervision from epoch 1. The reference detectors train for 20 epochs:

```bash
for CFG in fcooper/native fcooper/geoscaffold ego_only/native early_fusion/native late_fusion/native; do
  python -m opencood.tools.train --hypes_yaml configs/urbaning/$CFG.yaml --seed 1669408686
done
```

These runs write to `opencood/logs/lidar/<name>_<time>/`.

## DiscoNet teacher

DiscoNet distills from a frozen Early Fusion teacher loaded from `model.args.teacher.checkpoint_path`, which points to `checkpoints/urbaning/early_fusion/native/net_epoch20.pth`. To use your own Early Fusion run, copy its `net_epoch20.pth` to that path or change the path in the DiscoNet configs.

## Evaluating your runs

`--model_dir` takes a run folder, and inference loads the newest checkpoint in it:

```bash
python -m opencood.tools.inference --model_dir runs/cobevt/geoscaffold --fusion_method intermediate --gpu cuda:0
```

The [README](../README.md#evaluation) lists the fusion flags of the reference detectors and explains `evaluation_seed`.
