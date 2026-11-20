# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

from __future__ import annotations


import argparse
import json
import math
import os
import random
import statistics
import time
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import torch
import tqdm
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, DistributedSampler

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils
from opencood.tools import multi_gpu_utils
from opencood.data_utils.datasets import build_dataset
from opencood.tools import train_utils
import torch.distributed as dist
from opencood.utils.seg_utils import cal_iou_training


os.environ['MASTER_PORT'] = '29600'
os.environ['WORLD_SIZE'] = "2"


def _build_dataloader_generator(seed: int, *, offset: int = 0) -> torch.Generator:
    generator = torch.Generator()
    generator.manual_seed(int(seed) + int(offset))
    return generator


def _seed_dataloader_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


def _reset_model_side_rng(seed: int) -> None:
    train_utils.set_random_seed(int(seed))


def _release_iteration_cache(criterion) -> None:
    """Release completed dynamic graphs before returning CUDA blocks."""

    loss_dict = getattr(criterion, 'loss_dict', None)
    if isinstance(loss_dict, dict):
        loss_dict.clear()
    torch.cuda.empty_cache()


def _set_training_epoch_if_supported(component, completed_epochs: int) -> bool:
    """Advance staged model/loss components without method-specific branches."""
    if isinstance(component, (tuple, list)):
        updated = False
        for item in component:
            updated = _set_training_epoch_if_supported(
                item,
                completed_epochs,
            ) or updated
        return updated
    setter = getattr(component, 'set_training_epoch', None)
    if setter is None:
        return False
    setter(int(completed_epochs))
    return True


def _backbone_health(model: torch.nn.Module) -> dict[str, object]:
    """Detect numeric underflow, not a scientific AP/convergence threshold."""
    backbone = model.backbone
    weight = next(module.weight.detach() for module in backbone.modules()
                  if isinstance(module, torch.nn.Conv2d))
    variances = [module.running_var.detach() for module in backbone.modules()
                 if isinstance(module, torch.nn.modules.batchnorm._BatchNorm)
                 and module.running_var is not None]
    if not variances:
        raise ValueError('Backbone-health monitoring requires BatchNorm running statistics')
    values = torch.cat([value.flatten() for value in variances])
    weight_max = float(weight.abs().max().item())
    variance_max = float(values.max().item())
    health = {
        'finite': bool(torch.isfinite(weight).all() and torch.isfinite(values).all()),
        'first_conv_abs_max': weight_max,
        'bn_variance_min': float(values.min().item()),
        'bn_variance_median': float(values.median().item()),
        'bn_variance_max': variance_max,
        'bn_layers': len(variances),
        'numerical_collapse': (weight_max < torch.finfo(weight.dtype).tiny
                               and variance_max < torch.finfo(values.dtype).tiny),
    }
    return {key: None if isinstance(value, float) and not math.isfinite(value) else value
            for key, value in health.items()}


def _build_loss_target(batch_ego):
    """Bundle optional native auxiliary labels without changing standard loss input."""
    auxiliary_keys = ('ego_label_dict', 'label_dict_single')
    if not any(key in batch_ego for key in auxiliary_keys):
        return batch_ego['label_dict']
    target = {'label_dict': batch_ego['label_dict']}
    for key in auxiliary_keys:
        if key in batch_ego:
            target[key] = batch_ego[key]
    return target


def _load_initial_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    *,
    allowed_missing_prefixes: Sequence[str] = (),
) -> dict[str, list[str]]:
    """Load weight initialization without changing the training start epoch."""

    checkpoint = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Initial checkpoint not found: {checkpoint}")
    state = torch.load(str(checkpoint), map_location='cpu')
    if isinstance(state, Mapping) and 'state_dict' in state:
        state = state['state_dict']
    if not isinstance(state, Mapping):
        raise ValueError("Initial checkpoint must contain a model state mapping.")
    incompatible = model.load_state_dict(state, strict=False)
    missing = list(incompatible.missing_keys)
    unexpected = list(incompatible.unexpected_keys)
    unallowed_missing = [
        key
        for key in missing
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unallowed_missing or unexpected:
        raise RuntimeError(
            "Initial checkpoint is incompatible: "
            f"unallowed_missing={unallowed_missing}, unexpected={unexpected}"
        )
    return {"missing_keys": missing, "unexpected_keys": unexpected}


def _atomic_json(path: str, payload: dict[str, object]) -> None:
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_progress(
    path: str | None,
    *,
    stage: str,
    completed: int,
    total: int,
    start_time: float,
    throughput: float | None,
    extra: dict[str, object] | None = None,
) -> None:
    if not path:
        return
    elapsed = max(time.monotonic() - start_time, 0.0)
    remaining = max(total - completed, 0)
    eta = None
    if throughput is not None and throughput > 0.0:
        eta = remaining / throughput
    payload: dict[str, object] = {
        "event": "progress",
        "stage": stage,
        "completed": int(completed),
        "total": int(total),
        "elapsed_seconds": float(elapsed),
        "throughput_per_second": None if throughput is None else float(throughput),
        "eta_seconds": None if eta is None else float(eta),
    }
    if extra:
        payload.update(extra)
    _atomic_json(path, payload)


def train_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument("--hypes_yaml", type=str,
                        required=True,
                        help='data generation yaml file needed ')
    parser.add_argument('--model_dir', default='',
                        help='Continued training path')
    parser.add_argument("--half", action='store_true',
                        help="whether train with half precision.")
    parser.add_argument('--distributed', action='store_true',
                        help='whether to use distributed training')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--gpu', default=0, type=int,
                        help='ID of gpu to use (if available)')
    parser.add_argument('--progress_path', default='',
                        help='Optional JSON heartbeat path')
    parser.add_argument('--progress_every', default=50, type=int,
                        help='Heartbeat frequency in training steps')
    parser.add_argument('--empty_cache_every', default=0, type=int,
                        help='Release CUDA cache every N completed steps; 0 disables it')
    parser.add_argument('--initial_checkpoint', default='',
                        help='Optional weight-only initialization; training starts at epoch zero')
    parser.add_argument('--initial_checkpoint_allowed_missing_prefix',
                        action='append', default=[],
                        help='Allowed missing-key prefix for weight-only initialization')
    opt = parser.parse_args()
    return opt


def run_training(hypes_yaml, model_dir=None, seed=0, distributed=False, half=False, gpu=0,
                 progress_path: str | None = None, progress_every: int = 50,
                 empty_cache_every: int = 0,
                 initial_checkpoint: str | None = None,
                 initial_checkpoint_allowed_missing_prefix: Sequence[str] = ()):
    """
    Run training for the model.

    Parameters
    ----------
    hypes_yaml : str
        yaml file path.
    model_dir : str
        continued training path.
    seed : int
        random seed.
    distributed : bool
        whether to use distributed training.
    half : bool
        whether to use half precision training.
    gpu : int
        gpu id.

    Returns
    -------
    None
    """
    hypes = yaml_utils.load_yaml(hypes_yaml, model_dir)
    monitor_backbone = bool(hypes.get('train_params', {}).get('monitor_backbone_health', False))

    if distributed:
        distributed = multi_gpu_utils.init_distributed_mode()
    start_time = time.monotonic()
    progress_path = progress_path or None
    progress_every = max(int(progress_every), 1)
    empty_cache_every = int(empty_cache_every)
    if empty_cache_every < 0:
        raise ValueError('empty_cache_every must be non-negative')

    print('-----------------Seed Setting----------------------')
    seed = train_utils.init_random_seed(None if seed == 0 else seed)
    hypes['train_params']['seed'] = seed
    print('Set seed to %d' % seed)
    train_utils.set_random_seed(seed)

    print('-----------------Dataset Building------------------')
    opencood_train_dataset = build_dataset(hypes, visualize=False, train=True)
    opencood_validation_dataset = build_dataset(hypes, visualize=False, train=False)

    train_loader_generator = _build_dataloader_generator(seed)
    val_loader_generator = _build_dataloader_generator(seed, offset=1)

    if distributed:
        sampler_train = DistributedSampler(
            opencood_train_dataset,
            shuffle=True,
            seed=seed,
        )
        sampler_val = DistributedSampler(opencood_validation_dataset,
                                         shuffle=False,
                                         seed=seed)

        batch_sampler_train = torch.utils.data.BatchSampler(
            sampler_train, hypes['train_params']['batch_size'], drop_last=True)

        train_loader = DataLoader(opencood_train_dataset,
                                  batch_sampler=batch_sampler_train,
                                  num_workers=16,
                                  collate_fn=opencood_train_dataset.collate_batch,
                                  worker_init_fn=_seed_dataloader_worker,
                                  generator=train_loader_generator)
        val_loader = DataLoader(opencood_validation_dataset,
                                sampler=sampler_val,
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch,
                                worker_init_fn=_seed_dataloader_worker,
                                generator=val_loader_generator)
    else:
        train_loader = DataLoader(opencood_train_dataset,
                                    batch_size=hypes['train_params']['batch_size'],
                                    num_workers=16,
                                    collate_fn=opencood_train_dataset.collate_batch,
                                    shuffle=True,
                                    pin_memory=True,
                                    worker_init_fn=_seed_dataloader_worker,
                                    generator=train_loader_generator)
        val_loader = DataLoader(opencood_validation_dataset,
                                batch_size=hypes['train_params']['batch_size'],
                                num_workers=16,
                                collate_fn=opencood_train_dataset.collate_batch,
                                shuffle=False,
                                pin_memory=True,
                                worker_init_fn=_seed_dataloader_worker,
                                generator=val_loader_generator)

    print('---------------Creating Model------------------')
    model = train_utils.create_model(hypes)
    if not distributed and gpu:
        device = torch.device(f'cuda:{gpu}' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # if we want to train from last checkpoint.
    if model_dir:
        saved_path = model_dir
        if initial_checkpoint:
            init_epoch = 0
            initialization_report = _load_initial_checkpoint(
                model,
                initial_checkpoint,
                allowed_missing_prefixes=initial_checkpoint_allowed_missing_prefix,
            )
            print(f'Loaded weight-only initialization: {initialization_report}')
        else:
            init_epoch, model = train_utils.load_saved_model(saved_path, model)
    else:
        init_epoch = 0
        # if we train the model from scratch, we need to create a folder
        # to save the model,
        saved_path = train_utils.setup_train(hypes)
        if initial_checkpoint:
            initialization_report = _load_initial_checkpoint(
                model,
                initial_checkpoint,
                allowed_missing_prefixes=initial_checkpoint_allowed_missing_prefix,
            )
            print(f'Loaded weight-only initialization: {initialization_report}')

    # we assume gpu is necessary
    model.to(device)
    
    model_without_ddp = model

    if distributed:
        model = \
            torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[gpu],
                find_unused_parameters=True
            )
        model_without_ddp = model.module

    # define the loss
    criterion = train_utils.create_loss(hypes)

    # optimizer setup
    optimizer = train_utils.setup_optimizer(hypes, model_without_ddp)
    # lr scheduler setup
    num_steps = len(train_loader)
    scheduler = train_utils.setup_lr_scheduler(hypes, optimizer, num_steps)

    # record training
    writer = SummaryWriter(saved_path)

    # half precision training
    if half:
        scaler = torch.cuda.amp.GradScaler()

    _reset_model_side_rng(seed)
    print('Training start')
    epoches = hypes['train_params']['epoches']
    # used to help schedule learning rate

    best_val_loss = 1e10
    total_train_steps = len(train_loader) * max(epoches - init_epoch, 0)
    completed_train_steps = 0
    _write_progress(
        progress_path,
        stage='setup',
        completed=0,
        total=max(total_train_steps, 1),
        start_time=start_time,
        throughput=None,
        extra={
            'epoch_start': int(init_epoch),
            'epoch_end': int(epoches),
            'device': str(device),
            'seed': int(seed),
        },
    )

    for epoch in range(init_epoch, max(epoches, init_epoch)):
        _set_training_epoch_if_supported((model_without_ddp, criterion), epoch)
        if hasattr(model_without_ddp, 'geometry_scaffold_start_epoch'):
            print('GeoScaffold epoch=%d start_epoch=%d active=%s' % (
                epoch + 1, model_without_ddp.geometry_scaffold_start_epoch,
                model_without_ddp.geometry_scaffold_active,
            ), flush=True)
        if hypes['lr_scheduler']['core_method'] != 'cosineannealwarm':
            scheduler.step(epoch)
        if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
            scheduler.step_update(epoch * num_steps + 0)

        for param_group in optimizer.param_groups:
            print('learning rate %.7f' % param_group["lr"])

        pbar_train = tqdm.tqdm(total=len(train_loader), leave=True)

        # Run training for one epoch
        epoch_train_start = time.monotonic()
        for i, batch_data in enumerate(train_loader):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            batch_data = train_utils.to_device(batch_data, device)

            # case1 : late fusion train --> only ego needed,
            # and ego is random selected
            # case2 : early fusion train --> all data projected to ego
            # case3 : intermediate fusion --> ['ego']['processed_lidar']
            # becomes a list, which containing all data from other cavs
            # as well
            if not half:
                output_dict = model(batch_data['ego'])
                # first argument is always your output dictionary,
                # second argument is always your label dictionary.
                final_loss = criterion(
                    output_dict,
                    _build_loss_target(batch_data['ego']),
                )
            else:
                with torch.cuda.amp.autocast():
                    output_dict = model(batch_data['ego'])
                    final_loss = criterion(
                        output_dict,
                        _build_loss_target(batch_data['ego']),
                    )

            if monitor_backbone and not torch.isfinite(final_loss).all():
                raise RuntimeError('Non-finite training loss in the repaired backbone path')
            criterion.logging(epoch, i, len(train_loader), writer, pbar=pbar_train)
            pbar_train.update(1)

            if not half:
                final_loss.backward()
                optimizer.step()
            else:
                scaler.scale(final_loss).backward()
                scaler.step(optimizer)
                scaler.update()

            if hypes['lr_scheduler']['core_method'] == 'cosineannealwarm':
                scheduler.step_update(epoch * num_steps + i)

            completed_train_steps += 1
            if completed_train_steps % progress_every == 0 or completed_train_steps == total_train_steps:
                throughput = completed_train_steps / max(time.monotonic() - start_time, 1e-12)
                _write_progress(
                    progress_path,
                    stage='train',
                    completed=completed_train_steps,
                    total=max(total_train_steps, 1),
                    start_time=start_time,
                    throughput=throughput,
                    extra={
                        'epoch': int(epoch + 1),
                        'epoch_batch': int(i + 1),
                        'epoch_batches': int(len(train_loader)),
                        'epoch_elapsed_seconds': float(time.monotonic() - epoch_train_start),
                    },
                )
            if (
                empty_cache_every > 0
                and completed_train_steps % empty_cache_every == 0
            ):
                del output_dict, final_loss, batch_data
                _release_iteration_cache(criterion)

        # Run evaluation
        if epoch % hypes['train_params']['eval_freq'] == 0 or epoch == epoches - 1:
            valid_avg_loss = []

            pbar_val = tqdm.tqdm(total=len(val_loader), leave=True)
            val_start = time.monotonic()
            with torch.no_grad():
                for i, batch_data in enumerate(val_loader):
                    model.eval()

                    batch_data = train_utils.to_device(batch_data, device)
                    output_dict = model(batch_data['ego'])

                    final_loss = criterion(
                        output_dict,
                        _build_loss_target(batch_data['ego']),
                    )

                    valid_avg_loss.append(final_loss.item())

                    pbar_val.update(1)
                    if (i + 1) % progress_every == 0 or i + 1 == len(val_loader):
                        throughput = (i + 1) / max(time.monotonic() - val_start, 1e-12)
                        _write_progress(
                            progress_path,
                            stage='validate',
                            completed=i + 1,
                            total=max(len(val_loader), 1),
                            start_time=val_start,
                            throughput=throughput,
                            extra={
                                'epoch': int(epoch + 1),
                                'train_steps_completed': int(completed_train_steps),
                            },
                        )
                    if empty_cache_every > 0 and (i + 1) % empty_cache_every == 0:
                        del output_dict, final_loss, batch_data
                        _release_iteration_cache(criterion)

            valid_avg_loss = statistics.mean(valid_avg_loss)
            print('At epoch %d, the validation loss is %f' % (epoch, valid_avg_loss))
            writer.add_scalar('Validate_Loss', valid_avg_loss, epoch)
        
            if valid_avg_loss < best_val_loss:
                best_val_loss = valid_avg_loss
                torch.save(model_without_ddp.state_dict(),
                           os.path.join(saved_path, 'best_model.pth'))
    
        # Save the model
        if epoch % hypes['train_params']['save_freq'] == 0 or epoch == epoches - 1:
            torch.save(model_without_ddp.state_dict(),
                os.path.join(saved_path, 'net_epoch%d.pth' % (epoch + 1)))

        if monitor_backbone:
            health = dict(_backbone_health(model_without_ddp), epoch=int(epoch + 1))
            _atomic_json(os.path.join(saved_path, 'backbone_health.json'), health)
            print('Backbone health: ' + json.dumps(health), flush=True)
            if not health['finite'] or health['numerical_collapse']:
                raise RuntimeError('Backbone numerical collapse; retain checkpoint and stop this arm')
    
        # reinitialize the dataset
        opencood_train_dataset.reinitialize()

    print('Training Finished, checkpoints saved to %s' % saved_path)
    _write_progress(
        progress_path,
        stage='complete',
        completed=max(total_train_steps, completed_train_steps),
        total=max(total_train_steps, completed_train_steps, 1),
        start_time=start_time,
        throughput=completed_train_steps / max(time.monotonic() - start_time, 1e-12)
        if completed_train_steps
        else None,
        extra={
            'saved_path': saved_path,
            'best_val_loss': float(best_val_loss),
        },
    )


def main():
    # python -m torch.distributed.run --nproc_per_node=2 --nnodes=1 opencood/tools/train.py --distributed --hypes_yaml /home/dominik/Git_Repos/Private/Ingolstadt_Crossing_Dataset_DL/opencood/hypes_yaml/lidar/thi_dataset/point_pillar_cobevt.yaml
    opt = train_parser()
    hypes_yaml = opt.hypes_yaml
    model_dir = opt.model_dir
    seed = opt.seed
    distributed = opt.distributed
    half = opt.half
    gpu = opt.gpu

    if distributed:
        gpu = opt.gpu
    
    run_training(
        hypes_yaml,
        model_dir,
        seed,
        distributed,
        half,
        gpu,
        progress_path=opt.progress_path,
        progress_every=opt.progress_every,
        empty_cache_every=opt.empty_cache_every,
        initial_checkpoint=opt.initial_checkpoint or None,
        initial_checkpoint_allowed_missing_prefix=(
            opt.initial_checkpoint_allowed_missing_prefix
        ),
    )

    

if __name__ == '__main__':
    main()
