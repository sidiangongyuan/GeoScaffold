# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>, Hao Xiang <haxiang@g.ucla.edu>, Yifan Lu <yifan_lu@sjtu.edu.cn>
# License: TDG-Attribution-NonCommercial-NoDistrib


import argparse
import copy
import json
import math
import os
import random
import time
import statistics
import numpy as np
from tqdm import tqdm
import torch
import open3d as o3d
from torch.utils.data import DataLoader, Dataset

import opencood.hypes_yaml.yaml_utils as yaml_utils
from opencood.tools import train_utils, inference_utils
from opencood.data_utils.datasets import build_dataset
from opencood.utils import eval_utils
from opencood.visualization import vis_utils


def _atomic_json(path, payload):
    temporary = f"{path}.tmp.{os.getpid()}"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    os.replace(temporary, path)


def _write_progress(path, *, stage, completed, total, start_time, throughput, extra=None):
    if not path:
        return
    elapsed = max(time.monotonic() - start_time, 0.0)
    remaining = max(total - completed, 0)
    eta = None
    if throughput is not None and throughput > 0.0:
        eta = remaining / throughput
    payload = {
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


def _communication_rate_path(save_dir):
    return os.path.join(save_dir, 'communication_rate.json')


def _maybe_finite_scalar(value):
    if value is None:
        return None
    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            return None
        value = value.detach().item()
    elif hasattr(value, "shape") and hasattr(value, "item"):
        try:
            value = value.item()
        except (TypeError, ValueError):
            return None
    try:
        scalar = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(scalar):
        return None
    return scalar


def _set_training_epoch_if_supported(component, completed_epochs):
    setter = getattr(component, 'set_training_epoch', None)
    if setter is None:
        return False
    setter(int(completed_epochs))
    return True


class _CommunicationRateCollector:
    def __init__(self):
        self.values = []

    def hook(self, module, inputs, output):
        del module, inputs
        if not isinstance(output, dict):
            return
        scalar = _maybe_finite_scalar(output.get('com'))
        if scalar is not None:
            self.values.append(scalar)

    def finalize(self, save_dir):
        output_path = _communication_rate_path(save_dir)
        if not self.values:
            if os.path.exists(output_path):
                os.remove(output_path)
            return None
        payload = {
            'count': int(len(self.values)),
            'mean': float(statistics.mean(self.values)),
            'min': float(min(self.values)),
            'max': float(max(self.values)),
        }
        _atomic_json(output_path, payload)
        return payload


def test_parser():
    parser = argparse.ArgumentParser(description="synthetic data generation")
    parser.add_argument('--model_dir', type=str,
                        required=True,
                        help='Continued training path')
    parser.add_argument('--fusion_method',
                        #required=True,
                        type=str,
                        default='intermediate',
                        help='nofusion, late, early or intermediate')
    parser.add_argument('--show_vis', action='store_true',
                        help='whether to show image visualization result')
    parser.add_argument('--show_sequence', action='store_true',
                        help='whether to show video visualization result.'
                             'it can note be set true with show_vis together ')
    parser.add_argument('--save_vis',action='store_true',
                        help='whether to save visualization result')
    parser.add_argument('--save_npy', action='store_true',
                        help='whether to save prediction and gt result'
                             'in npy_test file')
    parser.add_argument('--global_sort_detections', action='store_true',
                        help='whether to globally sort detections by confidence score.'
                             'If set to True, it is the mainstream AP computing method,'
                             'but would increase the tolerance for FP (False Positives).')
    parser.add_argument('--gpu', type=str, default='cuda:1',
                        help='gpu number to use')
    parser.add_argument('--load_best', action='store_true',
                        help='whether to load the best model')
    parser.add_argument('--progress_path', default='',
                        help='Optional JSON heartbeat path')
    parser.add_argument('--progress_every', default=25, type=int,
                        help='Heartbeat frequency in evaluation batches')
    parser.add_argument('--gt_config', default=None,
                        help='Matched collaborative evaluation config for intermediate nofusion GT')

    opt = parser.parse_args()
    return opt


def _evaluation_distance_keys(dataset):
    """Native V2V4Real overall uses its rectangular extent, without a radius cut."""
    is_v2v4real = getattr(dataset, 'params', {}).get('dataset_format') == 'v2v4real'
    overall = '0-all' if is_v2v4real else '0-100'
    return [overall, '0-30', '30-50', '50-100']


def _inference_dataset_config(hypes: dict, fusion_method: str) -> dict:
    """Keep the No Fusion input restriction separate from evaluation annotations."""
    config = copy.deepcopy(hypes)
    if fusion_method == 'nofusion':
        core = config.get('fusion', {}).get('core_method')
        if core == 'LidarIntermediateFusionDataset':
            if int(config.get('train_params', {}).get('max_cav', 0)) != 1:
                raise ValueError('Intermediate nofusion model input must have max_cav=1')
        elif core != 'LidarLateFusionDataset':
            raise ValueError('nofusion requires a LateFusionDataset or single-agent intermediate input')
        elif config.get('dataset_format') == 'v2v4real':
            # This two-vehicle benchmark evaluates the union of both annotations,
            # including targets unobserved by the ego. Training/model settings stay intact.
            train = config.setdefault('train_params', {})
            train['max_cav'] = max(2, int(train.get('max_cav', 2)))
    return config


class _FrameSeededEvaluationDataset(Dataset):
    """Opt-in repeatable preprocessing, independent of model initialization and workers."""

    def __init__(self, dataset, seed):
        self.dataset = dataset
        self.seed = int(seed)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'dataset'), name)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        torch_state = torch.random.get_rng_state()
        frame_seed = (self.seed + int(index)) % (2 ** 32)
        try:
            np.random.seed(frame_seed)
            random.seed(frame_seed)
            torch.random.default_generator.manual_seed(frame_seed)
            return self.dataset[index]
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)


class _EgoInputFullGTDataset(Dataset):
    """Keep the native single-agent input and borrow only the matched full evaluation GT.

    The reference dataset runs its existing GT projection, deduplication, range and
    communication rules. Its features are discarded and never reach the ego detector.
    """

    def __init__(self, ego_dataset, gt_dataset):
        self.ego_dataset = ego_dataset
        self.gt_dataset = gt_dataset
        if len(ego_dataset) != len(gt_dataset):
            raise ValueError('No Fusion input and GT datasets have different frame coverage')
        for key in ('validate_dir',):
            if os.path.realpath(ego_dataset.params[key]) != os.path.realpath(gt_dataset.params[key]):
                raise ValueError('No Fusion input and GT must refer to the same split')
        for index, input_scene in ego_dataset.scenario_database.items():
            gt_scene = gt_dataset.scenario_database[index]
            input_ego = next(key for key, value in input_scene.items() if value['ego'])
            gt_ego = next(key for key, value in gt_scene.items() if value['ego'])
            if input_ego != gt_ego:
                raise ValueError('No Fusion input and GT must use the same receiver')
            input_stamps = [key for key, value in input_scene[input_ego].items()
                            if isinstance(value, dict) and 'yaml' in value]
            gt_stamps = [key for key, value in gt_scene[gt_ego].items()
                         if isinstance(value, dict) and 'yaml' in value]
            if input_stamps != gt_stamps:
                raise ValueError('No Fusion input and GT timestamps differ')
        if ego_dataset.len_record != gt_dataset.len_record:
            raise ValueError('No Fusion input and GT scene/frame indexing differs')
        if ego_dataset.object_types != gt_dataset.object_types:
            raise ValueError('No Fusion input and GT class mappings differ')
        if not np.array_equal(ego_dataset.gt_range, gt_dataset.gt_range):
            raise ValueError('No Fusion input and GT evaluation ranges differ')

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, 'ego_dataset'), name)

    def __len__(self):
        return len(self.ego_dataset)

    def __getitem__(self, index):
        sample = self.ego_dataset[index]
        if sample['ego']['cav_num'] != 1:
            raise ValueError('No Fusion sample contains collaborator model inputs')
        # Additional GT preprocessing must not perturb later ego point sampling.
        numpy_state, python_state = np.random.get_state(), random.getstate()
        try:
            reference = self.gt_dataset[index]
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
        for key in ('object_bbx_center', 'object_bbx_mask', 'object_ids'):
            sample['ego'][key] = reference['ego'][key]
        return sample


def _accumulate_distance_result(boxes, scores, gt, statistics, distance, iou,
                                distance_mode='legacy_corners'):
    if distance == '0-all':
        eval_utils.caluclate_tp_fp(boxes, scores, gt, statistics[distance], iou)
    else:
        minimum, maximum = distance.split('-')
        eval_utils.caluclate_tp_fp(
            boxes, scores, gt, statistics, iou, minimum, maximum,
            distance_mode=distance_mode)


def _normalize_empty_predictions(boxes, scores, gt):
    """An empty prediction frame still contributes all its GT to recall."""
    if boxes is None or scores is None:
        if ((boxes is not None and len(boxes)) or (scores is not None and len(scores))):
            raise ValueError('Prediction boxes and scores disagree on empty output')
        return gt.new_empty((0, *gt.shape[1:])), gt.new_empty((0, 2))
    return boxes, scores


def run_point_cloud_inference(save_dir, model, data_loader, device, fusion_method, save_results=True,
                              show_vis=False, show_sequence=False, save_vis=False,
                              save_npy=False, global_sort_detections=False,
                              progress_path=None, progress_every=25):
    dataset = data_loader.dataset
    total_batches = len(data_loader)
    progress_path = progress_path or None
    progress_every = max(int(progress_every), 1)
    start_time = time.monotonic()

    # create save directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    communication_rate_collector = _CommunicationRateCollector()
    communication_hook = None
    if hasattr(model, "register_forward_hook"):
        communication_hook = model.register_forward_hook(
            communication_rate_collector.hook
        )
    _write_progress(
        progress_path,
        stage='setup',
        completed=0,
        total=max(total_batches, 1),
        start_time=start_time,
        throughput=None,
        extra={
            'fusion_method': fusion_method,
            'device': str(device),
            'save_dir': save_dir,
        },
    )

    # returns eval statistics
    distances_to_check = _evaluation_distance_keys(dataset)
    distance_mode = ('bev_centroid' if getattr(dataset, 'params', {}).get('dataset_format')
                     == 'v2v4real' else 'legacy_corners')
    _atomic_json(os.path.join(save_dir, 'evaluation_metrics_protocol.json'), {
        'distance_mode': distance_mode,
        'distance_bins': distances_to_check,
        'global_sort_detections': bool(global_sort_detections),
        'overall_distance_filter': 'none' if '0-all' in distances_to_check else 'legacy',
    })

    object_types = dataset.object_types

    ious_to_check = [0.1, 0.2, 0.3, 0.5, 0.7]
    result_stat = {}
    for object_type in object_types:
        result_stat[object_type] = {}
        for distance_key in distances_to_check:
            result_stat[object_type][distance_key] = {}
            for iou in ious_to_check:
                result_stat[object_type][distance_key][iou] = {
                    'tp': [],
                    'fp': [],
                    'gt': 0,
                    'score': []
                }

    if show_sequence:
        vis = o3d.visualization.Visualizer()
        vis.create_window()

        vis.get_render_option().background_color = [0.05, 0.05, 0.05]
        vis.get_render_option().point_size = 1.0
        vis.get_render_option().show_coordinate_frame = True

        # used to visualize lidar points
        vis_pcd = o3d.geometry.PointCloud()
        # used to visualize object bounding box, maximum 50
        vis_aabbs_gt = []
        vis_aabbs_pred = []
        for _ in range(100):
            vis_aabbs_gt.append(o3d.geometry.LineSet())
            vis_aabbs_pred.append(o3d.geometry.LineSet())

    try:
        for i, batch_data in tqdm(enumerate(data_loader), total=total_batches):
            with torch.no_grad():
                batch_data = train_utils.to_device(batch_data, device)
                if fusion_method == 'nofusion':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils.inference_nofusion(batch_data, model, dataset)
                elif fusion_method == 'late':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils.inference_late_fusion(
                            batch_data,
                            model,
                            dataset)
                elif fusion_method == 'early':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils.inference_early_fusion(
                            batch_data,
                            model,
                            dataset)
                elif fusion_method == 'intermediate':
                    pred_box_tensor, pred_score, gt_box_tensor, gt_label_tensor = \
                        inference_utils.inference_intermediate_fusion(
                            batch_data,
                            model,
                            dataset)
                else:
                    raise NotImplementedError('Only nofusion, early, late and intermediate'
                                            'fusion is supported.')
                pred_box_tensor, pred_score = _normalize_empty_predictions(
                    pred_box_tensor, pred_score, gt_box_tensor)
                for class_id, class_name in enumerate(result_stat.keys()):
                    class_id += 1
                    keep_index_pred = pred_score[:, -1] == class_id
                    keep_index_gt = gt_label_tensor == class_id

                    preds_class_tensor = pred_box_tensor[keep_index_pred, ...]
                    preds_class_score = pred_score[keep_index_pred, 0]
                    gt_class_box_tensor = gt_box_tensor[keep_index_gt, ...]
                    for distances in result_stat[class_name].keys():
                        for iou in result_stat[class_name][distances].keys():
                            _accumulate_distance_result(
                                preds_class_tensor,
                                preds_class_score,
                                gt_class_box_tensor,
                                result_stat[class_name],
                                distances,
                                iou,
                                distance_mode=distance_mode,
                            )

                if save_npy:
                    npy_save_path = os.path.join(save_dir, 'npy')
                    if not os.path.exists(npy_save_path):
                        os.makedirs(npy_save_path)
                    inference_utils.save_prediction_gt(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'][0],
                        i,
                        npy_save_path)

                if show_vis or save_vis:
                    vis_save_path = ''
                    if save_vis:
                        vis_save_path = os.path.join(save_dir, 'vis')
                        if not os.path.exists(vis_save_path):
                            os.makedirs(vis_save_path)
                        vis_save_path = os.path.join(vis_save_path, '%05d.png' % i)

                    dataset.visualize_result(
                        pred_box_tensor,
                        gt_box_tensor,
                        batch_data['ego']['origin_lidar'],
                        None,
                        show_vis,
                        vis_save_path,
                        dataset=dataset)

                if show_sequence:
                    pcd, pred_o3d_box, gt_o3d_box = \
                        vis_utils.visualize_inference_sample_dataloader_with_map(
                            pred_box_tensor,
                            gt_box_tensor,
                            batch_data['ego']['origin_lidar'],
                            None,
                            vis_pcd,
                            mode='constant'
                            )
                    if i == 0:
                        vis.add_geometry(pcd)
                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_pred,
                                                    pred_o3d_box,
                                                    update_mode='add')

                        vis_utils.linset_assign_list(vis,
                                                    vis_aabbs_gt,
                                                    gt_o3d_box,
                                                    update_mode='add')

                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_pred,
                                                pred_o3d_box)
                    vis_utils.linset_assign_list(vis,
                                                vis_aabbs_gt,
                                                gt_o3d_box)
                    vis.update_geometry(pcd)
                    vis.poll_events()
                    vis.update_renderer()
                    time.sleep(0.001)
            batch_index = i + 1
            if batch_index % progress_every == 0 or batch_index == total_batches:
                throughput = batch_index / max(time.monotonic() - start_time, 1e-12)
                _write_progress(
                    progress_path,
                    stage='inference',
                    completed=batch_index,
                    total=max(total_batches, 1),
                    start_time=start_time,
                    throughput=throughput,
                    extra={'fusion_method': fusion_method},
                )

        eval_results = eval_utils.eval_final_results(
            result_stat,
            save_dir,
            global_sort_detections,
            save_results
        )
        communication_rate_collector.finalize(save_dir)

        if show_sequence:
            vis.destroy_window()

        throughput = total_batches / max(time.monotonic() - start_time, 1e-12) if total_batches else None
        _write_progress(
            progress_path,
            stage='complete',
            completed=total_batches,
            total=max(total_batches, 1),
            start_time=start_time,
            throughput=throughput,
            extra={
                'fusion_method': fusion_method,
                'save_dir': save_dir,
            },
        )

        return eval_results
    finally:
        if communication_hook is not None:
            communication_hook.remove()


def prepare_inference_dataloader(hypes, specific_scenario=None, gt_hypes=None, **dataloader_kwargs):
    standard_dl_kwargs = {
        'num_workers': 16,
        'shuffle': False,
        'pin_memory': False,
        'drop_last': False
    }
    standard_dl_kwargs.update(dataloader_kwargs)
    standard_dl_kwargs['batch_size'] = 1  # always 1 for inference

    opencood_dataset = build_dataset(hypes, visualize=True, train=False, specific_scenario=specific_scenario)
    if gt_hypes is not None:
        reference = build_dataset(
            gt_hypes, visualize=False, train=False, specific_scenario=specific_scenario)
        opencood_dataset = _EgoInputFullGTDataset(opencood_dataset, reference)

    is_camera_dataset = True if 'camera' in opencood_dataset.__class__.__name__.lower() else False

    if hypes.get('evaluation_seed') is not None:
        opencood_dataset = _FrameSeededEvaluationDataset(
            opencood_dataset, hypes['evaluation_seed'])

    data_loader = DataLoader(
        opencood_dataset,
        collate_fn=opencood_dataset.collate_batch if is_camera_dataset else opencood_dataset.collate_batch_test,
        **standard_dl_kwargs)
    
    return data_loader, is_camera_dataset


def prepare_inference_device(gpu='cuda:0'):
    device = 'cpu' if gpu is None else gpu
    if gpu is not None and torch.cuda.is_available():
        device = torch.device(gpu)
    else:
        device = torch.device('cpu')
        
    return device


def prepare_inference_model(hypes, device):
    model = train_utils.create_model(hypes)
    model.to(device)

    return model


def load_inference_model_weights(model, saved_path, load_best: bool = False, epoch=None):
    if epoch:
        model = train_utils.load_specific_epoch_model(saved_path, model, epoch)
        _set_training_epoch_if_supported(model, int(epoch))
    else:
        initial_epoch, model = train_utils.load_saved_model(
            saved_path,
            model,
            load_best=load_best,
        )
        if isinstance(initial_epoch, int) and initial_epoch > 0:
            _set_training_epoch_if_supported(model, initial_epoch)
    
    model.eval()

    return model


def run_inference(
        model_dir, fusion_method='intermediate', show_vis=False,
        show_sequence=False, save_vis=False,
        save_npy=False, global_sort_detections=False,
        gpu='cuda:0', load_best=False,
        progress_path=None, progress_every=25, gt_config=None):
    
    assert os.path.exists(model_dir), '%s not found' % model_dir

    assert fusion_method in ['nofusion', 'late', 'early', 'intermediate']
    assert not (show_vis and show_sequence), \
        'you can only visualize the results in single image mode or video mode'

    hypes = yaml_utils.load_yaml(None, model_dir)

    dataset_config = _inference_dataset_config(hypes, fusion_method)
    gt_hypes = yaml_utils.load_yaml(gt_config) if gt_config else None
    if gt_hypes is not None and fusion_method != 'nofusion':
        raise ValueError('--gt_config is scoped to No Fusion evaluation')
    if (fusion_method == 'nofusion'
            and hypes['fusion']['core_method'] == 'LidarIntermediateFusionDataset'
            and gt_hypes is None):
        raise ValueError('Intermediate nofusion requires --gt_config from matched full evaluation')
    data_loader, is_camera_dataset = prepare_inference_dataloader(dataset_config, gt_hypes=gt_hypes)
    if fusion_method == 'nofusion':
        _atomic_json(os.path.join(model_dir, 'evaluation_protocol.json'), {
            'fusion_method': 'nofusion', 'prediction_agents': ['ego'],
            'gt_source': 'all_dataset_annotations_before_prediction_agent_selection',
            'dataset_format': hypes.get('dataset_format', 'urbaning'),
            'dataset_max_cav': (gt_hypes or dataset_config)['train_params']['max_cav'],
            'gt_config': os.path.realpath(gt_config) if gt_config else None,
            'prediction_max_cav': 1,
            'frames': len(data_loader.dataset),
        })

    device = prepare_inference_device(gpu)

    model = prepare_inference_model(hypes, device)

    model = load_inference_model_weights(model, model_dir, load_best=load_best)
    model.to(device)
    model.eval()

    save_dir = os.path.join(model_dir, os.path.basename(hypes['validate_dir']))
    if 'specific_ego_id' in hypes['train_params'] and hypes['train_params']['max_cav'] > 1:
        save_dir = os.path.join(save_dir, str(hypes['train_params']['specific_ego_id']))

    run_point_cloud_inference(
        save_dir, model, data_loader, device, save_results=True,
        fusion_method=fusion_method, show_vis=show_vis,
        show_sequence=show_sequence, save_vis=save_vis,
        save_npy=save_npy, global_sort_detections=global_sort_detections,
        progress_path=progress_path, progress_every=progress_every)


def main():
    opt = test_parser()
    model_dir = opt.model_dir
    fusion_method = opt.fusion_method
    show_vis = opt.show_vis
    show_sequence = opt.show_sequence
    save_vis = opt.save_vis
    save_npy = opt.save_npy
    global_sort_detections = opt.global_sort_detections
    gpu = opt.gpu
    load_best = opt.load_best
    progress_path = opt.progress_path
    progress_every = opt.progress_every

    run_inference(
        model_dir, fusion_method, show_vis=show_vis,
        show_sequence=show_sequence, save_vis=save_vis,
        save_npy=save_npy, global_sort_detections=global_sort_detections,
        gpu=gpu, load_best=load_best,
        progress_path=progress_path, progress_every=progress_every, gt_config=opt.gt_config
    )


if __name__ == '__main__':
    main()
