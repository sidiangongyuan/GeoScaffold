"""Dependency-light, uncertainty-weighted SE(2) box alignment for CoAlign.

The historical ``legacy`` path is retained for reproducibility.  An explicit
``scipy_se2`` path provides a joint agent--landmark least-squares objective
matching the graph semantics of the original g2o implementation.  The ego
pose is fixed throughout; frames without enough shared landmarks are returned
byte-for-byte unchanged.  Metric uncertainty must be declared explicitly for
the new path because stage-1 native box-code variances cannot be converted
without the generating anchor.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


STAGE1_SCHEMA_VERSION = 1


def _metadata_value(
    frame: Mapping[str, Any],
    artifact: Mapping[str, Any],
    *keys: str,
) -> Any:
    """Read frame-local metadata, falling back to artifact scope metadata."""
    for key in keys:
        if key in frame:
            return frame[key]
    for key in keys:
        if key in artifact:
            return artifact[key]
    return None


def _normalise_identity(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _timestamp_number(value: Any, *, field_name: str) -> float:
    try:
        parsed = float(str(value))
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"CoAlign {field_name} must be numeric, got {value!r}."
        ) from error
    if not math.isfinite(parsed):
        raise ValueError(f"CoAlign {field_name} must be finite, got {value!r}.")
    return parsed


def _timestamps_equal(left: Any, right: Any) -> bool:
    """Compare timestamp keys while tolerating equivalent zero padding."""
    left_text = _normalise_identity(left)
    right_text = _normalise_identity(right)
    if left_text == right_text:
        return True
    if left_text is None or right_text is None:
        return False
    try:
        return _timestamp_number(left_text, field_name="timestamp") == _timestamp_number(
            right_text,
            field_name="timestamp",
        )
    except ValueError:
        return False


def _normalise_agent_entries(
    frame: Mapping[str, Any],
) -> tuple[list[str], list[Any], list[Any], list[Any], list[str] | None]:
    """Convert rich or legacy stage-1 frame forms to parallel arrays."""
    agents = frame.get("agents")
    if agents is not None:
        if isinstance(agents, Mapping):
            entries = []
            for cav_id, value in agents.items():
                if not isinstance(value, Mapping):
                    raise ValueError(
                        f"CoAlign agent {cav_id!r} must be a mapping."
                    )
                entries.append((str(cav_id), value))
        elif isinstance(agents, Sequence) and not isinstance(agents, (str, bytes)):
            entries = []
            for value in agents:
                if not isinstance(value, Mapping) or "cav_id" not in value:
                    raise ValueError(
                        "Rich CoAlign agents must provide mapping entries with cav_id."
                    )
                entries.append((str(value["cav_id"]), value))
        else:
            raise ValueError("CoAlign frame agents must be a mapping or list.")
        if not entries:
            raise ValueError("CoAlign frame agents cannot be empty.")
        cav_ids: list[str] = []
        boxes: list[Any] = []
        uncertainties: list[Any] = []
        class_ids: list[Any] = []
        timestamps: list[str] = []
        for cav_id, entry in entries:
            if cav_id in cav_ids:
                raise ValueError(f"Duplicate CoAlign agent id {cav_id!r}.")
            required = ("boxes_hwl", "uncertainties", "class_ids")
            missing = [key for key in required if key not in entry]
            source_timestamp = next(
                (
                    entry[key]
                    for key in ("timestamp", "source_hist", "history_timestamp")
                    if key in entry
                ),
                None,
            )
            if source_timestamp is None:
                missing.append("timestamp/source_hist")
            if missing:
                raise ValueError(
                    f"CoAlign agent {cav_id!r} misses required fields {missing}."
                )
            cav_ids.append(cav_id)
            boxes.append(entry["boxes_hwl"])
            uncertainties.append(entry["uncertainties"])
            class_ids.append(entry["class_ids"])
            timestamps.append(str(source_timestamp))
        return cav_ids, boxes, uncertainties, class_ids, timestamps

    required = ("cav_ids", "boxes_hwl", "uncertainties", "class_ids")
    if any(key not in frame for key in required):
        missing = [key for key in required if key not in frame]
        raise ValueError(f"CoAlign frame misses required fields {missing}.")
    cav_ids = [str(value) for value in frame["cav_ids"]]
    boxes = list(frame["boxes_hwl"])
    uncertainties = list(frame["uncertainties"])
    class_ids = list(frame["class_ids"])
    lengths = [len(cav_ids), len(boxes), len(uncertainties), len(class_ids)]
    if len(set(lengths)) != 1:
        raise ValueError("CoAlign frame has inconsistent agent arrays.")

    raw_timestamps = None
    for key in (
        "agent_timestamps",
        "timestamps",
        "history_timestamps",
        "source_hist",
        "source_history",
    ):
        if key in frame:
            raw_timestamps = frame[key]
            break
    if raw_timestamps is None:
        timestamps = None
    elif isinstance(raw_timestamps, Mapping):
        missing = [cav_id for cav_id in cav_ids if cav_id not in raw_timestamps]
        if missing:
            raise ValueError(
                f"CoAlign frame misses timestamps for agents {missing}."
            )
        timestamps = [str(raw_timestamps[cav_id]) for cav_id in cav_ids]
    elif isinstance(raw_timestamps, Sequence) and not isinstance(
        raw_timestamps,
        (str, bytes),
    ):
        if len(raw_timestamps) != len(cav_ids):
            raise ValueError("CoAlign agent timestamps must align with cav_ids.")
        timestamps = [str(value) for value in raw_timestamps]
    else:
        raise ValueError("CoAlign agent timestamps must be a mapping or list.")
    return cav_ids, boxes, uncertainties, class_ids, timestamps


def _as_boxes(value: Any) -> np.ndarray:
    boxes = np.asarray(value, dtype=np.float64)
    if boxes.size == 0:
        return np.empty((0, 7), dtype=np.float64)
    if boxes.ndim != 2 or boxes.shape[1] != 7:
        raise ValueError("Stage-1 boxes must have shape [N, 7].")
    if not np.isfinite(boxes).all():
        raise ValueError("Stage-1 boxes contain non-finite values.")
    return boxes


def _as_uncertainty(value: Any, count: int) -> np.ndarray:
    uncertainty = np.asarray(value, dtype=np.float64)
    if uncertainty.size == 0:
        return np.zeros((count, 3), dtype=np.float64)
    if uncertainty.ndim != 2 or uncertainty.shape[0] != count:
        raise ValueError("Uncertainty rows must align with stage-1 boxes.")
    if uncertainty.shape[1] not in (2, 3):
        raise ValueError("CoAlign uncertainty must contain x/y or x/y/yaw.")
    if uncertainty.shape[1] == 2:
        uncertainty = np.pad(uncertainty, ((0, 0), (0, 1)))
    if not np.isfinite(uncertainty).all():
        raise ValueError("Stage-1 uncertainty contains non-finite values.")
    return uncertainty


def _as_classes(value: Any, count: int) -> np.ndarray:
    if value is None:
        return np.ones(count, dtype=np.int64)
    classes = np.asarray(value, dtype=np.int64).reshape(-1)
    if classes.shape[0] != count:
        raise ValueError("Class ids must align with stage-1 boxes.")
    return classes


def _canonical_frame(
    frame: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and expose a stage-1 frame through one internal shape."""
    if not isinstance(frame, Mapping):
        raise ValueError("CoAlign frame must be a mapping.")
    cav_ids, boxes, uncertainties, class_ids, timestamps = _normalise_agent_entries(
        frame
    )
    if len(set(cav_ids)) != len(cav_ids):
        raise ValueError("CoAlign frame contains duplicate cav_ids.")
    canonical_boxes = []
    canonical_uncertainties = []
    canonical_classes = []
    for boxes_value, uncertainty_value, classes_value in zip(
        boxes,
        uncertainties,
        class_ids,
    ):
        boxes_array = _as_boxes(boxes_value)
        canonical_boxes.append(boxes_array.tolist())
        canonical_uncertainties.append(
            _as_uncertainty(uncertainty_value, boxes_array.shape[0]).tolist()
        )
        canonical_classes.append(
            _as_classes(classes_value, boxes_array.shape[0]).tolist()
        )
    result: dict[str, Any] = {
        "cav_ids": cav_ids,
        "boxes_hwl": canonical_boxes,
        "uncertainties": canonical_uncertainties,
        "class_ids": canonical_classes,
    }
    if timestamps is not None:
        result["agent_timestamps"] = timestamps
    for key in (
        "sample_idx",
        "dataset_root",
        "split",
        "dataset_split",
        "scenario",
        "scenario_id",
        "timestamp",
        "current_timestamp",
        "uncertainty_space",
        "uncertainty_semantics",
    ):
        if key in frame:
            result[key] = frame[key]
    return result


def _wrap_radians(angle: np.ndarray | float) -> np.ndarray | float:
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _world_measurements(
    boxes: np.ndarray,
    pose: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    yaw = math.radians(float(pose[4]))
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]],
        dtype=np.float64,
    )
    centers = boxes[:, :2] @ rotation.T + pose[None, :2]
    headings = _wrap_radians(boxes[:, 6] + yaw)
    return centers, np.asarray(headings, dtype=np.float64)


def _components(
    centers: list[np.ndarray],
    class_ids: list[np.ndarray],
    match_distance: float,
) -> list[list[tuple[int, int]]]:
    nodes = [
        (agent, index)
        for agent, agent_centers in enumerate(centers)
        for index in range(agent_centers.shape[0])
    ]
    parent = list(range(len(nodes)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for left, (left_agent, left_index) in enumerate(nodes):
        for right in range(left + 1, len(nodes)):
            right_agent, right_index = nodes[right]
            if left_agent == right_agent:
                continue
            if class_ids[left_agent][left_index] != class_ids[right_agent][right_index]:
                continue
            distance = np.linalg.norm(
                centers[left_agent][left_index] - centers[right_agent][right_index]
            )
            if distance < match_distance:
                union(left, right)

    grouped: dict[int, list[tuple[int, int]]] = {}
    for index, node in enumerate(nodes):
        grouped.setdefault(find(index), []).append(node)

    components: list[list[tuple[int, int]]] = []
    for component in grouped.values():
        if len({agent for agent, _ in component}) < 2:
            continue
        # Chained matches can place multiple detections from one source in one
        # component. Keep the one nearest the provisional component center.
        provisional = np.mean(
            [centers[agent][index] for agent, index in component], axis=0
        )
        unique: list[tuple[int, int]] = []
        for agent in sorted({agent for agent, _ in component}):
            candidates = [node for node in component if node[0] == agent]
            unique.append(
                min(
                    candidates,
                    key=lambda node: (
                        float(np.linalg.norm(centers[node[0]][node[1]] - provisional)),
                        node[1],
                    ),
                )
            )
        components.append(unique)
    return components


def _weighted_rigid_fit(
    source: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray] | None:
    if source.shape[0] < 2:
        return None
    weights = np.maximum(weights.astype(np.float64), 1e-12)
    weights /= weights.sum()
    source_center = np.sum(source * weights[:, None], axis=0)
    target_center = np.sum(target * weights[:, None], axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ (target_zero * weights[:, None])
    if np.linalg.matrix_rank(covariance) < 1:
        return None
    left, _, right_t = np.linalg.svd(covariance)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation


def _align_agent_poses_legacy(
    boxes_by_agent: Sequence[Any],
    lidar_poses: Any,
    *,
    uncertainties_by_agent: Sequence[Any] | None = None,
    class_ids_by_agent: Sequence[Any] | None = None,
    match_distance: float = 1.5,
    minimum_landmarks: int = 2,
    max_iterations: int = 20,
    convergence_tolerance: float = 1e-4,
) -> np.ndarray:
    """Refine remote x/y/yaw poses from shared stage-1 detections.

    ``boxes_by_agent`` are local-frame ``[x,y,z,h,w,l,yaw]`` predictions.
    Log-variance uncertainty weights are used when available. The first agent
    is the fixed receiver, matching the repository's intermediate-fusion order.
    """

    if match_distance <= 0.0:
        raise ValueError("match_distance must be positive.")
    if minimum_landmarks < 2:
        raise ValueError("minimum_landmarks must be at least two.")
    poses = np.asarray(lidar_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 5:
        raise ValueError("lidar_poses must have shape [agents, >=5].")
    if len(boxes_by_agent) != poses.shape[0]:
        raise ValueError("Every agent pose must have one stage-1 box array.")
    original = poses.copy()
    boxes = [_as_boxes(value) for value in boxes_by_agent]
    if uncertainties_by_agent is None:
        uncertainties = [np.zeros((value.shape[0], 3)) for value in boxes]
    else:
        if len(uncertainties_by_agent) != len(boxes):
            raise ValueError("Uncertainty list must align with agents.")
        uncertainties = [
            _as_uncertainty(value, box.shape[0])
            for value, box in zip(uncertainties_by_agent, boxes)
        ]
    if class_ids_by_agent is None:
        classes = [np.ones(value.shape[0], dtype=np.int64) for value in boxes]
    else:
        if len(class_ids_by_agent) != len(boxes):
            raise ValueError("Class-id list must align with agents.")
        classes = [
            _as_classes(value, box.shape[0])
            for value, box in zip(class_ids_by_agent, boxes)
        ]

    refined = poses.copy()
    saw_valid_graph = False
    for _ in range(int(max_iterations)):
        centers: list[np.ndarray] = []
        headings: list[np.ndarray] = []
        for agent, agent_boxes in enumerate(boxes):
            world_centers, world_headings = _world_measurements(
                agent_boxes, refined[agent]
            )
            centers.append(world_centers)
            headings.append(world_headings)
        components = _components(centers, classes, float(match_distance))
        if len(components) < int(minimum_landmarks):
            return original
        saw_valid_graph = True

        landmark_centers: list[np.ndarray] = []
        landmark_headings: list[float] = []
        for component in components:
            component_weights = np.asarray(
                [
                    math.exp(
                        -float(
                            np.clip(
                                uncertainties[agent][index, :2].mean(),
                                -10.0,
                                10.0,
                            )
                        )
                    )
                    for agent, index in component
                ],
                dtype=np.float64,
            )
            component_weights /= component_weights.sum()
            landmark_centers.append(
                np.sum(
                    np.stack([centers[a][i] for a, i in component])
                    * component_weights[:, None],
                    axis=0,
                )
            )
            complex_heading = np.sum(
                np.exp(1j * np.asarray([headings[a][i] for a, i in component]))
                * component_weights
            )
            landmark_headings.append(float(np.angle(complex_heading)))

        previous = refined.copy()
        for agent in range(1, refined.shape[0]):
            local_points: list[np.ndarray] = []
            target_points: list[np.ndarray] = []
            fit_weights: list[float] = []
            yaw_offsets: list[float] = []
            for component_index, component in enumerate(components):
                observations = [node for node in component if node[0] == agent]
                if not observations:
                    continue
                _, box_index = observations[0]
                local_points.append(boxes[agent][box_index, :2])
                target_points.append(landmark_centers[component_index])
                weight = math.exp(
                    -float(
                        np.clip(
                            uncertainties[agent][box_index, :2].mean(),
                            -10.0,
                            10.0,
                        )
                    )
                )
                fit_weights.append(weight)
                yaw_offsets.append(
                    float(
                        _wrap_radians(
                            landmark_headings[component_index]
                            - boxes[agent][box_index, 6]
                        )
                    )
                )
            if len(local_points) < int(minimum_landmarks):
                continue
            fit = _weighted_rigid_fit(
                np.stack(local_points),
                np.stack(target_points),
                np.asarray(fit_weights),
            )
            if fit is None:
                continue
            rotation, translation = fit
            geometric_yaw = math.atan2(rotation[1, 0], rotation[0, 0])
            if yaw_offsets:
                yaw_vector = np.sum(
                    np.exp(1j * np.asarray(yaw_offsets))
                    * np.asarray(fit_weights)
                )
                measured_yaw = float(np.angle(yaw_vector))
                # Center registration is better conditioned; heading provides
                # the native CoAlign orientation measurement when available.
                yaw = float(np.angle(np.exp(1j * geometric_yaw) + np.exp(1j * measured_yaw)))
            else:
                yaw = geometric_yaw
            refined[agent, 0:2] = translation
            refined[agent, 4] = math.degrees(yaw)

        delta = np.max(np.abs(refined[:, [0, 1, 4]] - previous[:, [0, 1, 4]]))
        if delta <= float(convergence_tolerance):
            break

    return refined if saw_valid_graph else original


def _normalise_uncertainty_space(value: Any) -> str:
    """Return the explicit unit contract used by the pose-graph backend."""
    if value is None:
        raise ValueError(
            "The scipy_se2 backend requires explicit uncertainty_space='metric' "
            "when uncertainty predictions are provided."
        )
    space = str(value).strip().lower().replace("_", "-")
    if space not in {"metric", "metric-log-variance"}:
        raise ValueError(
            "scipy_se2 accepts only uncertainty_space='metric' (log variances "
            "in m^2, m^2, rad^2); native box-code uncertainty requires an "
            "anchor-aware conversion before alignment."
        )
    return "metric"


def _prepare_metric_log_variances(
    uncertainties: Sequence[np.ndarray],
    *,
    uncertainty_space: Any,
    use_uncertainty: bool,
    uncertainty_was_provided: bool,
) -> list[np.ndarray]:
    """Validate the metric uncertainty contract without guessing anchor sizes.

    Stage-1 predicts log variances in native regression coordinates.  This
    backend intentionally consumes only values that have already been mapped
    to metric units by an anchor-aware replay.  In particular, it must not
    infer an anchor from the predicted class: class labels and anchor indices
    are independent in the class-aware head.
    """
    if not use_uncertainty or not uncertainty_was_provided:
        return [np.zeros_like(value, dtype=np.float64) for value in uncertainties]
    if any(value.shape[0] > 0 for value in uncertainties):
        _normalise_uncertainty_space(uncertainty_space)
    # The stage-1 yaw target is sin(delta yaw).  Treating its variance as
    # rad^2 is a small-angle approximation and is recorded as such by the
    # artifact producer; this module does not claim a full likelihood model.
    return [
        np.clip(value.astype(np.float64, copy=False), -30.0, 30.0)
        for value in uncertainties
    ]


def _edge_whitening(log_variance: np.ndarray) -> np.ndarray:
    """Convert diagonal metric log variances to residual whitening factors."""
    return np.exp(-0.5 * np.clip(log_variance.astype(np.float64), -30.0, 30.0))


def _align_agent_poses_scipy(
    boxes_by_agent: Sequence[Any],
    lidar_poses: Any,
    *,
    uncertainties_by_agent: Sequence[Any] | None = None,
    class_ids_by_agent: Sequence[Any] | None = None,
    uncertainty_space: Any = None,
    match_distance: float = 1.5,
    minimum_landmarks: int = 2,
    max_iterations: int = 20,
    convergence_tolerance: float = 1e-4,
    landmark_se2: bool = True,
    adaptive_landmark: bool = False,
    abandon_hard_cases: bool = False,
    drop_hard_boxes: bool = False,
    drop_unsure_edge: bool = False,
    use_uncertainty: bool = True,
    yaw_var_thres: float = 0.2,
) -> np.ndarray:
    """Optimize the CoAlign agent--landmark graph with SciPy.

    This is an opt-in, dependency-light port of the objective constructed by
    CoAlign's ``box_align_v2.py``.  Agent poses are world-from-agent SE(2)
    transforms, the first agent is fixed, and every active edge measures a
    local box pose from an agent to a shared landmark.  Unlike the historical
    alternating solver, translation and yaw are optimized jointly and every
    uncertainty dimension has its own diagonal weight.

    The implementation is numerically equivalent in objective semantics to
    the official g2o graph, but is not bitwise equivalent to g2o.  It accepts
    only explicit metric log variances.  The legacy backend remains the
    default in :func:`align_agent_poses` for result reproducibility.
    """
    # Import lazily: the legacy path should not acquire a SciPy runtime
    # dependency merely by importing the CoAlign module.
    from scipy.optimize import least_squares
    from scipy.sparse import lil_matrix

    if match_distance <= 0.0:
        raise ValueError("match_distance must be positive.")
    if minimum_landmarks < 2:
        raise ValueError("minimum_landmarks must be at least two.")
    if max_iterations < 1:
        raise ValueError("max_iterations must be positive.")
    if yaw_var_thres < 0.0:
        raise ValueError("yaw_var_thres must be non-negative.")

    poses = np.asarray(lidar_poses, dtype=np.float64)
    if poses.ndim != 2 or poses.shape[1] < 5:
        raise ValueError("lidar_poses must have shape [agents, >=5].")
    if len(boxes_by_agent) != poses.shape[0]:
        raise ValueError("Every agent pose must have one stage-1 box array.")
    original = poses.copy()
    boxes = [_as_boxes(value) for value in boxes_by_agent]

    if uncertainties_by_agent is None:
        uncertainties = [
            np.zeros((value.shape[0], 3), dtype=np.float64) for value in boxes
        ]
        uncertainty_was_provided = False
    else:
        if len(uncertainties_by_agent) != len(boxes):
            raise ValueError("Uncertainty list must align with agents.")
        uncertainties = [
            _as_uncertainty(value, box.shape[0])
            for value, box in zip(uncertainties_by_agent, boxes)
        ]
        uncertainty_was_provided = True

    if class_ids_by_agent is None:
        classes = [
            np.ones(value.shape[0], dtype=np.int64) for value in boxes
        ]
    else:
        if len(class_ids_by_agent) != len(boxes):
            raise ValueError("Class-id list must align with agents.")
        classes = [
            _as_classes(value, box.shape[0])
            for value, box in zip(class_ids_by_agent, boxes)
        ]

    metric_log_variances = _prepare_metric_log_variances(
        uncertainties,
        uncertainty_space=uncertainty_space,
        use_uncertainty=use_uncertainty,
        uncertainty_was_provided=uncertainty_was_provided,
    )

    # Correspondence is deliberately frozen from the observed (noisy) pose,
    # as in the official CoAlign construction.  Re-matching while optimizing
    # would change the graph and make the result depend on solver trajectory.
    initial_centers: list[np.ndarray] = []
    initial_headings: list[np.ndarray] = []
    for agent, agent_boxes in enumerate(boxes):
        centers, headings = _world_measurements(agent_boxes, original[agent])
        initial_centers.append(centers)
        initial_headings.append(headings)
    components = _components(initial_centers, classes, float(match_distance))
    if len(components) < int(minimum_landmarks):
        return original

    component_info: list[dict[str, Any]] = []
    for component in components:
        yaw_values = np.asarray(
            [initial_headings[agent][index] for agent, index in component],
            dtype=np.float64,
        )
        yaw_variance = float(np.var(yaw_values))
        if adaptive_landmark and yaw_variance > float(yaw_var_thres):
            component_landmark_se2 = False
        else:
            component_landmark_se2 = bool(landmark_se2)
        first_agent, first_index = component[0]
        initial_landmark = [
            float(initial_centers[first_agent][first_index, 0]),
            float(initial_centers[first_agent][first_index, 1]),
        ]
        if component_landmark_se2:
            initial_landmark.append(float(initial_headings[first_agent][first_index]))
        component_info.append(
            {
                "nodes": component,
                "active": not (
                    drop_hard_boxes and yaw_variance > float(yaw_var_thres)
                ),
                "landmark_se2": component_landmark_se2,
                "initial": np.asarray(initial_landmark, dtype=np.float64),
                "yaw_variance": yaw_variance,
            }
        )

    # Match the official hard-case switch explicitly.  These are protocol
    # switches, not hidden safeguards: callers must opt in to either policy.
    if abandon_hard_cases:
        if len(component_info) <= 3:
            return original
        varying = sum(
            info["yaw_variance"] > float(yaw_var_thres)
            for info in component_info
        )
        if varying >= 0.5 * len(component_info):
            return original

    active_component_count = sum(bool(info["active"]) for info in component_info)
    if active_component_count < int(minimum_landmarks):
        return original

    # Each edge is (agent index, box index, landmark index, SE2?, whitening).
    edges: list[tuple[int, int, int, bool, np.ndarray]] = []
    for landmark_index, info in enumerate(component_info):
        if not info["active"]:
            continue
        for agent, box_index in info["nodes"]:
            if use_uncertainty:
                whitening = _edge_whitening(
                    metric_log_variances[agent][box_index]
                )
            else:
                whitening = np.ones(3, dtype=np.float64)
            if drop_unsure_edge:
                precision_sum = float(
                    np.square(whitening[:3 if info["landmark_se2"] else 2]).sum()
                )
                if precision_sum < 100.0:
                    continue
            edges.append(
                (
                    int(agent),
                    int(box_index),
                    int(landmark_index),
                    bool(info["landmark_se2"]),
                    whitening,
                )
            )
    if not edges:
        return original

    remote_agents = list(range(1, poses.shape[0]))
    agent_offsets = {
        agent: offset for offset, agent in enumerate(remote_agents)
    }
    landmark_offsets: dict[int, int] = {}
    variable_count = 3 * len(remote_agents)
    for landmark_index, info in enumerate(component_info):
        if info["active"]:
            landmark_offsets[landmark_index] = variable_count
            variable_count += 3 if info["landmark_se2"] else 2
    if variable_count == 0:
        return original

    x0 = np.empty(variable_count, dtype=np.float64)
    for agent in remote_agents:
        offset = 3 * agent_offsets[agent]
        x0[offset : offset + 3] = [
            original[agent, 0],
            original[agent, 1],
            math.radians(float(original[agent, 4])),
        ]
    for landmark_index, offset in landmark_offsets.items():
        initial = component_info[landmark_index]["initial"]
        x0[offset : offset + initial.shape[0]] = initial

    residual_rows = sum(3 if edge[3] else 2 for edge in edges)
    jac_sparsity = lil_matrix(
        (residual_rows, variable_count), dtype=np.int8
    )
    row = 0
    for agent, _, landmark_index, landmark_is_se2, _ in edges:
        row_count = 3 if landmark_is_se2 else 2
        if agent != 0:
            agent_offset = 3 * agent_offsets[agent]
            jac_sparsity[row : row + row_count, agent_offset : agent_offset + 3] = 1
        landmark_offset = landmark_offsets[landmark_index]
        landmark_dim = 3 if landmark_is_se2 else 2
        jac_sparsity[row : row + row_count, landmark_offset : landmark_offset + landmark_dim] = 1
        row += row_count

    fixed_agent = np.asarray(
        [original[0, 0], original[0, 1], math.radians(float(original[0, 4]))],
        dtype=np.float64,
    )

    def agent_state(values: np.ndarray, agent: int) -> np.ndarray:
        if agent == 0:
            return fixed_agent
        offset = 3 * agent_offsets[agent]
        return values[offset : offset + 3]

    def residual(values: np.ndarray) -> np.ndarray:
        output = np.empty(residual_rows, dtype=np.float64)
        row_index = 0
        for agent, box_index, landmark_index, landmark_is_se2, whitening in edges:
            agent_pose = agent_state(values, agent)
            landmark_offset = landmark_offsets[landmark_index]
            landmark_dim = 3 if landmark_is_se2 else 2
            landmark = values[landmark_offset : landmark_offset + landmark_dim]
            local_box = boxes[agent][box_index]
            delta = landmark[:2] - agent_pose[:2]
            cosine = math.cos(float(agent_pose[2]))
            sine = math.sin(float(agent_pose[2]))
            # T_agent^{-1} * landmark: the residual lives in the local
            # measurement frame, matching g2o::EdgeSE2 semantics.
            predicted_local = np.asarray(
                [
                    cosine * delta[0] + sine * delta[1],
                    -sine * delta[0] + cosine * delta[1],
                ],
                dtype=np.float64,
            )
            translation_error = predicted_local - local_box[:2]
            if landmark_is_se2:
                # g2o::EdgeSE2 computes
                # inv(measurement) * (T_agent^{-1} * T_landmark).
                # Consequently, translation residuals are expressed in the
                # measurement frame before a diagonal information matrix is
                # applied.  This rotation matters for anisotropic x/y
                # uncertainty and is not present for EdgeSE2PointXY.
                measurement_cosine = math.cos(float(local_box[6]))
                measurement_sine = math.sin(float(local_box[6]))
                translation_error = np.asarray(
                    [
                        measurement_cosine * translation_error[0]
                        + measurement_sine * translation_error[1],
                        -measurement_sine * translation_error[0]
                        + measurement_cosine * translation_error[1],
                    ],
                    dtype=np.float64,
                )
            output[row_index : row_index + 2] = (
                translation_error * whitening[:2]
            )
            row_index += 2
            if landmark_is_se2:
                predicted_yaw = float(
                    _wrap_radians(landmark[2] - agent_pose[2])
                )
                yaw_error = float(
                    _wrap_radians(predicted_yaw - local_box[6])
                )
                output[row_index] = yaw_error * whitening[2]
                row_index += 1
        return output

    try:
        result = least_squares(
            residual,
            x0,
            jac_sparsity=jac_sparsity.tocsr(),
            method="trf",
            loss="linear",
            max_nfev=int(max_iterations),
            xtol=float(convergence_tolerance),
            ftol=float(convergence_tolerance),
            gtol=float(convergence_tolerance),
        )
    except (FloatingPointError, ValueError, RuntimeError):
        return original
    if not result.success or not np.isfinite(result.x).all():
        return original

    refined = original.copy()
    for agent in remote_agents:
        offset = 3 * agent_offsets[agent]
        refined[agent, 0] = result.x[offset]
        refined[agent, 1] = result.x[offset + 1]
        refined[agent, 4] = math.degrees(
            float(_wrap_radians(result.x[offset + 2]))
        )
    # The receiver is a hard gauge anchor.  Reassigning it makes the exact
    # identity guarantee explicit even if future solver code is refactored.
    refined[0] = original[0]
    return refined


def align_agent_poses(
    boxes_by_agent: Sequence[Any],
    lidar_poses: Any,
    *,
    backend: str = "legacy",
    uncertainty_space: Any = None,
    **kwargs: Any,
) -> np.ndarray:
    """Align poses using the legacy solver or the opt-in ``scipy_se2`` port.

    ``backend='legacy'`` is the historical default and is intentionally
    unchanged.  ``backend='scipy_se2'`` (or the upstream-compatible
    ``'pose_graph'`` alias) requires explicit metric uncertainty metadata
    whenever uncertainty predictions are supplied; this prevents an old cache
    with native box-code variances from silently entering the new objective.
    """
    backend_name = str(backend).strip().lower()
    if backend_name == "legacy":
        if uncertainty_space is not None:
            # The legacy implementation cannot consume metric metadata, but
            # accepting it keeps callers that switch back to the old protocol
            # source-compatible.  Its historical weighting remains untouched.
            kwargs.pop("uncertainty_space", None)
        return _align_agent_poses_legacy(
            boxes_by_agent,
            lidar_poses,
            **kwargs,
        )
    if backend_name in {"scipy_se2", "scipy-se2", "pose_graph"}:
        # Accept the spelling used by the upstream box-alignment config while
        # keeping the implementation provenance explicit: this environment
        # has no g2o, so ``pose_graph`` selects the SciPy fallback.
        if "landmark_SE2" in kwargs and "landmark_se2" not in kwargs:
            kwargs["landmark_se2"] = kwargs.pop("landmark_SE2")
        if "thres" in kwargs and "match_distance" not in kwargs:
            kwargs["match_distance"] = kwargs.pop("thres")
        return _align_agent_poses_scipy(
            boxes_by_agent,
            lidar_poses,
            uncertainty_space=uncertainty_space,
            **kwargs,
        )
    raise ValueError(
        f"Unknown CoAlign alignment backend {backend!r}; expected 'legacy' "
        "or 'scipy_se2'."
    )


def load_stage1_artifact(path: str | Path) -> dict[str, Any]:
    artifact_path = Path(path).expanduser().resolve()
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("CoAlign stage-1 artifact must be a JSON object.")
    if payload.get("schema_version") != STAGE1_SCHEMA_VERSION:
        raise ValueError("Unsupported CoAlign stage-1 artifact schema.")
    frames = payload.get("frames")
    if not isinstance(frames, Mapping):
        raise ValueError("CoAlign stage-1 artifact must contain a frames mapping.")
    for key in (
        "dataset_root",
        "split",
        "dataset_split",
        "uncertainty_space",
        "uncertainty_semantics",
    ):
        if key in payload and not isinstance(payload[key], (str, Path)):
            raise ValueError(f"CoAlign artifact {key} must be a string.")
    for frame_id, frame in frames.items():
        try:
            _canonical_frame(frame)
        except ValueError as error:
            raise ValueError(f"Invalid CoAlign frame {frame_id!r}: {error}") from error
    return payload


def select_stage1_frame(
    artifact: Mapping[str, Any],
    *,
    sample_idx: Any,
    dataset_root: str | Path | None,
    split: str | None,
    scenario_id: Any,
    current_timestamp: Any,
    expected_agent_timestamps: Mapping[Any, Any],
    expected_agent_ids: Sequence[Any],
) -> dict[str, Any]:
    """Select and validate the stage-1 frame for one dataset sample.

    Enriched frames may be keyed by sample index or by any manifest key, but
    must carry scenario/timestamp identity and a timestamp for every agent.
    Legacy parallel-array frames are accepted only when all requested source
    timestamps equal the current timestamp; this is the no-delay case where
    an unqualified frame index remains unambiguous.
    """
    if not isinstance(artifact, Mapping):
        raise ValueError("CoAlign artifact must be a mapping.")
    frames = artifact.get("frames")
    if not isinstance(frames, Mapping):
        raise ValueError("CoAlign artifact must contain a frames mapping.")

    requested_idx = str(sample_idx)
    direct = frames.get(requested_idx)
    candidates = [(requested_idx, direct)] if direct is not None else list(frames.items())
    if not candidates:
        raise ValueError(f"CoAlign stage-1 frame {requested_idx!r} is missing.")

    expected_root = (
        os.path.realpath(os.path.expanduser(str(dataset_root)))
        if dataset_root is not None
        else None
    )
    expected_split = None if split is None else str(split)
    expected_scenario = _normalise_identity(scenario_id)
    expected_current = _normalise_identity(current_timestamp)
    expected_ids = [str(value) for value in expected_agent_ids]
    expected_id_set = set(expected_ids)
    expected_timestamps = {
        str(agent_id): str(timestamp)
        for agent_id, timestamp in expected_agent_timestamps.items()
    }
    missing_timestamp_metadata = [
        agent_id for agent_id in expected_ids if agent_id not in expected_timestamps
    ]
    if missing_timestamp_metadata:
        raise ValueError(
            "CoAlign runtime sample misses history timestamps for agents "
            f"{missing_timestamp_metadata}."
        )

    last_error: ValueError | None = None
    for frame_key, raw_frame in candidates:
        try:
            frame = _canonical_frame(raw_frame)
            # The producer may store the unit conversion at artifact scope;
            # carry it into the selected frame so the opt-in metric backend
            # cannot accidentally consume an unqualified legacy cache.
            for metadata_key in ("uncertainty_space", "uncertainty_semantics"):
                metadata_value = _metadata_value(frame, artifact, metadata_key)
                if metadata_value is not None:
                    frame[metadata_key] = metadata_value
            frame_idx = frame.get("sample_idx")
            if frame_idx is not None and str(frame_idx) != requested_idx:
                raise ValueError(
                    f"frame sample_idx {frame_idx!r} does not match {requested_idx!r}"
                )

            artifact_root = _metadata_value(frame, artifact, "dataset_root")
            if expected_root is not None:
                if artifact_root is None:
                    raise ValueError("stage-1 frame has no dataset_root identity")
                actual_root = os.path.realpath(os.path.expanduser(str(artifact_root)))
                if actual_root != expected_root:
                    raise ValueError(
                        f"dataset_root mismatch: artifact={actual_root!r}, "
                        f"runtime={expected_root!r}"
                    )

            artifact_split = _metadata_value(
                frame,
                artifact,
                "split",
                "dataset_split",
            )
            if expected_split is not None and artifact_split is not None:
                if str(artifact_split) != expected_split:
                    raise ValueError(
                        f"dataset split mismatch: artifact={artifact_split!r}, "
                        f"runtime={expected_split!r}"
                    )
            elif expected_split is not None:
                # The repair runner may omit a literal split label when the
                # artifact root is already the resolved split-specific
                # validate_dir.  Root identity then confirms the split; an
                # explicit conflicting label is still rejected above.
                if artifact_root is None or expected_root is None:
                    raise ValueError("stage-1 frame has no split identity")

            artifact_scenario = _metadata_value(
                frame,
                artifact,
                "scenario_id",
                "scenario",
            )
            if expected_scenario is not None:
                if artifact_scenario is None:
                    raise ValueError("stage-1 frame has no scenario identity")
                if str(artifact_scenario) != expected_scenario:
                    raise ValueError(
                        f"scenario mismatch: artifact={artifact_scenario!r}, "
                        f"runtime={expected_scenario!r}"
                    )

            artifact_current = _metadata_value(
                frame,
                artifact,
                "current_timestamp",
                "timestamp",
            )
            if expected_current is not None:
                if artifact_current is None:
                    raise ValueError("stage-1 frame has no current timestamp identity")
                if not _timestamps_equal(artifact_current, expected_current):
                    raise ValueError(
                        f"current timestamp mismatch: artifact={artifact_current!r}, "
                        f"runtime={expected_current!r}"
                    )

            frame_ids = [str(value) for value in frame["cav_ids"]]
            frame_id_set = set(frame_ids)
            missing_ids = sorted(expected_id_set - frame_id_set)
            if missing_ids:
                raise ValueError(
                    f"stage-1 frame is missing active agents {missing_ids}"
                )

            frame_timestamps = frame.get("agent_timestamps")
            if frame_timestamps is None:
                # A legacy frame has no way to identify historical source
                # observations.  It remains valid only for clean samples;
                # delayed runs must use a prejoined artifact generated from
                # the delayed raw LiDAR source frames.
                if any(
                    not _timestamps_equal(
                        expected_timestamps[agent_id],
                        expected_current,
                    )
                    for agent_id in expected_ids
                ):
                    raise ValueError(
                        "legacy stage-1 frame has no per-agent timestamps; "
                        "delayed CoAlign requires a prejoined artifact"
                    )
            else:
                frame_timestamp_by_id = {
                    agent_id: str(timestamp)
                    for agent_id, timestamp in zip(frame_ids, frame_timestamps)
                }
                for agent_id in expected_ids:
                    actual_timestamp = frame_timestamp_by_id.get(agent_id)
                    expected_timestamp = expected_timestamps[agent_id]
                    if actual_timestamp is None:
                        raise ValueError(
                            f"stage-1 frame has no timestamp for active agent {agent_id!r}"
                        )
                    if not _timestamps_equal(actual_timestamp, expected_timestamp):
                        raise ValueError(
                            f"stage-1 timestamp mismatch for agent {agent_id!r}: "
                            f"artifact={actual_timestamp!r}, runtime={expected_timestamp!r}"
                        )
                    if expected_current is not None and _timestamp_number(
                        actual_timestamp,
                        field_name="agent timestamp",
                    ) > _timestamp_number(expected_current, field_name="current timestamp"):
                        raise ValueError(
                            f"stage-1 timestamp for agent {agent_id!r} is in the future"
                        )

            # Preserve the original frame key for audit/debug consumers while
            # exposing canonical parallel arrays to the alignment function.
            frame["frame_key"] = str(frame_key)
            return frame
        except ValueError as error:
            last_error = error
            if direct is not None:
                break

    detail = f": {last_error}" if last_error is not None else ""
    raise ValueError(
        f"No valid CoAlign stage-1 frame for sample {requested_idx!r}{detail}"
    ) from last_error


def refine_base_data_poses(
    base_data_dict: Mapping[Any, Any],
    frame: Mapping[str, Any],
    **alignment_args: Any,
) -> dict[str, Any]:
    """Apply one validated stage-1 frame to an intermediate-fusion sample."""

    from opencood.utils.transformation_utils import x1_to_x2

    frame = _canonical_frame(frame)
    artifact_ids = [str(value) for value in frame["cav_ids"]]
    artifact_index = {cav_id: index for index, cav_id in enumerate(artifact_ids)}
    base_ids = list(base_data_dict)
    receiver_ids = [
        cav_id for cav_id in base_ids if bool(base_data_dict[cav_id]["ego"])
    ]
    if len(receiver_ids) != 1 or receiver_ids[0] != base_ids[0]:
        raise ValueError("CoAlign requires exactly one receiver in the first position.")
    receiver_id = receiver_ids[0]
    if str(receiver_id) not in artifact_index:
        # Stage-1 pre-calculation uses a fixed receiver, while training may
        # choose another receiver for the same scenario.  Without a stage-1
        # measurement for the active receiver there is no anchored pose graph;
        # the only valid behavior is to preserve the sample exactly.
        return {
            "applied": False,
            "reason": "receiver_missing_from_stage1",
            "agent_count": 0,
        }
    active_ids = [
        cav_id for cav_id in base_ids if str(cav_id) in artifact_index
    ]

    poses = np.asarray(
        [
            base_data_dict[cav_id]["params"].get(
                "coalign_observed_lidar_pose",
                base_data_dict[cav_id]["params"]["lidar_pose"],
            )
            for cav_id in active_ids
        ],
        dtype=np.float64,
    )
    boxes = [
        frame["boxes_hwl"][artifact_index[str(cav_id)]] for cav_id in active_ids
    ]
    uncertainties = [
        frame["uncertainties"][artifact_index[str(cav_id)]] for cav_id in active_ids
    ]
    class_ids = [
        frame["class_ids"][artifact_index[str(cav_id)]] for cav_id in active_ids
    ]
    if "uncertainty_space" not in alignment_args:
        alignment_args["uncertainty_space"] = frame.get("uncertainty_space")
    refined = align_agent_poses(
        boxes,
        poses,
        uncertainties_by_agent=uncertainties,
        class_ids_by_agent=class_ids,
        **alignment_args,
    )
    ego_pose = refined[0].tolist()
    correction = np.abs(refined[:, [0, 1, 4]] - poses[:, [0, 1, 4]])
    for index, cav_id in enumerate(active_ids):
        params = base_data_dict[cav_id]["params"]
        if "coalign_observed_lidar_pose" in params:
            params["coalign_observed_lidar_pose"] = refined[index].tolist()
            transform_pose = params["coalign_observed_lidar_pose"]
        else:
            # Preserve the direct helper's historical behavior for callers
            # that construct base-data dictionaries themselves.  Dataset
            # samples always carry the explicit observed field, so native
            # clean-pose semantics remain protected in production.
            params["lidar_pose"] = refined[index].tolist()
            transform_pose = params["lidar_pose"]
        params["transformation_matrix"] = x1_to_x2(
            transform_pose, ego_pose
        )
    return {
        "applied": True,
        "agent_count": len(active_ids),
        "max_abs_pose_correction": float(correction.max(initial=0.0)),
    }
