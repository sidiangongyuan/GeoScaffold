"""UrbanIng four-class anchors with the native CoAlign direction decoder."""

from opencood.data_utils.post_processor.coalign_direction import apply_direction_bins
from opencood.data_utils.post_processor.voxel_postprocessor import VoxelPostprocessor


class CoAlignVoxelPostprocessor(VoxelPostprocessor):
    """Retain UrbanIng assignment, class scoring and NMS; decode direction bins."""

    def generate_label(self, **kwargs):
        # The legacy negative tensor contains only the final class's anchors.
        # CoAlign consumes explicit negatives, so request the existing complete,
        # class-major mask rather than repeating the last class's two channels.
        kwargs['include_all_negative_channels'] = True
        return super().generate_label(**kwargs)

    def select_decoded_boxes(self, boxes3d, selected_mask, output):
        direction = self.params['dir_args']
        logits = output.get('dir_preds', output.get('dm'))
        if logits is None:
            raise KeyError('CoAlign direction decoding requires dir_preds or dm')
        return apply_direction_bins(
            boxes3d, logits, selected_mask,
            dir_offset=float(direction['dir_offset']), num_bins=int(direction['num_bins']),
        )
