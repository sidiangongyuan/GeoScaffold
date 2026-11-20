"""PointPillar detector using the official S-AdaFusion BEV reducer."""

from __future__ import annotations

from opencood.models.fusion_modules.s_adafusion_fuse import SpatialFusion
from opencood.models.lidar.point_pillar_fcooper import PointPillarFCooper


class PointPillarSAdaFusion(PointPillarFCooper):
    """Current UrbanIng PointPillar heads with S-AdaFusion source aggregation.

    The parent supplies the existing four-class ``psm``/``rm`` heads and the
    shared training-only GeoScaffold integration.  Only its fusion operator is
    replaced, so the detector decoder and evaluation path remain unchanged.
    """

    def __init__(self, args):
        super().__init__(args)
        self.fusion_net = SpatialFusion()

