"""ResNet BEV decoder used by the author CoAlign implementation.

The repository already contains the same residual BEV implementation under
``res_bev_backbone``.  This small, explicitly named adapter keeps the CoAlign
model path source-local and makes the architectural contract visible: three
residual levels are encoded independently, fused at each level, and decoded
with the corresponding upsampling blocks.
"""

from __future__ import annotations

from opencood.models.sub_modules.res_bev_backbone import ResBEVBackbone


class ResNetBEVBackbone(ResBEVBackbone):
    """Author-compatible three-level residual BEV backbone.

    ``ResBEVBackbone`` uses ``ResNetLayers`` with ``BasicBlock`` and exposes
    ``get_multiscale_feature``/``decode_multiscale_feature``, which is the
    interface used by the official CoAlign ``PointPillarBaselineMultiscale``.
    The alias is intentionally separate from the generic backbone so a future
    CoAlign-specific change cannot silently alter other methods.
    """

    pass

