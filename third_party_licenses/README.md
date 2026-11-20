# Third-party code and licenses

GeoScaffold builds on the OpenCOOD fork of the UrbanIng-V2X repository. The top-level [LICENSE](../LICENSE), the UCLA Academic Software License of OpenCOOD, covers this code base; `OpenCOOD_LICENSE` is a copy of it.

| Source | License | Files in this repository |
|---|---|---|
| [OpenCOOD](https://github.com/DerrickXuNu/OpenCOOD) | UCLA Academic Software License (`OpenCOOD_LICENSE`) | the framework in `opencood/`, including the AttFuse, F-Cooper, V2VNet, V2X-ViT, CoBEVT, Where2comm, DiscoNet, and CoAlign fusion modules |
| [UrbanIng-V2X](https://github.com/thi-ad/UrbanIng-V2X) | MIT (`UrbanIng-V2X_LICENSE`) | the UrbanIng-V2X data loading and evaluation in `opencood/` |
| [S-AdaFusion](https://github.com/DonghaoQiao/Adaptive-Feature-Fusion-for-Cooperative-Perception) | MIT (`S-AdaFusion_LICENSE`) | `opencood/models/fusion_modules/s_adafusion_fuse.py`, `opencood/models/lidar/point_pillar_s_adafusion.py` |
| [ERMVP](https://github.com/Terry9a/ERMVP) | Apache License 2.0 (`ERMVP_LICENSE`) | `opencood/models/sub_modules/sampler.py`, `opencood/models/sub_modules/cluster.py`, `opencood/models/sub_modules/ermvp_fusion_modules.py`, `opencood/models/lidar/point_pillar_ermvp.py`, `opencood/loss/point_pillar_ermvp_loss.py` |
| [CoAlign](https://github.com/yifanlu0227/CoAlign) | UCLA Academic Software License | `opencood/models/lidar/point_pillar_coalign_native.py`, `opencood/models/sub_modules/coalign_pose.py`, `opencood/models/sub_modules/base_bev_backbone_resnet_coalign.py`, `opencood/data_utils/post_processor/coalign_voxel_postprocessor.py`, `opencood/data_utils/post_processor/coalign_direction.py`, `opencood/loss/point_pillar_coalign_loss.py` |

The S-AdaFusion, ERMVP, and CoAlign files were adapted to this code base. The HEAL pyramid fusion (`opencood/models/sub_modules/heal_pyramid.py`) was written for this code base after the [HEAL](https://github.com/yifanlu0227/HEAL) paper and configuration.
