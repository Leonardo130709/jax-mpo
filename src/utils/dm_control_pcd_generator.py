from typing import NamedTuple, Iterable, Tuple, List, TypedDict, Union

import numpy as np
from dm_control.mujoco.engine import Physics

BAD_GEOMS = ("wall", "ground", "floor")
PointCloud = np.ndarray


class IntrinsicParams(NamedTuple):
    width: float
    height: float
    fx: float
    fy: float
    cx: float
    cy: float


class CameraParams(TypedDict):
    width: float
    height: float
    camera_id: Union[int, str]


def partition_geoms_by_name(physics: Physics,
                            names: Iterable[str],
                            all_geom_ids: Iterable[int],
                            ) -> Tuple[List[int], List[int]]:
    """Splits geom_ids in two groups by input names."""
    names = tuple(names)
    in_and_out = ([], [])
    for _id in all_geom_ids:
        geom_name = physics.model.id2name(_id, "geom")
        in_and_out[geom_name not in names].append(_id)

    return in_and_out


def point_cloud_from_depth_map(depth: np.ndarray,
                               intrinsic_params: IntrinsicParams
                               ) -> PointCloud:
    """Makes point cloud from the depth map."""
    assert depth.shape == (intrinsic_params.height, intrinsic_params.width), \
        "Incompatible shapes"
    yy, xx = np.mgrid[:intrinsic_params.height, :intrinsic_params.width]
    xx = (xx - intrinsic_params.cx) / intrinsic_params.fx * depth
    yy = (yy - intrinsic_params.cy) / intrinsic_params.fy * depth

    return np.stack([xx, yy, depth], axis=-1).reshape(-1, 3)


def intrinsic_params_from_physics(physics: Physics,
                                  camera_params: CameraParams
                                  ) -> IntrinsicParams:
    """Conversion to explicit type."""
    width, height, camera_id = map(
        camera_params.get,
        ("width", "height", "camera_id")
    )
    fov = physics.named.model.cam_fovy[camera_id]
    f = (1. / np.tan(np.deg2rad(fov) / 2.)) * height / 2.0
    cx = (width - 1) / 2.
    cy = (height - 1) / 2.

    return IntrinsicParams(
        height=height, width=width,
        fx=-f, fy=f,
        cx=cx, cy=cy
    )


class PointCloudGenerator:
    """Extracts point cloud from camera view and physics."""

    def __init__(self,
                 pn_number: int,
                 cameras_params: Iterable[CameraParams],
                 stride: int = -1,
                 apply_translation: bool = False
                 ):
        self.stride = stride
        self.pn_number = pn_number
        self._cams = tuple(cameras_params)
        self._apply_translation = apply_translation

    def __call__(self, physics):
        """Merge cameras views to single point cloud."""
        pcd = np.concatenate([
            self.call(physics, cam) for cam in self._cams
        ])
        pcd = self._apply_stride(pcd)

        return self._to_fixed_number(pcd)

    def call(self,
             physics: Physics,
             render_kwargs: CameraParams
             ) -> PointCloud:
        """Per camera pcd generation."""
        depth = physics.render(depth=True, **render_kwargs)
        intrinsic_params = intrinsic_params_from_physics(physics, render_kwargs)
        pcd = point_cloud_from_depth_map(depth, intrinsic_params)
        mask = self._mask(physics, pcd, render_kwargs)
        pcd = pcd[mask]

        data = physics.named.data
        rot = data.cam_xmat[render_kwargs["camera_id"]].reshape(3, 3)
        pos = data.cam_xpos[render_kwargs["camera_id"]]
        pcd = - pcd @ rot.T
        if self._apply_translation:
            pcd += pos

        return pcd

    def _apply_stride(self, pcd: PointCloud) -> PointCloud:
        if self.stride < 0:
            adaptive_stride = pcd.shape[0] // self.pn_number
            return pcd[::max(adaptive_stride, 1)]
        else:
            return pcd[::self.stride]

    def _mask(self,
              physics: Physics,
              point_cloud: PointCloud,
              render_kwargs: CameraParams
              ) -> PointCloud:
        """Segmentation mask: cuts floor, walls, etc."""
        segmentation = physics.render(segmentation=True, **render_kwargs)
        geom_ids = np.unique(segmentation[..., 0]).tolist()
        _, geom_ids = partition_geoms_by_name(physics, BAD_GEOMS, geom_ids)
        geom_ids.remove(-1)  # sky renders infinity
        segmentation = np.isin(segmentation[..., 0].flatten(), geom_ids)

        truncate = point_cloud[..., 2] < 10.
        return np.logical_and(segmentation, truncate)

    def _to_fixed_number(self, pcd: PointCloud) -> PointCloud:
        n = pcd.shape[0]
        if n == 0:
            pcd = np.zeros((self.pn_number, 3))
        elif n <= self.pn_number:
            pcd = np.pad(pcd, ((0, self.pn_number - n), (0, 0)), mode="edge")
        else:
            pcd = np.random.permutation(pcd)[:self.pn_number]

        return pcd
