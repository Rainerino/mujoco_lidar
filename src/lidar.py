import attr
import mujoco
import numpy as np


@attr.define
class LidarSensor:
    model: mujoco.MjModel = attr.field()
    data: mujoco.MjData = attr.field()
    site_name: str = attr.field()
    ray_phi: np.ndarray = attr.field()
    ray_theta: np.ndarray = attr.field()
    site_id: int = attr.field(init=False)
    __parent_id: int = attr.field(init=False)
    cutoff_dist: float = attr.field(default=100.0)
    pcl_frame: np.ndarray = attr.field(init=False)
    
    # For Optimization
    __nray: int = attr.field(init=False)
    __dist: np.ndarray = attr.field(init=False)
    __geomid: np.ndarray = attr.field(init=False)
    
    def __attrs_post_init__(self) -> None:
        self.__nray = self.ray_phi.shape[0]
        # Find site in model. If not found, raise error
        try:
            self.site_id = self.model.site(self.site_name).id
            self.__parent_id = self.model.site(self.site_id).bodyid
        except ValueError as e:
            raise ValueError(f"Site '{self.site_name}' not found in model") from e

        if self.ray_phi.shape[0] != self.ray_theta.shape[0]:
            raise ValueError("ray_phi and ray_theta must have the same shape")
        
        # Initialize 
        self.__dist = np.full(self.__nray, self.cutoff_dist, dtype=np.float64)
        self.__geomid = np.full(self.__nray, 0, dtype=np.int32)

    def update(self) -> None:
        # Uniformly generate vec from site's pose and lidar settings
        # Note that all the vec are in the local frame.
        site_pos = self.data.site_xpos[self.site_id]
        pnt = np.array([site_pos]).T
        site_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
        x = np.cos(self.ray_phi) * np.cos(self.ray_theta)
        y = np.cos(self.ray_phi) * np.sin(self.ray_theta)
        z = np.sin(self.ray_phi)
        local_vecs = np.stack((x, y, z), axis=-1)
        world_vecs = (site_mat @ local_vecs.T).T
        world_vecs /=np.linalg.norm(world_vecs, axis=1, keepdims=True)
        world_vecs_flat = world_vecs.flatten()
    
        # Get the ray casting results
        mujoco.mj_multiRay(
            m=self.model,
            d=self.data,
            pnt=pnt,
            vec=world_vecs_flat,
            geomgroup=None,
            flg_static=1,
            bodyexclude=-1,
            geomid=self.__geomid,
            dist=self.__dist,
            nray=self.__nray,
            cutoff=self.cutoff_dist
        )
        # Calculate the point's position in local frame from vec + dist
        self.__dist[self.__geomid == -1] = self.cutoff_dist

        # Update the pcl frame with local frame data
        self.pcl_frame = local_vecs * self.__dist[:, np.newaxis]

    def get_data_in_local_frame(self) -> np.ndarray:
        return self.pcl_frame

    def get_data_in_world_frame(self) -> np.ndarray:
        pcl_local = self.get_data_in_local_frame()
        # Transform to the global frame.
        site_pos = self.data.site_xpos[self.site_id]
        site_mat = self.data.site_xmat[self.site_id].reshape(3, 3)
        pcl_world = np.add(pcl_local, site_pos)
        return pcl_world

    def exclude_geom(self, geom_id: int) -> None:
        raise NotImplementedError
