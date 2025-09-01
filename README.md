# Mujoco Basic Lidar API

LiDARSensor
__init__：mujoco.mjModel site_name
update：
parameters：
mjData
rays theta（水平） shape=(N,)
rays phi（垂直）shape=(N,)
return：
point in site_name's local frame shape=(N,3)，site_name坐标系下的点云的xyz坐标
N的典型值：1.8k-2w
场景mesh的面片数量的典型值：1k-10w
帧率需求：> 24fps

```python
def generate_grid_scan_pattern(num_ray_cols, num_ray_rows, theta_range=(-np.pi, np.pi), phi_range=(-np.pi/3, np.pi/3)):
    """
    生成网格状扫描模式

    参数:
        num_ray_cols: 水平方向射线数
        num_ray_rows: 垂直方向射线数

    返回:
        (ray_theta, ray_phi): 水平角和垂直角数组
    """
   # 创建网格扫描模式
    theta_grid, phi_grid = np.meshgrid(
        np.linspace(theta_range[0], theta_range[1], num_ray_cols),  # 水平角
        np.linspace(phi_range[0], phi_range[1], num_ray_rows)  # 垂直角
    )

    # 展平网格为一维数组
    ray_phi = phi_grid.flatten()
    ray_theta = theta_grid.flatten()

    # 打印扫描范围信息
    print(f"扫描模式：phi范围[{ray_phi.min():.2f}, {ray_phi.max():.2f}], theta范围[{ray_theta.min():.2f}, {ray_theta.max():.2f}]")
    return ray_theta, ray_phi
```
