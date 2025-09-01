# Load scene 1 and attach lidar onto the site
import mujoco
import mujoco.viewer
import os
import sys
import numpy as np
import time
from mujoco_lidar import generate_grid_scan_pattern, generate_HDL64, LivoxGenerator


# Add the parent directory to the Python path to import from src
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)
from src.lidar import LidarSensor

ray_theta, ray_phi = generate_grid_scan_pattern(
    num_ray_cols=50,
    num_ray_rows=50,
)
# ray_theta, ray_phi = generate_HDL64()
# ray_theta, ray_phi = LivoxGenerator("mid360").sample_ray_angles()

RANDOM_RANGE = 5
CUTOFF = 5

def test_scene1():
    # Get the path to the scene1.xml file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    filename = os.path.join(current_dir, "scene1.xml")
    try:
        with open(filename, "r") as f:
            original_xml = f.read()
    except FileNotFoundError:
        print(f"Error: The file '{filename}' was not found.")
        exit()
    boxes_xml = ""
    
    for i in range(ray_theta.shape[0]):
        # Unique position and color for each box
        start_pos = f"{np.random.uniform(-RANDOM_RANGE, RANDOM_RANGE)} {np.random.uniform(-RANDOM_RANGE, RANDOM_RANGE)} {np.random.uniform(0, RANDOM_RANGE/2)}"
        color = f"{np.random.uniform(0, 1)} {np.random.uniform(0, 1)} {np.random.uniform(0, 1)} 1"
        boxes_xml += f"""
            <body name="point_{i}" mocap="true" pos="{start_pos}">
                <site name="site_{i}" type="sphere" size="0.02" rgba="{color}"/>
            </body>
          """
    if "<worldbody>" in original_xml:
        full_xml = original_xml.replace("<worldbody>", f"<worldbody>{boxes_xml}", 1)
    else:
        print(f"Error: Could not find <worldbody> tag in {filename}.")
        exit()
    # Load the MuJoCo model
    model = mujoco.MjModel.from_xml_string(full_xml)
    data = mujoco.MjData(model)
    
    # Initialize lidar sensor
    lidar_sensor = LidarSensor(
        model=model,
        data=data,
        site_name="lidar_site",
        ray_theta=ray_theta,
        ray_phi=ray_phi,
        cutoff_dist=CUTOFF,
    )

    # Initalize M x N number of boxes, and visualize them.
    mocap_ids = [model.body(f"point_{i}").mocapid[0] for i in range(ray_theta.shape[0])]
    # Create viewer for visualization
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Initialize timing variables
        total_time = 0.0
        frame_count = 0
        timing_interval = 60  # Print stats every 60 frames

        # Run simulation
        while viewer.is_running():
            # Start timing the main loop
            loop_start = time.perf_counter()

            # Step the simulation
            mujoco.mj_step(model, data)

            # Update lidar sensor
            lidar_sensor.update()

            # Get data in local frame
            pcl: np.ndarray = lidar_sensor.get_data_in_world_frame()
            for i, pt in enumerate(pcl):
                mocap_id = mocap_ids[i]
                data.mocap_pos[mocap_id] = pt
            # TODO: how to visualize the point cloud?

            # Synchronize viewer
            viewer.sync()

            # End timing and calculate loop duration
            loop_end = time.perf_counter()
            loop_time = loop_end - loop_start

            # Update timing statistics
            total_time += loop_time
            frame_count += 1

            # Print timing stats periodically
            if frame_count % timing_interval == 0:
                avg_time = total_time / frame_count
                fps = 1.0 / avg_time if avg_time > 0 else 0
                print(f"Frame {frame_count}: Avg loop time: {avg_time*1000:.2f}ms, FPS: {fps:.1f}")
                print(f"Last loop time: {loop_time*1000:.2f}ms")


# Run the test
if __name__ == "__main__":
    test_scene1()
