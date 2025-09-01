import mujoco
import numpy as np

# 1. Define a simple model
xml = """
<mujoco>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="2 2 0.1" rgba=".8 .8 .8 1"/>
    <body name="target_box" pos="1 0 0.5">
      <joint type="free"/>
      <geom name="box_geom" type="box" size="0.2 0.2 0.2" rgba="1 0 0 1"/>
    </body>
    <body name="target_sphere" pos="-1 0 0.5">
      <joint type="free"/>
      <geom name="sphere_geom" type="sphere" size="0.3" rgba="0 1 0 1"/>
    </body>
  </worldbody>
</mujoco>
"""

# 2. Load model and create data
model = mujoco.MjModel.from_xml_string(xml)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# 3. Define the rays with the correct data types
nray = 2
pnt = np.array([[0, 0, 0.5]], dtype=np.float64).T  # FIX 1: Use float64

vec = np.array([[-1, 0, 0], [1, 0, 0]], dtype=np.float64)  # FIX 2: Use float64

print(vec)
vec /= np.linalg.norm(vec, axis=1, keepdims=True)
vec = vec.flatten()
print(vec)
# 4. Pre-allocate the output arrays with the correct data types
cutoff = 2.0
geomid = np.zeros(nray, dtype=np.int32)  # FIX 3: Use int32 for geomid
dist = np.full(nray, cutoff, dtype=np.float64)

print(pnt.shape)
print(pnt)
print(vec.shape)
print(dist)
print(geomid)
# 5. Call mj_multiRay with all required arguments
mujoco.mj_multiRay(
    m=model,
    d=data,
    pnt=pnt,
    vec=vec,
    geomgroup=None,
    flg_static=0,
    bodyexclude=-1,
    geomid=geomid,
    dist=dist,
    nray=nray,  # FIX 4: Explicitly pass nray
    cutoff=cutoff,  # FIX 5: Explicitly pass cutoff
)


# 6. Process the results
print(f"Ray intersection results (cutoff distance = {cutoff}):")
print("-" * 40)
for i in range(nray):
    if geomid[i] != -1:
        geom_name = model.geom(geomid[i]).name
        print(f"Ray {i}: Hit geom '{geom_name}' at distance {dist[i]:.3f}")
    else:
        print(f"Ray {i}: Missed all geoms.")
