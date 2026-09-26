import mujoco
import mujoco.viewer
import math

model = mujoco.MjModel.from_xml_path("v2/pedestal/throw_multi_scene.xml")
data = mujoco.MjData(model)

integral_1 = 0
integral_2 = 0.0
integral_3 = 0.0
integral_limit = 20.0  # anti-windup clamp, tune this

prev_error2 = 0.0
prev_error3 = 0.0
curr_time = 0


with mujoco.viewer.launch_passive(model, data) as viewer:

    while viewer.is_running():
        mujoco.mj_step(model, data)


        viewer.sync()