from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SCENE_XML = REPO_ROOT / "v2" / "pedestal" / "throw_multi_scene.xml"

COLORS = ("orange", "red", "green", "blue", "purple")

COLOR_RGBA = {
    "orange": (0.95, 0.5, 0.1, 1.0),
    "red": (0.85, 0.15, 0.15, 1.0),
    "green": (0.15, 0.7, 0.2, 1.0),
    "blue": (0.15, 0.3, 0.85, 1.0),
    "purple": (0.6, 0.2, 0.8, 1.0),
}

BALL_BODIES = tuple(f"ball_{c}" for c in COLORS)
BIN_BODIES = tuple(f"bin{i}" for i in range(5))

RIGHT_JOINTS = tuple(f"openarm_right_joint{i}" for i in range(1, 8))
RIGHT_ACTUATORS = tuple(f"right_joint{i}_ctrl" for i in range(1, 8)) + ("right_finger1_ctrl",)
LEFT_ACTUATORS = tuple(f"left_joint{i}_ctrl" for i in range(1, 8)) + ("left_finger1_ctrl",)

# Finger joint: 0 = closed (tips ~9 mm apart), -0.7854 = fully open (~15 cm).
GRIPPER_OPEN = -0.7854
GRIPPER_CLOSED = 0.0
GRIP_CLOSE_CMD = -0.4  # gripper command above this counts as "closed" for failure bookkeeping

# Ball-centre point between the fingers, in openarm_right_ee_base_link frame. The palm collider
# reaches z=-0.111 and fingertips end at ~-0.155, so a 3 cm ball centres at -0.145.
GRASP_OFFSET = (0.0, 0.0, -0.145)

HOME_CTRL_LEFT = (0.0, 0.0, 0.0, 1.570796, 0.0, 0.0, 0.0, 0.0)
THROW_READY_RIGHT = (-0.720369, 2.27095, 0.290977, 1.90389, 1.23759, 0.780000, 0.558776)

TRAIN_TEMPLATES = (
    "throw the {ball} ball into the {bin} bucket",
    "throw the {ball} ball into the {bin} bin",
    "put the {ball} ball in the {bin} bin",
    "toss the {ball} ball into the {bin} bucket",
    "place the {ball} ball into the {bin} bucket",
    "throw the {ball} ball in the {bin} bin",
    "toss the {ball} ball in the {bin} bin",
    "pick up the {ball} ball and throw it into the {bin} bucket",
    "grab the {ball} ball and toss it in the {bin} bin",
    "get the {ball} ball into the {bin} bucket",
    "throw the {ball} ball to the {bin} bucket",
    "put the {ball} ball into the {bin} bucket",
)
# Never used in training; reserved for language generalization tests.
HELDOUT_TEMPLATES = (
    "could you lob the {ball} ball into the {bin} container",
    "the {ball} ball goes in the {bin} bin",
    "i want the {ball} ball in the {bin} bucket",
)
# (ball color, bin color) pairs never collected for training/validation.
HELDOUT_PAIRS = (("red", "blue"), ("green", "orange"), ("purple", "red"))

STATE_DIM = 15  # 7 qpos + 7 qvel + 1 gripper
ACTION_DIM = 8  # 7 arm + 1 gripper
CONTROL_HZ = 50
PHYSICS_DT = 0.001
N_SUBSTEPS = 20
