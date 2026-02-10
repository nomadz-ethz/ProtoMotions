from protomotions.robot_configs.base import (
    RobotConfig,
    RobotAssetConfig,
    ControlConfig,
    ControlType,
    SimulatorParams,
)
from protomotions.simulator.isaacgym.config import (
    IsaacGymSimParams,
    IsaacGymPhysXParams,
)
from protomotions.simulator.isaaclab.config import (
    IsaacLabSimParams,
    IsaacLabPhysXParams,
)
from protomotions.simulator.genesis.config import GenesisSimParams
from protomotions.simulator.newton.config import NewtonSimParams
from protomotions.components.pose_lib import ControlInfo
from typing import List, Dict
from dataclasses import dataclass, field

# Parameters from booster_train (https://github.com/BoosterRobotics/booster_train/blob/main/source/booster_train/booster_train/assets/robots/booster.py)
ARMATURE_6416 = 0.095625
ARMATURE_4310 = 0.0282528
ARMATURE_6408 = 0.0478125
ARMATURE_4315 = 0.0339552
ARMATURE_8112 = 0.0523908
ARMATURE_8116 = 0.0636012
ARMATURE_ROB_14 = 0.001

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_6416 = 80.
STIFFNESS_4310 = 80.
STIFFNESS_6408 = 80.
STIFFNESS_4315 = 80.
STIFFNESS_ROB_14 = 4.

DAMPING_6416 = 2.0
DAMPING_4310 = 2.0
DAMPING_6408 = 2.0
DAMPING_4315 = 2.0
DAMPING_ROB_14 = 1.

@dataclass
class K1RobotConfig(RobotConfig):

    """
    Naming convention for the links in the USDA and MJCF files:

    'trunk_link', 'Head_1', 'Left_Arm_1', 'Right_Arm_1', 'Left_Hip_Pitch', 'Right_Hip_Pitch', 'head_pitch_link', 'Left_Arm_2', 'Right_Arm_2', 'Left_Hip_Roll', 'Right_Hip_Roll', 'Left_Arm_3', 'Right_Arm_3', 'Left_Hip_Yaw', 'Right_Hip_Yaw', 'left_forearm_pitch_link', 'right_forearm_pitch_link', 'Left_Shank', 'Right_Shank', 'Left_Ankle_Cross', 'Right_Ankle_Cross', 'left_foot_link', 'right_foot_link'
    """

    common_naming_to_robot_body_names: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "all_left_foot_bodies": ["left_foot_link"],
            "all_right_foot_bodies": ["right_foot_link"],
            "all_left_hand_bodies": ["left_hand_link"],
            "all_right_hand_bodies": ["right_hand_link"],
            "head_body_name": ["Head_1"],
            "torso_body_name": ["trunk_link"],
        }
    )

    trackable_bodies_subset: List[str] = field(
        default_factory=lambda: [
            "trunk_link",
            "Head_1",
            "right_foot_link",
            "left_foot_link",
            "left_hand_link",
            "right_hand_link",
        ]
    )

    default_root_height: float = 0.57
    anchor_body_name: str = "trunk_link" 

    asset: RobotAssetConfig = field(
        default_factory=lambda: RobotAssetConfig(
            asset_file_name="mjcf/K1_22dof.xml",
            usd_asset_file_name="usd/K1/k1.usda",
            usd_bodies_root_prim_path="/World/envs/env_.*/Robot/trunk_link/",
            replace_cylinder_with_capsule=True,
            thickness=0.01,
            max_angular_velocity=1000.0,
            max_linear_velocity=1000.0,
            density=0.001,
            angular_damping=0.0,
            linear_damping=0.0,
        )
    )

    control: ControlConfig = field(
        default_factory=lambda: ControlConfig(
            control_type=ControlType.BUILT_IN_PD,
            override_control_info={

                # Legs
                ".*_Hip_Pitch": ControlInfo(
                    stiffness=STIFFNESS_6408,
                    damping=DAMPING_6408,
                    effort_limit=30.,
                    velocity_limit=8.,
                    armature=ARMATURE_6408,
                ),
                ".*_Hip_Roll": ControlInfo(
                    stiffness=STIFFNESS_4315,
                    damping=DAMPING_4315,
                    effort_limit=35.,
                    velocity_limit=12.9,
                    armature=ARMATURE_4315,
                ),
                ".*_Hip_Yaw": ControlInfo(
                    stiffness=STIFFNESS_4310,
                    damping=DAMPING_4310,
                    effort_limit=20.,
                    velocity_limit=18.,
                    armature=ARMATURE_4310,
                ),
                ".*_Knee_Pitch": ControlInfo(
                    stiffness=STIFFNESS_6416,
                    damping=DAMPING_6416,
                    effort_limit=40.,
                    velocity_limit=12.5,
                    armature=ARMATURE_6416,
                ),

                # Feet
                ".*_Ankle_Pitch": ControlInfo(
                    stiffness=30.0,
                    damping=2.0,
                    effort_limit=20.0,
                    velocity_limit=18.0,
                    armature=2.0 * ARMATURE_4310,
                ),
                ".*_Ankle_Roll": ControlInfo(
                    stiffness=30.0,
                    damping=2.0,
                    effort_limit=20.0,
                    velocity_limit=18.0,
                    armature=2.0 * ARMATURE_4310,
                ),

                # Arms
                ".*_Shoulder_Pitch": ControlInfo(
                    stiffness=STIFFNESS_ROB_14,
                    damping=DAMPING_ROB_14,
                    effort_limit=14.0,
                    velocity_limit=18.0,
                    armature=ARMATURE_ROB_14,
                ),
                ".*_Shoulder_Roll": ControlInfo(
                    stiffness=STIFFNESS_ROB_14,
                    damping=DAMPING_ROB_14,
                    effort_limit=14.0,
                    velocity_limit=18.0,
                    armature=ARMATURE_ROB_14,
                ),
                ".*_Elbow_Pitch": ControlInfo(
                    stiffness=STIFFNESS_ROB_14,
                    damping=DAMPING_ROB_14,
                    effort_limit=14.0,
                    velocity_limit=18.0,
                    armature=ARMATURE_ROB_14,
                ),
                ".*_Elbow_Yaw": ControlInfo(
                    stiffness=STIFFNESS_ROB_14,
                    damping=DAMPING_ROB_14,
                    effort_limit=14.0,
                    velocity_limit=18.0,
                    armature=ARMATURE_ROB_14,
                ),

                # Head
                ".*Head.*": ControlInfo(
                    stiffness=4.0,
                    damping=1.0,
                    effort_limit=6.0,
                    velocity_limit=20.0,
                    armature=0.001,
                ),
            },
        )
    )

    simulation_params: SimulatorParams = field(
        default_factory=lambda: SimulatorParams(
            isaacgym=IsaacGymSimParams(
                fps=100,
                decimation=2,
                substeps=2,
                physx=IsaacGymPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            isaaclab=IsaacLabSimParams(
                fps=200,
                decimation=4,
                physx=IsaacLabPhysXParams(
                    num_position_iterations=8,
                    num_velocity_iterations=4,
                    max_depenetration_velocity=1,
                ),
            ),
            genesis=GenesisSimParams(
                fps=100,
                decimation=2,
                substeps=2,
            ),
            newton=NewtonSimParams(
                fps=200,
                decimation=4,
            ),
        )
    )
