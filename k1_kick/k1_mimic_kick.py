"""
Adapted from Tutorial 7: DeepMimic Agent but with Booster K1 and kicking motion.
"""

import argparse

parser = argparse.ArgumentParser()

# Simulator selection and options
parser.add_argument(
    "--simulator",
    type=str,
    default="isaaclab",
    help="Simulator to use (e.g., 'isaacgym', 'isaaclab', 'newton', 'genesis')",
)
parser.add_argument(
    "--cpu-only",
    action="store_true",
    default=False,
    help="Use CPU only for simulation (experimental, GPU is default)",
)
parser.add_argument(
    "--headless",
    action="store_true",
    default=False,
    help="Run simulator in headless mode (no rendering, faster performance)",
)

# Training parameters
parser.add_argument(
    "--num-envs",
    type=int,
    default=32,
    help="Number of parallel environments to run",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=30,
    help="Number of training epochs",
)
parser.add_argument(
    "--steps",
    type=int,
    default=32,
    help="Number of steps per environment per update",
)

args = parser.parse_args()

# Import simulator before torch - isaacgym/isaaclab must be imported before torch
# This also returns AppLauncher if using isaaclab, None otherwise
from protomotions.utils.simulator_imports import import_simulator_before_torch  # noqa: E402

AppLauncher = import_simulator_before_torch(args.simulator)

# Now safe to import everything else including torch
from protomotions.simulator.base_simulator.config import SimulatorConfig  # noqa: E402
from protomotions.envs.base_env.env import BaseEnv  # noqa: E402
from protomotions.envs.base_env.config import EnvConfig, RewardComponentConfig, TerminationComponentConfig  # noqa: E402
from protomotions.envs.motion_manager.config import MimicMotionManagerConfig  # noqa: E402
from protomotions.envs.obs.observation_component import ObservationComponentConfig  # noqa: E402
from protomotions.envs.control.mimic_control import MimicControlConfig  # noqa: E402
from protomotions.envs.obs import (  # noqa: E402
    max_coords_obs_factory,
    previous_actions_factory,
    mimic_target_poses_max_coords_factory,
)
from protomotions.envs.rewards import mean_squared_error_exp, rotation_error_exp, norm  # noqa: E402
from protomotions.envs.terminations import max_joint_err  # noqa: E402
from protomotions.components.motion_lib import MotionLibConfig  # noqa: E402
from protomotions.components.terrains.config import TerrainConfig  # noqa: E402
from protomotions.agents.ppo.config import (  # noqa: E402
    PPOAgentConfig,
    PPOActorConfig,
    PPOModelConfig,
)
from protomotions.agents.common.config import MLPWithConcatConfig, MLPLayerConfig  # noqa: E402
from protomotions.agents.base_agent.config import OptimizerConfig  # noqa: E402
from protomotions.agents.evaluators.config import MimicEvaluatorConfig  # noqa: E402
from protomotions.utils.hydra_replacement import get_class  # noqa: E402
import torch  # noqa: E402
from pathlib import Path  # noqa: E402

device = torch.device("cuda:0") if not args.cpu_only else torch.device("cpu")

# Import factory functions
from protomotions.simulator.factory import simulator_config  # noqa: E402
from protomotions.robot_configs.factory import robot_config  # noqa: E402

robot_cfg = robot_config("k1")

print("\n=== Robot Configuration ===")
print("Robot type: k1")
print(f"Robot config class: {type(robot_cfg).__name__}")
print(f"Number of actions: {robot_cfg.number_of_actions}")
print(f"Number of DOFs: {robot_cfg.kinematic_info.num_dofs}")
print(f"Number of bodies: {robot_cfg.kinematic_info.num_bodies}")
print(f"Contact bodies: {robot_cfg.contact_bodies}")

# Extra simulator parameters
extra_simulator_params = {}
if args.simulator == "isaaclab":

    app_launcher_flags = {"headless": args.headless, 
                          "device": str(device),
                          }
    app_launcher = AppLauncher(app_launcher_flags)
    simulation_app = app_launcher.app
    extra_simulator_params["simulation_app"] = simulation_app

# Create simulator configuration
simulator_cfg: SimulatorConfig = simulator_config(
    args.simulator,
    robot_cfg,
    headless=args.headless,
    num_envs=args.num_envs,
    experiment_name="k1_mimic_kick",
)

print("\n=== Simulator Configuration ===")
print(f"Simulator type: {args.simulator}")
print(f"Simulator class: {get_class(simulator_cfg._target_).__name__}")
print(f"Number of environments: {simulator_cfg.num_envs}")
print(f"Device: {device}")
print(f"Headless: {simulator_cfg.headless}")

# Motion file for imitation learning
motion_file = "gmr_proto/kick_soogon_2.motion"

print("\n=== DeepMimic Configuration ===")
print(f"Motion file: {motion_file}")

# Configure modular components - this is key for imitation learning
print("\n=== Modular Component Configuration ===")
print("Using modular component system for DeepMimic:")
print("  → MimicControl: Manages reference motion tracking")
print("  → Observation components: Robot state + target poses")
print("  → Reward components: Tracking rewards for imitation")
print("  → Termination components: End on tracking failure")

# Control component - MimicControl manages reference motion
control_components = {
    "mimic": MimicControlConfig(
        bootstrap_on_episode_end=True,  # Continue at end of motion
    )
}

# Observation components - using factory functions
observation_components = {
    # Current robot state
    "max_coords_obs": max_coords_obs_factory(),
    # Previous actions
    "previous_actions": previous_actions_factory(),
    # Mimic target poses - reference motion for policy to track
    "mimic_target_poses": mimic_target_poses_max_coords_factory(
        with_velocities=True,
        num_future_steps=1,
    ),
}

# Reward components - tracking rewards for imitation learning
reward_components = {
    "action_smoothness": RewardComponentConfig(
        function=norm,
        variables={"x": "current_actions - previous_actions"},
        weight=-0.02,
    ),
    "position_tracking": RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_pos",
            "ref_x": "ref_state_rigid_body_pos",
            "coefficient": -100.0,
        },
        weight=0.5,
    ),
    "rotation_tracking": RewardComponentConfig(
        function=rotation_error_exp,
        variables={
            "q": "current_state_rigid_body_rot",
            "ref_q": "ref_state_rigid_body_rot",
            "coefficient": -5.0,
        },
        weight=0.3,
    ),
    "velocity_tracking": RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_vel",
            "ref_x": "ref_state_rigid_body_vel",
            "coefficient": -0.5,
        },
        weight=0.1,
    ),
    "angular_velocity_tracking": RewardComponentConfig(
        function=mean_squared_error_exp,
        variables={
            "x": "current_state_rigid_body_ang_vel",
            "ref_x": "ref_state_rigid_body_ang_vel",
            "coefficient": -0.1,
        },
        weight=0.1,
    ),
}

# Termination components - end episode on tracking failure
termination_components = {
    "tracking_error": TerminationComponentConfig(
        function=max_joint_err,
        variables={
            "current_rigid_body_pos": "current_state_rigid_body_pos",
            "ref_rigid_body_pos": "ref_state_rigid_body_pos",
            "threshold": 0.5,  # Terminate if any joint > 0.5m from reference
        },
    ),
}

print("\nControl Components:")
print("  - 'mimic': MimicControl for reference motion management")
print("\nObservation Components:")
print("  - 'max_coords_obs': Current robot state")
print("  - 'previous_actions': Action history")
print("  - 'mimic_target_poses': Reference poses to track")
print("\nReward Components:")
print("  - Position, rotation, velocity tracking + action smoothness")
print("\nTermination Components:")
print("  - 'tracking_error': End episode if tracking error > 0.5m")

from protomotions.components.scene_lib import (  # noqa: E402
    ObjectOptions,
    MeshSceneObject,
    Scene,
    SceneLibConfig,
    SceneLib,
)

scene = Scene(humanoid_motion_id=0)

# Create environment configuration with modular components
env_config = EnvConfig(
    max_episode_length=300,
    # Modular components
    control_components=control_components,
    observation_components=observation_components,
    reward_components=reward_components,
    termination_components=termination_components,
    # Motion manager configuration
    motion_manager=MimicMotionManagerConfig(
        init_start_prob=1.0,  # Always start from beginning for consistent training
        resample_on_reset=False,  # Reset to current motion time instead of resampling
    ),
)

print("\n=== Environment Configuration ===")
print("Environment type: BaseEnv with modular MimicControl")
print(f"Episode length: {env_config.max_episode_length}")
print(f"Control components: {list(control_components.keys())}")
print(f"Observation components: {list(observation_components.keys())}")
print(f"Reward components: {list(reward_components.keys())}")
print(f"Termination components: {list(termination_components.keys())}")
print(f"Resample on reset: {env_config.motion_manager.resample_on_reset}")


# Create terrain with simple flat configuration
from protomotions.components.terrains.terrain import Terrain  # noqa: E402
from protomotions.simulator.base_simulator.utils import convert_friction_for_simulator  # noqa: E402

terrain_config = TerrainConfig()

# Convert friction settings for the specific simulator
# Newton requires CombineMode.MAX, IsaacGym requires CombineMode.AVERAGE
# This utility handles the conversion automatically
terrain_config, simulator_cfg = convert_friction_for_simulator(terrain_config, simulator_cfg)

terrain = Terrain(config=terrain_config, num_envs=simulator_cfg.num_envs, device=device)

scene_lib_config = SceneLibConfig(scene_file=None)
scene_lib = SceneLib(
    config=scene_lib_config,
    num_envs=simulator_cfg.num_envs,
    scenes=[scene],
    device=device,
    terrain=terrain,
)

motion_lib_config = MotionLibConfig(motion_file=motion_file)
from protomotions.components.motion_lib import MotionLib  # noqa: E402

motion_lib = MotionLib(config=motion_lib_config, device=device)

from protomotions.utils.hydra_replacement import get_class  # noqa: E402

SimulatorClass = get_class(simulator_cfg._target_)
simulator = SimulatorClass(
    config=simulator_cfg,
    robot_config=robot_cfg,
    terrain=terrain,
    scene_lib=scene_lib,
    device=device,
    **extra_simulator_params,
)

# Create the environment with modular components
env = BaseEnv(
    config=env_config,
    robot_config=robot_cfg,
    device=device,
    simulator=simulator,
    motion_lib=motion_lib,
    terrain=terrain,
    scene_lib=scene_lib,
)

print("\n=== Environment Initialization ===")
print("BaseEnv with MimicControl created successfully")
print(f"Motion library loaded: {env.motion_lib is not None}")
print(f"Motion manager type: {type(env.motion_manager).__name__}")
print(f"Control components: {list(env.control_manager.components.keys())}")

# Reset environment and get initial observations
print("\n=== Environment Reset ===")
env.reset()
print("Environment reset completed")

obs = env.get_obs()
print(f"Observations: {obs is not None}")
print(f"Motion IDs: {env.motion_manager.motion_ids}")
print(f"Motion times: {env.motion_manager.motion_times}")

# Analyze observation structure
print("\n=== Observation Structure from Reset ===")
print(f"Observation keys from reset: {list(obs.keys())}")
for key, value in obs.items():
    print(
        f"  '{key}': shape {value.shape}, range [{value.min().item():.3f}, {value.max().item():.3f}]"
    )

# Now create the PPO agent configuration
print("\n=== PPO Agent Configuration ===")

# Define observation keys used by both actor and critic
obs_keys = ["max_coords_obs", "mimic_target_poses"]

# Actor configuration - maps observations to actions
# Uses MLPWithConcatConfig: concatenates all observation keys and processes through MLP
actor_config = PPOActorConfig(
    num_out=robot_cfg.kinematic_info.num_dofs,
    actor_logstd=-2.9,  # Initial log standard deviation for action noise
    in_keys=obs_keys,  # Observation keys to process
    mu_key="actor_trunk_out",  # Output key for the mean action
    mu_model=MLPWithConcatConfig(
        in_keys=obs_keys,  # Same observation keys
        out_keys=["actor_trunk_out"],  # Output key
        normalize_obs=True,  # Normalize observations
        norm_clamp_value=5,  # Clamp normalized values
        num_out=robot_cfg.number_of_actions,  # Output size (robot DOFs)
        layers=[  # Network architecture
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=512, activation="relu"),
            MLPLayerConfig(units=256, activation="relu"),
        ],
        output_activation="tanh",  # Bound actions to [-1, 1]
    ),
)

# Critic configuration - maps observations to value estimates
# Uses same MLPWithConcatConfig pattern as actor
critic_config = MLPWithConcatConfig(
    in_keys=obs_keys,  # Same observation keys as actor
    out_keys=["value"],  # Output key for value estimate
    normalize_obs=True,  # Normalize observations
    norm_clamp_value=5,  # Clamp normalized values
    num_out=1,  # Single value output
    layers=[  # Slightly smaller network for critic
        MLPLayerConfig(units=512, activation="relu"),
        MLPLayerConfig(units=256, activation="relu"),
    ],
)

print("Actor configuration:")
print(f"  - Output size: {actor_config.num_out} (robot DOFs)")
print(f"  - Log std: {actor_config.actor_logstd}")
print(f"  - Input keys: {actor_config.in_keys}")
print(f"  - Network layers: {len(actor_config.mu_model.layers)}")

print("Critic configuration:")
print(f"  - Output size: {critic_config.num_out} (value estimate)")
print(f"  - Input keys: {critic_config.in_keys}")
print(f"  - Network layers: {len(critic_config.layers)}")

# Create PPO agent configuration
agent_config = PPOAgentConfig(
    model=PPOModelConfig(
        in_keys=obs_keys,  # Observation keys for the model
        out_keys=["action", "mean_action", "neglogp", "value"],  # Output keys
        actor=actor_config,
        critic=critic_config,
        actor_optimizer=OptimizerConfig(
            _target_="torch.optim.Adam",
            lr=2e-5,  # Learning rate for actor
        ),
        critic_optimizer=OptimizerConfig(
            _target_="torch.optim.Adam",
            lr=1e-4,  # Learning rate for critic (higher than actor)
        ),
    ),
    batch_size=args.num_envs * args.steps,  # envs * steps
    training_max_steps=args.epochs * args.num_envs * args.steps,  # epochs * envs * steps
    num_steps=args.steps,  # Steps per rollout
    num_mini_epochs=4,  # Mini epochs per update
    gradient_clip_val=50.0,  # Gradient clipping for stability
    clip_critic_loss=True,  # Clip critic loss for stability
    evaluator=MimicEvaluatorConfig(
        eval_metric_keys=["gt_err", "gr_err"]  # Key metrics to track
    ),
)

print("\n=== PPO Agent Configuration ===")
print("Agent type: PPO (Proximal Policy Optimization)")
print(f"Batch size: {agent_config.batch_size}")
print(f"Training max steps: {agent_config.training_max_steps}")
print(f"Steps per rollout: {agent_config.num_steps}")
print(f"Mini epochs per update: {agent_config.num_mini_epochs}")
print(f"Actor learning rate: {agent_config.model.actor_optimizer.lr}")
print(f"Critic learning rate: {agent_config.model.critic_optimizer.lr}")
print(f"Gradient clipping: {agent_config.gradient_clip_val}")
print(f"Evaluation metrics: {agent_config.evaluator.eval_metric_keys}")

print("\nAgent configuration finalized:")
print(f"  - Model input keys: {agent_config.model.in_keys}")
print(f"  - Model output keys: {agent_config.model.out_keys}")
print(f"  - Actor output size: {agent_config.model.actor.num_out}")
print(f"  - Critic output size: {agent_config.model.critic.num_out}")

from lightning.fabric import Fabric  # noqa: E402
from protomotions.utils.fabric_config import FabricConfig  # noqa: E402

fabric_config = FabricConfig(
    devices=1,
    num_nodes=1,
    loggers=[],
    callbacks=[],
)

from dataclasses import asdict  # noqa: E402
fabric: Fabric = Fabric(**asdict(fabric_config))
fabric.launch()

print("\n=== Fabric Configuration ===")
print(f"Fabric accelerator: {fabric_config.accelerator}")
print(f"Fabric device: {fabric.device}")
print(f"Fabric precision: {fabric_config.precision}")

# Create the agent with Fabric
from protomotions.agents.ppo.agent import PPO  # noqa: E402

agent = PPO(
    fabric=fabric,
    env=env,
    config=agent_config,
    root_dir=Path("/output/k1_mimic_kick_output"),  # Directory for saving checkpoints
)
agent.setup()

print("\n=== Agent Initialization ===")
print("PPO agent created successfully")
print(f"Agent device: {agent.device}")
print(f"Max epochs calculated: {agent.max_epochs}")
print(f"Agent has actor: {hasattr(agent, 'actor')}")
print(f"Agent has critic: {hasattr(agent, 'critic')}")
print(f"Training will run for {agent.max_epochs} epochs")

# Reset environment and get initial observations
print("\n=== Environment and Agent Reset ===")
env.reset()
obs = env.get_obs()

print("Environment reset completed")
print(f"Observation keys from reset: {list(obs.keys())}")

# Show how agent processes observations
print("\n=== Agent Action Generation ===")

from tensordict import TensorDict  # noqa: E402

with torch.no_grad():  # No gradients needed for inference

    # * Convert to TensorDict to prevent error due to no batch_size.
    obs_tensordict = TensorDict(
        {key: torch.as_tensor(val) for key, val in obs.items()},
        batch_size=[args.num_envs],
    )
    agent_outs = agent.model(obs_tensordict)

print("Agent processed observations:")
print(f"  - Input observation keys: {list(obs.keys())}")
print(f"  - Generated actions shape: {agent_outs['action'].shape}")
print(f"  - Action log probabilities shape: {agent_outs['neglogp'].shape}")
print(f"  - Value estimates shape: {agent_outs['value'].shape}")
print(f"  - Actions range: [{agent_outs['action'].min().item():.3f}, {agent_outs['action'].max().item():.3f}]")
print(f"  - Values range: [{agent_outs['value'].min().item():.3f}, {agent_outs['value'].max().item():.3f}]")

# Run actual training using the agent's fit function
print("\n=== Starting Agent Training ===")
print(f"Running actual PPO training for {args.epochs} epochs:")
print("  - Agent will learn to imitate the reference motion")
print("  - Training progress will be displayed")
print("  - You can watch the robot improve over time")
print("Camera controls during training:")
print("  L - start/stop recording")
print("  ; - cancel recording")
print("  O - toggle camera target")
print("  Q - close simulator")

try:
    # Use the agent's fit function for training
    # This runs the complete PPO training loop
    # Training parameters are set in the agent_config above
    agent.fit()

    print("\nTraining completed successfully!")

except KeyboardInterrupt:
    print("\nTraining stopped by user")
finally:
    env.close()