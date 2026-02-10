"""
Convert GMR pickle motion files to ProtoMotions .motion format.
Written with the help of LLM Agent and tested using motion_libs_visualizer.py to verify correctness of the output motions.

Usage:
# GMR->Proto
python data/scripts/convert_gmr_to_proto.py main   gmr_results/   gmr_proto   --input-fps 60 --output-fps 30   --mjcf-path protomotions/data/assets/mjcf/K1_22dof.xml   --root-rot-format xyzw   --joint-position-units radians   --yaml-output-name motions.yaml --ignore-motion-filter --ignore-motion-filter
# Proto->MotionLib
python protomotions/components/motion_lib.py --motion-path gmr_proto/motions.yaml --output-file gmr_proto/gmr.pt
# Run visualizer
python examples/motion_libs_visualizer.py --motion_files gmr_proto/gmr.pt --robot k1 --simulator isaaclab
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from pathlib import Path
import hashlib
import json
import os
import pickle

import numpy as np
import torch
import typer
import yaml
from scipy.spatial.transform import Rotation as R

from protomotions.components.pose_lib import (
	extract_kinematic_info,
	fk_batch_mjcf_with_velocities,
)
from protomotions.simulator.base_simulator.simulator_state import RobotState
from contact_detection import compute_contact_labels_from_pos_and_vel
from motion_filter import passes_exclude_motion_filter

app = typer.Typer(pretty_exceptions_enable=True)

# GMR Motor names (in order for K1) - from GMR configuration
# NOTE: K1_22dof.xml HAS head joints! Names must match exact MJCF capitalization.
GMR_K1_MOTOR_NAMES = [
    "AAHead_Yaw",                 # 0 - matches K1 DOF 0
    "Head_Pitch",                 # 1 - matches K1 DOF 1
    "ALeft_Shoulder_Pitch",        # 2
    "Left_Shoulder_Roll",          # 3
    "Left_Elbow_Pitch",            # 4
    "Left_Elbow_Yaw",              # 5
    "ARight_Shoulder_Pitch",       # 6
    "Right_Shoulder_Roll",         # 7
    "Right_Elbow_Pitch",           # 8
    "Right_Elbow_Yaw",             # 9
    "Left_Hip_Pitch",              # 10
    "Left_Hip_Roll",               # 11
    "Left_Hip_Yaw",                # 12
    "Left_Knee_Pitch",             # 13
    "Left_Ankle_Pitch",            # 14
    "Left_Ankle_Roll",             # 15
    "Right_Hip_Pitch",             # 16
    "Right_Hip_Roll",              # 17
    "Right_Hip_Yaw",               # 18
    "Right_Knee_Pitch",            # 19
    "Right_Ankle_Pitch",           # 20
    "Right_Ankle_Roll",            # 21
]


def _as_numpy(value: object) -> np.ndarray:
	if isinstance(value, np.ndarray):
		return value
	if torch.is_tensor(value):
		return value.detach().cpu().numpy()
	return np.asarray(value)


def _stack_frames(frames: Sequence[Sequence[object]]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	base_translation = np.stack([_as_numpy(frame[0]) for frame in frames], axis=0)
	base_rotation = np.stack([_as_numpy(frame[1]) for frame in frames], axis=0)
	joint_positions = np.stack([_as_numpy(frame[2]) for frame in frames], axis=0)
	return base_translation, base_rotation, joint_positions


def _load_joint_names(path: Path) -> List[str]:
	if not path.exists():
		raise FileNotFoundError(f"Joint name file not found: {path}")

	if path.suffix in {".yaml", ".yml"}:
		with open(path, "r") as f:
			data = yaml.safe_load(f)
		if isinstance(data, dict) and "joint_names" in data:
			return list(data["joint_names"])
		if isinstance(data, list):
			return list(data)
		raise ValueError("YAML joint name file must be a list or have a 'joint_names' key")

	if path.suffix == ".json":
		with open(path, "r") as f:
			data = json.load(f)
		if isinstance(data, dict) and "joint_names" in data:
			return list(data["joint_names"])
		if isinstance(data, list):
			return list(data)
		raise ValueError("JSON joint name file must be a list or have a 'joint_names' key")

	with open(path, "r") as f:
		return [line.strip() for line in f if line.strip()]


def _resolve_field(data: Dict[str, object], keys: Iterable[str]) -> Optional[object]:
	for key in keys:
		if key in data:
			return data[key]
	return None


def _load_gmr_pkl(
	pkl_path: Path,
	base_translation_key: Optional[str],
	base_rotation_key: Optional[str],
	joint_positions_key: Optional[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
	with open(pkl_path, "rb") as f:
		data = pickle.load(f)

	if isinstance(data, dict):
		base_translation = None
		base_rotation = None
		joint_positions = None

		if base_translation_key:
			base_translation = data.get(base_translation_key)
		if base_rotation_key:
			base_rotation = data.get(base_rotation_key)
		if joint_positions_key:
			joint_positions = data.get(joint_positions_key)

		if base_translation is None:
			base_translation = _resolve_field(
				data,
				[
					"robot_base_translation",
					"base_translation",
					"root_pos",
					"base_pos",
					"translation",
				],
			)
		if base_rotation is None:
			base_rotation = _resolve_field(
				data,
				[
					"robot_base_rotation",
					"base_rotation",
					"root_rot",
					"base_rot",
					"rotation",
				],
			)
		if joint_positions is None:
			joint_positions = _resolve_field(
				data,
				[
					"robot_joint_positions",
					"joint_positions",
					"dof_pos",
					"joint_pos",
					"joints",
				],
			)

		if base_translation is None or base_rotation is None or joint_positions is None:
			raise ValueError(
				"Missing required fields in pickle. Provide explicit key names or "
				"use standard keys for base translation, base rotation, and joint positions."
			)

		return _as_numpy(base_translation), _as_numpy(base_rotation), _as_numpy(joint_positions)

	if isinstance(data, (list, tuple)):
		if len(data) == 3:
			return _as_numpy(data[0]), _as_numpy(data[1]), _as_numpy(data[2])
		if len(data) > 0 and isinstance(data[0], (list, tuple)) and len(data[0]) == 3:
			return _stack_frames(data)

	raise ValueError("Unsupported pickle format. Expected dict or list of frame tuples.")


def _convert_root_rotation(
	base_rotation: np.ndarray,
	root_rot_format: str,
) -> np.ndarray:
	root_rot_format = root_rot_format.lower()

	if root_rot_format in {"wxyz", "quat_wxyz"}:
		return base_rotation
	if root_rot_format in {"xyzw", "quat_xyzw"}:
		return base_rotation[..., [3, 0, 1, 2]]
	if root_rot_format in {"rotmat", "matrix", "rotation_matrix"}:
		rot = R.from_matrix(base_rotation)
		quat_xyzw = rot.as_quat()
		return quat_xyzw[..., [3, 0, 1, 2]]
	if root_rot_format in {"euler_xyz", "euler"}:
		rot = R.from_euler("xyz", base_rotation, degrees=False)
		quat_xyzw = rot.as_quat()
		return quat_xyzw[..., [3, 0, 1, 2]]

	raise ValueError(f"Unsupported root rotation format: {root_rot_format}")


def _apply_y_up_to_z_up(
	root_pos: np.ndarray,
	root_rot_wxyz: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
	rot = R.from_euler("x", -np.pi / 2.0, degrees=False)
	root_pos_rot = rot.apply(root_pos)
	quat_xyzw = rot.as_quat()
	root_rot_xyzw = root_rot_wxyz[..., [1, 2, 3, 0]]
	root_rot_xyzw = (R.from_quat(quat_xyzw) * R.from_quat(root_rot_xyzw)).as_quat()
	root_rot_wxyz = root_rot_xyzw[..., [3, 0, 1, 2]]
	return root_pos_rot, root_rot_wxyz


def _compute_dof_vel(dof_pos: torch.Tensor, fps: int) -> torch.Tensor:
	if dof_pos.shape[0] < 2:
		return torch.zeros_like(dof_pos)
	dof_vel = torch.zeros_like(dof_pos)
	dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) * fps
	dof_vel[-1] = dof_vel[-2]
	return dof_vel


def _downsample(data: np.ndarray, factor: int) -> np.ndarray:
	if factor == 1:
		return data
	return data[::factor]


def _iter_pkl_files(input_path: Path) -> Iterable[Path]:
	if input_path.is_file() and input_path.suffix == ".pkl":
		yield input_path
		return
	for root, _, files in os.walk(input_path):
		for file_name in files:
			if file_name.endswith(".pkl"):
				yield Path(root) / file_name


@app.command()
def print_mjcf_dofs(mjcf_path: Path = Path("protomotions/data/assets/mjcf/K1_22dof.xml")):
	"""Print DOF names and count from MJCF file to help with joint mapping."""
	kinematic_info = extract_kinematic_info(str(mjcf_path))
	print(f"\n=== MJCF DOF Information ({mjcf_path}) ===")
	print(f"Total DOFs (including root): {kinematic_info.nq}")
	print(f"Non-root DOFs: {kinematic_info.num_dofs}")
	print(f"\nDOF Names (non-root only):")
	for i, name in enumerate(kinematic_info.dof_names):
		print(f"  {i}: {name}")
	print(f"\nGMR Motor Names (for reference):")
	for i, name in enumerate(GMR_K1_MOTOR_NAMES):
		print(f"  {i}: {name}")


@app.command()
def print_gmr_k1_mapping(mjcf_path: Path = Path("protomotions/data/assets/mjcf/K1_22dof.xml")):
	"""Print or create GMR-to-K1 joint mapping based on motor names."""
	kinematic_info = extract_kinematic_info(str(mjcf_path))
	reorder_indices = []
	print(f"\n=== GMR to K1 Joint Mapping ===")
	for gmr_idx, gmr_name in enumerate(GMR_K1_MOTOR_NAMES):
		try:
			mjcf_idx = kinematic_info.dof_names.index(gmr_name)
			reorder_indices.append(mjcf_idx)
			print(f"GMR motor {gmr_idx:2d} ({gmr_name:25s}) -> MJCF DOF {mjcf_idx:2d}")
		except ValueError:
			print(f"WARNING: GMR motor {gmr_idx:2d} ({gmr_name:25s}) NOT FOUND in MJCF DOFs")
			reorder_indices.append(-1)
	print(f"\nReorder indices for --joint-names-path: {reorder_indices}")


@app.command()
def main(
	gmr_pkl_path: Path,
	output_path: Path,
	input_fps: int = 30,
	output_fps: int = 30,
	mjcf_path: Path = Path("protomotions/data/assets/mjcf/K1_22dof.xml"),
	root_rot_format: str = "xyzw",
	base_translation_key: Optional[str] = None,
	base_rotation_key: Optional[str] = None,
	joint_positions_key: Optional[str] = None,
	joint_names_path: Optional[Path] = None,
	joint_position_units: str = "radians",
	convert_y_up_to_z_up: bool = False,
	num_rank: int = 1,
	slurm_rank: int = 0,
	min_height_threshold: float = -0.05,
	max_velocity_threshold: float = 15.0,
	max_dof_vel_threshold: float = 40.0,
	duration_height_filter: float = 0.2,
	duration_height_seconds: float = 1.0,
	ignore_motion_filter: bool = False,
	yaml_output_name: Optional[str] = None,
	force_remake: bool = False,
):
	if yaml_output_name is not None:
		yaml_output = output_path / yaml_output_name
	else:
		yaml_output = None

	device = torch.device("cpu")
	dtype = torch.float32

	if input_fps % output_fps != 0:
		raise typer.Exit("input_fps must be divisible by output_fps")

	kinematic_info = extract_kinematic_info(str(mjcf_path))
	kinematic_info = kinematic_info.to(device, dtype)

	if joint_names_path is not None:
		input_joint_names = _load_joint_names(joint_names_path)
		if len(input_joint_names) != kinematic_info.num_dofs:
			raise typer.Exit(
				f"Joint name count ({len(input_joint_names)}) does not match MJCF DOFs ({kinematic_info.num_dofs})"
			)
		reorder_indices = [input_joint_names.index(name) for name in kinematic_info.dof_names]
	else:
		# Use default GMR motor names for K1 if no mapping provided
		reorder_indices = None
		if len(GMR_K1_MOTOR_NAMES) != kinematic_info.num_dofs:
			print(f"WARNING: GMR motor count ({len(GMR_K1_MOTOR_NAMES)}) does not match MJCF DOFs ({kinematic_info.num_dofs})")
		else:
			try:
				reorder_indices = [kinematic_info.dof_names.index(name) for name in GMR_K1_MOTOR_NAMES]
				print(f"Using default GMR-to-K1 joint mapping")
			except ValueError as e:
				print(f"WARNING: Could not apply default GMR-to-K1 mapping: {e}")
				reorder_indices = None

	output_motions_yaml = []
	output_yaml_idx = 0
	output_path.mkdir(parents=True, exist_ok=True)

	for pkl_file_path in _iter_pkl_files(gmr_pkl_path):
		rel_path = pkl_file_path.parent.relative_to(gmr_pkl_path) if gmr_pkl_path.is_dir() else Path(".")
		file_hash = int(
			hashlib.sha256(str(rel_path / pkl_file_path.name).encode("utf-8")).hexdigest(), 16
		)
		if file_hash % num_rank != slurm_rank:
			continue

		output_dir = output_path / rel_path
		output_dir.mkdir(parents=True, exist_ok=True)
		file_name = pkl_file_path.name.replace(".pkl", ".motion").replace(" ", "_")
		output_file_relative = rel_path / file_name
		output_file = output_dir / file_name

		if not force_remake and output_file.exists():
			continue

		try:
			base_translation, base_rotation, joint_positions = _load_gmr_pkl(
				pkl_file_path,
				base_translation_key=base_translation_key,
				base_rotation_key=base_rotation_key,
				joint_positions_key=joint_positions_key,
			)
		except Exception as exc:
			print(f"Skipping {pkl_file_path}: {exc}")
			continue

		base_translation = _as_numpy(base_translation)
		base_rotation = _as_numpy(base_rotation)
		joint_positions = _as_numpy(joint_positions)

		if base_translation.ndim != 2 or base_translation.shape[1] != 3:
			print(f"Skipping {pkl_file_path}: base translation shape {base_translation.shape} is invalid")
			continue

		if base_rotation.ndim not in {2, 3}:
			print(f"Skipping {pkl_file_path}: base rotation shape {base_rotation.shape} is invalid")
			continue

		base_rotation = _convert_root_rotation(base_rotation, root_rot_format)
		if base_rotation.shape[-1] != 4:
			print(f"Skipping {pkl_file_path}: base rotation shape {base_rotation.shape} is invalid after conversion")
			continue

		if joint_positions.ndim != 2:
			print(f"Skipping {pkl_file_path}: joint positions shape {joint_positions.shape} is invalid")
			continue

		if base_translation.shape[0] != base_rotation.shape[0] or base_translation.shape[0] != joint_positions.shape[0]:
			print(f"Skipping {pkl_file_path}: frame counts do not match")
			continue

		downsample_factor = input_fps // output_fps
		base_translation = _downsample(base_translation, downsample_factor)
		base_rotation = _downsample(base_rotation, downsample_factor)
		joint_positions = _downsample(joint_positions, downsample_factor)

		if joint_position_units.lower() == "degrees":
			joint_positions = np.deg2rad(joint_positions)
		elif joint_position_units.lower() != "radians":
			raise typer.Exit("joint_position_units must be 'radians' or 'degrees'")

		if reorder_indices is not None:
			joint_positions = joint_positions[:, reorder_indices]

		if joint_positions.shape[1] != kinematic_info.num_dofs:
			print(
				f"Skipping {pkl_file_path}: joint count {joint_positions.shape[1]} does not match MJCF DOFs {kinematic_info.num_dofs}"
			)
			continue

		if convert_y_up_to_z_up:
			base_translation, base_rotation = _apply_y_up_to_z_up(base_translation, base_rotation)

		root_rot_norm = base_rotation / np.linalg.norm(base_rotation, axis=-1, keepdims=True)

		qpos = np.zeros((base_translation.shape[0], kinematic_info.nq), dtype=np.float32)
		qpos[:, 0:3] = base_translation
		qpos[:, 3:7] = root_rot_norm
		qpos[:, 7:] = joint_positions

		qpos_t = torch.from_numpy(qpos).to(device=device, dtype=dtype)
		motion: RobotState = fk_batch_mjcf_with_velocities(
			kinematic_info=kinematic_info,
			qpos=qpos_t,
			fps=output_fps,
			compute_velocities=True,
			velocity_max_horizon=3,
		)

		dof_pos_t = torch.from_numpy(joint_positions).to(device=device, dtype=dtype)
		motion.dof_pos = dof_pos_t
		motion.dof_vel = _compute_dof_vel(dof_pos_t, output_fps)

		if motion.rigid_body_vel is not None:
			motion.rigid_body_contacts = compute_contact_labels_from_pos_and_vel(
				positions=motion.rigid_body_pos,
				velocity=motion.rigid_body_vel,
				vel_thres=0.15,
				height_thresh=0.1,
			).to(torch.bool)

		if not ignore_motion_filter and not passes_exclude_motion_filter(
			motion,
			min_height_threshold=min_height_threshold,
			max_velocity_threshold=max_velocity_threshold,
			max_dof_vel_threshold=max_dof_vel_threshold,
			duration_height_filter=duration_height_filter,
			duration_height_seconds=duration_height_seconds,
		):
			print(f"Skipping {pkl_file_path}: failed motion filter")
			continue

		torch.save(motion.to_dict(), str(output_file))
		print(f"Saved to {output_file}")

		if yaml_output is not None:
			output_motions_yaml.append(
				{
					"file": str(output_file_relative),
					"fps": output_fps,
					"weight": 1.0,
					"idx": output_yaml_idx,
				}
			)
			output_yaml_idx += 1

	if yaml_output is not None:
		with open(yaml_output, "w") as f:
			yaml.dump({"motions": output_motions_yaml}, f)
		print(f"Saved motions list to {yaml_output}")


if __name__ == "__main__":
	app()
