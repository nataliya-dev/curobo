#!/usr/bin/env python3
# Standard Library
import json
import os
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D
import h5py

# CuRobo
from curobo.geom.types import Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.util.trajectory import InterpolateType
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


# ============================================================================
# CONFIGURATION LOADER
# ============================================================================

def load_config(config_path="config.json"):
    """
    Load configuration from JSON file.

    Args:
        config_path: Path to JSON configuration file

    Returns:
        dict: Configuration dictionary with values extracted
    """
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Extract values from the config structure
    extracted = {}
    for section, params in config.items():
        extracted[section] = {}
        for param, values in params.items():
            # Use the 'value' if it's not None, otherwise use 'default'
            extracted[section][param] = values['value'] if values['value'] is not None else values['default']

    return extracted


# ============================================================================
# OBSTACLE GENERATOR
# ============================================================================

class ObstacleGenerator:
    """Generate random obstacles for the workspace."""

    def __init__(self, obstacle_config: Dict[str, Any]):
        """
        Initialize obstacle generator.

        Args:
            obstacle_config: Dictionary with obstacle generation parameters
        """
        self.num_obstacles_min = obstacle_config['num_obstacles_min']
        self.num_obstacles_max = obstacle_config['num_obstacles_max']
        self.radius_min = obstacle_config['radius_min']
        self.radius_max = obstacle_config['radius_max']
        self.x_range = obstacle_config['x_range']
        self.y_range = obstacle_config['y_range']
        self.z_range = obstacle_config['z_range']

    def generate(self) -> Tuple[list, WorldConfig]:
        """
        Generate random spherical obstacles.

        Returns:
            obstacles: List of obstacle dictionaries for visualization
            world_config: CuRobo WorldConfig with obstacles
        """
        # Random number of obstacles
        num_obstacles = np.random.randint(self.num_obstacles_min, self.num_obstacles_max + 1)

        obstacles = []
        sphere_list = []

        for i in range(num_obstacles):
            # Random position
            position = [
                np.random.uniform(self.x_range[0], self.x_range[1]),
                np.random.uniform(self.y_range[0], self.y_range[1]),
                np.random.uniform(self.z_range[0], self.z_range[1]),
            ]

            # Random radius
            radius = np.random.uniform(self.radius_min, self.radius_max)

            # Store for visualization
            obstacles.append({
                'position': position,
                'radius': radius,
            })

            # Create CuRobo sphere (pose is [x, y, z, qw, qx, qy, qz])
            sphere = Sphere(
                name=f"obstacle_{i}",
                pose=position + [1, 0, 0, 0],  # Identity quaternion
                radius=radius,
            )
            sphere_list.append(sphere)

        # Create world configuration
        world_config = WorldConfig(sphere=sphere_list)

        # Convert world config to mesh representation
        # TODO(nn) confirm this keeps our spheres as ususal, had problems with this in the past
        world_config = WorldConfig.create_mesh_world(world_config)

        return obstacles, world_config


# ============================================================================
# DATA STORAGE
# ============================================================================
# EXAMPLE USAGE FOR LOADING DATA FOR TRAINING:
# import h5py
# import numpy as np

# # Load all trajectories at once
# with h5py.File('data/trajectories.h5', 'r') as f:
#     starts = []
#     goals = []
#     plans = []
#     obstacles_list = []

#     for traj_name in f.keys():
#         traj = f[traj_name]
#         starts.append(traj['start'][:])
#         goals.append(traj['goal'][:])
#         plans.append(traj['interpolated_plan'][:])
#         obstacles_list.append({
#             'positions': traj['obstacle_positions'][:],
#             'radii': traj['obstacle_radii'][:]
#         })

#     # Convert to arrays for batch training
#     starts = np.array(starts)      # [N, 3]
#     goals = np.array(goals)        # [N, 3]
#     # plans need padding if different lengths
# ============================================================================


# ============================================================================
# DATA STORAGE
# ============================================================================

class DataStorage:
    """
    Store trajectory data in HDF5 format and save visualization images.

    WHAT WE SAVE:
    =============
    For each trajectory, we save:

    1. JOINT CONFIGURATIONS (numpy arrays):
       - start: [dof] - Starting joint angles
       - goal: [dof] - Goal joint angles
       - optimized_plan: [n_waypoints, dof] - Optimized trajectory waypoints
       - interpolated_plan: [n_steps, dof] - Full interpolated trajectory

    2. TIMING DATA (numpy arrays):
       - optimized_dt: [n_waypoints-1] - Time between optimized waypoints

    3. OBSTACLE DATA (numpy arrays):
       - obstacle_positions: [n_obstacles, 3] - XYZ positions of obstacles
       - obstacle_radii: [n_obstacles] - Radius of each obstacle sphere

    4. METADATA (HDF5 attributes - scalars):
       - trajectory_id, num_obstacles, solve_time, motion_time,
       - interpolation_dt, plan lengths, timestamp

    FILE VERSIONING:
    ===============
    If trajectories.h5 exists, creates trajectories_001.h5, trajectories_002.h5, etc.
    This prevents accidental data loss from overwriting existing files.
    """

    def __init__(self, output_directory: str, storage_config: Dict[str, Any],
                 enabled: bool = True):
        """
        Initialize data storage.

        Args:
            output_directory: Base directory for storing data
            storage_config: Configuration for data storage
            enabled: If False, skip all saving operations (for testing)
        """
        self.enabled = enabled

        if not self.enabled:
            print("DataStorage initialized in DISABLED mode (no saving)")
            return

        self.output_directory = Path(output_directory)
        self.storage_config = storage_config

        # Create directory structure
        self.output_directory.mkdir(parents=True, exist_ok=True)

        # Get base filename and find next available version
        base_filename = storage_config['hdf5_filename']
        self.hdf5_path = self._get_versioned_filepath(base_filename)

        # Images directory
        self.images_dir = self.output_directory / storage_config['debug_images_dir']
        self.images_dir.mkdir(parents=True, exist_ok=True)

        # Image settings
        self.image_format = storage_config['image_format']
        self.image_dpi = storage_config['image_dpi']

        # HDF5 compression settings
        self.compression = storage_config['compression']
        self.compression_opts = storage_config['compression_opts']

        # Flush settings
        self.flush_interval = storage_config.get('flush_interval', 1)
        self.trajectories_since_flush = 0

        # HDF5 file handle (initially None, opened with open())
        self.hdf5_file = None

        print(f"DataStorage initialized (ENABLED):")
        print(f"  Output directory: {self.output_directory}")
        print(f"  HDF5 file: {self.hdf5_path}")
        print(f"  Images directory: {self.images_dir}")
        print(f"  Flush interval: every {self.flush_interval} trajectory(ies)")

    def _get_versioned_filepath(self, base_filename: str) -> Path:
        """
        Get next available versioned filepath to avoid overwriting.

        Args:
            base_filename: Base filename (e.g., "trajectories.h5")

        Returns:
            Path to versioned file (e.g., "trajectories_001.h5" if base exists)
        """
        base_path = self.output_directory / base_filename

        # If file doesn't exist, use base name
        if not base_path.exists():
            print(f"  Using new file: {base_filename}")
            return base_path

        # File exists - find next available version
        stem = base_path.stem  # "trajectories"
        suffix = base_path.suffix  # ".h5"

        version = 1
        while True:
            versioned_path = self.output_directory / f"{stem}_{version:03d}{suffix}"
            if not versioned_path.exists():
                print(f"  File {base_filename} exists - using versioned file: {versioned_path.name}")
                return versioned_path
            version += 1

            # Safety check to prevent infinite loop
            if version > 9999:
                raise RuntimeError(f"Too many versions of {base_filename} (>9999)!")

    def open(self) -> None:
        """Open HDF5 file for batch writing."""
        if not self.enabled:
            print("[DataStorage] Skipping open (disabled)")
            return

        if self.hdf5_file is not None:
            print("[DataStorage] WARNING: File already open, skipping open()")
            return

        try:
            print(f"[DataStorage] Opening HDF5 file: {self.hdf5_path}")
            self.hdf5_file = h5py.File(self.hdf5_path, 'w')

            # Store global metadata
            self.hdf5_file.attrs['created_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            self.hdf5_file.attrs['version'] = '1.0'
            self.hdf5_file.attrs['flush_interval'] = self.flush_interval

            self.trajectories_since_flush = 0
            print(f"[DataStorage] ✓ HDF5 file opened successfully")

        except Exception as e:
            print(f"[DataStorage] ✗ ERROR opening HDF5 file!")
            print(f"  Error: {e}")
            raise

    def close(self) -> None:
        """Close HDF5 file and flush any remaining data."""
        if not self.enabled:
            print("[DataStorage] Skipping close (disabled)")
            return

        if self.hdf5_file is None:
            print("[DataStorage] WARNING: File not open, skipping close()")
            return

        try:
            print(f"[DataStorage] Closing HDF5 file...")

            # Final flush
            self.hdf5_file.flush()
            print(f"  ✓ Final flush complete")

            # Update metadata
            self.hdf5_file.attrs['closed_at'] = time.strftime('%Y-%m-%d %H:%M:%S')
            total_trajectories = len(self.hdf5_file.keys())
            self.hdf5_file.attrs['total_trajectories'] = total_trajectories
            print(f"  ✓ Saved {total_trajectories} trajectories")

            # Close file
            self.hdf5_file.close()
            self.hdf5_file = None
            print(f"[DataStorage] ✓ HDF5 file closed: {self.hdf5_path}")

        except Exception as e:
            print(f"[DataStorage] ✗ ERROR closing HDF5 file!")
            print(f"  Error: {e}")
            # Still set to None to avoid re-closing
            self.hdf5_file = None
            raise

    def _save_array_dataset(self, group, name: str, data: np.ndarray,
                            verbose: bool = False) -> None:
        """
        Helper to save array dataset with proper compression handling.

        Args:
            group: HDF5 group to save to
            name: Dataset name
            data: Numpy array to save
            verbose: If True, print debug information
        """
        # Convert to numpy array if needed
        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # Only apply compression to non-scalar datasets
        if data.shape == () or data.size == 1:
            # Scalar or single-element - no compression
            group.create_dataset(name, data=data)
            if verbose:
                print(f"    [DEBUG] Saved scalar dataset '{name}': shape={data.shape}")
        else:
            # Array - apply compression
            group.create_dataset(
                name,
                data=data,
                compression=self.compression,
                compression_opts=self.compression_opts
            )
            if verbose:
                print(f"    [DEBUG] Saved compressed dataset '{name}': "
                      f"shape={data.shape}, size={data.nbytes/1024:.2f} KB")

    def save_trajectory(self, trajectory_id: int, trajectory_data: Dict[str, Any],
                        verbose: bool = False) -> None:
        """
        Save trajectory data to HDF5 file.

        Args:
            trajectory_id: Unique identifier for the trajectory
            trajectory_data: Dictionary containing trajectory information
            verbose: If True, print detailed debug information
        """
        if not self.enabled:
            if verbose:
                print(f"  [SKIP] Saving disabled - would save trajectory {trajectory_id}")
            return

        if self.hdf5_file is None:
            raise RuntimeError("HDF5 file is not open! Call open() before saving.")

        group_name = f"trajectory_{trajectory_id:06d}"

        try:
            if verbose:
                print(f"  Saving to HDF5 group: {group_name}")

            # Create group for this trajectory
            if group_name in self.hdf5_file:
                if verbose:
                    print(f"    [DEBUG] Overwriting existing group")
                del self.hdf5_file[group_name]

            grp = self.hdf5_file.create_group(group_name)

            # Save trajectory arrays
            if verbose:
                print(f"    [DEBUG] Saving trajectory arrays...")

            self._save_array_dataset(grp, 'start', trajectory_data['start'], verbose)
            self._save_array_dataset(grp, 'goal', trajectory_data['goal'], verbose)
            self._save_array_dataset(grp, 'optimized_plan',
                                     trajectory_data['optimized_plan'], verbose)
            self._save_array_dataset(grp, 'optimized_dt',
                                     trajectory_data['optimized_dt'], verbose)
            self._save_array_dataset(grp, 'interpolated_plan',
                                     trajectory_data['interpolated_plan'], verbose)

            # Save obstacle data
            if verbose:
                print(f"    [DEBUG] Saving obstacle data...")

            obstacles = trajectory_data['obstacles']
            if len(obstacles) > 0:
                obstacle_positions = np.array([obs['position'] for obs in obstacles])
                obstacle_radii = np.array([obs['radius'] for obs in obstacles])
            else:
                # Empty arrays if no obstacles
                obstacle_positions = np.empty((0, 3))
                obstacle_radii = np.empty(0)

            self._save_array_dataset(grp, 'obstacle_positions',
                                     obstacle_positions, verbose)
            self._save_array_dataset(grp, 'obstacle_radii',
                                     obstacle_radii, verbose)

            # Save metadata as attributes
            if verbose:
                print(f"    [DEBUG] Saving metadata attributes...")

            grp.attrs['trajectory_id'] = trajectory_id
            grp.attrs['num_obstacles'] = trajectory_data['num_obstacles']
            grp.attrs['solve_time'] = float(trajectory_data['solve_time'])

            # Handle motion_time - might be scalar or array
            motion_time = trajectory_data['motion_time']
            if isinstance(motion_time, np.ndarray):
                grp.attrs['motion_time'] = float(motion_time.item())
            else:
                grp.attrs['motion_time'] = float(motion_time)

            grp.attrs['interpolation_dt'] = float(trajectory_data['interpolation_dt'])
            grp.attrs['optimized_plan_length'] = len(trajectory_data['optimized_plan'])
            grp.attrs['interpolated_plan_length'] = len(trajectory_data['interpolated_plan'])
            grp.attrs['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')

            # Periodic flush for data safety
            self.trajectories_since_flush += 1
            if self.trajectories_since_flush >= self.flush_interval:
                self.hdf5_file.flush()
                self.trajectories_since_flush = 0
                print(
                    f"  [FLUSH] Data committed to disk (every {self.flush_interval} trajectories)")

            if not verbose:
                print(f"  ✓ Saved trajectory data to HDF5: {group_name}")

        except Exception as e:
            print(f"  ✗ ERROR saving trajectory {trajectory_id} to HDF5!")
            print(f"    Error type: {type(e).__name__}")
            print(f"    Error message: {e}")
            print(f"    Trajectory data keys: {trajectory_data.keys()}")
            print(f"    Data shapes:")
            for key, val in trajectory_data.items():
                if isinstance(val, np.ndarray):
                    print(f"      {key}: {val.shape} {val.dtype}")
                elif key == 'obstacles':
                    print(f"      obstacles: {len(val)} obstacles")
                else:
                    print(f"      {key}: {type(val)}")

            # Emergency flush to try to save what we can
            try:
                print(f"  [EMERGENCY FLUSH] Attempting to save data before raising error...")
                self.hdf5_file.flush()
                print(f"  ✓ Emergency flush successful")
            except Exception as flush_error:
                print(f"  ✗ Emergency flush failed: {flush_error}")

            raise

    def save_image(self, trajectory_id: int, image_type: str, figure) -> None:
        """
        Save trajectory visualization image.

        Args:
            trajectory_id: Unique identifier for the trajectory
            image_type: Type of image ('start_goal' or 'trajectory')
            figure: Matplotlib figure object
        """
        if not self.enabled:
            print(f"  [SKIP] Saving disabled - would save {image_type} image")
            plt.close(figure)
            return

        filename = f"traj_{trajectory_id:06d}_{image_type}.{self.image_format}"
        filepath = self.images_dir / filename

        try:
            figure.savefig(filepath, dpi=self.image_dpi, bbox_inches='tight')
            print(f"  ✓ Saved {image_type} image: {filename}")

        except Exception as e:
            print(f"  ✗ ERROR saving image {filename}!")
            print(f"    Error: {e}")
            raise
        finally:
            plt.close(figure)

    def __enter__(self):
        """Context manager entry - opens file."""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes file even on error."""
        if exc_type is not None:
            print(f"\n[DataStorage] Exception occurred during context: {exc_type.__name__}")
            print(f"  Closing file to preserve data...")

        self.close()

        # Don't suppress the exception
        return False


# ============================================================================
# TRAJECTORY GENERATOR
# ============================================================================


class TrajectoryGenerator:
    """Generate trajectories using MotionGen for a robot arm."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the trajectory generator.

        Args:
            config: Configuration dictionary
        """
        print("Initializing TrajectoryGenerator...")
        start_time = time.time()

        self.config = config
        device = config['general']['device']
        self.tensor_args = TensorDeviceType(device=torch.device(device))

        # Store plan config
        self.plan_config_dict = config['plan_config']

        # Initialize with empty world (will be updated per trajectory)
        empty_world = WorldConfig(sphere=[])

        # Load MotionGen configuration
        config_start = time.time()

        mg_config = config['motion_gen_config']
        robot_file = config['general']['robot_config_file']

        # Handle interpolation_type conversion
        interpolation_type_str = mg_config['interpolation_type']
        if interpolation_type_str is not None:
            interpolation_type = getattr(InterpolateType, interpolation_type_str)
        else:
            interpolation_type = InterpolateType.LINEAR_CUDA

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            robot_file,
            empty_world,  # Start with empty world
            self.tensor_args,
            # Seed configuration
            num_ik_seeds=mg_config['num_ik_seeds'],
            num_graph_seeds=mg_config['num_graph_seeds'],
            num_trajopt_seeds=mg_config['num_trajopt_seeds'],
            num_batch_ik_seeds=mg_config['num_batch_ik_seeds'],
            num_batch_trajopt_seeds=mg_config['num_batch_trajopt_seeds'],
            num_trajopt_noisy_seeds=mg_config['num_trajopt_noisy_seeds'],
            # Convergence thresholds
            position_threshold=mg_config['position_threshold'],
            rotation_threshold=mg_config['rotation_threshold'],
            cspace_threshold=mg_config['cspace_threshold'],
            # Config files
            base_cfg_file=mg_config['base_cfg_file'],
            particle_ik_file=mg_config['particle_ik_file'],
            gradient_ik_file=mg_config['gradient_ik_file'],
            graph_file=mg_config['graph_file'],
            particle_trajopt_file=mg_config['particle_trajopt_file'],
            gradient_trajopt_file=mg_config['gradient_trajopt_file'],
            finetune_trajopt_file=mg_config['finetune_trajopt_file'],
            # Trajectory configuration
            trajopt_tsteps=mg_config['trajopt_tsteps'],
            interpolation_steps=mg_config['interpolation_steps'],
            interpolation_dt=mg_config['interpolation_dt'],
            interpolation_type=interpolation_type,
            # CUDA configuration
            use_cuda_graph=mg_config['use_cuda_graph'],
            # Collision parameters
            self_collision_check=mg_config['self_collision_check'],
            self_collision_opt=mg_config['self_collision_opt'],
            collision_activation_distance=mg_config['collision_activation_distance'],
            collision_max_outside_distance=mg_config['collision_max_outside_distance'],
            collision_checker_type=mg_config['collision_checker_type'],
            collision_cache=mg_config['collision_cache'],
            n_collision_envs=mg_config['n_collision_envs'],
            # Optimization iterations
            grad_trajopt_iters=mg_config['grad_trajopt_iters'],
            graph_trajopt_iters=mg_config['graph_trajopt_iters'],
            finetune_trajopt_iters=mg_config['finetune_trajopt_iters'],
            ik_opt_iters=mg_config['ik_opt_iters'],
            # Optimization settings
            trajopt_seed_ratio=mg_config['trajopt_seed_ratio'],
            ik_particle_opt=mg_config['ik_particle_opt'],
            trajopt_particle_opt=mg_config['trajopt_particle_opt'],
            use_gradient_descent=mg_config['use_gradient_descent'],
            # Evolutionary strategy
            use_es_ik=mg_config['use_es_ik'],
            use_es_trajopt=mg_config['use_es_trajopt'],
            es_ik_learning_rate=mg_config['es_ik_learning_rate'],
            es_trajopt_learning_rate=mg_config['es_trajopt_learning_rate'],
            # Fixed samples
            use_ik_fixed_samples=mg_config['use_ik_fixed_samples'],
            use_trajopt_fixed_samples=mg_config['use_trajopt_fixed_samples'],
            # Trajectory settings
            minimize_jerk=mg_config['minimize_jerk'],
            filter_robot_command=mg_config['filter_robot_command'],
            optimize_dt=mg_config['optimize_dt'],
            # Timing constraints
            trajopt_dt=mg_config['trajopt_dt'],
            js_trajopt_dt=mg_config['js_trajopt_dt'],
            js_trajopt_tsteps=mg_config['js_trajopt_tsteps'],
            minimum_trajectory_dt=mg_config['minimum_trajectory_dt'],
            maximum_trajectory_time=mg_config['maximum_trajectory_time'],
            maximum_trajectory_dt=mg_config['maximum_trajectory_dt'],
            # Scaling
            velocity_scale=mg_config['velocity_scale'],
            acceleration_scale=mg_config['acceleration_scale'],
            jerk_scale=mg_config['jerk_scale'],
            finetune_dt_scale=mg_config['finetune_dt_scale'],
            # Other settings
            evaluate_interpolated_trajectory=mg_config['evaluate_interpolated_trajectory'],
            partial_ik_iters=mg_config['partial_ik_iters'],
            fixed_iters_trajopt=mg_config['fixed_iters_trajopt'],
            trim_steps=mg_config['trim_steps'],
            smooth_weight=mg_config['smooth_weight'],
            finetune_smooth_weight=mg_config['finetune_smooth_weight'],
            state_finite_difference_mode=mg_config['state_finite_difference_mode'],
            project_pose_to_goal_frame=mg_config['project_pose_to_goal_frame'],
            # Debug settings
            store_ik_debug=mg_config['store_ik_debug'],
            store_trajopt_debug=mg_config['store_trajopt_debug'],
            store_debug_in_result=mg_config['store_debug_in_result'],
            # Random seeds
            ik_seed=mg_config['ik_seed'],
            graph_seed=mg_config['graph_seed'],
            # Precision
            high_precision=mg_config['high_precision'],
            # End effector
            ee_link_name=mg_config['ee_link_name'],
            # Sync
            sync_cuda_time=mg_config['sync_cuda_time'],
            # Evaluator
            traj_evaluator_config=mg_config['traj_evaluator_config'],
            traj_evaluator=mg_config['traj_evaluator'],
            # CUDA graph metrics
            use_cuda_graph_trajopt_metrics=mg_config['use_cuda_graph_trajopt_metrics'],
            # Terminal action
            trajopt_fix_terminal_action=mg_config['trajopt_fix_terminal_action'],
            trajopt_js_fix_terminal_action=mg_config['trajopt_js_fix_terminal_action'],
        )

        config_time = time.time() - config_start
        print(f"  Config load time: {config_time:.3f}s")

        self.motion_gen = MotionGen(self.motion_gen_config)

        # Get robot properties
        self.dof = self.motion_gen.kinematics.get_dof()
        self.joint_names = self.motion_gen.joint_names
        self.joint_limits = self.motion_gen.kinematics.get_joint_limits()

        print(f"  Robot DOF: {self.dof}")
        print(f"  Joint names: {self.joint_names}")

        total_time = time.time() - start_time
        print(f"  Total initialization time: {total_time:.3f}s")
        print("Ready!\n")

    def reset_graph_planner(self) -> None:
        """Reset graph planner buffer to prevent CUDA graph issues."""
        self.motion_gen.graph_planner.reset_buffer()

    def update_world(self, world_config: WorldConfig) -> None:
        """
        Update the collision world without reinitializing MotionGen.

        Args:
            world_config: New world configuration
        """
        self.motion_gen.update_world(world_config)

    def sample_random_joint_state(self) -> JointState:
        """Sample a random joint configuration within joint limits."""
        lower = self.joint_limits.position[0].unsqueeze(0)
        upper = self.joint_limits.position[1].unsqueeze(0)

        position = lower + torch.rand(
            1, self.dof,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        ) * (upper - lower)

        return JointState.from_position(position, joint_names=self.joint_names)

    def sample_collision_free_state(self, max_attempts: int = 100) -> Optional[JointState]:
        """
        Sample a random collision-free joint configuration.

        Args:
            max_attempts: Maximum number of sampling attempts

        Returns:
            JointState: Collision-free joint state, or None if failed
        """
        for _ in range(max_attempts):
            # Sample random state
            state = self.sample_random_joint_state()

            # Check if state is collision-free
            result = self.motion_gen.check_start_state(state)
            valid, status = result

            if valid:
                return state

        print(f"    [WARN] Could not find collision-free state after {max_attempts} attempts")
        return None

    def plan_trajectory(self, start_state: JointState, goal_state: JointState) -> Tuple[bool, Optional[Dict[str, Any]]]:
        """
        Plan a single trajectory from start to goal.

        Args:
            start_state: Start joint configuration
            goal_state: Goal joint configuration

        Returns:
            success: Whether planning succeeded
            trajectory_data: Dictionary containing trajectory data (None if failed)
        """
        # Configure planning from config
        pc = self.plan_config_dict
        plan_config = MotionGenPlanConfig(
            enable_graph=pc['enable_graph'],
            enable_opt=pc['enable_opt'],
            use_nn_ik_seed=pc['use_nn_ik_seed'],
            need_graph_success=pc['need_graph_success'],
            max_attempts=pc['max_attempts'],
            timeout=pc['timeout'],
            enable_graph_attempt=pc['enable_graph_attempt'],
            disable_graph_attempt=pc['disable_graph_attempt'],
            ik_fail_return=pc['ik_fail_return'],
            partial_ik_opt=pc['partial_ik_opt'],
            num_ik_seeds=pc['num_ik_seeds'],
            num_graph_seeds=pc['num_graph_seeds'],
            num_trajopt_seeds=pc['num_trajopt_seeds'],
            success_ratio=pc['success_ratio'],
            fail_on_invalid_query=pc['fail_on_invalid_query'],
            use_start_state_as_retract=pc['use_start_state_as_retract'],
            pose_cost_metric=pc['pose_cost_metric'],
            enable_finetune_trajopt=pc['enable_finetune_trajopt'],
            parallel_finetune=pc['parallel_finetune'],
            finetune_dt_scale=pc['finetune_dt_scale'],
            finetune_attempts=pc['finetune_attempts'],
            finetune_dt_decay=pc['finetune_dt_decay'],
            time_dilation_factor=pc['time_dilation_factor'],
            check_start_validity=pc['check_start_validity'],
            finetune_js_dt_scale=pc['finetune_js_dt_scale'],
        )

        # Plan trajectory
        result = self.motion_gen.plan_single_js(
            start_state=start_state,
            goal_state=goal_state,
            plan_config=plan_config,
        )

        # Check success
        if not result.success.item():
            return False, None

        # Extract trajectory data
        interpolated_plan = result.get_interpolated_plan()

        trajectory_data = {
            'start': start_state.position.cpu().numpy(),
            'goal': goal_state.position.cpu().numpy(),
            'optimized_plan': result.optimized_plan.position.cpu().numpy(),
            'optimized_dt': result.optimized_dt.cpu().numpy(),
            'interpolated_plan': interpolated_plan.position.cpu().numpy(),
            'interpolation_dt': result.interpolation_dt,
            'motion_time': result.motion_time if isinstance(result.motion_time, float) else result.motion_time.cpu().numpy(),
            'solve_time': result.solve_time,
        }

        return True, trajectory_data


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_trajectory(generator: TrajectoryGenerator, trajectory_data: Dict[str, Any],
                         obstacles: list, n_frames: int = 5):
    """
    Visualize trajectory with obstacles.

    Args:
        generator: TrajectoryGenerator instance
        trajectory_data: Dictionary containing trajectory data
        obstacles: List of obstacle dictionaries
        n_frames: Show every n-th frame

    Returns:
        Matplotlib figure object
    """
    positions = trajectory_data['interpolated_plan']

    # Select frame indices to visualize
    frame_indices = range(0, len(positions), n_frames)

    # Create colormap for trajectory progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(frame_indices)))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles
    for obs in obstacles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obs['radius'] * np.outer(np.cos(u), np.sin(v)) + obs['position'][0]
        y = obs['radius'] * np.outer(np.sin(u), np.sin(v)) + obs['position'][1]
        z = obs['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obs['position'][2]
        ax.plot_surface(x, y, z, color='red', alpha=0.3)

    # Plot robot states for each selected frame
    for idx, frame in enumerate(frame_indices):
        # Get joint configuration
        q = torch.tensor(positions[frame], device=generator.tensor_args.device).unsqueeze(0)

        # Get robot spheres
        robot_spheres = generator.motion_gen.kinematics.get_robot_as_spheres(q)[0]

        # Plot robot spheres
        for sph in robot_spheres:
            if sph.radius > 0:
                u = np.linspace(0, 2 * np.pi, 5)
                v = np.linspace(0, np.pi, 5)
                x = sph.radius * np.outer(np.cos(u), np.sin(v)) + sph.pose[0]
                y = sph.radius * np.outer(np.sin(u), np.sin(v)) + sph.pose[1]
                z = sph.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sph.pose[2]
                ax.plot_surface(x, y, z, color=colors[idx], alpha=0.4)

        # Get link poses
        state = generator.motion_gen.kinematics.get_state(q)

        # Plot link positions as points
        if state.links_position is not None:
            link_pos = state.links_position[0].cpu().numpy()
            ax.scatter(link_pos[:, 0], link_pos[:, 1], link_pos[:, 2],
                       c=[colors[idx]], s=100, marker='o',
                       label=f'Frame {frame}')

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Trajectory Visualization (every {n_frames} frames)')
    ax.legend()

    ax.view_init(elev=90, azim=-90)  # Top-down view of X-Y plane

    plt.tight_layout()

    return fig


def visualize_start_goal(generator: TrajectoryGenerator, q_start: np.ndarray,
                         q_goal: np.ndarray, obstacles: list):
    """
    Visualize start and goal configurations.

    Args:
        generator: TrajectoryGenerator instance
        q_start: Start joint configuration (numpy array)
        q_goal: Goal joint configuration (numpy array)
        obstacles: List of obstacle dictionaries

    Returns:
        Matplotlib figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles
    for obs in obstacles:
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = obs['radius'] * np.outer(np.cos(u), np.sin(v)) + obs['position'][0]
        y = obs['radius'] * np.outer(np.sin(u), np.sin(v)) + obs['position'][1]
        z = obs['radius'] * np.outer(np.ones(np.size(u)), np.cos(v)) + obs['position'][2]
        ax.plot_surface(x, y, z, color='red', alpha=0.3)

    # Convert to torch tensors
    q_start_torch = torch.tensor(q_start, device=generator.tensor_args.device).unsqueeze(0)
    q_goal_torch = torch.tensor(q_goal, device=generator.tensor_args.device).unsqueeze(0)

    # Plot start configuration (green)
    start_spheres = generator.motion_gen.kinematics.get_robot_as_spheres(q_start_torch)[0]
    for sph in start_spheres:
        if sph.radius > 0:
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = sph.radius * np.outer(np.cos(u), np.sin(v)) + sph.pose[0]
            y = sph.radius * np.outer(np.sin(u), np.sin(v)) + sph.pose[1]
            z = sph.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sph.pose[2]
            ax.plot_surface(x, y, z, color='green', alpha=0.6)

    # Plot goal configuration (blue)
    goal_spheres = generator.motion_gen.kinematics.get_robot_as_spheres(q_goal_torch)[0]
    for sph in goal_spheres:
        if sph.radius > 0:
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = sph.radius * np.outer(np.cos(u), np.sin(v)) + sph.pose[0]
            y = sph.radius * np.outer(np.sin(u), np.sin(v)) + sph.pose[1]
            z = sph.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sph.pose[2]
            ax.plot_surface(x, y, z, color='blue', alpha=0.6)

    ax.set_xlim([-1, 1])
    ax.set_ylim([-1, 1])
    ax.set_zlim([-1, 1])
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Start (Green) and Goal (Blue) Configurations')
    ax.view_init(elev=90, azim=-90)
    plt.tight_layout()

    return fig


# ============================================================================
# TRAJECTORY GENERATION ORCHESTRATOR
# ============================================================================

class TrajectoryDatasetGenerator:
    """Orchestrate generation of multiple trajectories."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize dataset generator.

        Args:
            config: Full configuration dictionary
        """
        self.config = config
        self.gen_config = config['generation']
        self.viz_config = config['visualization']

        # Initialize components
        print("=" * 70)
        print("Trajectory Dataset Generator")
        print("=" * 70)
        print()

        # Initialize obstacle generator
        self.obstacle_generator = ObstacleGenerator(config['obstacles'])

        # Initialize trajectory generator (once)
        self.trajectory_generator = TrajectoryGenerator(config)

        # Initialize data storage

        self.data_storage = DataStorage(
            config['general']['output_directory'],
            config['data_storage'],
            enabled=config['data_storage']['enabled']  # Add this parameter
        )

        print()

    def generate_single_trajectory(self, trajectory_id: int) -> bool:
        """
        Generate a single trajectory with retry logic.

        Args:
            trajectory_id: Unique identifier for this trajectory

        Returns:
            bool: True if successful, False otherwise
        """
        print(f"\n{'='*70}")
        print(f"Generating Trajectory {trajectory_id}")
        print(f"{'='*70}")

        max_attempts = self.gen_config['max_attempts_per_trajectory']
        regenerate_obstacles = self.gen_config['regenerate_obstacles_on_failure']

        for attempt in range(max_attempts):
            print(f"\nAttempt {attempt + 1}/{max_attempts}")

            # Generate obstacles (regenerate on retry if enabled)
            if attempt == 0 or regenerate_obstacles:
                print("  Generating obstacles...")
                obstacles, world_config = self.obstacle_generator.generate()
                print(f"    Created {len(obstacles)} obstacles")

                # Update world
                # TODO(nn) verify this resetting, why is planning taking too long, should be miiliseonds
                # self.trajectory_generator.reset_graph_planner()
                self.trajectory_generator.update_world(world_config)

            # Sample start and goal states
            print("  Sampling start state...")
            start_state = self.trajectory_generator.sample_collision_free_state()
            if start_state is None:
                print("    Failed to find collision-free start state")
                continue

            print("  Sampling goal state...")
            goal_state = self.trajectory_generator.sample_collision_free_state()
            if goal_state is None:
                print("    Failed to find collision-free goal state")
                continue

            print(f"    Start: {start_state.position.cpu().numpy().flatten()}")
            print(f"    Goal:  {goal_state.position.cpu().numpy().flatten()}")

            # Plan trajectory
            print("  Planning trajectory...")
            success, trajectory_data = self.trajectory_generator.plan_trajectory(
                start_state, goal_state
            )

            if not success:
                print("    Planning failed")
                continue

            # Success! Process results
            print(f"  [SUCCESS] Trajectory generated!")
            print(f"    Solve time: {trajectory_data['solve_time']:.3f}s")
            print(f"    Motion time: {trajectory_data['motion_time']:.3f}s")

            # Add obstacle data
            trajectory_data['obstacles'] = obstacles
            trajectory_data['num_obstacles'] = len(obstacles)
            trajectory_data['trajectory_id'] = trajectory_id

            # Save trajectory data
            print("  Saving trajectory data...")
            self.data_storage.save_trajectory(trajectory_id, trajectory_data)

            # Generate and save visualizations
            if self.gen_config['save_images']:
                print("  Generating visualizations...")

                if self.gen_config['save_start_goal_images']:
                    fig_sg = visualize_start_goal(
                        self.trajectory_generator,
                        start_state.position.cpu().numpy().flatten(),
                        goal_state.position.cpu().numpy().flatten(),
                        obstacles
                    )
                    self.data_storage.save_image(trajectory_id, 'start_goal', fig_sg)

                if self.gen_config['save_trajectory_images']:
                    fig_traj = visualize_trajectory(
                        self.trajectory_generator,
                        trajectory_data,
                        obstacles,
                        n_frames=self.viz_config['n_frames']
                    )
                    self.data_storage.save_image(trajectory_id, 'trajectory', fig_traj)

            return True

        # Failed after all attempts
        print(f"\n[FAILED] Could not generate trajectory after {max_attempts} attempts")
        return False

    def generate_dataset(self) -> Dict[str, Any]:
        """
        Generate the full dataset of trajectories.

        Returns:
            dict: Summary statistics of generation
        """
        num_trajectories = self.gen_config['num_trajectories']

        print(f"\n{'='*70}")
        print(f"Starting Dataset Generation")
        print(f"  Target: {num_trajectories} trajectories")
        print(f"{'='*70}")

        start_time = time.time()

        successful = 0
        failed = 0

        # Open HDF5 file once for all trajectories
        self.data_storage.open()

        try:
            for i in range(num_trajectories):
                success = self.generate_single_trajectory(i)

                if success:
                    successful += 1
                else:
                    failed += 1

        except KeyboardInterrupt:
            print("\n" + "=" * 70)
            print("INTERRUPTED BY USER (Ctrl+C)")
            print("=" * 70)
            print(f"Saving {successful} successful trajectories before exit...")
            # File will be closed in finally block

        except Exception as e:
            print("\n" + "=" * 70)
            print("UNEXPECTED ERROR DURING GENERATION")
            print("=" * 70)
            print(f"Error: {e}")
            print(f"Saving {successful} successful trajectories before exit...")
            # File will be closed in finally block
            raise

        finally:
            # Always close the file
            self.data_storage.close()

        total_time = time.time() - start_time

        # Print summary (rest stays the same)
        print(f"\n{'='*70}")
        print("GENERATION SUMMARY")
        print(f"{'='*70}")
        print(f"Total trajectories requested: {num_trajectories}")
        print(f"Successful:                   {successful}")
        print(f"Failed:                       {failed}")
        print(f"Success rate:                 {100 * successful / num_trajectories:.1f}%")
        print(f"Total time:                   {total_time:.2f}s")
        print(f"Average time per trajectory:  {total_time / num_trajectories:.2f}s")
        print(f"{'='*70}")

        return {
            'num_requested': num_trajectories,
            'num_successful': successful,
            'num_failed': failed,
            'success_rate': successful / num_trajectories,
            'total_time': total_time,
            'avg_time_per_trajectory': total_time / num_trajectories,
        }


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate trajectory dataset."""
    # Load configuration
    print("Loading configuration from config.json...")
    config = load_config("config.json")
    print()

    # Create dataset generator
    dataset_generator = TrajectoryDatasetGenerator(config)

    # Generate dataset
    summary = dataset_generator.generate_dataset()

    print("\nDataset generation complete!")


if __name__ == "__main__":
    main()
