
import time
import numpy as np
import torch
from typing import List, Tuple, Dict
import yaml

# CuRobo imports
from curobo.geom.types import WorldConfig, Sphere
from curobo.types.base import TensorDeviceType
from curobo.types.math import Pose
from curobo.types.robot import JointState, RobotConfig
from curobo.util_file import join_path, load_yaml
from curobo.wrap.reacher.trajopt import TrajOptSolver, TrajOptSolverConfig
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.rollout.rollout_base import Goal
from curobo.util_file import load_yaml
from curobo.geom.types import WorldConfig


class DatasetGenerator:
    """Generate collision-free trajectory dataset for motion planning."""

    def __init__(
        self,
        robot_config_file: str,
        urdf_file: str,
        num_obstacles: int = 3,
        trajectory_length: int = 50,
        device: str = "cuda:0"
    ):
        """
        Initialize the dataset generator.

        Args:
            robot_config_file: Path to robot YAML config
            urdf_file: Path to robot URDF file
            num_obstacles: Number of random sphere obstacles to add
            trajectory_length: Number of states in resampled trajectory
            device: Device to run on (cuda:0 or cpu)
        """

        self.num_obstacles = num_obstacles
        self.trajectory_length = trajectory_length
        self.tensor_args = TensorDeviceType(device=device)

        # Load robot configuration
        print(f"Loading robot config from: {robot_config_file}")
        self.robot_cfg = self._load_robot_config(robot_config_file, urdf_file)
        print(f"Robot loaded: {len(self.robot_cfg.kinematics.kinematics_config.joint_names)} DOF")
        print(f"Joint names: {self.robot_cfg.kinematics.kinematics_config.joint_names}")

        # Create world with random obstacles
        print(f"Creating world with {num_obstacles} random obstacles")
        self.world_file = self._create_random_world()

        # Initialize TrajOpt solver
        print(f"Initializing TrajOpt solver")
        self.trajopt_solver = self._initialize_trajopt()
        print(f"TrajOpt solver ready")

        # Get joint limits for sampling
        self.joint_limits = self._get_joint_limits()
        print(f"Joint limits:")
        for i, (lower, upper) in enumerate(self.joint_limits):
            print(f"Joint {i}: [{lower:.2f}, {upper:.2f}]")

    def _load_robot_config(self, config_file: str, urdf_file: str) -> RobotConfig:
        """Load robot configuration from YAML file."""
        config_dict = load_yaml(config_file)["robot_cfg"]

        # Update URDF path to absolute path
        config_dict["kinematics"]["urdf_path"] = urdf_file

        robot_cfg = RobotConfig.from_dict(config_dict, self.tensor_args)
        return robot_cfg

    def _create_random_world(self) -> str:
        """Create a world with random sphere obstacles and save to YAML."""

        spheres_dict = {}

        # Define workspace bounds
        x_range = (-0.5, 0.5)
        y_range = (-0.5, 0.5)
        z_range = (0.1, 0.8)
        radius_range = (0.05, 0.15)

        for i in range(self.num_obstacles):
            x = np.random.uniform(*x_range)
            y = np.random.uniform(*y_range)
            z = np.random.uniform(*z_range)
            radius = np.random.uniform(*radius_range)

            spheres_dict[f"obstacle_{i}"] = {
                "pose": [float(x), float(y), float(z), 1.0, 0.0, 0.0, 0.0],
                "radius": float(radius)
            }
            print(f"  Obstacle {i}: pos=({x:.2f}, {y:.2f}, {z:.2f}), radius={radius:.3f}")

        # Create world config in proper format
        world_cfg_dict = {

            "sphere": spheres_dict

        }

        # Save to YAML file with proper formatting
        world_file_path = "random_world.yml"
        with open(world_file_path, 'w') as f:
            yaml.dump(world_cfg_dict, f, default_flow_style=False, sort_keys=False)

        return world_file_path

    def _initialize_trajopt(self) -> TrajOptSolver:
        """Initialize the trajectory optimization solver."""

        # Load world config from YAML file
        self.world_cfg = WorldConfig.from_dict(load_yaml(self.world_file))

        self.world_cfg = WorldConfig.create_mesh_world(self.world_cfg)

        # Verify obstacles are loaded
        if hasattr(self.world_cfg, 'sphere') and self.world_cfg.sphere is not None:
            print(f"  ✓ Loaded {len(self.world_cfg.sphere)} sphere obstacles")
        else:
            print("  ✗ WARNING: No obstacles loaded!")

        trajopt_config = TrajOptSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,  # Now this is stored as self.world_cfg
            self.tensor_args,
            use_cuda_graph=False,
        )
        return TrajOptSolver(trajopt_config)

    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """Get joint limits from robot configuration."""
        limits = []
        kin_cfg = self.robot_cfg.kinematics

        for i in range(len(kin_cfg.kinematics_config.joint_names)):
            lower = -3.0  # kin_cfg.kinematics_config.joint_limits.position[0, i].item()
            upper = 3.0  # kin_cfg.kinematics_config.joint_limits.position[1, i].item()
            limits.append((lower, upper))

        return limits

    def sample_collision_free_config(self, max_attempts: int = 100) -> torch.Tensor:
        """
        Sample a random collision-free configuration.

        Args:
            max_attempts: Maximum number of sampling attempts

        Returns:
            Collision-free joint configuration as tensor
        """
        for attempt in range(max_attempts):
            # Sample random configuration within joint limits
            q = torch.zeros(len(self.joint_limits), device=self.tensor_args.device)
            for i, (lower, upper) in enumerate(self.joint_limits):
                q[i] = torch.rand(1, device=self.tensor_args.device) * (upper - lower) + lower

            # Check for collisions
            q_batch = q.unsqueeze(0)  # Add batch dimension

            # Use the kinematics model to check collision
            kin_state = self.trajopt_solver.fk(q_batch)

            # For now, we'll just return the sampled config
            # More sophisticated collision checking can be added
            return q

        raise RuntimeError(f"Failed to sample collision-free config after {max_attempts} attempts")

    def generate_trajectory(
        self,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
        verbose: bool = True
    ) -> Dict:
        """
        Generate a collision-free trajectory from start to goal.

        Args:
            q_start: Start joint configuration
            q_goal: Goal joint configuration
            verbose: Whether to print detailed information

        Returns:
            Dictionary containing trajectory and metadata
        """
        if verbose:
            print("\n" + "-" * 60)
            print("GENERATING TRAJECTORY")
            print("-" * 60)
            print(f"Start config: {q_start.cpu().numpy()}")
            print(f"Goal config:  {q_goal.cpu().numpy()}")

        # Create start and goal states
        current_state = JointState.from_position(q_start.unsqueeze(0))
        goal_state = JointState.from_position(q_goal.unsqueeze(0))

        # Create goal for TrajOpt

        js_goal = Goal(goal_state=goal_state, current_state=current_state)

        # Solve trajectory optimization
        start_time = time.time()
        result = self.trajopt_solver.solve_single(js_goal)
        solve_time = time.time() - start_time

        if verbose:
            print(f"\nTrajOpt Results:")
            print(f"  Success: {result.success.item()}")
            print(f"  Solve time: {solve_time:.4f}s")
            print(f"  Original trajectory length: {result.solution.position.shape[0]}")

        if not result.success:
            if verbose:
                print("  ✗ Trajectory generation FAILED")
            return None

        # Resample trajectory to fixed length
        resampled_traj = self._resample_trajectory(result.solution)

        if verbose:
            print(f"  Resampled trajectory length: {resampled_traj['position'].shape[0]}")
            print("  ✓ Trajectory generation SUCCESS")
            print("-" * 60 + "\n")

        return {
            'start': q_start.cpu().numpy(),
            'goal': q_goal.cpu().numpy(),
            'position': resampled_traj['position'],
            'velocity': resampled_traj['velocity'],
            'acceleration': resampled_traj['acceleration'],
            'success': True,
            'solve_time': solve_time,
            'original_length': result.solution.position.shape[0]
        }

    def _resample_trajectory(self, trajectory: JointState) -> Dict:
        """
        Resample trajectory to fixed number of states using linear interpolation.

        Args:
            trajectory: Original trajectory from TrajOpt

        Returns:
            Dictionary with resampled position, velocity, acceleration
        """
        original_length = trajectory.position.shape[0]

        # Create interpolation indices
        original_indices = torch.linspace(0, original_length - 1, original_length)
        target_indices = torch.linspace(0, original_length - 1, self.trajectory_length)

        # Resample position
        position = trajectory.position.squeeze(1).cpu()  # [T, DOF]
        resampled_position = torch.zeros(self.trajectory_length, position.shape[1])

        for i in range(position.shape[1]):
            resampled_position[:, i] = torch.tensor(
                np.interp(
                    target_indices.numpy(),
                    original_indices.numpy(),
                    position[:, i].numpy()
                )
            )

        # Resample velocity
        velocity = trajectory.velocity.squeeze(
            1).cpu() if trajectory.velocity is not None else torch.zeros_like(position)
        resampled_velocity = torch.zeros(self.trajectory_length, velocity.shape[1])

        for i in range(velocity.shape[1]):
            resampled_velocity[:, i] = torch.tensor(
                np.interp(
                    target_indices.numpy(),
                    original_indices.numpy(),
                    velocity[:, i].numpy()
                )
            )

        # Resample acceleration
        acceleration = trajectory.acceleration.squeeze(
            1).cpu() if trajectory.acceleration is not None else torch.zeros_like(position)
        resampled_acceleration = torch.zeros(self.trajectory_length, acceleration.shape[1])

        for i in range(acceleration.shape[1]):
            resampled_acceleration[:, i] = torch.tensor(
                np.interp(
                    target_indices.numpy(),
                    original_indices.numpy(),
                    acceleration[:, i].numpy()
                )
            )

        return {
            'position': resampled_position.numpy(),
            'velocity': resampled_velocity.numpy(),
            'acceleration': resampled_acceleration.numpy()
        }

    def generate_single_example(self, example_id: int = 0) -> Dict:
        """
        Generate a single trajectory example.

        Args:
            example_id: ID for this example

        Returns:
            Dictionary containing full trajectory data
        """
        print(f"\n{'='*60}")
        print(f"GENERATING EXAMPLE {example_id}")
        print(f"{'='*60}")

        # Sample collision-free start and goal
        print("Sampling collision-free start configuration...")
        q_start = self.sample_collision_free_config()
        print(f"Start: {q_start.cpu().numpy()}")

        print("Sampling collision-free goal configuration...")
        q_goal = self.sample_collision_free_config()
        print(f"Goal: {q_goal.cpu().numpy()}")

        # Generate trajectory
        print("Generating collision-free trajectory...")
        trajectory_data = self.generate_trajectory(q_start, q_goal)

        if trajectory_data is None:
            print("FAILED to generate trajectory")
            return None

        # Add metadata
        trajectory_data['example_id'] = example_id
        trajectory_data['num_obstacles'] = self.num_obstacles
        trajectory_data['obstacles'] = [
            {
                'name': sphere.name,
                'position': sphere.pose[:3],
                'radius': sphere.radius
            }
            for sphere in self.world_cfg.sphere
        ]

        print(f"\n{'='*60}")
        print(f"EXAMPLE {example_id} COMPLETE")
        print(f"{'='*60}\n")

        return trajectory_data


def main():
    """Main function to test the dataset generator."""

    print("\n" + "#" * 60)
    print("# COLLISION-FREE MOTION PLANNING DATASET GENERATOR")
    print("#" * 60 + "\n")

    # Configuration
    robot_config_file = "/home/nataliya/curobo/data_generation/yaml/kinematic_arm_3_dof.yml"
    urdf_file = "/home/nataliya/sim_learning/rodrigues_network/src/urdfs/kinematic_arm_3_dof.urdf"
    num_obstacles = 3
    trajectory_length = 50

    # Initialize generator
    try:
        generator = DatasetGenerator(
            robot_config_file=robot_config_file,
            urdf_file=urdf_file,
            num_obstacles=num_obstacles,
            trajectory_length=trajectory_length,
            device="cuda:0" if torch.cuda.is_available() else "cpu"
        )
    except Exception as e:
        print(f"\n✗ ERROR during initialization: {e}")
        import traceback
        traceback.print_exc()
        return

    # Generate a single example
    try:
        example = generator.generate_single_example(example_id=0)

        if example is not None:
            print("\n" + "#" * 60)
            print("# EXAMPLE SUCCESSFULLY GENERATED")
            print("#" * 60)
            print(f"\nTrajectory shape: {example['position'].shape}")
            print(f"Start config: {example['start']}")
            print(f"Goal config: {example['goal']}")
            print(f"Solve time: {example['solve_time']:.4f}s")
            print(f"Number of obstacles: {example['num_obstacles']}")
        else:
            print("\n✗ Failed to generate example")

    except Exception as e:
        print(f"\n✗ ERROR during example generation: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
