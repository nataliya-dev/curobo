
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
from curobo.geom.sdf.world import CollisionQueryBuffer

# plotting
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
import torch


class DatasetGenerator:
    """Generate collision-free trajectory dataset for motion planning."""

    def __init__(
        self,
        robot_config_file: str,
        urdf_file: str,
        num_obstacles: int = 3,
        trajectory_length: int = 10,
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

        self._collision_weight = self.tensor_args.to_device([1.0])
        self._collision_activation_distance = self.tensor_args.to_device([0.02])

        # Load robot configuration
        print(f"Loading robot config from: {robot_config_file}")
        self.robot_cfg = self._load_robot_config(robot_config_file, urdf_file)
        print(f"Robot loaded: {len(self.robot_cfg.kinematics.kinematics_config.joint_names)} DOF")
        print(f"Joint names: {self.robot_cfg.kinematics.kinematics_config.joint_names}")

        # Create world with random obstacles
        print(f"Creating world with {num_obstacles} random obstacles")
        self.world_cfg_dict = self._create_random_world()

        # Initialize TrajOpt solver
        print(f"Initializing TrajOpt solver")
        self.trajopt_solver = self._initialize_trajopt()
        print(f"TrajOpt solver ready")

        # Get joint limits for sampling
        self.joint_limits = self._get_joint_limits()
        print(f"Joint limits:")
        for i, (lower, upper) in enumerate(self.joint_limits):
            print(f"Joint {i}: [{lower:.2f}, {upper:.2f}]")

        sample_q = torch.zeros(1, self.trajopt_solver.dof, device=self.tensor_args.device)
        sample_kin = self.trajopt_solver.fk(sample_q)
        sphere_shape = sample_kin.link_spheres_tensor.unsqueeze(1).shape  # [1, 1, n_spheres, 4]

        self._collision_query_buffer = CollisionQueryBuffer.initialize_from_shape(
            sphere_shape,
            self.tensor_args,
            self.trajopt_solver.world_coll_checker.collision_types
        )

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
        # TODO(nn) need to come from some config
        x_range = (-0.7, 0.7)
        y_range = (-0.7, 0.7)
        z_range = (0.0, 0.0)
        radius_range = (0.05, 0.02)

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

        return world_cfg_dict

    def _initialize_trajopt(self) -> TrajOptSolver:
        """Initialize the trajectory optimization solver."""

        # Load world config from dictionary
        self.world_cfg = WorldConfig.from_dict(self.world_cfg_dict)

        # Create mesh world representation
        self.world_cfg = WorldConfig.create_mesh_world(self.world_cfg)

        # Verify obstacles are loaded
        # print(f"Loaded {len(self.world_cfg.mesh)} mesh obstacles")
        print(f"Loaded {len(self.world_cfg.mesh)} mesh obstacles")

        trajopt_config = TrajOptSolverConfig.load_from_robot_config(
            self.robot_cfg,
            self.world_cfg,
            self.tensor_args,
            use_cuda_graph=True,
            traj_tsteps=self.trajectory_length,


            num_seeds=4,  # Increase from default 2 to 4-8
            seed_ratio={"linear": 0.2, "bias": 0.8, "start": 0.0, "goal": 0.0},  # Add bias seeds!
            grad_trajopt_iters=150,  # Increase L-BFGS iterations (default ~100)
            collision_activation_distance=self._collision_activation_distance,  # Add safety buffer around obstacles
            trajopt_dt=0.25,  # Increase dt for more flexible timing

        )
        return TrajOptSolver(trajopt_config)

    def _get_joint_limits(self) -> List[Tuple[float, float]]:
        """Get joint limits from robot configuration."""
        limits = []
        kin_cfg = self.robot_cfg.kinematics

        for i in range(len(kin_cfg.kinematics_config.joint_names)):
            lower = -3.14  # kin_cfg.kinematics_config.joint_limits.position[0, i].item()
            upper = 3.14  # kin_cfg.kinematics_config.joint_limits.position[1, i].item()
            limits.append((lower, upper))

        return limits

    def sample_collision_free_config(self, max_attempts: int = 100) -> torch.Tensor:
        for attempt in range(max_attempts):
            # Sample random configuration
            q = torch.zeros(len(self.joint_limits), device=self.tensor_args.device)
            for i, (lower, upper) in enumerate(self.joint_limits):
                q[i] = torch.rand(1, device=self.tensor_args.device) * (upper - lower) + lower

            q_batch = q.unsqueeze(0)  # [1, DOF]

            # Get forward kinematics
            kin_state = self.trajopt_solver.fk(q_batch)
            sphere_tensor = kin_state.link_spheres_tensor.unsqueeze(1)  # [1, 1, n_spheres, 4]

            # Check collision
            collision_result = self.trajopt_solver.world_coll_checker.get_sphere_collision(
                sphere_tensor,
                self._collision_query_buffer,
                self._collision_weight,
                self._collision_activation_distance,
            )

            # If no collision detected, return this configuration
            if not collision_result.any():
                return q

        raise RuntimeError(f"Failed to sample collision-free config after {max_attempts} attempts")

    def generate_trajectory(
        self,
        q_start: torch.Tensor,
        q_goal: torch.Tensor,
    ) -> Dict:
        """
        Generate a collision-free trajectory from start to goal.

        Args:
            q_start: Start joint configuration
            q_goal: Goal joint configuration

        Returns:
            Dictionary containing trajectory and metadata
        """
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

        print(f"\nTrajOpt Results:")
        print(f"  Success: {result.success.item()}")
        print(f"  Solve time: {solve_time:.4f}s")
        print(f"  Original trajectory length: {result.solution.position.shape[0]}")

        if not result.success:
            print("Trajectory generation FAILED")
            return None

        return {
            'start': q_start.cpu().numpy(),
            'goal': q_goal.cpu().numpy(),
            'position': result.solution.position.squeeze(1).cpu().numpy(),  # [T, DOF]
            'velocity': result.solution.velocity.squeeze(1).cpu().numpy() if result.solution.velocity is not None else None,
            'acceleration': result.solution.acceleration.squeeze(1).cpu().numpy() if result.solution.acceleration is not None else None,
            'success': True,
            'solve_time': solve_time,
        }

    def generate_single_example(self, example_id: int = 0) -> Dict:

        # Sample collision-free start and goal
        print("Sampling collision-free start configuration...")
        start = time.time()
        q_start = self.sample_collision_free_config()
        print(f"Sampling time: {time.time() - start:.4f}s")
        print(f"Start: {q_start.cpu().numpy()}")

        print("Sampling collision-free goal configuration...")
        start = time.time()
        q_goal = self.sample_collision_free_config()
        print(f"Sampling time: {time.time() - start:.4f}s")
        print(f"Goal: {q_goal.cpu().numpy()}")

        # Generate trajectory
        print("Generating collision-free trajectory...")
        start = time.time()
        trajectory_data = self.generate_trajectory(q_start, q_goal)
        print(f"Generation time: {time.time() - start:.4f}s")

        if trajectory_data is None:
            print("FAILED to generate trajectory")
            visualize_start_goal_only(self, q_start, q_goal)
            return None

        # Add metadata
        trajectory_data['example_id'] = example_id
        trajectory_data['num_obstacles'] = self.num_obstacles
        trajectory_data['obstacles'] = [
            {
                'name': mesh.name,
                'position': mesh.pose[:3],
                'radius': self._estimate_mesh_radius(mesh)
            }
            for mesh in self.world_cfg.mesh
        ]

        print(trajectory_data['obstacles'])

        return trajectory_data

    def _estimate_mesh_radius(self, mesh):
        """Estimate sphere radius from mesh bounds."""
        trimesh_obj = mesh.get_trimesh_mesh()
        bounds = trimesh_obj.bounds  # [[min_x, min_y, min_z], [max_x, max_y, max_z]]
        # Use half of the maximum extent as radius estimate
        extents = bounds[1] - bounds[0]
        return max(extents) / 2.0


def visualize_trajectory_simple(generator, trajectory_data, n_frames=5):
    """
    Visualize every n-th frame of trajectory on the same plot.

    Args:
        generator: DatasetGenerator instance
        trajectory_data: Dictionary containing trajectory data
        n_frames: Show every n-th frame (default: 5)
    """
    positions = trajectory_data['position']
    obstacles = trajectory_data['obstacles']

    # Select frame indices to visualize
    frame_indices = range(0, len(positions), n_frames)

    # Create colormap for trajectory progression
    colors = plt.cm.viridis(np.linspace(0, 1, len(frame_indices)))

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles once (they don't move)
    print(f"Plotting {len(obstacles)} obstacles")
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
        robot_spheres = generator.trajopt_solver.kinematics.get_robot_as_spheres(q)[0]

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
        state = generator.trajopt_solver.kinematics.get_state(q)

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
    plt.show()


def visualize_start_goal_only(generator, q_start, q_goal):
    """
    Visualize just the start and goal configurations with obstacles.

    Args:
        generator: DatasetGenerator instance
        q_start: Start joint configuration (torch.Tensor)
        q_goal: Goal joint configuration (torch.Tensor)
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles
    for mesh in generator.world_cfg.mesh:
        trimesh_obj = mesh.get_trimesh_mesh()
        bounds = trimesh_obj.bounds
        center = mesh.pose[:3]
        radius = max(bounds[1] - bounds[0]) / 2.0

        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 20)
        x = radius * np.outer(np.cos(u), np.sin(v)) + center[0]
        y = radius * np.outer(np.sin(u), np.sin(v)) + center[1]
        z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + center[2]
        ax.plot_surface(x, y, z, color='red', alpha=0.3)

    # Plot start configuration (green)
    q_start_batch = q_start.unsqueeze(0)
    start_spheres = generator.trajopt_solver.kinematics.get_robot_as_spheres(q_start_batch)[0]
    for sph in start_spheres:
        if sph.radius > 0:
            u = np.linspace(0, 2 * np.pi, 10)
            v = np.linspace(0, np.pi, 10)
            x = sph.radius * np.outer(np.cos(u), np.sin(v)) + sph.pose[0]
            y = sph.radius * np.outer(np.sin(u), np.sin(v)) + sph.pose[1]
            z = sph.radius * np.outer(np.ones(np.size(u)), np.cos(v)) + sph.pose[2]
            ax.plot_surface(x, y, z, color='green', alpha=0.6)

    # Plot goal configuration (blue)
    q_goal_batch = q_goal.unsqueeze(0)
    goal_spheres = generator.trajopt_solver.kinematics.get_robot_as_spheres(q_goal_batch)[0]
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
    plt.show()


def main():
    # Configuration
    robot_config_file = "/home/nataliya/curobo/data_generation/yaml/kinematic_arm_3_dof.yml"
    urdf_file = "/home/nataliya/sim_learning/rodrigues_network/src/urdfs/kinematic_arm_3_dof.urdf"
    num_obstacles = 2
    trajectory_length = 50

    generator = DatasetGenerator(
        robot_config_file=robot_config_file,
        urdf_file=urdf_file,
        num_obstacles=num_obstacles,
        trajectory_length=trajectory_length,
        device="cuda:0" if torch.cuda.is_available() else "cpu"
    )

    example = generator.generate_single_example(example_id=0)

    if example is not None:
        print(f"Trajectory shape: {example['position'].shape}")
        print(f"Start config: {example['start']}")
        print(f"Goal config: {example['goal']}")
        print(f"Solve time: {example['solve_time']:.4f}s")
        print(f"Number of obstacles: {example['num_obstacles']}")

        print(f"\nTrajectory shape: {example['position'].shape}")
        print(f"Start config: {example['start']}")
        print(f"Goal config: {example['goal']}")
        print(f"Solve time: {example['solve_time']:.4f}s")
        print(f"Number of obstacles: {example['num_obstacles']}")
        visualize_trajectory_simple(generator, example)

    else:
        print("Failed to generate example")


if __name__ == "__main__":
    main()
