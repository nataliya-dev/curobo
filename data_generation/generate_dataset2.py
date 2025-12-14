#!/usr/bin/env python3
# Standard Library
import time

# Third Party
import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D

# CuRobo
from curobo.geom.types import Sphere, WorldConfig
from curobo.types.base import TensorDeviceType
from curobo.types.robot import JointState
from curobo.wrap.reacher.motion_gen import MotionGen, MotionGenConfig, MotionGenPlanConfig


# ============================================================================
# HARDCODED PARAMETERS
# ============================================================================

# Robot Configuration
ROBOT_CONFIG_FILE = "/home/nataliya/curobo/data_generation/yaml/kinematic_arm_3_dof.yml"
DEVICE = "cuda:0"

# Obstacle Generation Settings
NUM_OBSTACLES = 2
OBSTACLE_RADIUS_MIN = 0.05
OBSTACLE_RADIUS_MAX = 0.15
OBSTACLE_X_RANGE = [-0.8, 0.8]
OBSTACLE_Y_RANGE = [-0.8, 0.8]
OBSTACLE_Z_RANGE = [0.0, 0.0]

# Seed Configuration (number of parallel optimization attempts)
NUM_IK_SEEDS = 32              # Seeds for inverse kinematics
NUM_TRAJOPT_SEEDS = 12         # Seeds for trajectory optimization (12-16 recommended)
NUM_GRAPH_SEEDS = 8            # Seeds for graph planner
NUM_TRAJOPT_NOISY_SEEDS = 1    # Augmented trajectories per seed (keep at 1)

# Optimization Iterations
GRAD_TRAJOPT_ITERS = 350       # Iterations for gradient-based trajectory optimization
GRAPH_TRAJOPT_ITERS = 400      # Iterations when using graph-seeded trajectories
FINETUNE_TRAJOPT_ITERS = 300   # Iterations for final refinement

# Trajectory Time Configuration
TRAJOPT_TSTEPS = 50            # Number of waypoints in optimized trajectory
TRAJOPT_DT = 0.15              # Time step between waypoints (seconds)
INTERPOLATION_DT = 0.05        # Time step for interpolated output (seconds)
INTERPOLATION_STEPS = 200      # Buffer size for interpolated trajectory

# Collision Configuration
COLLISION_ACTIVATION_DISTANCE = 0.02   # Distance to activate collision cost (meters)
SELF_COLLISION_CHECK = True            # Enable self-collision checking
SELF_COLLISION_OPT = True              # Enable self-collision cost in optimization

# Convergence Thresholds
POSITION_THRESHOLD = 0.005     # Position error threshold (meters)
ROTATION_THRESHOLD = 0.05      # Rotation error threshold
CSPACE_THRESHOLD = 0.05        # Joint space error threshold (radians)

# Planning Configuration
MAX_PLANNING_ATTEMPTS = 20     # Maximum attempts for entire planning loop
PLANNING_TIMEOUT = 15.0        # Overall planning timeout (seconds)
ENABLE_GRAPH = True            # Use graph planner for seed generation
ENABLE_FINETUNE_TRAJOPT = True  # Enable trajectory refinement
ENABLE_GRAPH_ATTEMPT = 8       # Attempt number to enable graph if disabled
NEED_GRAPH_SUCCESS = False     # Don't require graph to succeed

# Advanced Settings
USE_CUDA_GRAPH = True          # Use CUDA graphs for speedup
OPTIMIZE_DT = True             # Optimize time-optimal trajectories
MINIMIZE_JERK = True           # Minimize jerk in trajectories
FILTER_ROBOT_COMMAND = False   # Filter to remove artifacts

# Output Settings
SAVE_PLOT = True
PLOT_FILENAME = "trajectory.png"


# ============================================================================
# OBSTACLE GENERATION
# ============================================================================

def generate_random_obstacles():
    """
    Generate random spherical obstacles within defined ranges.

    Returns:
        obstacles (list): List of obstacle dictionaries
        world_config (WorldConfig): CuRobo world configuration
    """
    obstacles = []
    sphere_list = []

    for i in range(NUM_OBSTACLES):
        # Random position
        position = [
            np.random.uniform(OBSTACLE_X_RANGE[0], OBSTACLE_X_RANGE[1]),
            np.random.uniform(OBSTACLE_Y_RANGE[0], OBSTACLE_Y_RANGE[1]),
            np.random.uniform(OBSTACLE_Z_RANGE[0], OBSTACLE_Z_RANGE[1]),
        ]

        # Random radius
        radius = np.random.uniform(OBSTACLE_RADIUS_MIN, OBSTACLE_RADIUS_MAX)

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

    # Store and convert world config to mesh representation
    world_config = WorldConfig.create_mesh_world(world_config)

    return obstacles, world_config


# ============================================================================
# TRAJECTORY GENERATOR CLASS
# ============================================================================

class TrajectoryGenerator:
    """Generate trajectories using MotionGen for a 3DOF robot arm."""

    def __init__(self, obstacles, world_config):
        """
        Initialize the trajectory generator.

        Args:
            obstacles: List of obstacle dictionaries for visualization
            world_config: CuRobo WorldConfig with obstacles
        """
        print("Initializing TrajectoryGenerator...")
        start_time = time.time()

        self.tensor_args = TensorDeviceType(device=torch.device(DEVICE))
        self.obstacles = obstacles  # Store for visualization

        # Load MotionGen configuration
        config_start = time.time()

        self.motion_gen_config = MotionGenConfig.load_from_robot_config(
            ROBOT_CONFIG_FILE,
            world_config,
            self.tensor_args,
            # Seed configuration
            num_ik_seeds=NUM_IK_SEEDS,
            num_trajopt_seeds=NUM_TRAJOPT_SEEDS,
            num_graph_seeds=NUM_GRAPH_SEEDS,
            num_trajopt_noisy_seeds=NUM_TRAJOPT_NOISY_SEEDS,
            # Optimization iterations
            grad_trajopt_iters=GRAD_TRAJOPT_ITERS,
            graph_trajopt_iters=GRAPH_TRAJOPT_ITERS,
            finetune_trajopt_iters=FINETUNE_TRAJOPT_ITERS,
            # Trajectory configuration
            trajopt_tsteps=TRAJOPT_TSTEPS,
            trajopt_dt=TRAJOPT_DT,
            interpolation_dt=INTERPOLATION_DT,
            interpolation_steps=INTERPOLATION_STEPS,
            # Collision parameters
            collision_activation_distance=COLLISION_ACTIVATION_DISTANCE,
            self_collision_check=SELF_COLLISION_CHECK,
            self_collision_opt=SELF_COLLISION_OPT,
            # Convergence thresholds
            position_threshold=POSITION_THRESHOLD,
            rotation_threshold=ROTATION_THRESHOLD,
            cspace_threshold=CSPACE_THRESHOLD,
            # Advanced settings
            use_cuda_graph=USE_CUDA_GRAPH,
            optimize_dt=OPTIMIZE_DT,
            minimize_jerk=MINIMIZE_JERK,
            filter_robot_command=FILTER_ROBOT_COMMAND,
        )

        config_time = time.time() - config_start
        print(f"  Config load time: {config_time:.3f}s")

        self.motion_gen = MotionGen(self.motion_gen_config)

        # Get robot properties
        self.dof = self.motion_gen.kinematics.get_dof()
        self.joint_names = self.motion_gen.joint_names
        self.joint_limits = self.motion_gen.kinematics.get_joint_limits()

        print(f"Robot DOF: {self.dof}")
        print(f"Joint names: {self.joint_names}")
        print(f"Number of obstacles: {len(self.obstacles)}")

        total_time = time.time() - start_time
        print(f"Total initialization time: {total_time:.3f}s")
        print("Ready!\n")

    def sample_random_joint_state(self):
        """Sample a random joint configuration within joint limits."""
        lower = self.joint_limits.position[0].unsqueeze(0)
        upper = self.joint_limits.position[1].unsqueeze(0)

        position = lower + torch.rand(
            1, self.dof,
            device=self.tensor_args.device,
            dtype=self.tensor_args.dtype
        ) * (upper - lower)

        return JointState.from_position(position, joint_names=self.joint_names)

    def sample_collision_free_state(self, max_attempts=100):
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

            if not valid:
                print(status)

            if valid:
                return state

        print(f"[WARN] Could not find collision-free state after {max_attempts} attempts")
        return None

    def generate_trajectory(self, start_state, goal_state):
        """
        Generate a single trajectory from random start to random goal.

        Returns:
            success (bool): Whether planning succeeded
            trajectory_data (dict): Dictionary containing trajectory data
        """
        total_start = time.time()

        print(f"Start: {start_state.position.cpu().numpy().flatten()}")
        print(f"Goal:  {goal_state.position.cpu().numpy().flatten()}")

        # Configure planning
        plan_config = MotionGenPlanConfig(
            enable_graph=ENABLE_GRAPH,
            enable_opt=True,
            max_attempts=MAX_PLANNING_ATTEMPTS,
            timeout=PLANNING_TIMEOUT,
            enable_finetune_trajopt=ENABLE_FINETUNE_TRAJOPT,
            enable_graph_attempt=ENABLE_GRAPH_ATTEMPT,
            need_graph_success=NEED_GRAPH_SUCCESS,
        )

        # Plan trajectory
        print("Planning trajectory...")
        plan_start = time.time()
        result = self.motion_gen.plan_single_js(
            start_state=start_state,
            goal_state=goal_state,
            plan_config=plan_config,
        )
        plan_time = time.time() - plan_start

        # Check success
        if not result.success.item():
            print(f"[FAILED] Planning failed: {result.status}")
            print(f"  Planning time: {plan_time:.3f}s")
            return False, None

        # Extract trajectory data
        extract_start = time.time()
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
            'obstacles': self.obstacles,  # Include obstacles for visualization
        }
        extract_time = time.time() - extract_start

        total_time = time.time() - total_start

        print(f"[SUCCESS] Trajectory generated!")
        print(f"  Solve time: {trajectory_data['solve_time']:.3f}s")
        print(f"  Motion time: {trajectory_data['motion_time']:.3f}s")
        print(f"  Optimized shape: {trajectory_data['optimized_plan'].shape}")
        print(f"  Interpolated shape: {trajectory_data['interpolated_plan'].shape}")
        print(f"  Data extraction time: {extract_time:.3f}s")
        print(f"  Total generation time: {total_time:.3f}s")

        return True, trajectory_data


# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def visualize_trajectory_simple(generator, trajectory_data, n_frames=5):
    """
    Visualize every n-th frame of trajectory on the same plot.

    Args:
        generator: TrajectoryGenerator instance
        trajectory_data: Dictionary containing trajectory data
        n_frames: Show every n-th frame (default: 5)
    """
    positions = trajectory_data['interpolated_plan']
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

    print(f"Plotting {len(frame_indices)} frames")

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

    if SAVE_PLOT:
        plt.savefig(PLOT_FILENAME, dpi=150)
        print(f"Plot saved to {PLOT_FILENAME}")
    else:
        plt.show()

    plt.close()


def visualize_start_goal_only(generator, q_start, q_goal, obstacles):
    """
    Visualize just the start and goal configurations.

    Args:
        generator: TrajectoryGenerator instance
        q_start: Start joint configuration (numpy array)
        q_goal: Goal joint configuration (numpy array)
        obstacles: List of obstacle dictionaries
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot obstacles
    print(f"Plotting {len(obstacles)} obstacles")
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

    if SAVE_PLOT:
        plt.savefig(PLOT_FILENAME.replace('.png', '_start_goal.png'), dpi=150)
        print(f"Start/Goal plot saved to {PLOT_FILENAME.replace('.png', '_start_goal.png')}")
    else:
        plt.show()

    plt.close()


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Generate a single trajectory and visualize it."""
    print("=" * 70)
    print("3DOF Robot Arm Trajectory Generation with MotionGen")
    print("=" * 70)
    print()

    total_start = time.time()

    # Generate random obstacles
    print("Generating random obstacles...")
    obs_start = time.time()
    obstacles, world_config = generate_random_obstacles()
    obs_time = time.time() - obs_start
    print(f"  Generated {len(obstacles)} obstacles")
    print(f"  Obstacle generation time: {obs_time:.3f}s\n")

    # Create generator
    init_start = time.time()
    generator = TrajectoryGenerator(obstacles, world_config)
    init_time = time.time() - init_start

    # Sample random start and goal
    sample_start = time.time()
    start_state = generator.sample_collision_free_state()
    goal_state = generator.sample_collision_free_state()
    sample_time = time.time() - sample_start
    print(f"  Sampling time: {sample_time:.3f}s")

    print("\nVisualizing start and goal...")
    sg_start = time.time()
    visualize_start_goal_only(
        generator,
        start_state.position.cpu().numpy().flatten(),
        goal_state.position.cpu().numpy().flatten(),
        obstacles
    )
    # Generate single trajectory
    gen_start = time.time()
    success, traj_data = generator.generate_trajectory(start_state, goal_state)
    gen_time = time.time() - gen_start

    if not success:
        print("\nFailed to generate trajectory.")
        return

    # Visualize trajectory
    print("\nVisualizing trajectory...")
    viz_start = time.time()
    visualize_trajectory_simple(generator, traj_data, n_frames=5)
    viz_time = time.time() - viz_start
    print(f"  Visualization time: {viz_time:.3f}s")

    sg_time = time.time() - sg_start
    print(f"  Start/Goal visualization time: {sg_time:.3f}s")

    total_time = time.time() - total_start

    print("\n" + "=" * 70)
    print("TIMING SUMMARY")
    print("=" * 70)
    print(f"Obstacle Gen:       {obs_time:8.3f}s")
    print(f"Initialization:     {init_time:8.3f}s")
    print(f"Trajectory Gen:     {gen_time:8.3f}s")
    print(f"Visualization:      {viz_time:8.3f}s")
    print(f"Start/Goal Viz:     {sg_time:8.3f}s")
    print(f"-" * 70)
    print(f"TOTAL:              {total_time:8.3f}s")
    print("=" * 70)


if __name__ == "__main__":
    main()
