#!/usr/bin/env python3
"""
Load and analyze trajectory dataset from HDF5.
Simple statistics and verification - no plotting.
"""

import h5py
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# ============================================================================
# CONFIGURATION
# ============================================================================

HDF5_PATH = "data/trajectories.h5"
VERIFY_DATA = True  # Run integrity checks
SHOW_DETAILED_STATS = True  # Show per-trajectory details
MAX_TRAJECTORIES_TO_SHOW = 10  # Limit detailed output


# ============================================================================
# TRAJECTORY LOADER
# ============================================================================

class TrajectoryDataset:
    """Load and analyze trajectory dataset."""

    def __init__(self, hdf5_path: str):
        """Initialize and load dataset summary."""
        self.hdf5_path = Path(hdf5_path)

        if not self.hdf5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.hdf5_path}")

        print("=" * 70)
        print(f"TRAJECTORY DATASET LOADER")
        print("=" * 70)
        print(f"File: {self.hdf5_path}")
        print(f"Size: {self.hdf5_path.stat().st_size / 1024 / 1024:.2f} MB")

        self.trajectory_ids = self._get_trajectory_ids()
        print(f"Trajectories: {len(self.trajectory_ids)}")

        with h5py.File(self.hdf5_path, 'r') as f:
            print(f"Created: {f.attrs.get('created_at', 'Unknown')}")

        print("=" * 70)

    def _get_trajectory_ids(self) -> List[int]:
        """Get sorted list of trajectory IDs."""
        with h5py.File(self.hdf5_path, 'r') as f:
            groups = [k for k in f.keys() if k.startswith('trajectory_')]
            ids = [int(name.split('_')[1]) for name in groups]
        return sorted(ids)

    def load_trajectory(self, traj_id: int) -> Dict[str, Any]:
        """Load single trajectory data."""
        group_name = f"trajectory_{traj_id:06d}"

        with h5py.File(self.hdf5_path, 'r') as f:
            if group_name not in f:
                raise KeyError(f"Trajectory {traj_id} not found")

            grp = f[group_name]

            # Helper to load dataset (handles both scalars and arrays)
            def load_dataset(name):
                dset = grp[name]
                if dset.shape == ():  # Scalar dataset
                    return dset[()]
                else:  # Array dataset
                    return dset[:]

            return {
                'id': traj_id,
                'start': load_dataset('start'),
                'goal': load_dataset('goal'),
                'optimized_plan': load_dataset('optimized_plan'),
                'optimized_dt': load_dataset('optimized_dt'),
                'interpolated_plan': load_dataset('interpolated_plan'),
                'obstacle_positions': load_dataset('obstacle_positions'),
                'obstacle_radii': load_dataset('obstacle_radii'),
                'num_obstacles': grp.attrs['num_obstacles'],
                'solve_time': grp.attrs['solve_time'],
                'motion_time': grp.attrs['motion_time'],
                'interpolation_dt': grp.attrs['interpolation_dt'],
            }

    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics."""
        print("\n" + "=" * 70)
        print("COMPUTING STATISTICS")
        print("=" * 70)

        stats = {
            'num_trajectories': len(self.trajectory_ids),
            'solve_times': [],
            'motion_times': [],
            'num_obstacles': [],
            'obstacle_radii': [],
            'obstacle_positions': [],
            'trajectory_lengths': [],
            'joint_ranges': [],
        }

        for traj_id in self.trajectory_ids:
            data = self.load_trajectory(traj_id)

            stats['solve_times'].append(data['solve_time'])
            stats['motion_times'].append(data['motion_time'])
            stats['num_obstacles'].append(data['num_obstacles'])
            stats['trajectory_lengths'].append(len(data['interpolated_plan']))

            if data['num_obstacles'] > 0:
                stats['obstacle_radii'].extend(data['obstacle_radii'].tolist())
                stats['obstacle_positions'].extend(data['obstacle_positions'].tolist())

            # Joint angle ranges
            all_configs = np.vstack([data['start'], data['goal'], data['interpolated_plan']])
            joint_min = all_configs.min(axis=0)
            joint_max = all_configs.max(axis=0)
            stats['joint_ranges'].append(joint_max - joint_min)

        # Convert lists to arrays
        stats['solve_times'] = np.array(stats['solve_times'])
        stats['motion_times'] = np.array(stats['motion_times'])
        stats['num_obstacles'] = np.array(stats['num_obstacles'])
        stats['trajectory_lengths'] = np.array(stats['trajectory_lengths'])
        stats['joint_ranges'] = np.array(stats['joint_ranges'])

        if stats['obstacle_radii']:
            stats['obstacle_radii'] = np.array(stats['obstacle_radii'])
            stats['obstacle_positions'] = np.array(stats['obstacle_positions'])

        return stats

    def print_statistics(self, stats: Dict[str, Any]):
        """Print formatted statistics."""
        print("\n" + "-" * 70)
        print("DATASET STATISTICS")
        print("-" * 70)

        # Solve times
        print(f"\nSolve Time (seconds):")
        print(f"  Mean:   {stats['solve_times'].mean():.3f}")
        print(f"  Min:    {stats['solve_times'].min():.3f}")
        print(f"  Max:    {stats['solve_times'].max():.3f}")
        print(f"  Std:    {stats['solve_times'].std():.3f}")

        # Motion times
        print(f"\nMotion Time (seconds):")
        print(f"  Mean:   {stats['motion_times'].mean():.3f}")
        print(f"  Min:    {stats['motion_times'].min():.3f}")
        print(f"  Max:    {stats['motion_times'].max():.3f}")
        print(f"  Std:    {stats['motion_times'].std():.3f}")

        # Trajectory lengths
        print(f"\nTrajectory Length (interpolated steps):")
        print(f"  Mean:   {stats['trajectory_lengths'].mean():.1f}")
        print(f"  Min:    {stats['trajectory_lengths'].min()}")
        print(f"  Max:    {stats['trajectory_lengths'].max()}")

        # Obstacles
        print(f"\nObstacles per Trajectory:")
        print(f"  Mean:   {stats['num_obstacles'].mean():.2f}")
        print(f"  Min:    {stats['num_obstacles'].min()}")
        print(f"  Max:    {stats['num_obstacles'].max()}")
        print(f"  Distribution: {np.bincount(stats['num_obstacles']).tolist()}")

        if len(stats['obstacle_radii']) > 0:
            print(f"\nObstacle Radii:")
            print(f"  Mean:   {stats['obstacle_radii'].mean():.3f}")
            print(f"  Min:    {stats['obstacle_radii'].min():.3f}")
            print(f"  Max:    {stats['obstacle_radii'].max():.3f}")

            print(f"\nObstacle Positions (XYZ):")
            obs_pos = stats['obstacle_positions']
            print(f"  X range: [{obs_pos[:, 0].min():.2f}, {obs_pos[:, 0].max():.2f}]")
            print(f"  Y range: [{obs_pos[:, 1].min():.2f}, {obs_pos[:, 1].max():.2f}]")
            print(f"  Z range: [{obs_pos[:, 2].min():.2f}, {obs_pos[:, 2].max():.2f}]")

        # Joint ranges
        print(f"\nJoint Motion Range (radians):")
        mean_ranges = stats['joint_ranges'].mean(axis=0)
        for i, r in enumerate(mean_ranges):
            print(f"  Joint {i+1}: {r:.3f} rad ({np.rad2deg(r):.1f}°)")

        print("-" * 70)

    def verify_all(self) -> Dict[str, int]:
        """Verify all trajectories."""
        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)

        passed = 0
        failed = 0
        issues = []

        for traj_id in self.trajectory_ids:
            try:
                data = self.load_trajectory(traj_id)

                # Check 1: Shapes
                dof = len(data['start'])
                assert len(data['goal']) == dof, f"Goal shape mismatch"
                assert data['optimized_plan'].shape[1] == dof, f"Optimized plan shape"
                assert data['interpolated_plan'].shape[1] == dof, f"Interpolated plan shape"

                # Check 2: No NaN/Inf
                for key in ['start', 'goal', 'optimized_plan', 'interpolated_plan']:
                    assert not np.any(np.isnan(data[key])), f"{key} has NaN"
                    assert not np.any(np.isinf(data[key])), f"{key} has Inf"

                # Check 3: Obstacle consistency
                n_obs = data['num_obstacles']
                assert len(data['obstacle_positions']) == n_obs, f"Obstacle count mismatch"
                assert len(data['obstacle_radii']) == n_obs, f"Radii count mismatch"

                # Check 4: Trajectory endpoints
                start_match = np.allclose(data['interpolated_plan'][0], data['start'], atol=0.1)
                goal_match = np.allclose(data['interpolated_plan'][-1], data['goal'], atol=0.1)

                if not start_match:
                    issues.append(f"Traj {traj_id}: Start mismatch")
                if not goal_match:
                    issues.append(f"Traj {traj_id}: Goal mismatch")

                passed += 1

            except Exception as e:
                failed += 1
                issues.append(f"Traj {traj_id}: {str(e)}")

        # Print results
        print(f"\nResults: {passed} passed, {failed} failed")

        if issues:
            print(f"\nIssues found ({len(issues)}):")
            for issue in issues[:10]:  # Show first 10
                print(f"  - {issue}")
            if len(issues) > 10:
                print(f"  ... and {len(issues) - 10} more")
        else:
            print("✓ All checks passed!")

        print("=" * 70)

        return {'passed': passed, 'failed': failed}

    def show_sample_trajectories(self, max_show: int = MAX_TRAJECTORIES_TO_SHOW):
        """Show details of sample trajectories."""
        print("\n" + "=" * 70)
        print(f"SAMPLE TRAJECTORIES (first {max_show})")
        print("=" * 70)

        for traj_id in self.trajectory_ids[:max_show]:
            data = self.load_trajectory(traj_id)

            print(f"\nTrajectory {traj_id}:")
            print(f"  Start:     {data['start']}")
            print(f"  Goal:      {data['goal']}")
            print(f"  Obstacles: {data['num_obstacles']}")
            print(f"  Solve:     {data['solve_time']:.3f}s")
            print(f"  Motion:    {data['motion_time']:.3f}s")
            print(f"  Steps:     {len(data['interpolated_plan'])}")
            # print(data["interpolated_plan"])

        if len(self.trajectory_ids) > max_show:
            print(f"\n... and {len(self.trajectory_ids) - max_show} more trajectories")

        print("=" * 70)


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Load and analyze dataset."""

    # Load dataset
    dataset = TrajectoryDataset(HDF5_PATH)

    # Compute and print statistics
    stats = dataset.compute_statistics()
    dataset.print_statistics(stats)

    # Verify data
    if VERIFY_DATA:
        results = dataset.verify_all()

    # Show sample trajectories
    if SHOW_DETAILED_STATS:
        dataset.show_sample_trajectories()

    print("\n✓ Analysis complete!\n")


if __name__ == "__main__":
    main()
