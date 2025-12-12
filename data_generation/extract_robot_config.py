"""
Extract robot configuration information for CuRobo YAML files.
Generates collision_link_names, collision_spheres, and self_collision_ignore.
"""

import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


class RobotConfigExtractor:
    """Extract configuration info from URDF for CuRobo."""

    def __init__(self, urdf_path: str):
        """
        Initialize with URDF file.

        Args:
            urdf_path: Path to robot URDF file
        """
        self.urdf_path = Path(urdf_path)

        # Parse URDF
        self.tree = ET.parse(self.urdf_path)
        self.root = self.tree.getroot()

        # Extract information
        self._extract_link_names()
        self._extract_collision_spheres()
        self._extract_adjacent_links()
        self._extract_joint_names()

    def _extract_link_names(self):
        """Get all link names (excluding world)."""
        self.link_names = []

        for link in self.root.findall('link'):
            link_name = link.get('name')
            if link_name and link_name != "world":
                self.link_names.append(link_name)

        print(f"Found {len(self.link_names)} links")

    def _extract_collision_spheres(self):
        """Extract collision sphere information from URDF."""
        self.collision_spheres = {}

        # Find all links with collision geometry
        for link in self.root.findall('link'):
            link_name = link.get('name')

            # Skip world
            if link_name == "world":
                continue

            # Find collision spheres
            collision = link.find('collision')
            if collision is not None:
                geometry = collision.find('geometry')
                if geometry is not None:
                    sphere = geometry.find('sphere')
                    if sphere is not None:
                        radius = float(sphere.get('radius'))

                        # Get collision origin (defaults to [0,0,0])
                        origin = collision.find('origin')
                        if origin is not None:
                            xyz_str = origin.get('xyz', '0 0 0')
                            center = [float(x) for x in xyz_str.split()]
                        else:
                            center = [0.0, 0.0, 0.0]

                        # Add sphere to dictionary
                        if link_name not in self.collision_spheres:
                            self.collision_spheres[link_name] = []

                        self.collision_spheres[link_name].append({
                            'center': center,
                            'radius': radius
                        })

        print(f"Found collision spheres for {len(self.collision_spheres)} links")

    def _extract_adjacent_links(self):
        """Extract adjacent link pairs (connected by joints)."""
        self.adjacent_links = defaultdict(set)

        # Get all joint connections
        for joint in self.root.findall('joint'):
            parent_elem = joint.find('parent')
            child_elem = joint.find('child')

            if parent_elem is not None and child_elem is not None:
                parent_name = parent_elem.get('link')
                child_name = child_elem.get('link')

                # Skip world connections
                if parent_name == "world" or child_name == "world":
                    continue

                # Add bidirectional adjacency
                self.adjacent_links[parent_name].add(child_name)
                self.adjacent_links[child_name].add(parent_name)

        # Convert sets to sorted lists
        self.adjacent_links = {
            link: sorted(list(adjacent))
            for link, adjacent in self.adjacent_links.items()
        }

        print(f"Built adjacency for {len(self.adjacent_links)} links")

    def print_collision_link_names(self):
        """Print collision_link_names for YAML config."""
        print("\n" + "=" * 70)
        print("COLLISION_LINK_NAMES (for robot config YAML)")
        print("=" * 70)
        print("collision_link_names:")
        print("  [")
        for i, link in enumerate(self.link_names):
            comma = "," if i < len(self.link_names) - 1 else ""
            print(f'    "{link}"{comma}')
        print("  ]")

    def save_collision_spheres_yaml(self, output_path: str = "collision_spheres.yml"):
        """Save collision spheres to YAML file."""
        # Prepare data in the format CuRobo expects
        spheres_data = {"collision_spheres": {}}

        for link_name, spheres in self.collision_spheres.items():
            spheres_data["collision_spheres"][link_name] = []
            for sphere in spheres:
                spheres_data["collision_spheres"][link_name].append({
                    "center": sphere["center"],
                    "radius": sphere["radius"]
                })

        # Save to file
        output_file = Path(output_path)
        with open(output_file, 'w') as f:
            yaml.dump(spheres_data, f, default_flow_style=False, sort_keys=False)

        print(f"\n" + "=" * 70)
        print(f"COLLISION SPHERES saved to: {output_file.absolute()}")
        print("=" * 70)
        print(f"Add to your robot config YAML:")
        print(f'  collision_spheres: "spheres/{output_file.name}"')

    def print_self_collision_ignore(self):
        """Print self_collision_ignore dictionary."""
        print("\n" + "=" * 70)
        print("SELF_COLLISION_IGNORE (for robot config YAML)")
        print("=" * 70)
        print("self_collision_ignore:")
        print("  {")

        link_list = sorted(self.adjacent_links.keys())
        for i, link_name in enumerate(link_list):
            adjacent = self.adjacent_links[link_name]
            adjacent_str = ", ".join([f'"{adj}"' for adj in adjacent])
            comma = "," if i < len(link_list) - 1 else ""
            print(f'    "{link_name}": [{adjacent_str}]{comma}')

        print("  }")

    def save_all_configs(self, spheres_file: str = "collision_spheres.yml"):
        """Generate and save/print all configuration information."""
        print("\n" + "#" * 70)
        print("# ROBOT CONFIGURATION EXTRACTOR")
        print("#" * 70)

        # Print collision link names
        self.print_collision_link_names()

        # Save collision spheres to YAML
        self.save_collision_spheres_yaml(spheres_file)

        # Print self-collision ignore
        self.print_self_collision_ignore()

        # Print joint names
        self.print_joint_names()

        print("\n" + "#" * 70)
        print("# EXTRACTION COMPLETE")
        print("#" * 70)

    def _extract_joint_names(self):
        """Extract actuated joint names (excluding fixed joints)."""
        self.joint_names_list = []

        for joint in self.root.findall('joint'):
            joint_type = joint.get('type')
            joint_name = joint.get('name')

            # Only include actuated joints (not fixed)
            if joint_type != 'fixed' and joint_name:
                self.joint_names_list.append(joint_name)

        print(f"Found {len(self.joint_names_list)} actuated joints")

    def print_joint_names(self):
        """Print joint_names for YAML config."""
        print("\n" + "=" * 70)
        print("JOINT_NAMES (for cspace in robot config YAML)")
        print("=" * 70)
        print("joint_names: [", end="")
        for i, name in enumerate(self.joint_names_list):
            comma = ", " if i < len(self.joint_names_list) - 1 else ""
            print(f'"{name}"{comma}', end="")
        print("]")


def main():
    """Main function."""

    # Path to your robot URDF
    urdf_path = "/home/nataliya/sim_learning/rodrigues_network/src/urdfs/kinematic_arm_3_dof.urdf"

    # Output file for collision spheres
    output_spheres = "yaml/kinematic_arm_3_dof_spheres.yml"

    # Extract configuration
    extractor = RobotConfigExtractor(urdf_path)

    # Generate all outputs
    extractor.save_all_configs(output_spheres)

    print("\nUsage:")
    print("1. Copy 'collision_link_names' section into your robot config YAML")
    print("2. Place the collision_spheres YAML in a 'spheres/' directory")
    print("3. Reference it in your config: collision_spheres: 'spheres/kinematic_arm_3_dof_spheres.yml'")
    print("4. Copy 'self_collision_ignore' section into your robot config YAML")


if __name__ == "__main__":
    main()
