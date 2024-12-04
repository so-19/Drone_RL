import os
import pybullet as p
import time


# Function to create a simple drone URDF file
def create_drone_urdf():
    # Define the path to save the URDF file
    urdf_path = "drone.urdf"

    # Open the URDF file for writing
    with open(urdf_path, 'w') as file:
        # Write the URDF structure
        file.write("""<?xml version="1.0"?>
<robot name="simple_drone">
    <!-- Link representing the body of the drone -->
    <link name="base_link">
        <visual>
            <geometry>
                <cylinder radius="0.2" length="0.05"/>
            </geometry>
            <material name="blue"/>
        </visual>
        <collision>
            <geometry>
                <cylinder radius="0.2" length="0.05"/>
            </geometry>
        </collision>
    </link>

    <!-- 4 Thrusters attached to the drone -->
    <link name="thruster1">
        <visual>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
            <material name="red"/>
        </visual>
        <collision>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
        </collision>
    </link>

    <link name="thruster2">
        <visual>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
            <material name="red"/>
        </visual>
        <collision>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
        </collision>
    </link>

    <link name="thruster3">
        <visual>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
            <material name="red"/>
        </visual>
        <collision>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
        </collision>
    </link>

    <link name="thruster4">
        <visual>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
            <material name="red"/>
        </visual>
        <collision>
            <geometry>
                <cylinder radius="0.05" length="0.1"/>
            </geometry>
        </collision>
    </link>

    <!-- Joints connecting the base to the thrusters -->
    <joint name="base_to_thruster1" type="revolute">
        <parent link="base_link"/>
        <child link="thruster1"/>
        <origin xyz="0.2 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit effort="100" lower="0" upper="3.1416" velocity="1"/>
    </joint>

    <joint name="base_to_thruster2" type="revolute">
        <parent link="base_link"/>
        <child link="thruster2"/>
        <origin xyz="-0.2 0 0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit effort="100" lower="0" upper="3.1416" velocity="1"/>
    </joint>

    <joint name="base_to_thruster3" type="revolute">
        <parent link="base_link"/>
        <child link="thruster3"/>
        <origin xyz="0.2 0 -0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit effort="100" lower="0" upper="3.1416" velocity="1"/>
    </joint>

    <joint name="base_to_thruster4" type="revolute">
        <parent link="base_link"/>
        <child link="thruster4"/>
        <origin xyz="-0.2 0 -0.1" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit effort="100" lower="0" upper="3.1416" velocity="1"/>
    </joint>

    <!-- Additional links and joints for the drone's components can be added here -->
</robot>
""")
    print(f"URDF file created at: {urdf_path}")


# Function to load the URDF into the simulation and run it in PyBullet
def view_drone():
    # Connect to PyBullet in GUI mode
    p.connect(p.GUI)  # Use p.DIRECT for non-GUI mode

    # Load the URDF file (ensure the file path is correct)
    drone_id = p.loadURDF("drone.urdf", basePosition=[0, 0, 2])

    # Run the simulation for a few seconds
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1. / 240)  # Slow down the simulation to visualize it

    # Disconnect from the simulation
    p.disconnect()


# Create the URDF file
create_drone_urdf()

# Visualize the drone in PyBullet
view_drone()
