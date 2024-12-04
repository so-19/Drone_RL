import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
import time

class RoboticArmEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.action_space = spaces.Box(-1, 1, (2,), np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, (9,), np.float32)

        self.model = mujoco.MjModel.from_xml_string("""
        <mujoco>
            <option gravity="0 0 -9.81" />
            <visual>
                <global offwidth="1920" offheight="1080" />
            </visual>
            <worldbody>
                <light diffuse=".8 .8 .8" pos="0 0 3" dir="0 0 -1"/>
                <geom name="ground" type="plane" size="2 2 0.1" rgba=".9 .9 .9 1"/>
                <body name="base" pos="0 0 0.1">
                    <joint name="root" type="free"/>
                    <geom name="base" type="cylinder" size=".1 .05" rgba=".2 .2 .8 1"/>
                    <body name="link1" pos="0 0 0.05">
                        <joint name="joint1" type="hinge" axis="0 0 1"/>
                        <geom name="link1" type="capsule" size=".05" fromto="0 0 0 0.5 0 0" rgba=".8 .2 .2 1"/>
                        <body name="link2" pos="0.5 0 0">
                            <joint name="joint2" type="hinge" axis="0 1 0"/>
                            <geom name="link2" type="capsule" size=".05" fromto="0 0 0 0.5 0 0" rgba=".2 .8 .2 1"/>
                            <site name="end_effector" pos="0.5 0 0" size="0.05" rgba="0 0 0 1"/>
                        </body>
                    </body>
                </body>
            </worldbody>
            <actuator>
                <motor joint="joint1" name="motor1" gear="100" />
                <motor joint="joint2" name="motor2" gear="100" />
            </actuator>
        </mujoco>
        """)
        self.data = mujoco.MjData(self.model)
        self.viewport = mujoco.MjrRect(0, 0, 1920, 1080)
        self.scene = mujoco.MjvScene(self.model, maxgeom=10000)
        self.cam = mujoco.MjvCamera()
        self.opt = mujoco.MjvOption()

        # Create the correct rendering context using just the model and a flag (integer)
        self.context = mujoco.MjrContext(self.model, 0)  # Use 0 or another integer flag if needed

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.data.qpos = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        self.data.qvel = np.zeros(8)
        return self._get_obs(), {}

    def step(self, action):
        self.data.ctrl = action
        mujoco.mj_step(self.model, self.data)
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = False
        truncated = False
        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        return np.concatenate([self.data.qpos, self.data.qvel, self.data.qacc])

    def _compute_reward(self):
        target_pos = np.array([0.5, 0.5, 0.0])
        current_pos = self._get_end_effector_pos()
        distance = np.linalg.norm(target_pos - current_pos)
        return -distance

    def _get_end_effector_pos(self):
        return self.data.site_xpos[0]

    def render(self, mode='human'):
        mujoco.mj_step(self.model, self.data)
        img = np.zeros((self.viewport.height, self.viewport.width, 3), dtype=np.uint8)
        mujoco.mjr_render(self.viewport, self.scene, self.context)  # Now using MjrContext correctly
        mujoco.mjr_readPixels(img, None, self.viewport, self.context)  # Read the pixels
        return np.flipud(img)

def simulate_and_visualize():
    env = RoboticArmEnv()
    state, _ = env.reset()
    frames = []

    for _ in range(1000):
        action = np.random.uniform(-1, 1, size=2)
        next_state, _, terminated, truncated, _ = env.step(action)
        frames.append(env.render())
        if terminated or truncated:
            break
        state = next_state
        time.sleep(0.01)

    return frames

if __name__ == "__main__":
    frames = simulate_and_visualize()
    print("Simulation completed. You can now view the animation on your phone.")
