import numpy as np
import logging
from dm_control import mujoco
from dm_control.rl.control import Environment
from dm_control import viewer
import dm_env
from dm_env import specs


class SimpleTask:
    def initialize_episode(self, physics):
        # sample a target position on the table in front of the robot
        # expressed in world coordinates
        # choose a point near the robot base: x in [0.3,0.6], y in [-0.2,0.2], z in [0.0,0.2]
        self.target = np.array([
            float(np.random.uniform(0.3, 0.6)),
            float(np.random.uniform(-0.2, 0.2)),
            float(np.random.uniform(0.0, 0.2))
        ], dtype=np.float32)
        # success threshold (meters)
        self.pick_threshold = 0.06
        # logging helpers
        self.step_count = 0
        self.last_action = None
        self.prev_ee = None
        logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    def before_step(self, action, physics):
        # store last action and apply control
        self.last_action = action
        try:
            physics.data.ctrl[:] = action
        except Exception:
            pass

    def after_step(self, physics):
        # logging: compute ee position, distance to target, reward, detect spikes
        try:
            ee = np.array(physics.named.data.xpos['attachment_site'], dtype=np.float32)
        except Exception:
            ee = np.array(physics.data.qpos[:3], dtype=np.float32)

        dist = np.linalg.norm(ee - self.target)
        reward = self.get_reward(physics)

        # detect large EE jump (spasm)
        spike = False
        if self.prev_ee is not None:
            delta = np.linalg.norm(ee - self.prev_ee)
            if delta > 0.1:  # threshold for a large jump (meters)
                spike = True

        # detect if ee or target below ground
        fallen = False
        if ee[2] < 0.0 or self.target[2] < 0.0:
            fallen = True

        # print concise status
        logging.info(f"step={self.step_count} ee={ee.tolist()} target={self.target.tolist()} dist={dist:.3f} reward={reward:.3f} action={getattr(self.last_action, 'tolist', lambda: self.last_action)()} spike={spike} fallen={fallen}")

        self.prev_ee = ee.copy()
        self.step_count += 1

    def action_spec(self, physics):
        try:
            n = int(physics.model.nu)
        except Exception:
            n = int(np.shape(physics.data.ctrl)[0])
        return specs.BoundedArray((n,), np.float32, -1.0, 1.0, name='action')

    def get_observation(self, physics):
        # end-effector / attachment site position
        try:
            ee = np.array(physics.named.data.xpos['attachment_site'], dtype=np.float32)
        except Exception:
            # fallback: use first three qpos values (not ideal)
            ee = np.array(physics.data.qpos[:3], dtype=np.float32)

        obs = {
            'qpos': np.array(physics.data.qpos, dtype=np.float32),
            'qvel': np.array(physics.data.qvel, dtype=np.float32),
            'ee_pos': ee,
            'target_pos': np.array(self.target, dtype=np.float32),
            'ee_to_target': (self.target - ee).astype(np.float32)
        }
        return obs

    def get_reward(self, physics):
        try:
            ee = np.array(physics.named.data.xpos['attachment_site'], dtype=np.float32)
        except Exception:
            ee = np.array(physics.data.qpos[:3], dtype=np.float32)
        dist = np.linalg.norm(ee - self.target)
        # negative distance as reward, plus a large bonus on successful pick
        reward = -float(dist)
        if dist < self.pick_threshold:
            reward += 10.0
        return float(reward)

    def get_termination(self, physics):
        try:
            ee = np.array(physics.named.data.xpos['attachment_site'], dtype=np.float32)
        except Exception:
            ee = np.array(physics.data.qpos[:3], dtype=np.float32)
        dist = np.linalg.norm(ee - self.target)
        if dist < self.pick_threshold:

            return 0.0
        return None


def main():
    physics = mujoco.Physics.from_xml_path(
        '/home/hiwi/Desktop/Mujoco/mujoco_menagerie/universal_robots_ur5e/ur5e.xml')
    task = SimpleTask()
    env = Environment(physics, task)

    def random_policy(timestep):
        spec = env.action_spec()
        shape = spec.shape
        low = np.array(spec.minimum, dtype=np.float32)
        high = np.array(spec.maximum, dtype=np.float32)
        if shape == () or shape is None:
            val = low + (high - low) * np.random.rand()
            return np.array(val, dtype=np.float32)
        rnd = np.random.rand(*shape).astype(np.float32)
        low_b = np.broadcast_to(low, shape).astype(np.float32)
        high_b = np.broadcast_to(high, shape).astype(np.float32)
        return (low_b + (high_b - low_b) * rnd).astype(np.float32)

    viewer.launch(env, policy=random_policy)


if __name__ == '__main__':
    main()