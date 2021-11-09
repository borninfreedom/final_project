from stable_baselines3 import PPO
from easy_logx.easy_logx import EasyLog
from kuka_visual_grasp2 import KukaVisualGraspEnv
import logging

if __name__ == '__main__':

    logger = EasyLog(log_level=logging.INFO)
    logger.add_filehandler()

    env=KukaVisualGraspEnv(is_render=True,skip=4)

    model=PPO.load("best_model.zip",env=env)

    obs=env.reset()
    for _ in range(1000):
        action,_states=model.predict(obs)
        obs,rewards,dones,info=env.step(action)
        if dones:
            obs=env.reset()
