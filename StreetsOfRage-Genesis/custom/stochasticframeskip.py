import gymnasium as gym
import torch as th
import torch.nn as nn
import numpy as np
import random
import os
import json


from custom.reward_manager import HierarchicalReward
from custom.pygame_renderer import PygameRenderer
from custom.load_config import config
config.load(os.path.join('settings', 'config.yaml'))


def update_max_values(self):
        if self.cumulative_reward > self.max_reward:
            self.max_reward = self.cumulative_reward
        if self.cumulative_score > self.max_score:
            self.max_score = self.cumulative_score
        if self.cumulative_map > self.max_map:
            self.max_map = self.cumulative_map
        if self.cumulative_health < self.max_hea:
            self.max_hea = self.cumulative_health
        if self.cumulative_damage > self.max_dam:
            self.max_dam = self.cumulative_damage

def save_env_data(env_id, reset, data):
    directory = os.path.join(os.getcwd(), "env_log")
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except FileExistsError:
        pass
    filename = os.path.join(directory, f"env{env_id:02d}.json")
    env_data = {"latest": data}
    with open(filename, 'w') as f:
        json.dump(env_data, f)



class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob, env_id):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.env_id = env_id
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")


        self.reward_system = HierarchicalReward()

        self.cumulative_reward = 0
        self.cumulative_score = 0
        self.cumulative_map = 0
        self.cumulative_health = 0
        self.cumulative_damage = 0
        self.health = 0
        self.damage = 0
        self.map = 0
        self.tempdamage = 0
        self.wei_sco = 0.10
        self.wei_stp = 0.00001
        self.wei_map = 0.01
        self.wei_dam = 0.25
        self.wei_hea = 0.25
        self.score_temp = 0
        self.time = 0
        self.go = 0
        self.special = -1
        self.steps_without_reward = 0
        self.max_reward = 0
        self.max_score = 0
        self.max_map = 0
        self.max_hea = 0
        self.max_dam = 0



        if self.env_id == 0 and config.get('train_render'):
            self.pygame_renderer = PygameRenderer(env)
        else:
            self.pygame_renderer = None
        self.reset_count = 0
        self.max_steps_without_reward = 20000
        self.steps_for_log = 0
        self.previous_stats = None
        self.previous_stats_temp = None


    def send_data(self, info):
        render_data = {
            'info': info,
            'steps_without_reward': self.steps_without_reward,
            'max_steps_without_reward': self.max_steps_without_reward,
            'cumulative_health': self.cumulative_health,
            'max_hea': self.max_hea,
            'cumulative_reward': self.cumulative_reward,
            'max_reward': self.max_reward,
            'cumulative_score': self.cumulative_score,
            'max_score': self.max_score,
            'cumulative_map': self.cumulative_map,
            'max_map': self.max_map,
            'cumulative_damage': self.cumulative_damage,
            'max_dam': self.max_dam,
            'reset_count': self.reset_count,
            'curac': self.curac,
            'steps_for_log': self.steps_for_log,
            'previous_stats': self.previous_stats,
            'previous_stats_temp': self.previous_stats_temp
        }
        return render_data



    def reset(self, **kwargs):
        update_max_values(self)

        self.reward_system.reset()

        self.cumulative_reward = 0
        self.cumulative_score = 0
        self.cumulative_damage = 0
        self.cumulative_map = 0
        self.cumulative_health = 0
        self.health = 0
        self.damage = 0
        self.map = 0
        self.tempdamage = 0
        self.score_temp = 0
        self.time = 0
        self.go = 0
        self.special = -1
        self.steps_without_reward = 0


        data = {
            "cumulative_reward": round(self.cumulative_reward),
            "cumulative_score": round(self.cumulative_score, 3),
            "cumulative_map": round(self.cumulative_map, 3),
            "cumulative_health": abs(round(self.cumulative_health, 3)),
            "cumulative_damage": round(self.cumulative_damage, 3)
        }
        save_env_data(self.env_id, self.reset_count, data)
        self.curac = None
        self.reset_count += 1
        self.steps_without_reward = 0
        return self.env.reset(**kwargs)
    def step(self, ac):
        self.steps_for_log += 1
        terminated = False
        truncated = False
        playing_mode = config.get('train_render') == True and self.env_id == 0
        playing_mode = False
        for i in range(self.n):
            if playing_mode:
                epsilon = 0.005  # Keep the epsilon for random exploration
                if np.random.random() < epsilon:
                    self.curac = self.env.action_space.sample()
                else:
                    if isinstance(ac, np.ndarray):
                        # Replace NaN values with 0
                        ac = np.nan_to_num(ac, nan=0.0)
                        # Check if all values are zero
                        if np.all(ac == 0):
                            # Use uniform distribution if all values are zero
                            best_action = np.random.choice(len(ac))
                        else:
                            # Choose the action with the highest probability
                            best_action = np.argmax(ac)
                        # Set a high probability for the best action (e.g., 0.9)
                        best_action_prob = 0.5
                        # Distribute the remaining probability among other actions
                        other_probs = (1 - best_action_prob) / (len(ac) - 1)
                        ac_probs = np.full(len(ac), other_probs)
                        ac_probs[best_action] = best_action_prob
                        # Sample from the modified action distribution
                        self.curac = np.random.choice(len(ac_probs), p=ac_probs)
                        # Convert to one-hot encoding
                        one_hot = np.zeros_like(ac)
                        one_hot[self.curac] = 1
                        self.curac = one_hot
                    else:
                        self.curac = ac
            else:
                # Original training logic
                if self.curac is None:
                    self.curac = ac
                elif i == 0:
                    if self.rng.rand() > self.stickprob:
                        self.curac = ac
                elif i == 1:
                    self.curac = ac
            
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)

            rew = self.reward_system.calculate_reward(info, ac)
            self.reward_system.update_common_rewards(info)
            self.cumulative_reward = self.reward_system.get_cumulative_reward()
            self.cumulative_health = self.reward_system.get_cumulative_health()
            self.cumulative_map = self.reward_system.get_cumulative_map()
            self.cumulative_damage = self.reward_system.get_cumulative_damage()
            self.cumulative_score = self.reward_system.get_cumulative_score()
            self.steps_without_reward = self.reward_system.get_steps_without_reward()


            if self.steps_without_reward >= self.max_steps_without_reward:
                terminated = True
                truncated = True 
            if terminated or truncated:
                break
        #print(rew)
        playing_mode = config.get('train_render') == True and self.env_id == 0
        if playing_mode:
            update_max_values(self)
            render_data = self.send_data(info)
            self.pygame_renderer.render(render_data)
        return ob, rew, terminated, truncated, info