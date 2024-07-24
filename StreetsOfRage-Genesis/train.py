import os
import json

import retro

from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
)

from custom.load_config import config
from custom.statslogger import StatsLogger
from custom.stochasticframeskip import StochasticFrameSkip
from custom.policies import CustomCNNLSTMPolicy

def make_retro(*, game, state=None, max_episode_steps=None, record='./record', env_id=0, **kwargs):
    env = retro.make(game, state, use_restricted_actions=retro.Actions.FILTERED, record=record, **kwargs)
    #print(env.buttons)
    env = StochasticFrameSkip(env, n=2, stickprob=0.15, env_id=env_id)
    return env


def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env


def make_env(env_id, game, state, scenario, record, render, max_episode_steps):
    env = make_retro(game=game, state=state, scenario=scenario, record=record, env_id=env_id, max_episode_steps=max_episode_steps, render_mode="rgb_array")
    env = wrap_deepmind_retro(env)
    return env





def main():
    
    config.load(os.path.join('settings', 'config.yaml'))
    best_params_path = os.path.join(config.get('tune_path'), 'best_params.json')
    with open(best_params_path, 'r') as f:
        best_params = json.load(f)

    def make_env_fn(env_id):
        return make_env(env_id, config.get('game_name'), retro.State.DEFAULT, None, os.path.join(config.get('train_record_path')), config.get('train_render'),
                        config.get('train_num_timesteps'))

    venv = VecTransposeImage(
        VecFrameStack(SubprocVecEnv([lambda i=i: make_env_fn(i) for i in range(config.get('train_env_num'))]), n_stack=config.get('train_n_stack')))

    model_path = os.path.join(config.get('train_path'), config.get('train_model_name'))

    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        model = PPO.load(model_path, env=venv, custom_objects={"policy_class": CustomCNNLSTMPolicy})
    else:
        print("Creating a new model")
        model = PPO(
            policy=CustomCNNLSTMPolicy,
            env=venv,
            learning_rate=best_params['learning_rate'],
            n_steps=best_params['n_steps'],
            batch_size=best_params['batch_size'],
            n_epochs=best_params['n_epochs'],
            gamma=best_params['gamma'],
            gae_lambda=best_params['gae_lambda'],
            clip_range=best_params['clip_range'],
            ent_coef=best_params['ent_coef'],
            verbose=1,
        )

    timesteps_per_iteration = config.get('train_num_timesteps') // config.get('train_iterations')

    # Create log directory
    os.makedirs('./logs/', exist_ok=True)

    # Set up logger
    stats_logger = StatsLogger(csv_path=os.path.join('./logs/training_log.csv'))

    for i in range(config.get('train_iterations')):
        print(f"Starting iteration {i + 1}/{config.get('train_iterations')}")
        model.learn(
            total_timesteps=timesteps_per_iteration,
            log_interval=1,
            progress_bar=True,
            reset_num_timesteps=False,  # This ensures the timestep count continues across iterations
            callback=[stats_logger]
        )
        model.save(model_path)

    # Save the final model
    model.save(model_path)
    print(f"Model saved to {model_path}")
    print(f"Recording saved in {config.train_path} directory")

if __name__ == "__main__":
    main()
