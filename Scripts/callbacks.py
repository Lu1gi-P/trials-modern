from loguru import logger
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.evaluation import evaluate_policy
from typing import Callable
import gymnasium as gym
import numpy as np
import os

from utils import Action


class TradingEvalCallback(EventCallback):
    """
    Callback for evaluating a trading worker.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.

    :param eval_env: The environment used for initialization
    :param test_env: The test environment used for initialization
    """

    def __init__(
        self,
        name,
        test_env: gym.Env | VecEnv,
        max_steps: int | None = None, 
    ):  
        super().__init__(self, verbose=0)
        self.name = name
        self.test_env = test_env
        self.close_action = Action.close
        self.max_steps = max_steps 
        self.num_steps = 0

    def _on_step(self) -> bool:
        self.num_steps += 1
        self.log_env(self.test_env)
        return True

    def log_env(
        self,
        env,
    ):
        
        obs, _ = env.reset()
        action_list = []
        reward_list = []
        net_value_list = []
        sharpe_ratio_list = []
        step_count = 0
        
        while True:
            
            action, _ = self.model.predict(obs, deterministic=True)

            obs, reward, terminated, _, info = env.step(action)
            step_count += 1
            
            #print(f"Action: {action}, Reward: {reward}, Info: {info}")
            done = terminated
            action_list.append(action)
            reward_list.append(reward)
            net_value_list.append(info["net_value"])
            sharpe_ratio_list.append(info["sharpe_ratio"])

            if self.max_steps is not None and step_count >= self.max_steps:
                logger.info(f"Evaluation stopped early at {step_count} steps (max_steps={self.max_steps})")
                break

            if done:
                action_list[-1] = int(self.close_action)
                break
        self.best_metric = {
            "num_timesteps": self.parent.num_timesteps,
            "actions": action_list,
            "rewards": reward_list,
            "returns": net_value_list,
            "sharpe_ratios": sharpe_ratio_list,
            "sharpe_ratio": sharpe_ratio_list[-1],
        }
        trading_metric = {
            f"{self.name}/return": net_value_list[-1],
            f"{self.name}/sharpe_ratio": sharpe_ratio_list[-1],
        }
        
        # wandb.log(
        #     trading_metric,
        #     step=self.num_steps
        # )
        logger.info(
            f"Evaluation {self.name} environment: "
            f"return {net_value_list[-1]}, "
            f"sharpe_ratio {sharpe_ratio_list[-1]}"
        )


class EvalModelCallback(BaseCallback):
    """Run trading results on test env"""

    def __init__(
        self,
        test_env,
        verbose: int = 1,
        train_env=None,
        valid_env=None,
    ):
        super(EvalModelCallback, self).__init__(verbose)
        self.field_names = [
            "num_timesteps",
            "actions",
            "rewards",
            "euclidean_dist",
        ]
        self.test_env = test_env
        self.valid_env = valid_env
        self.train_env = train_env

    def _on_step(self) -> bool:
        self.test_env.worker_model = self.train_env.worker_model
        self.test_env.eval(self.model, self.num_timesteps)
        if self.valid_env is not None:
            self.valid_env.worker_model = self.train_env.worker_model
            self.valid_env.eval(self.model, self.num_timesteps)
        return True



class EvalCallback(EventCallback):
    """
    Callback for evaluating a manager.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq``
    :param best_model_save_path: Path to a folder where the best model
        according to performance on the eval env will be saved.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose:
    """

    def __init__(
        self,
        eval_env: gym.Env | VecEnv,
        train_env: gym.Env | VecEnv,
        training_env_ori: gym.Env | VecEnv,
        callback_on_new_best: BaseCallback | None = None,
        patience_steps: int = 1_000_000,
        n_eval_episodes: int = 5,
        eval_freq: int = 10_000,
        best_model_save_path: str | None = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        exclude_names: list[str] | None = None,
        metric_fn: Callable[
            [list[float], list[float], list[float], list[float]], float
        ] | None = None,
    ):
        super().__init__(callback_on_new_best, verbose=verbose)
        self.callback_on_new_best = callback_on_new_best
        if self.callback_on_new_best is not None:
            self.callback_on_new_best.parent = self
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.deterministic = deterministic
        self.render = render
        self.patience_steps = patience_steps

        self.eval_env = eval_env
        self.train_env = train_env
        self.best_model_save_path = best_model_save_path

        self.exclude_names = exclude_names
        self.metric_fn = metric_fn
        self.best_metric = -np.inf
        self.training_env_ori = training_env_ori

    def _init_callback(self) -> None:
        if not isinstance(self.training_env_ori, type(self.eval_env)):
            logger.error("Training and eval env are not of the same type ")

        if self.best_model_save_path is not None: 
            os.makedirs(self.best_model_save_path, exist_ok=True)

        if self.callback_on_new_best is not None:
            self.callback_on_new_best.init_callback(self.model)


    def _on_step(self) -> bool:
        if (self.eval_freq > 0
            and self.n_calls % self.eval_freq == 0
            and self.num_timesteps >= self.patience_steps):

            self.eval_env.worker_model = self.train_env.worker_model

            backup = self.eval_env.is_eval
            self.eval_env.is_eval = True
            episode_eval_rewards, episode_eval_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
                return_episode_rewards=True
            )
            self.eval_env.is_eval = backup

            backup = self.train_env.is_eval
            self.train_env.is_eval = True
            episode_train_rewards, episode_train_lengths = evaluate_policy(
                self.model,
                self.train_env,
                n_eval_episodes=self.n_eval_episodes,
                deterministic=self.deterministic,
                render=self.render,
                return_episode_rewards=True
            )
            self.train_env.is_eval = backup

            mean_eval_reward, std_reward = np.mean(episode_eval_rewards), np.std(episode_eval_rewards)
            mean_eval_ep_length, std_ep_length = np.mean(episode_eval_lengths), np.std(episode_eval_lengths)
            mean_train_reward = np.mean(episode_train_rewards)
            mean_train_ep_length = np.mean(episode_train_lengths)

            logger.info(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"mean_episode_reward={mean_eval_reward:.2f} ±  {std_reward:.2f}, "
                f"mean_episode_length={mean_eval_ep_length:.2f}± {std_ep_length:.2f}, "
                f"mean_training_reward={mean_train_reward:.2f}"
            )

            metric = self.metric_fn(
                episode_eval_rewards,
                episode_eval_lengths,
                episode_train_rewards,
                episode_train_lengths
            ) if self.metric_fn is not None else mean_eval_reward

            logger.info(f"Present metric {metric} | SOTA {self.best_metric}")

            if metric > self.best_metric:
                logger.info(f"New best metric: {metric} (previous: {self.best_metric})")
                self.best_metric = metric

                if self.best_model_save_path is not None:
                    model_path = os.path.join(self.best_model_save_path, "best_model")
                    self.model.save(model_path)

                    # Save worker trading model
                    if hasattr(self.train_env, 'worker_model') and self.train_env.worker_model is not None:
                        worker_model_path = os.path.join(self.best_model_save_path, "best_worker_model")
                        self.train_env.worker_model.save(worker_model_path)
                        logger.info(f"Saved worker model to: {worker_model_path}")

                if self.callback_on_new_best is not None:
                    self.callback_on_new_best._on_step()

        return True



    def update_child_locals(self, locals_: dict[str, object]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        None

def eval_reward_metric(
    episode_eval_rewards: list[float],
    episode_eval_lengths: list[float],
    episode_training_rewards: list[float],
    episode_training_lengths: list[float],
) -> float:
    return np.mean(episode_eval_rewards).item()