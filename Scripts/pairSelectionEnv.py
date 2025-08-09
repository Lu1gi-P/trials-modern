import gymnasium as gym
import numpy as np
import pandas as pd
import empyrical
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.a2c import A2C
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.preprocessing import preprocess_obs
from pairTradingEnv import pairsTradingEnv
from feature_extractor import TRADING_FEATURE_EXTRACTORS
from callbacks import TradingEvalCallback

from loguru import logger
from utils import plot_assets, get_valid_action_indexes

class pairSelectionEnv(gym.Env):
    def __init__(
        self,
        name: str,
        form_date: list[str],
        trad_date: list[str],
        asset_name: list[str],
        form_asset_features: np.array,
        trad_asset_features: np.array,
        form_asset_log_prices: np.array,
        trad_asset_log_prices: np.array,
        feature_dim: int,
        **kwargs,
    ):
        """
        :param form_date: date of the formative period
        :param trad_date: date of the trading period
        :param asset_name: stock symbols
        :param form_asset_features: N x T x M Characteristics of the formative period
        :param trad_asset_features: N x T x M Characteristics of the trading period
        :param form_asset_log_prices: N x 1 Opening prices of the formative period
        :param trad_asset_log_prices: N x 1 Opening prices of the trading period
        :param feature_dim: the dimension of the features
        :param kwargs: Additional parameters including:
            - init_net_value: Initial net value
            - serial_selection: Whether to use MultiCategoricalAction
            - trading_threshold: CointegrationStateMachine trading threshold
            - stop_loss_threshold: CointegrationStateMachine stop loss threshold
            - enable_reinforcement: Whether to enable RL training mode
            - trading_train_steps: Training steps for RL
            - window_size: Window size for RL mode
        """
        super(pairSelectionEnv, self).__init__()
        self.name = name
        self.is_eval = False
        self.asset_num = len(asset_name)
        self.serial_selection = kwargs.get("serial_selection", True)
        self.asset_attention = kwargs.get("asset_attention", True)
        
        # action space setup
        self.action_space = (
            gym.spaces.MultiDiscrete([self.asset_num, self.asset_num]) # select asset x and then asset y 
            if self.serial_selection 
            else gym.spaces.Discrete(self.asset_num * (self.asset_num - 1)// 2) # select asset x and y simultaneously (parrallel selection)
        )
        self.form_len = len(form_date)
        self.trad_len = len(trad_date)
        self.feature_dim = feature_dim
        self.form_flatten_len = self.form_len * self.asset_num * feature_dim
        self.trad_flatten_len = self.trad_len * self.asset_num * feature_dim
        self.observation_space = gym.spaces.Dict(
            {
                "assets": gym.spaces.Box(
                    -5, 5, shape=(self.form_flatten_len,), dtype=np.float32
                )
            }
        )
        self.form_date = form_date
        self.trad_date = trad_date
        self.asset_name = asset_name
        self.form_asset_features = form_asset_features
        self.trad_asset_features = trad_asset_features
        self.form_asset_log_prices = form_asset_log_prices
        self.trad_asset_log_prices = trad_asset_log_prices

        # trading parameters
        self.trading_threshold = kwargs.get("trading_threshold", 1.0)
        self.stop_loss_threshold = kwargs.get("stop_loss_threshold", 1.0)
        self.init_net_value = kwargs.get("init_net_value", 1.0)

        self.euclidian_distance = float("inf") 
        self.metric = float("-inf")  

        self.train_step = kwargs.get("trading_train_steps", 5)
        self.window_size = kwargs.get("window_size", 20)

        self.observation = {"assets": form_asset_features.flatten()}

        # only for non serial selection 
        if not self.serial_selection:
            self.index_map = {}
            for first_asset_index in range(self.asset_num -1 ):
                for second_asset_index in range(first_asset_index + 1, self.asset_num):
                    self.index_map[len(self.index_map)] = (
                        first_asset_index,
                        second_asset_index,
                    )

        # start of training 
        trad_date = list(self.form_date[-self.window_size:])
        trad_date.extend(self.trad_date)
        self.trad_date = trad_date

        self.trad_asset_log_prices = np.concatenate(
            [
                self.form_asset_log_prices[:,-self.window_size:],
                self.trad_asset_log_prices,
            ], 
            axis=1)
        
        new_trad_asset_log_prices = np.exp(self.trad_asset_log_prices)
        new_trad_asset_log_prices = (new_trad_asset_log_prices / new_trad_asset_log_prices[:, :1])
        self.trad_asset_log_prices = np.log(new_trad_asset_log_prices)

        new_form_asset_log_prices = np.exp(self.form_asset_log_prices)
        new_form_asset_log_prices = (new_form_asset_log_prices / new_form_asset_log_prices[:, :1])
        self.form_asset_log_prices = np.log(new_form_asset_log_prices)


        def initialize_env(name, date, names, prices):
            return pairsTradingEnv(
                name=name,
                date=date,
                asset_name=names,
                log_prices=prices,
                window_size=self.window_size,
                max_len=self.form_len,
                )
            
        self.train_env = initialize_env(
            name="train",
            date=self.form_date,
            names=self.asset_name,
            prices=self.form_asset_log_prices,
        )


        self.test_env = initialize_env(
            name="test",
            date=self.trad_date,
            names=self.asset_name,
            prices=self.trad_asset_log_prices,
        )

        policy_kwargs = {
            "features_extractor_class": TRADING_FEATURE_EXTRACTORS[
                kwargs["trading_feature_extractor"]
            ],
            "features_extractor_kwargs": {
                "feature_dim": kwargs["trading_feature_extractor_feature_dim"],
                "num_layers": kwargs["trading_feature_extractor_num_layers"],
                "dropout": kwargs["trading_dropout"],
            },
        }

        if kwargs.get("worker_model") is not None:
            self.worker_model = kwargs["worker_model"]
        else:
            logger.info("Initialize new worker model") 
            self.worker_model = A2C(
                policy="MultiInputPolicy",
                env=self.train_env,
                n_steps=20,
                learning_rate=kwargs.get("learning_rate", 0.0001),
                tensorboard_log=kwargs.get("tensorboard_log", None),
                seed=kwargs.get("seed", None),
                gamma=kwargs.get("trading_rl_gamma", 1.0),
                ent_coef=kwargs.get("trading_ent_coef", 0.0001),
                policy_kwargs=policy_kwargs,
                verbose=kwargs.get("verbose", 0),
                device=kwargs.get("device", "cpu"),
            )
        
        self.trading_callback = TradingEvalCallback(name=self.name, test_env=self.test_env)
        
        logger.info(
            f"Initialized {self.name} environment with {self.asset_num} assets, "
            f"formative period length {self.form_len}, trading period length {self.trad_len}, "
            f"feature dimension {self.feature_dim}"
        )


    def render(self):
        
        pass

    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        self.euclidian_distance = float("inf")
        return self.observation, {}

    def get_map_action(self, action):
        """Map action according to selection mode

        :param action: ([th.Tensor]) the log probabilities of actions
        """
        if self.serial_selection:
            return action
        if isinstance(action, np.ndarray):
            action = action.item()
        return self.index_map[action]

    def step(self, action):
        """Reward according to action

        :param action: ([th.Tensor]) the log probabilities of actions
        """
        x_index, y_index = self.get_map_action(action)

        if x_index == y_index:
            logger.info("Selected the same asset for both x and y")
            return (
                    self.observation,
                    -20,
                    True,  
                    False,  
                    {
                        "sharpe_ratio": -20,
                        "euclidean_dist": 0,
                        "annual_return": -20,
                        "annual_volatility": 20,
                        "max_drawdown": -20,
                        "returns": [],
                        "actions": [],
                    },
                )
        logger.info(
            f"Selected asset x: {x_index},{self.asset_name[x_index]}, asset y: {y_index},{self.asset_name[y_index]}"
        )

        trading_indexes = [x_index, y_index]
        self.train_env.__setattr__("trading_indexes", trading_indexes)
        self.test_env.__setattr__("trading_indexes", trading_indexes)

        if self.train_step > 0 and not self.is_eval:
            self.worker_model.learn(
                total_timesteps=self.train_step,
                reset_num_timesteps=False,
            )
        
        self.trading_callback.model = self.worker_model
        self.trading_callback.on_step()
        info = self.trading_callback.best_metric

        reward = np.log(info["returns"][-1])
        net_value = info["returns"]
        returns = [0]
        for i in range(1, len(net_value)):
            daily_return = (net_value[i] - net_value[i - 1]) / net_value[i - 1]
            returns.append(daily_return)
        
        if sum(returns) == 0:
            annual_return = -2.0
            annual_volatility = -2.0
            max_drawdown = -2.0
        else:
            returns = pd.Series(returns)
            annual_return = empyrical.annual_return(returns, period="daily", annualization=252)
            annual_volatility = empyrical.annual_volatility(returns, period="daily", alpha=2.0, annualization=252)
            max_drawdown = empyrical.max_drawdown(returns)

        info["annual_return"] = annual_return
        info["annual_volatility"] = annual_volatility
        info["max_drawdown"] = max_drawdown

        if reward == 0:
            reward = -2.0
        # if self.name == "test":
        #     logger.info(
        #         f"{self.name} | Present action: {action}, sharpe: {info['sharpe_ratio']}"
        #     )
        
        return self.observation, reward, True, False, info
    
    def eval(self, model, step=None, return_figure=True, return_probabilities=True):
        """Evaluate model performance
        :param model: The model to evaluate
        :param step: The current step for logging
        :param return_figure: Whether to return the trajectory figure
        :param return_probabilities: Whether to return pair probabilities and asset representations
        """
        obs, _ = self.reset()
        tensor_obs = obs_as_tensor(obs, model.device)
        probs = model.policy.get_distribution(tensor_obs)
        tensor_obs = preprocess_obs(
            tensor_obs,
            model.policy.observation_space,
            normalize_images=model.policy.normalize_images,
        )
        
        results = {}
        
        # Get pair probabilities (useful for analysis)
        if return_probabilities:
            outputs = model.policy.features_extractor(
                tensor_obs, attention_output=True
            )
            pair_probs = np.zeros([self.asset_num, self.asset_num])
            
            if isinstance(probs.distribution, list):
                prob_distributions = [dis.probs for dis in probs.distribution]
                for first_asset in range(self.asset_num):
                    for second_asset in range(first_asset + 1, self.asset_num):
                        pair_probs[first_asset][second_asset] = (
                            prob_distributions[0][0][first_asset]
                            + prob_distributions[1][0][second_asset]
                        )
                        pair_probs[second_asset][first_asset] = (
                            prob_distributions[1][0][first_asset]
                            + prob_distributions[0][0][second_asset]
                        )
            else:
                prob_dist = probs.distribution.probs[0]
                for first_asset in range(self.asset_num):
                    for second_asset in range(first_asset + 1, self.asset_num):
                        pair_index = (
                            first_asset * self.asset_num
                            + second_asset
                            - (first_asset + 2) * (first_asset + 1) // 2
                        )
                        pair_probs[first_asset][second_asset] = prob_dist[pair_index]
                        pair_probs[second_asset][first_asset] = prob_dist[pair_index]
            
            results['pair_probabilities'] = pair_probs
            results['asset_representations'] = outputs[0][0].reshape(self.asset_num, -1).cpu().detach().numpy()
            
            if outputs[1] is not None:
                results['temporal_attention'] = outputs[1][:, 0].cpu().detach().numpy()
            if self.asset_attention and len(outputs) > 2:
                results['asset_attention'] = outputs[2][0].cpu().detach().numpy()

        # Evaluate model performance
        action, _ = model.predict(obs, deterministic=True)
        self.is_eval = True
        obs, reward, _, _, info = self.step(action)
        self.is_eval = False
        
        # Create trajectory plot
        if return_figure:
            x_index, y_index = self.get_map_action(action)
            figure = self.plot_trajectory(
                self.trad_date,
                [self.asset_name[asset_index] for asset_index in [x_index, y_index]],
                self.trad_asset_log_prices[x_index, :],
                self.trad_asset_log_prices[y_index, :],
                info["actions"],
                info["returns"],
            )
            results['trajectory_figure'] = figure

        logger.info(
            f"Evaluation {self.name} environment: "
            f"asset x: {self.asset_name[x_index]}, "
            f"asset y: {self.asset_name[y_index]}, "
            f"return {info['returns'][-1]}, "
            f"sharpe_ratio {info['sharpe_ratio']}, "
            f"annual_return {info['annual_return']}, "
            f"annual_volatility {info['annual_volatility']}, "
            f"max_drawdown {info['max_drawdown']}"
        )
        
        results.update({
            'reward': reward,
            'sharpe_ratio': info["sharpe_ratio"],
            'annual_return': info["annual_return"],
            'annual_volatility': info["annual_volatility"],
            'max_drawdown': info["max_drawdown"],
            'returns': info["returns"],
            'actions': info["actions"]
        })
        
        return results
    
    def plot_trajectory(
        self,
        trading_dates: list[str],
        asset_names: list[str],
        x_prices: list[float],
        y_prices: list[float],
        action_list: list[int],
        net_value_list: list[float],
        figsize: tuple[int, int] = (15, 5),
        net_value_limit: tuple[float, float] = (0.9, 1.1),
    ):
        assert len(net_value_list) == len(action_list)
        long_idxs, short_idxs, close_idxs = get_valid_action_indexes(
            action_list
        )
        net_value_limit = (
            min(net_value_list) - 0.1,
            max(net_value_list) + 0.1,
        )

        start_index = getattr(self, "window_size")
        #logger.info(f"start index is {start_index}")
        start_index = start_index or 1
        start_index = start_index - 1
        figure = plot_assets(
            date=np.array(trading_dates[start_index:], dtype="datetime64"),
            asset_x=np.array(x_prices[start_index:]),
            asset_x_label=asset_names[0],
            asset_y=np.array(y_prices[start_index:]),
            asset_y_label=asset_names[1],
            net_value=np.array(net_value_list),
            long_idxs=np.array(long_idxs),
            short_idxs=np.array(short_idxs),
            close_idxs=np.array(close_idxs),
            figsize=figsize,
            net_value_limit=net_value_limit,
        )

        return figure