import gymnasium as gym
import numpy as np

from utils import PositionState, Action, get_curr_net_value, plot_assets, get_valid_action_indexes
import empyrical

class pairsTradingEnv(gym.Env):
    def __init__(
        self,
        name: str,
        date: list[str],
        asset_name: list[str],
        log_prices: np.array,
        commission_rate: float = 0.001,
        fund_ratio: float = 1.0,
        init_net_value: float = 1.0,
        risk_free: float = 0.000085,
        successive_close_reward: float = 0,
        window_size: int = 1,
        max_len: int = 252,
    ):
        super(pairsTradingEnv, self).__init__()
        self.name = name
        self.asset_name = asset_name
        self.log_prices = log_prices
        self.action_space = gym.spaces.Discrete(3)
        self.trading_indexes = [0, 0]
        self.window_size = window_size
        assert window_size + 1 <= max_len - 1

        # we make use of the spaces.Box here to represent sequences of observations
        self.observation_space = gym.spaces.Dict({
            "asset_x": gym.spaces.Box(
                -5, 5, shape=(max_len,), dtype=np.float32
            ),
            "asset_y": gym.spaces.Box(
                -5, 5, shape=(max_len,), dtype=np.float32
            ),
            "net_value": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.float32
            ),
            "unrealized_net_value": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.float32
            ),
            "sharpe_ratio": gym.spaces.Box(
                -40, 40, shape=(max_len,), dtype=np.float32
            ),
            "position": gym.spaces.Box(0, 2, shape=(max_len,), dtype=np.int32),  
            "next_end": gym.spaces.Box(
                0, 1, shape=(max_len,), dtype=np.int32  
            ),  #  (Will the next step be done?)
            "hold_threshold": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.int32  
            ),
            "hold_indicator": gym.spaces.Box(
                0, 2, shape=(max_len,), dtype=np.int32  #
            ),
            "action": gym.spaces.Box(-1, 2, shape=(max_len,), dtype=np.int32), 
            "mask_len": gym.spaces.Box(0, max_len, shape=(1,), dtype=np.int32),  
        })

        # env settings
        self.date = date
        self.commission_rate = commission_rate
        self.fund_ratio = fund_ratio
        self.risk_free = risk_free
        self.init_net_value = init_net_value
        self.max_len = min(max_len, len(self.log_prices[0, :]) + 1)
        self.successive_close_reward = successive_close_reward
        assert len(self.log_prices[0, :]) >= window_size + 1

        self.position: int | None = PositionState.bear 
        self.start_idx: int | None = None
        self.curr_idx: int | None = None
        self.last_buy_idx: int | None = None
        self.last_net_value: float | None = None

        # funds (initialized as 1.0)
        self.net_value: np.ndarray = np.array(
            [self.init_net_value] * max_len, dtype=np.float32
        )

        # worth calculated based on current prices (initialized as 1.0)
        self.unrealized_net_value: np.ndarray = np.array(
            np.full(max_len, self.init_net_value, dtype=np.float32)
        )

        self.action_list = []

        self.observation = self._get_obs(max_len)

    def _get_obs(self, max_len):
        return {
            "asset_x": np.zeros(max_len, dtype=np.float32),
            "asset_y": np.zeros(max_len, dtype=np.float32),
            "net_value": np.full(max_len, self.init_net_value, dtype=np.float32),
            "unrealized_net_value": np.full(max_len, self.init_net_value, dtype=np.float32),
            "sharpe_ratio": np.zeros(max_len, dtype=np.float32),
            "position": np.full(max_len, PositionState.bear, dtype=np.int32),
            "next_end": np.zeros(max_len, dtype=np.int32), 
            "hold_threshold": np.zeros(max_len, dtype=np.int32),
            "hold_indicator": np.zeros(max_len, dtype=np.int32),
            "action": np.full(max_len, -1, dtype=np.int32),  # -1 indicates no action taken
            "mask_len": np.array([1], dtype=np.int32), 
        }


    def reset(self, *, seed: int | None = None, options: dict[str, object] | None = None):
        # for plotting 
        self.pair_name = [
            self.asset_name[self.trading_indexes[0]],
            self.asset_name[self.trading_indexes[1]],
        ] 

        self.asset_price = [
            self.log_prices[self.trading_indexes[0], :].tolist(),
            self.log_prices[self.trading_indexes[1], :].tolist(),
        ]
        self.curr_idx = self.window_size - 1
        self.start_idx = self.curr_idx
        self.position = PositionState.bear
        self.net_value.fill(self.init_net_value)
        self.unrealized_net_value.fill(self.init_net_value)
        self.action_list.clear()

        self.observation["asset_x"].fill(0)
        self.observation["asset_y"].fill(0)
        self.observation["net_value"].fill(self.init_net_value)
        self.observation["unrealized_net_value"].fill(self.init_net_value)
        self.observation["sharpe_ratio"].fill(0)
        self.observation["position"].fill(int(PositionState.bear))
        self.observation["next_end"].fill(0)
        self.observation["mask_len"].fill(1)
        self.observation["hold_threshold"].fill(0)
        self.observation["hold_indicator"].fill(0)
        self.observation["action"].fill(-1)

        # fill the window with initial values
        for i, j in enumerate(range(self.curr_idx + 1 - self.window_size, self.curr_idx + 1)):
            self.observation["asset_x"][i] = self.asset_price[0][j]
            self.observation["asset_y"][i] = self.asset_price[1][j]
        self.observation["mask_len"][0] = self.window_size
        self.last_buy_idx = None
        self.last_net_value = self.init_net_value

        return self.observation, {}

    def step(self, action):
        # take action_0
        reward = 0
        if (
            self.curr_idx - self.start_idx + self.window_size
            == self.max_len - 1
        ):
            action = Action.close

        if action == Action.short:
            if self.position == PositionState.bear:
                self.last_buy_idx = self.curr_idx
            elif self.position == PositionState.long:
                reward += self.get_net_value_change()
                self.last_net_value += self.get_net_value_change()
                self.last_buy_idx = self.curr_idx
            self.position = PositionState.short
        elif action == Action.long:
            if self.position == PositionState.bear:
                self.last_buy_idx = self.curr_idx
            elif self.position == PositionState.short:
                reward += self.get_net_value_change()
                self.last_net_value += self.get_net_value_change()
                self.last_buy_idx = self.curr_idx
            self.position = PositionState.long
        elif action == Action.close:
            if (
                self.position == PositionState.long
                or self.position == PositionState.short
            ):
                reward += self.get_net_value_change()
                self.last_net_value += self.get_net_value_change()
            else:
                reward += self.successive_close_reward
            self.last_buy_idx = None
            self.position = PositionState.bear
        self.action_list.append(int(action))

        # state transition
        self.curr_idx += 1
        # if its done
        if self.curr_idx - self.start_idx + self.window_size == self.max_len:
            done = True
        else:
            done = False

        curr_net_value = self.last_net_value
        if (
            self.position == PositionState.short
            or self.position == PositionState.long
        ):
            curr_net_value = get_curr_net_value(
                self.asset_price[0],
                self.asset_price[1],
                self.fund_ratio,
                self.position,
                self.last_net_value,
                self.last_buy_idx,
                self.curr_idx - 1,
            )

        unrealized_net_value = self.last_net_value
        if (
            self.position == PositionState.short
            or self.position == PositionState.long
        ):
            unrealized_net_value = get_curr_net_value(
                self.asset_price[0],
                self.asset_price[1],
                self.fund_ratio,
                self.position,
                self.last_net_value,
                self.last_buy_idx,
                self.curr_idx,
            )

        obs_idx = self.curr_idx - self.start_idx + self.window_size - 1
        self.observation["action"][obs_idx - 1] = int(action)
        # sharpe_ratio dependencies unrealized_net_value updates
        self.net_value[obs_idx] = curr_net_value
        self.unrealized_net_value[obs_idx] = unrealized_net_value
        sharpe_ratio = self.get_curr_sharpe_ratio()

        self.observation["net_value"][obs_idx] = curr_net_value

        self.observation["unrealized_net_value"][
            obs_idx
        ] = unrealized_net_value

        self.observation["sharpe_ratio"][obs_idx] = sharpe_ratio
        self.observation["position"][obs_idx] = int(self.position)
        self.observation["mask_len"][0] = obs_idx + 1
        self.observation["hold_threshold"][obs_idx] = 0

        if done:
            self.observation["asset_x"][obs_idx] = -1
            self.observation["asset_y"][obs_idx] = -1
            self.observation["next_end"][obs_idx] = 0
        else:
            self.observation["asset_x"][obs_idx] = self.asset_price[0][
                self.curr_idx
            ]
            self.observation["asset_y"][obs_idx] = self.asset_price[1][
                self.curr_idx
            ]
            self.observation["next_end"][obs_idx] = int(
                obs_idx + 1 == self.max_len - 1
            )
        if self.position == PositionState.bear:
            self.observation["hold_indicator"].fill(0)
        elif (
            self.observation["position"][obs_idx]
            == self.observation["position"][obs_idx - 1]
        ):
            self.observation["hold_indicator"][obs_idx] = 1
        else:
            self.observation["hold_indicator"].fill(0)
            self.observation["hold_indicator"][obs_idx] = 1
            self.observation["hold_indicator"][obs_idx - 1] = 1

        info = {
            "curr_idx": self.curr_idx,
            "last_buy_idx": self.last_buy_idx,
            "last_net_value": self.last_net_value,
            "net_value": curr_net_value,
            "ur_net_value": unrealized_net_value,
            "position": self.position,
            "sharpe_ratio": sharpe_ratio,
        }
        print(f"Action taken: {action} at index {self.curr_idx}, Sharpy Ratio: {sharpe_ratio}")
        # print(f"Sharpy Ratio: {sharpe_ratio}, Net Value: {curr_net_value}, Unrealized Net Value: {unrealized_net_value}")
        return self.observation, reward, done, False, info 

    def get_curr_sharpe_ratio(self):
        net_values = self.unrealized_net_value[
            self.window_size
            - 1 : self.curr_idx
            - self.start_idx
            + self.window_size
        ]
        noncumulative_returns = (
            np.array(net_values)[1:] / np.array(net_values)[:-1] - 1
        )
        sharpe_ratio = float(
            empyrical.sharpe_ratio(noncumulative_returns, self.risk_free)
        )
        if np.isinf(sharpe_ratio) or np.isnan(sharpe_ratio):
            sharpe_ratio = 0
        sharpe_ratio = max(sharpe_ratio, -3.0)
        return sharpe_ratio
    
    def get_net_value_change(self):
        """
        :return: Returns the change in net value after executing
        """
        return (
            get_curr_net_value(
                self.asset_price[0],
                self.asset_price[1],
                self.fund_ratio,
                self.position,
                self.last_net_value,
                self.last_buy_idx,
                self.curr_idx,
            )
            * (1 - self.commission_rate)
            - self.last_net_value
        )

    def render(self):
        pass 

    def close(self):
        pass

    def plot_trajectory(
        self,
        action_list: list[int],
        net_value_list: list[float] | None = None,
        figsize: tuple[int, int] = (15, 5),
        net_value_limit: tuple[float, float] = (0.9, 1.1),
    ):
        # If net_value_list is not provided, extract from environment's internal state
        if net_value_list is None:
            episode_length = len(action_list)
            net_value_list = self.net_value[self.window_size-1:self.window_size-1+episode_length].tolist()
        
        print(f"Plotting trajectory with {len(net_value_list)} net values and {len(action_list)} actions and dates {len(self.date[self.window_size - 1 :  self.window_size - 1 + len(action_list)])}")
        
        assert (
            len(net_value_list)
            == len(action_list)
            == len(self.date[self.window_size - 1 : self.window_size - 1 + len(action_list)])
        )
        long_idxs, short_idxs, close_idxs = get_valid_action_indexes(
            action_list
        )
        figure = plot_assets(
            date=np.array(
                self.date[self.window_size - 1 :  self.window_size - 1 + len(action_list)], dtype="datetime64"
            ),
            asset_x=np.array(self.asset_price[0][self.window_size - 1 :self.window_size - 1 + len(action_list)]),
            asset_x_label=self.pair_name[0],
            asset_y=np.array(self.asset_price[1][self.window_size - 1 :self.window_size - 1 + len(action_list)]),
            asset_y_label=self.pair_name[1],
            net_value=np.array(net_value_list),
            long_idxs=np.array(long_idxs),
            short_idxs=np.array(short_idxs),
            close_idxs=np.array(close_idxs),
            figsize=figsize,
            net_value_limit=net_value_limit,
        )

        return figure