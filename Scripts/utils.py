import os
import pandas as pd
import numpy as np
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.ticker import MultipleLocator


def sub(file_name):
    end_point = file_name.index("_")
    dataset_type = file_name[:end_point]
    return dataset_type

def select_data(rolling_dataset_path, dataset_type):
    file_name = os.listdir(rolling_dataset_path)
    file_name = list(filter(lambda x: sub(x) == dataset_type, file_name))
    file_name.sort()
    return file_name

def load_data(path, file_name):
    df = pd.read_csv(
        os.path.join(path, file_name),
        encoding="gbk",
        header=[0, 1],
        thousands=",",
        index_col=0,
    )
    return df

def get_curr_net_value(
    asset_x_price: list[float],
    asset_y_price: list[float],
    fund_ratio: float,
    position: int,
    last_net_value: float,
    last_buy_idx: int,
    curr_idx: int,
) -> float:
    """
    :return: calculate the current net value 
    """
    if position == PositionState.long:
        return last_net_value * (
                np.exp(asset_x_price[curr_idx]) / np.exp(asset_x_price[last_buy_idx]) +
                fund_ratio * (
                        2 - np.exp(asset_y_price[curr_idx]) / np.exp(asset_y_price[last_buy_idx])
                )
        ) / (1 + fund_ratio)
    elif position == PositionState.short:
        return last_net_value * (
                (
                        2 - np.exp(asset_x_price[curr_idx]) / np.exp(asset_x_price[last_buy_idx])
                ) +
                fund_ratio * (
                        np.exp(asset_y_price[curr_idx]) / np.exp(asset_y_price[last_buy_idx])
                )
        ) / (1 + fund_ratio)
    else:
        return last_net_value
    


def build_dataset(train, valid, test, asset_number, feature_dim):
    """Build formation and trading for train, valid, and test"""
    logger.info(f"Start building dataset")
    asset_names = (
        train.columns.get_level_values(0).drop_duplicates().values.tolist()
    )
    logger.info(f"Assets: {asset_names}")
    train_size = train.shape[0]
    valid_size = valid.shape[0]
    test_size = test.shape[0]
    logger.info(
        f"Original dataset size: train {train_size} "
        f"| valid {valid_size} | test {test_size}"
    )
    assert test_size == valid_size
    trading_size = test_size
    formation_size = train_size - trading_size
    logger.info(
        f"Generate dataset size: trading {trading_size} "
        f"| formation {formation_size}"
    )

    # T x N x M  (time, asset_number, feature_dim)
    #print(train.values) #(T, N * M)
    train_value = train.values.astype(float).reshape(
        train_size, asset_number, feature_dim
    )
    valid_value = valid.values.astype(float).reshape(
        valid_size, asset_number, feature_dim
    )
    test_value = test.values.astype(float).reshape(
        test_size, asset_number, feature_dim
    )

    def log_price(data):
        data = np.transpose(data, (1, 0, 2))  # N x T x M
        return np.log(data[:, :, 1])

    def normalize(data):
        """Normalize features: log returns for prices, standardize volume"""
        data = np.transpose(data, (1, 0, 2))  # N x T x M
        data_normalized = data.copy()
        
        for i in range(2):  # First two features
            prices = data[:, :, i]
            if np.any(prices <= 0):
                logger.warning(f"Found non-positive values in feature {i}, replacing with small positive value")
                prices = np.maximum(prices, 1e-8)
            
            # Calculate log returns: log(P_t / P_{t-1})
            # Use the first price as reference for each asset
            price_ratios = prices / prices[:, 0:1] 
            data_normalized[:, :, i] = np.log(price_ratios)
        
        feature_2 = data[:, :, 2]
        mean_val = np.mean(feature_2, axis=1, keepdims=True)
        std_val = np.std(feature_2, axis=1, keepdims=True)
        # Avoid division by zero
        std_val = np.maximum(std_val, 1e-8)
        data_normalized[:, :, 2] = (feature_2 - mean_val) / std_val
        
        return data_normalized

    train_formation = normalize(np.array(train_value[:formation_size]))
    #print(train_formation)
    train_formation_log_price = log_price(np.array(train_value[:formation_size]))
    train_trading = normalize(np.array(train_value[formation_size:]))
    train_trading_log_price = log_price(np.array(train_value[formation_size:]))
    train_formation_dates = train.index.values[:formation_size].tolist()
    train_trading_dates = train.index.values[formation_size:].tolist()

    valid_formation = normalize(np.array(train_value[trading_size:]))
    valid_formation_log_price = log_price(np.array(train_value[trading_size:]))
    valid_trading = normalize(np.array(valid_value))
    valid_trading_log_price = log_price(np.array(valid_value))
    valid_formation_dates = train.index.values[trading_size:].tolist()
    valid_trading_dates = valid.index.values.tolist()

    logger.info(
        f"{np.array(train_value[(trading_size * 2):]).shape}, "
        f"{np.array(valid_value).shape}"
    )

    test_formation_data = np.concatenate(
        [
            np.array(train_value[(trading_size * 2) :]),
            np.array(valid_value),
        ],
        axis=0,
    )
    test_formation = normalize(np.array(test_formation_data))
    test_formation_log_price = log_price(np.array(test_formation_data))
    test_trading = normalize(np.array(test_value))
    test_trading_log_price = log_price(np.array(test_value))
    test_formation_dates = (
        train.index.values[(trading_size * 2) :].tolist() + valid_trading_dates
    )
    test_trading_dates = test.index.values.tolist()

    return (
        asset_names,
        (
            train_formation,
            train_formation_log_price,
            train_trading,
            train_trading_log_price,
            train_formation_dates,
            train_trading_dates,
        ),
        (
            valid_formation,
            valid_formation_log_price,
            valid_trading,
            valid_trading_log_price,
            valid_formation_dates,
            valid_trading_dates,
        ),
        (
            test_formation,
            test_formation_log_price,
            test_trading,
            test_trading_log_price,
            test_formation_dates,
            test_trading_dates,
        ),
    )

class Action:
    long = 0  # long A, short B
    short = 1  # short A, long B
    close = 2  # close position


class PositionState:
    long = 0  # long A, short B
    short = 1  # long B, short A
    bear = 2  # bear position

def plot_assets(
    date: np.ndarray,
    asset_x: np.ndarray,
    asset_x_label: str,
    asset_y: np.ndarray,
    asset_y_label: str,
    long_idxs: np.ndarray,
    short_idxs: np.ndarray,
    close_idxs: np.ndarray,
    net_value: np.ndarray | None = None,
    figsize: tuple[int, int] = (15, 5),
    net_value_limit: tuple[float, float] = (0.9, 1.1),
    spread: np.ndarray | None = None,
    trading_threshold: float | None = None,
    stop_loss_threshold: float | None = None,
) -> Figure:
    figure = plt.figure(figsize=figsize)
    if spread is not None:
        ax1: Axes = figure.add_subplot(2, 1, 1)
    else:
        ax1: Axes = figure.add_subplot(1, 1, 1)

    asset_x = np.exp(asset_x)
    asset_y = np.exp(asset_y)

    fontsize = 10
    scattersize = 50
    ax1.plot(date, asset_x, color="darkslategrey", label=asset_x_label)
    if long_idxs.size != 0:
        ax1.scatter(
            date[long_idxs],
            asset_x[long_idxs],
            color="red",
            marker="^",
            s=scattersize,
            label="long",
        )
        for a, b in zip(date[long_idxs], asset_x[long_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if short_idxs.size != 0:
        ax1.scatter(
            date[short_idxs],
            asset_x[short_idxs],
            color="green",
            marker="v",
            s=scattersize,
            label="short",
        )
        for a, b in zip(date[short_idxs], asset_x[short_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if close_idxs.size != 0:
        ax1.scatter(
            date[close_idxs],
            asset_x[close_idxs],
            color="navy",
            marker="x",
            s=scattersize,
            label="close",
        )
        for a, b in zip(date[close_idxs], asset_x[close_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    ax1.plot(date, asset_y, color="blue", label=asset_y_label)
    if long_idxs.size != 0:
        ax1.scatter(
            date[long_idxs],
            asset_y[long_idxs],
            color="green",
            marker="v",
            s=scattersize,
        )
        for a, b in zip(date[long_idxs], asset_y[long_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if short_idxs.size != 0:
        ax1.scatter(
            date[short_idxs],
            asset_y[short_idxs],
            color="red",
            marker="^",
            s=scattersize,
        )
        for a, b in zip(date[short_idxs], asset_y[short_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if close_idxs.size != 0:
        ax1.scatter(
            date[close_idxs],
            asset_y[close_idxs],
            color="navy",
            marker="x",
            s=scattersize,
        )
        for a, b in zip(date[close_idxs], asset_y[close_idxs]):
            ax1.text(
                a,
                b,
                f"{b:.1f}",
                horizontalalignment="center",
                verticalalignment="top",
                fontsize=fontsize,
            )

    if net_value is not None:
        ax2 = ax1.twinx()
        ax2.axhline(1, color="red", linestyle="--")
        ax2.plot(date, net_value, color="red", label="net value")
        if long_idxs.size != 0:
            ax2.scatter(
                date[long_idxs],
                net_value[long_idxs],
                color="red",
                marker="^",
                s=scattersize,
            )
            for a, b in zip(date[long_idxs], net_value[long_idxs]):
                ax2.text(
                    a,
                    b,
                    f"{b:.2f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=fontsize,
                )
        if short_idxs.size != 0:
            ax2.scatter(
                date[short_idxs],
                net_value[short_idxs],
                color="green",
                marker="v",
                s=scattersize,
            )
            for a, b in zip(date[short_idxs], net_value[short_idxs]):
                ax2.text(
                    a,
                    b,
                    f"{b:.2f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=fontsize,
                )
        if close_idxs.size != 0:
            ax2.scatter(
                date[close_idxs],
                net_value[close_idxs],
                color="navy",
                marker="x",
                s=scattersize,
            )
            for a, b in zip(date[close_idxs], net_value[close_idxs]):
                ax2.text(
                    a,
                    b,
                    f"{b:.2f}",
                    horizontalalignment="center",
                    verticalalignment="top",
                    fontsize=fontsize,
                )

    ax1.legend(loc="upper left")
    ax1.set_ylabel("price")
    if net_value is not None:
        ax2.legend(loc="upper right")
        ax2.set_ylabel("net value")
        ax2.set_ylim(bottom=net_value_limit[0], top=net_value_limit[1])

    x_major_locator = MultipleLocator(3)
    ax1.xaxis.set_major_locator(x_major_locator)
    ax1.set_xlim(
        left=date[0] - np.timedelta64(1, "D"),
        right=date[-1] + np.timedelta64(1, "D"),
    )
    for xtick in ax1.get_xticklabels():
        xtick.set_rotation(-90)

    if spread is not None:
        ax3 = figure.add_subplot(2, 1, 2)
        ax3.plot(date, spread, color="black", label="spread")
        ax3.axhline(trading_threshold, color="darkviolet", linestyle="--")
        ax3.axhline(-trading_threshold, color="darkviolet", linestyle="--")
        ax3.axhline(stop_loss_threshold, color="darkred", linestyle="--")
        ax3.axhline(-stop_loss_threshold, color="darkred", linestyle="--")
        ax3.axhline(0, color="black", linestyle="--")
        if long_idxs.size != 0:
            ax3.scatter(
                date[long_idxs], spread[long_idxs], color="red", marker="^"
            )
            for a, b in zip(date[long_idxs], spread[long_idxs]):
                ax3.text(
                    a,
                    b,
                    f"{b:.1f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=fontsize,
                )

        if short_idxs.size != 0:
            ax3.scatter(
                date[short_idxs], spread[short_idxs], color="green", marker="v"
            )
            for a, b in zip(date[short_idxs], spread[short_idxs]):
                ax3.text(
                    a,
                    b,
                    f"{b:.1f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=fontsize,
                )

        if close_idxs.size != 0:
            ax3.scatter(
                date[close_idxs], spread[close_idxs], color="navy", marker="x"
            )
            for a, b in zip(date[close_idxs], spread[close_idxs]):
                ax3.text(
                    a,
                    b,
                    f"{b:.1f}",
                    horizontalalignment="center",
                    verticalalignment="bottom",
                    fontsize=fontsize,
                )

        if net_value is not None:
            ax4 = ax3.twinx()
            ax4.plot(date, net_value, color="red", label="net value")
            if long_idxs.size != 0:
                ax4.scatter(
                    date[long_idxs],
                    net_value[long_idxs],
                    color="red",
                    marker="^",
                )
                for a, b in zip(date[long_idxs], net_value[long_idxs]):
                    ax4.text(
                        a,
                        b,
                        f"{b:.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=fontsize,
                    )
            if short_idxs.size != 0:
                ax4.scatter(
                    date[short_idxs],
                    net_value[short_idxs],
                    color="green",
                    marker="v",
                )
                for a, b in zip(date[short_idxs], net_value[short_idxs]):
                    ax4.text(
                        a,
                        b,
                        f"{b:.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=fontsize,
                    )
            if close_idxs.size != 0:
                ax4.scatter(
                    date[close_idxs],
                    net_value[close_idxs],
                    color="navy",
                    marker="x",
                )
                for a, b in zip(date[close_idxs], net_value[close_idxs]):
                    ax4.text(
                        a,
                        b,
                        f"{b:.2f}",
                        horizontalalignment="center",
                        verticalalignment="bottom",
                        fontsize=fontsize,
                    )

        ax3.legend(loc="upper left")
        ax3.set_ylabel("spread")
        if net_value is not None:
            ax4.legend(loc="upper right")
            ax4.set_ylabel("net value")
            ax4.set_ylim(bottom=net_value_limit[0], top=net_value_limit[1])

        x_major_locator = MultipleLocator(3)
        ax3.xaxis.set_major_locator(x_major_locator)
        ax3.set_xlim(
            left=date[0] - np.timedelta64(1, "D"),
            right=date[-1] + np.timedelta64(1, "D"),
        )
        ax3.set_ylim(bottom=-3, top=3)
        for xtick in ax3.get_xticklabels():
            xtick.set_rotation(-90)

    figure.tight_layout()

    return figure

def get_valid_action_indexes(
    actions: list[int],
) -> tuple[list[int], list[int], list[int]]:
    # Filter out useless operations
    long_idxs = []
    short_idxs = []
    close_idxs = []
    state = PositionState.bear
    for idx, action in enumerate(actions):
        if action == -1:
            continue
        elif action == Action.short and (
            state == PositionState.bear or state == PositionState.long
        ):
            state = PositionState.short
            short_idxs.append(idx)
        elif action == Action.long and (
            state == PositionState.bear or state == PositionState.short
        ):
            state = PositionState.long
            long_idxs.append(idx)
        elif action == Action.close and (
            state == PositionState.long or state == PositionState.short
        ):
            state = PositionState.bear
            close_idxs.append(idx)
    return long_idxs, short_idxs, close_idxs
