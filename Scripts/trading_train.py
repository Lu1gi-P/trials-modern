from stable_baselines3 import A2C
from loguru import logger
from pairTradingEnv import pairsTradingEnv
from feature_extractor import TRADING_FEATURE_EXTRACTORS
from callbacks import TradingEvalCallback

import matplotlib.pyplot as plt
import numpy as np
import torch
import empyrical
import pandas as pd
import utils
import os


# Configure logging to file
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "..", "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist

log_file = os.path.join(logs_dir, "trading_experiment_{time:YYYY-MM-DD_HH-mm-ss}.log")

# Add file sink with rotation and retention
logger.add(
    log_file,
    rotation="10 MB",      # Rotate when file reaches 10MB
    retention="7 days",    # Keep logs for 7 days
    compression="zip",     # Compress old logs
    level="INFO",          # Log level
    format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{function}:{line} - {message}",
    backtrace=True,        # Include traceback for errors
    diagnose=True          # Include variable values in traceback
)

logger.info("Logging configured - outputs will be saved to file and console")

# Automatic device detection
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {device}")
if device == "cuda":
    logger.info(f"CUDA device: {torch.cuda.get_device_name()}")
    logger.info(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def initialize_trading_env(name, date, names, prices, window_size, form_len):
    return pairsTradingEnv(
        name=name,
        date=date,
        asset_name=names,
        log_prices=prices,
        window_size=window_size,
        max_len=form_len,
    )


def create_trading_envs(name, dataset, names, window_size):
    form_date=dataset[4]
    trad_date=dataset[5]
    form_len=len(form_date)
    asset_name=names
    form_asset_log_prices=dataset[1]
    trad_asset_log_prices=dataset[3]


    new_trad_asset_log_prices = np.exp(trad_asset_log_prices)
    new_trad_asset_log_prices = (new_trad_asset_log_prices / new_trad_asset_log_prices[:, :1])
    trad_asset_log_prices = np.log(new_trad_asset_log_prices)

    new_form_asset_log_prices = np.exp(form_asset_log_prices)
    new_form_asset_log_prices = (new_form_asset_log_prices / new_form_asset_log_prices[:, :1])
    form_asset_log_prices = np.log(new_form_asset_log_prices)

    train_env = initialize_trading_env(
        name="train", 
        date=form_date, 
        names=asset_name, 
        prices=form_asset_log_prices, 
        window_size=window_size, 
        form_len=form_len
    )
    
    test_env = initialize_trading_env(
        name="test", 
        date=trad_date, 
        names=asset_name, 
        prices=trad_asset_log_prices, 
        window_size=window_size, 
        form_len=form_len
    )

    return train_env, test_env

def calculate_trading_metrics(info):
    """Calculate additional trading metrics"""
    net_value = info["returns"]
    returns = [0]  # First return is 0
    
    # Fix the variable name collision
    for j in range(1, len(net_value)):  # Use 'j' instead of 'i'
        daily_return = (net_value[j] - net_value[j - 1]) / net_value[j - 1]
        returns.append(daily_return)
    
    if sum(returns) == 0:
        return {
            "annual_return": -2.0,
            "annual_volatility": -2.0,
            "max_drawdown": -2.0
        }
    else:
        returns_series = pd.Series(returns)
        return {
            "annual_return": empyrical.annual_return(returns_series, period="daily", annualization=252),
            "annual_volatility": empyrical.annual_volatility(returns_series, period="daily", alpha=2.0, annualization=252),
            "max_drawdown": empyrical.max_drawdown(returns_series)
        }

def Trading_train(x_index, y_index, folder, cycles, seed=42):
   
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    data_path = os.path.join(script_dir, "..", "Data", "U.S.S&P500")

    save_path = os.path.join(script_dir, "..", "models", folder, f"seed_{seed}")

    os.makedirs(save_path, exist_ok=True)
    
    train = utils.select_data(
        rolling_dataset_path=data_path,
        dataset_type="train"
    )
    test = utils.select_data(
        rolling_dataset_path=data_path,
        dataset_type="test"
    )
    valid = utils.select_data(
        rolling_dataset_path=data_path,
        dataset_type="valid"
    )
    rolling_serial = 0
    # currently only using 1 rolling dataset
    if rolling_serial >= len(train):
        logger.warning("Rolling serial is larger than or equal to the length of the training data")
        rolling_serial = len(train) - 1

    df_train = utils.load_data(
        path=data_path,
        file_name=train[rolling_serial]
    )
    df_test = utils.load_data(
        path=data_path,
        file_name=test[rolling_serial]
    )
    df_valid = utils.load_data(
        path=data_path,
        file_name=valid[rolling_serial]
    )

    asset_num = df_train.columns.get_level_values(0).drop_duplicates().shape[0]
    feature_dim = df_train.shape[1] // asset_num
    logger.info(f"Feature dimension: {feature_dim}")
    logger.info(f"Asset number: {asset_num}")
    
    asset_names, train_dataset, valid_dataset, test_dataset = utils.build_dataset(
        df_train, df_valid, df_test, asset_num, feature_dim
    )

    def log_dataset(name, dataset):
        logger.info(
            f"Generated {name} dataset:\n  "
            f"Formation ({dataset[4][0]} - {dataset[4][-1]})\n  "
            f"Data size (N x T x M): {dataset[0].shape}\n  "
            f"Trading ({dataset[5][0]} - {dataset[5][-1]})\n  "
            f"Data size (N x T x M): {dataset[2].shape}"
        )

    dataset_names = ["train", "valid", "test"]
    datasets = [train_dataset, valid_dataset, test_dataset]
    [
        log_dataset(dataset_names[index], dataset)
        for index, dataset in enumerate(datasets)
    ]

    window_size = 20


    train_train_env, train_test_env = create_trading_envs(
        name="train",
        dataset=train_dataset,
        names=asset_names,
        window_size=window_size,
    )
    
    valid_train_env, valid_test_env = create_trading_envs(
        name = "valid",
        dataset=valid_dataset,
        names=asset_names,
        window_size=window_size,
    )


    test_train_env, test_test_env = create_trading_envs(
        name = "test",
        dataset=test_dataset,
        names=asset_names,
        window_size=window_size,
    )

    policy_kwargs = {
            "features_extractor_class": TRADING_FEATURE_EXTRACTORS[
                "lstm"
            ],
            "features_extractor_kwargs": {
                "features_dim": 16, 
                "num_layers": 2, 
                "dropout": 0.0, 
            },
    }
    logger.info("Initializing A2C model")
    model = A2C(
        "MultiInputPolicy",
        train_train_env,
        policy_kwargs=policy_kwargs,
        n_steps=20,
        learning_rate=3e-4,
        seed=seed,
        gamma=0.99,
        ent_coef=0.01,
        verbose=0,
        device=device,
    )

    trading_train_callback = TradingEvalCallback(name="train", test_env=train_test_env)
    trading_valid_callback = TradingEvalCallback(name="valid", test_env=valid_test_env)
    trading_test_callback = TradingEvalCallback(name="test", test_env=test_test_env)

    trading_indexes = [x_index, y_index ]
    best_previous_metric = float('-inf')

    for cycle in range(cycles):
        train_train_env.__setattr__("trading_indexes", trading_indexes)
        train_test_env.__setattr__("trading_indexes", trading_indexes)

        

        model.learn(
            total_timesteps=1000,
            reset_num_timesteps=False,
        )
        

        trading_train_callback.model = model
        trading_train_callback.on_step()
        train_info = trading_train_callback.best_metric

        train_metrics = calculate_trading_metrics(train_info)

        if cycle % 2 == 0 and cycle > 0:
            valid_test_env.__setattr__("trading_indexes", trading_indexes)
            trading_valid_callback.model = model
            trading_valid_callback.on_step()
            valid_info = trading_valid_callback.best_metric
            
            valid_metrics = calculate_trading_metrics(valid_info)

            logger.info(
                f"Evaluation valid environment:"
                f"Assets: {asset_names[x_index]} & {asset_names[y_index]}, "
                f"Return: {valid_info['returns'][-1]:.4f}, "
                f"Sharpe: {valid_info['sharpe_ratio']:.4f}, "
                f"Annual Return: {valid_metrics['annual_return']:.4f}, "
                f"Annual Volatility: {valid_metrics['annual_volatility']:.4f}, "
                f"Max Drawdown: {valid_metrics['max_drawdown']:.4f}"
            )

            # save a figure of the trading trajectory
            figure_valid = valid_test_env.plot_trajectory(
                action_list=valid_test_env.action_list,
                figsize=(15, 5),
                net_value_limit=(0.7, 1.3),
            )
            figure_valid.savefig(os.path.join(save_path, f"trading_trajectory_valid_{cycles}.png"))

            # Save best model based on test Sharpe ratio
            if valid_info['sharpe_ratio'] > best_previous_metric:
                logger.info(f"New best metric: {valid_info['sharpe_ratio']:.4f}, (previous: {best_previous_metric:.4f})")
                best_previous_metric = valid_info['sharpe_ratio']

                test_test_env.__setattr__("trading_indexes", trading_indexes)
                trading_test_callback.model = model
                trading_test_callback.on_step()
                test_info = trading_test_callback.best_metric
                
                test_metrics = calculate_trading_metrics(test_info)

                logger.info(
                    f"New best Test Results {cycle}/{cycles}:"
                    f"Assets: {asset_names[x_index]} & {asset_names[y_index]}, "
                    f"Return: {test_info['returns'][-1]:.4f}, "
                    f"Sharpe: {test_info['sharpe_ratio']:.4f}, "
                    f"Annual Return: {test_metrics['annual_return']:.4f}, "
                    f"Annual Volatility: {test_metrics['annual_volatility']:.4f}, "
                    f"Max Drawdown: {test_metrics['max_drawdown']:.4f}"
                )
                model.save(os.path.join(save_path, "best_trading_model"))

                # save a figure of the trading trajectory
                figure = test_test_env.plot_trajectory(
                    action_list=test_test_env.action_list,
                    figsize=(15, 5),
                    net_value_limit=(0.7, 1.3),
                )
                figure.savefig(os.path.join(save_path, f"trading_trajectory_{cycles}.png"))




def main():
    # this function is use to train the trading model on a single pair
    logger.info("Starting Test...")
    cycles = 150
    Trading_train(seed=13, x_index=21, y_index=28, folder="test_12", cycles=cycles)
    logger.info("Test Completed.")

if __name__ == "__main__":
    main()