from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import CheckpointCallback
from loguru import logger
from pairSelectionEnv import pairSelectionEnv
from feature_extractor import FEATURE_EXTRACTORS
from callbacks import EvalModelCallback, EvalCallback, eval_reward_metric

from policy_network import PairSelectionActorCriticPolicy

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import os


# Configure logging to file
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "..", "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist

log_file = os.path.join(logs_dir, "selection_experiment_{time:YYYY-MM-DD_HH-mm-ss}.log")

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



def Selection_train():
    policy: str = "simple_serial_selection"
    feature_extractor: str = "gru"
    trading_feature_extractor: str = "lstm"
    asset_attention: bool = False
    rolling_serial: int = 1
    asset_num: int = 30
    feature_dim: int = 3
    feature_extractor_hidden_dim: int = 64
    feature_extractor_num_layers: int = 1
    feature_extractor_num_heads: int = 2
    policy_network_hidden_dim: int = 64
    seed: int = 13
    patience_steps: int = 0
    eval_freq: int = 20
    train_steps: int = int(10000) 
    learning_rate: float = 1e-4
    dropout: float = 0.5
    rl_gamma: float = 0.99
    ent_coef: float = 1e-4
    trading_train_steps: int = int(1000) 
    trading_feature_extractor_feature_dim: int = 3
    trading_feature_extractor_num_layers: int = 1
    trading_feature_extractor_hidden_dim: int = 64
    trading_dropout: float = 0.0
    trading_feature_extractor_num_heads: int = 2
    trading_learning_rate: float = 1e-4
    trading_rl_gamma: float = 0.99
    trading_ent_coef: float = 1e-4

    
    # script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Data/U.S.S&P500 directory
    data_path = os.path.join(script_dir, "..", "Data", "U.S.S&P500")
    
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
    # has to be smaller then the length of the training data
    rolling_serial = 0
    if rolling_serial >= len(train):
        print("Warning: Rolling serial is larger than or equal to the length of the training data")
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
    # df_valid['AMAT']['close'].plot(title="AMAT Close Price")
    # plt.show()

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
    time_step = len(train_dataset[4])
    serial_selection = policy

    def initialize_env(
        name,
        names,
        dataset,
        feature_dim,
        serial_selection,
        asset_attention,
        trading_train_steps,
        worker_model,
    ):
        
        return pairSelectionEnv(
            name=name,
            form_date=dataset[4],
            trad_date=dataset[5],
            asset_name=names,
            form_asset_features=dataset[0],
            form_asset_log_prices=dataset[1],
            trad_asset_features=dataset[2],
            trad_asset_log_prices=dataset[3],
            feature_dim=feature_dim,
            serial_selection=serial_selection,
            asset_attention=asset_attention,
            trading_feature_extractor=trading_feature_extractor,
            trading_feature_extractor_feature_dim=trading_feature_extractor_feature_dim,
            trading_feature_extractor_num_layers=trading_feature_extractor_num_layers,
            trading_feature_extractor_hidden_dim=trading_feature_extractor_hidden_dim,
            trading_feature_extractor_num_heads=trading_feature_extractor_num_heads,
            trading_train_steps=trading_train_steps,
            #trading_num_process=1,
            trading_dropout=trading_dropout,
            policy=policy,
            trading_learning_rate=trading_learning_rate,
            #trading_log_dir=args.trading_log_dir,
            trading_rl_gamma=trading_rl_gamma,
            trading_ent_coef=trading_ent_coef,
            seed=seed,
            worker_model=worker_model,
            device=device,
        )
    

    
    train_env = initialize_env(
        "train",
        asset_names,
        train_dataset,
        feature_dim,
        serial_selection,
        asset_attention,
        trading_train_steps,
        None,
    )
    valid_env = initialize_env(
        "valid",
        asset_names,
        valid_dataset,
        feature_dim,
        serial_selection,
        asset_attention,
        0,
        train_env.worker_model,
    )
    test_env = initialize_env(
        "test",
        asset_names,
        test_dataset,
        feature_dim,
        serial_selection,
        asset_attention,
        0,
        train_env.worker_model,
    )

    policy_kwargs = {
        "features_extractor_class": FEATURE_EXTRACTORS[feature_extractor],
        "features_extractor_kwargs": {
            "asset_num": asset_num,
            "time_step": time_step,
            "input_feature": feature_dim,
            "hidden_dim": feature_extractor_hidden_dim,
            "num_layers": feature_extractor_num_layers,
            "num_heads": feature_extractor_num_heads,
            "asset_attention": asset_attention,
            "drouput": dropout,
        },
        "hidden_dim": feature_extractor_hidden_dim,
        "asset_num": asset_num,
        "feature_dim": feature_dim,
        "policy": policy,
        "num_heads": feature_extractor_num_heads,
        "latent_pi": (asset_num * 2)
        if serial_selection
        else (asset_num * (asset_num - 1) // 2),
        "latent_vf": policy_network_hidden_dim,
    }

    test_callback = EvalModelCallback(
        test_env=test_env,
        valid_env=valid_env,
        train_env=train_env,
    )

    eval_callback = EvalCallback(
        eval_env=valid_env,
        train_env=train_env,
        training_env_ori=initialize_env(
            "train",
            asset_names,
            train_dataset,
            feature_dim,
            serial_selection,
            asset_attention,
            0,
            train_env.worker_model,
        ),
        patience_steps=patience_steps,
        best_model_save_path="./models",
        callback_on_new_best=test_callback,
        verbose=0,
        n_eval_episodes=1,
        eval_freq=eval_freq,
        deterministic=True,
        render=False,
        exclude_names=None,
        metric_fn=eval_reward_metric,
    )

    model = A2C(
        PairSelectionActorCriticPolicy,
        train_env,
        learning_rate=learning_rate,
        seed=seed,
        gamma=rl_gamma,
        ent_coef=ent_coef,
        policy_kwargs=policy_kwargs,
        verbose=0,
        n_steps=1,
        device=device,
    )

    
    checkpoint_callback = CheckpointCallback(
        save_freq=100,  
        save_path="./models/checkpoints/",
        name_prefix="pairs_trading_model"
    )
  
    model.learn(
        total_timesteps=train_steps,
        callback=[checkpoint_callback, eval_callback],
    )



def main():
    logger.info("Starting Test...")
    Selection_train()
    logger.info("Test Completed.")

if __name__ == "__main__":
    main()