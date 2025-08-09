from stable_baselines3 import A2C
from loguru import logger
from pairTradingEnv import pairsTradingEnv
from pairSelectionEnv import pairSelectionEnv

import matplotlib.pyplot as plt
import numpy as np
import torch
import utils
import os


# Configure logging to file
script_dir = os.path.dirname(os.path.abspath(__file__))
logs_dir = os.path.join(script_dir, "..", "logs")
os.makedirs(logs_dir, exist_ok=True)  # Create logs directory if it doesn't exist

log_file = os.path.join(logs_dir, "model_evaluation_{time:YYYY-MM-DD_HH-mm-ss}.log")

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



def Evaluate_saved_model(selection_model_path, worker_model_path):
    policy: str = "simple_serial_selection"
    feature_extractor: str = "mlp"
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
    eval_freq: int = 32
    train_steps: int = int(10000) 
    learning_rate: float = 1e-4
    dropout: float = 0.5
    rl_gamma: float = 1
    ent_coef: float = 1e-4
    trading_train_steps: int = int(1000) 
    trading_feature_extractor_feature_dim: int = 3
    trading_feature_extractor_num_layers: int = 1
    trading_feature_extractor_hidden_dim: int = 64
    trading_dropout: float = 0.0
    trading_feature_extractor_num_heads: int = 2
    trading_learning_rate: float = 1e-4
    trading_rl_gamma: float = 1
    trading_ent_coef: float = 1e-4

    
    # script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Data/U.S.S&P500 directory
    data_path = os.path.join(script_dir, "..", "Data", "U.S.S&P500")

    train = utils.select_data(rolling_dataset_path=data_path, dataset_type="train")
    test = utils.select_data(rolling_dataset_path=data_path, dataset_type="test")
    valid = utils.select_data(rolling_dataset_path=data_path, dataset_type="valid")

    # has to be smaller then the length of the training data
    rolling_serial = 0
    if rolling_serial >= len(train):
        print("Warning: Rolling serial is larger than or equal to the length of the training data")
        rolling_serial = len(train) - 1

    df_train = utils.load_data(path=data_path, file_name=train[rolling_serial])
    df_test = utils.load_data(path=data_path, file_name=test[rolling_serial])
    df_valid = utils.load_data(path=data_path, file_name=valid[rolling_serial])


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
    
    worker_model = A2C.load(worker_model_path, device=device) 
    

    valid_env = initialize_env(
        "valid",
        asset_names,
        valid_dataset,
        feature_dim,
        serial_selection,
        asset_attention,
        0,
        worker_model
    )
    test_env = initialize_env(
        "test",
        asset_names,
        test_dataset,
        feature_dim,
        serial_selection,
        asset_attention,
        0,
        worker_model
    )

    logger.info(f"Loading model from: {selection_model_path}")
    model = A2C.load(selection_model_path, device=device)
    
    results = {}
    
    logger.info("Evaluating on validation set...")
    valid_results = valid_env.eval(
        model, 
        return_figure=True, 
        return_probabilities=True
    )
    results['validation'] = valid_results


    logger.info("Evaluating on test set...")
    test_results = test_env.eval(
        model,
        return_figure=True,
        return_probabilities=True
    )


    results['test'] = test_results

    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    for env_name, env_results in results.items():
        print(f"\n{env_name.upper()} SET:")
        print(f"  Return: {env_results['returns'][-1]:.4f}")
        print(f"  Sharpe Ratio: {env_results['sharpe_ratio']:.4f}")
        print(f"  Annual Return: {env_results['annual_return']:.4f}")
        print(f"  Annual Volatility: {env_results['annual_volatility']:.4f}")
        print(f"  Max Drawdown: {env_results['max_drawdown']:.4f}")

    print("\nDisplaying trajectory plots...")

    if 'trajectory_figure' in valid_results:
        valid_fig = valid_results['trajectory_figure']
        valid_fig.canvas.manager.set_window_title("Validation Set")
        plt.figure(valid_fig.number)
        plt.show(block=False)  

    if 'trajectory_figure' in test_results:
        test_fig = test_results['trajectory_figure']
        test_fig.canvas.manager.set_window_title("Test Set")
        plt.figure(test_fig.number)
        plt.show(block=False)  

    # Keep both figures open
    plt.show()  
    
    return results


def main():
    logger.info("Starting model evaluation script")
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models", "Trials", "seed_13","Checkpoint_2")
    selection_model_path = os.path.join(model_path, "best_model.zip")
    worker_model_path = os.path.join(model_path, "best_worker_model.zip")
    logger.info(f"Selection model path: {selection_model_path}")
    logger.info(f"Worker model path: {worker_model_path}")
    Evaluate_saved_model(selection_model_path=selection_model_path, worker_model_path=worker_model_path)

if __name__ == "__main__":
    main()