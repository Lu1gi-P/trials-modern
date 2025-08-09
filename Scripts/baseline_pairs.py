import math
import os

from loguru import logger
from statsmodels.tsa.stattools import coint
from scipy.stats import kendalltau, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')  # Suppress copula warnings
import numpy as np
import pandas as pd
import utils


dataset_names = ["train", "valid", "test", "formation"]
x_symbol = " "
y_symbol = " "


def log_func(x):
    if isinstance(x, str):
        x = float(x.replace(",", ""))
    return math.log(x)


def dist(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.sqrt(sum((a - b) ** 2)) / len(a)


def vertify_coint(asset_x, asset_y, p_threshold) -> bool:
    if len(asset_x) != len(asset_y):
        return False
    _, p_value, _ = coint(asset_x, asset_y)
    return p_value < p_threshold


def corr(a, b):
    a = np.array(a)
    b = np.array(b)
    corr = np.corrcoef(a, b)[0][1]
    return corr


def select_pairs_copula_spearman(*args: list) -> tuple[str, str]:
    """
    Pick the stock pair with the highest Spearman rank correlation.
    Well-established non-parametric measure of monotonic relationships.
    
    Args:
    args[0]: Contains data for all stocks in a rolling.
    
    Returns:
    Tuple of (x_symbol, y_symbol) with highest Spearman correlation
    """
    
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    
    max_correlation = -1.0
    best_x_symbol = ""
    best_y_symbol = ""
    
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            try:
                x_close = (
                    df[column_name[i * 3][0]]["close"]
                    .apply(lambda x: float(x.replace(",", "")))
                    .values.tolist()
                )
                y_close = (
                    df[column_name[j * 3][0]]["close"]
                    .apply(lambda x: float(x.replace(",", "")))
                    .values.tolist()
                )
                
                x_returns = np.diff(np.log(x_close))
                y_returns = np.diff(np.log(y_close))
                
                rho, p_value = spearmanr(x_returns, y_returns)
                
                abs_rho = abs(rho)
                
                if p_value < 0.05 and abs_rho > max_correlation:
                    max_correlation = abs_rho
                    best_x_symbol = column_name[i * 3][0]
                    best_y_symbol = column_name[j * 3][0]
                    
            except (ValueError, ZeroDivisionError):
                continue
    
    return best_x_symbol, best_y_symbol


def select_pairs_eucl(*args: list) -> None:
    """
    Pick the stock pair with the smallest Euclidean distance.

    Args:
    args[0]: Contains data for all stocks in a rolling.

    Returns:
    Nothing to see here.
    """
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)

    dis_min = float("inf")
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            x_close = (
                df[column_name[i * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            y_close = (
                df[column_name[j * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            # normilize values
            x_close = list(map(lambda x: x / x_close[0], x_close))
            y_close = list(map(lambda x: x / y_close[0], y_close))

            dis = dist(x_close, y_close)
            if dis < dis_min:
                dis_min = dis
                x_symbol = column_name[i * 3][0]
                y_symbol = column_name[j * 3][0]
    return x_symbol, y_symbol


def select_pairs_coin(*args: list) -> None:
    """
    Pick the stock pair with the smallest Euclidean distance
    where the value of P_value is below p_threshold.

    Args:
    args[0]: Contains data for all stocks in a rolling.
    args[1]: The maximum value of p_value

    Returns:
    Nothing to see here.
    """
    df = args[0]
    p_threshold = args[1]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)

    dis_min = float("inf")
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            x_close = (
                df[column_name[i * 3][0]]["close"]
                .apply(log_func)
                .values.tolist()
            )
            y_close = (
                df[column_name[j * 3][0]]["close"]
                .apply(log_func)
                .values.tolist()
            )

            if vertify_coint(x_close, y_close, p_threshold):
                x_close = list(map(lambda x: x / x_close[0], x_close))
                y_close = list(map(lambda x: x / y_close[0], y_close))
                dis = dist(x_close, y_close)
                if dis < dis_min:
                    dis_min = dis
                    x_symbol = column_name[i * 3][0]
                    y_symbol = column_name[j * 3][0]
    return x_symbol, y_symbol



def select_pairs_corr(*args: list) -> None:
    """
    Pick the stock pair with the maximum correlation.

    Args:
    args[0]: Contains data for all stocks in a rolling.

    Returns:
    Nothing to see here.
    """
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    corr_max = 0.0
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            x_close = (
                df[column_name[i * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            y_close = (
                df[column_name[j * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            # normilize values
            x_close = list(map(lambda x: x / x_close[0], x_close))
            y_close = list(map(lambda x: x / y_close[0], y_close))

            corr_value = corr(x_close, y_close)
            if corr_value > corr_max:
                corr_max = corr_value
                x_symbol = column_name[i * 3][0]
                y_symbol = column_name[j * 3][0]
    return x_symbol, y_symbol

def select_pairs_cluster_corr(*args: list, n_clusters: int = 6) -> tuple[str, str]:
    """
    Pick the stock pair with maximum correlation within the same cluster.
    
    Args:
    args[0]: Contains data for all stocks in a rolling.
    n_clusters: Number of clusters for k-means (default: 6 for ~5 assets per cluster)
    
    Returns:
    Tuple of (x_symbol, y_symbol) with highest correlation within same cluster
    """
    
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    
    price_data = []
    asset_names = []
    
    for i in range(column_num):
        try:
            close_prices = (
                df[column_name[i * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            # Normalize prices
            normalized_prices = [x / close_prices[0] for x in close_prices]
            price_data.append(normalized_prices)
            asset_names.append(column_name[i * 3][0])
            
        except (ValueError, ZeroDivisionError):
            continue
    
    price_data = np.array(price_data)
    
    scaler = StandardScaler()
    price_data_scaled = scaler.fit_transform(price_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(price_data_scaled)
    
    best_corr = -1.0
    best_x_symbol = ""
    best_y_symbol = ""
    
    for cluster_id in range(n_clusters):
        cluster_indices = np.where(cluster_labels == cluster_id)[0]
        
        if len(cluster_indices) < 2: 
            continue
            
        for i in range(len(cluster_indices)):
            for j in range(i + 1, len(cluster_indices)):
                idx_i = cluster_indices[i]
                idx_j = cluster_indices[j]
                
                corr_value = np.corrcoef(price_data[idx_i], price_data[idx_j])[0, 1]
                
                if corr_value > best_corr:
                    best_corr = corr_value
                    best_x_symbol = asset_names[idx_i]
                    best_y_symbol = asset_names[idx_j]
    
    return best_x_symbol, best_y_symbol


def select_pairs_pca(*args: list, n_components: int = 3) -> tuple[str, str]:
    """
    Pick the stock pair with the highest loadings on the same principal component.
    PCA identifies the main sources of variation, and pairs with high loadings
    on the same component are likely to mean-revert when they deviate.
    
    Args:
    args[0]: Contains data for all stocks in a rolling.
    n_components: Number of principal components to consider (default: 3)
    
    Returns:
    Tuple of (x_symbol, y_symbol) with highest same-component loadings
    """
    
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    
    price_data = []
    asset_names = []
    
    for i in range(column_num):
        try:
            close_prices = (
                df[column_name[i * 3][0]]["close"]
                .apply(lambda x: float(x.replace(",", "")))
                .values.tolist()
            )
            log_returns = np.diff(np.log(close_prices))
            price_data.append(log_returns)
            asset_names.append(column_name[i * 3][0])
            
        except (ValueError, ZeroDivisionError):
            continue
    
    returns_matrix = np.array(price_data)
    
    scaler = StandardScaler()
    returns_scaled = scaler.fit_transform(returns_matrix)
    
    pca = PCA(n_components=n_components)
    pca.fit(returns_scaled)
    
    loadings = pca.components_
    
    best_score = -1.0
    best_x_symbol = ""
    best_y_symbol = ""
    
    for component_idx in range(n_components):
        component_loadings = loadings[component_idx, :]
        
        for i in range(len(asset_names)):
            for j in range(i + 1, len(asset_names)):

                loading_i = abs(component_loadings[i])
                loading_j = abs(component_loadings[j])
                
                pair_score = loading_i * loading_j
                
                if pair_score > best_score:
                    best_score = pair_score
                    best_x_symbol = asset_names[i]
                    best_y_symbol = asset_names[j]
    
    return best_x_symbol, best_y_symbol

def select_pairs_copula_kendall(*args: list) -> tuple[str, str]:
    """
    Pick the stock pair with the highest copula dependence (Kendall's tau).
    
    Args:
    args[0]: Contains data for all stocks in a rolling.
    
    Returns:
    x_symbol, y_symbol with highest copula dependence
    """
    
    df = args[0]
    column_name = df.columns.values.tolist()
    column_num = int(df.shape[1] / 3)
    
    max_dependence = -1.0
    best_x_symbol = ""
    best_y_symbol = ""
    
    for i in range(0, column_num):
        for j in range(i + 1, column_num):
            try:
                # Get price data
                x_close = (
                    df[column_name[i * 3][0]]["close"]
                    .apply(lambda x: float(x.replace(",", "")))
                    .values.tolist()
                )
                y_close = (
                    df[column_name[j * 3][0]]["close"]
                    .apply(lambda x: float(x.replace(",", "")))
                    .values.tolist()
                )
                
                # Convert to returns instead of normalized prices
                x_returns = np.diff(np.log(x_close))
                y_returns = np.diff(np.log(y_close))
                
                # Calculate Kendall's tau (measures rank correlation)
                tau, p_value = kendalltau(x_returns, y_returns)
                
                # Use absolute tau to capture both positive and negative dependence
                abs_tau = abs(tau)
                
                # Only consider statistically significant relationships
                if p_value < 0.05 and abs_tau > max_dependence:
                    max_dependence = abs_tau
                    best_x_symbol = column_name[i * 3][0]
                    best_y_symbol = column_name[j * 3][0]
                    
            except (ValueError, ZeroDivisionError):
                # Skip pairs with insufficient data or calculation errors
                continue
    
    return best_x_symbol, best_y_symbol

METHODS = {
    "euclidean": select_pairs_eucl,
    "cointegration": select_pairs_coin,
    "correlation": select_pairs_corr,
    "copula_kendall": select_pairs_copula_kendall,
    "copula_spearman": select_pairs_copula_spearman,
    "cluster_correlation": select_pairs_cluster_corr,
    "pca": select_pairs_pca
}



def main(
    method: str,
    p_threshold: float = 0.05,
) -> None:
    """
    Pick a pair of stocks according to method_num from rolling_dataset_path

    Args:
    method: The methods for selecting stock pairs, including 'euclidean',
            'cointegration' and 'correlation'.
    p_threshold: When method is 'cointegration', p_threshold is meaningful,
                 and p_threshold represents the maximum value of p_value.
                 When method isn't 'cointegration', p_threshold has no meaning.

    Returns:
    Nothing to see here.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
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
    rolling_serial = 0  
    if rolling_serial >= len(train):
        logger.warning("Rolling serial is larger than or equal to the length of the training data")
        rolling_serial = len(train) - 1

    # store_path = f"{store_path}/{method}_pairs"
    # os.makedirs(store_path, exist_ok=True)
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

    asset_names = (df_train.columns.get_level_values(0).drop_duplicates().values.tolist())

    df_formation = pd.concat([df_train, df_valid], axis=0)
    df_rolling = [df_train, df_valid, df_test, df_formation]
    df_formation = df_formation.astype(str)
    args = (
        [df_formation]
        if method != "cointegration"
        else [df_formation, float(p_threshold)]
    )
    x_symbol, y_symbol = METHODS[method](*args)

    # instead of returning the symbols, we can give the indexes of the symbols
    x_index = asset_names.index(x_symbol)
    y_index = asset_names.index(y_symbol)

    logger.info(f"Selected pairs for method {method}: {x_symbol}, {y_symbol} at indexes {x_index}, {y_index}")
    # potentially create dataset with the selected pairs



if __name__ == "__main__":
    # main(method="correlation")
    # main(method="cointegration", p_threshold=0.2)
    # main(method="euclidean")
    # main(method="copula_kendall")
    # main(method="copula_spearman")
    # main(method="cluster_correlation")
    main(method="pca")
