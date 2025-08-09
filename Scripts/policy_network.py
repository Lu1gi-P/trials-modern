import torch as th
from torch import nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution
import gymnasium as gym
from typing import Callable


class PairSelectionDistribution(MultiCategoricalDistribution):
    """Custom distribution to allow directly logit output from proxy network"""

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        The layer that directly take the output as the distribution:
        The proxy network should be the same as the logits (flattened)
        You can then get probabilities using a softmax on each sub-space.
        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """

        action_logits = nn.Identity(latent_dim, sum(self.action_dims))
        return action_logits
    
class SimpleSerialSelection(nn.Module):
    """Simple serial selection policy network

    :param feature_dim: dimension of the features, which is N x M
    :param hidden_dim: dimension of the hidden representations as M
    :param asset_num: number of assets as N
    :param last_layer_dim_pi: (int) number of units of the policy network
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        asset_num: int,
        last_layer_dim_pi: int = 60,
        **kwargs,
    ):
        super(SimpleSerialSelection, self).__init__()
        # the output number should be equal to the flattened logits
        # which is twice as the number of assets
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.last_layer_dim_pi = last_layer_dim_pi
        # The market vector to calculate logits
        self.selection_vector = nn.Parameter(
            th.Tensor(self.hidden_dim).unsqueeze(0)
        )

    def forward(self, features: th.Tensor) -> th.Tensor:
        """Calculate logits and reverse them to select two assets

        :return: (th.Tensor) latent_policy
        """
        # B x N x M
        features = features.reshape(-1, self.asset_num, self.hidden_dim)
        batch_size = features.size(0)
        # B x N
        forward_logits = th.bmm(
            features,
            self.selection_vector.unsqueeze(-1).repeat(batch_size, 1, 1),
        ).squeeze(-1)
        backward_logits = -forward_logits

        return th.cat([forward_logits, backward_logits], dim=1)
    
POLICY_NETWORKS = {
    "simple_serial_selection": SimpleSerialSelection,
}
    
class PairSelectionNetwork(nn.Module):
    """Pair selection policy and mlp value function.
    It receives as input the features extracted by the feature extractor.

    :param feature_dim: dimension of the features.
    :param hidden_dim: dimension of the hidden representations as M
    :param asset_num: number of assets as N
    :param last_layer_dim_pi: (int) number of units of the policy network
    :param last_layer_dim_vf: (int) number of units of the value network
    """

    def __init__(
        self,
        policy: str,
        feature_dim: int,
        hidden_dim: int,
        asset_num: int,
        num_heads: int,
        last_layer_dim_pi: int = 60,
        last_layer_dim_vf: int = 64,
    ):
        super(PairSelectionNetwork, self).__init__()

        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        # Policy network
        self.policy_net = POLICY_NETWORKS[policy](
            feature_dim,
            hidden_dim,
            asset_num,
            last_layer_dim_pi=last_layer_dim_pi,
            num_heads=num_heads,
        )

        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(hidden_dim * asset_num, last_layer_dim_vf), nn.ReLU()
        )

    def forward(self, features: th.Tensor) -> tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value
            If all layers are shared, then ``latent_policy == latent_value``
        """
        batch_size = features.shape[0]
        return self.policy_net(features), self.value_net(
            features.reshape(batch_size, -1)
        )

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        batch_size = features.shape[0]
        return self.value_net(features.reshape(batch_size, -1))


class PairSelectionActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Callable[[float], float],
        net_arch: list[int | dict[str, list[int]]] | None = None,
        activation_fn: type[nn.Module] = nn.Tanh,
        feature_dim: int = 3,
        hidden_dim: int = 32,
        asset_num: int = 60,
        num_heads: int = 2,
        latent_pi: int = 60,
        latent_vf: int = 64,
        policy: str = "simple_serial",
        *args,
        **kwargs,
    ):
        self.hidden_dim = hidden_dim
        self.asset_num = asset_num
        self.policy = policy
        self.ortho_init = False
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        self.latent_pi = latent_pi
        self.latent_vf = latent_vf
        super(PairSelectionActorCriticPolicy, self).__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch,
            activation_fn,
            # Pass remaining arguments to base class
            *args,
            **kwargs,
        )
        # Disable orthogonal initialization
        if policy in ["simple_serial_selection"]:
            self.action_dist = PairSelectionDistribution(action_space.nvec)
            self.action_net = self.action_dist.proba_distribution_net(
                latent_dim=latent_pi
            )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = PairSelectionNetwork(
            self.policy,
            self.feature_dim,
            self.hidden_dim,
            self.asset_num,
            num_heads=self.num_heads,
            last_layer_dim_pi=self.latent_pi,
            last_layer_dim_vf=self.latent_vf,
        )
