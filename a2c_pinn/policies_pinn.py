from typing import Dict, List, Type, Union, Tuple

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines.common.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy

class KANLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spline_order = 3  # k
        self.grid_size = 5  # Number of grid intervals, adjustable
        steps = self.grid_size + 1
        grid = th.linspace(-1, 1, steps=steps).unsqueeze(0).expand(self.in_dim, steps)
        self.grid = nn.Parameter(self.extend_grid(grid, k_extend=self.spline_order))
        num_basis = self.grid_size + self.spline_order
        self.spline_coeff = nn.Parameter(th.randn(self.in_dim, self.out_dim, num_basis))
        self.base_weight = nn.Parameter(th.normal(0, 1, (self.in_dim, self.out_dim)))
        self.scale = nn.Parameter(th.ones(1))
        self.silu = nn.SiLU()

    def extend_grid(self, grid, k_extend):
        h = (grid[:, -1] - grid[:, 0]) / (grid.shape[1] - 1)
        h = h.unsqueeze(1)
        for _ in range(k_extend):
            grid = th.cat([grid[:, 0:1] - h, grid], dim=1)
            grid = th.cat([grid, grid[:, -1:] + h], dim=1)
        return grid

    def b_spline_basis(self, x, grid, order):
        x = x.unsqueeze(2)
        grid = grid.unsqueeze(0)
        if order == 0:
            value = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
        else:
            basis_lower = self.b_spline_basis(x[:, :, 0], grid[0], order - 1)
            left_coef = (x - grid[:, :, :-(order + 1)]) / (grid[:, :, order:-1] - grid[:, :, :-(order + 1)])
            left = left_coef * basis_lower[:, :, :-1]
            right_coef = (grid[:, :, order + 1:] - x) / (grid[:, :, order + 1:] - grid[:, :, 1:-(order)])
            right = right_coef * basis_lower[:, :, 1:]
            value = left + right
        return th.nan_to_num(value)

    def forward(self, x):
        silu = self.silu(x)
        basis = self.b_spline_basis(x, self.grid, self.spline_order)
        spline = th.einsum('bij,ikj->bik', basis, self.spline_coeff)
        base = silu[:, :, None] * self.base_weight[None, :, :]
        y = base + spline
        y = y * self.scale
        y = y.sum(dim=1)
        return y

class KANMlpExtractor(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[List[Union[int, Dict[str, List[int]]]], Dict[str, List[int]]],
        activation_fn: Type[nn.Module],
        device: Union[th.device, str] = "auto",
    ) -> None:
        super().__init__()
        device = get_device(device)
        if isinstance(net_arch, dict):
            net_arch = [net_arch]
        shared_net: List[nn.Module] = []
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        policy_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the policy network
        value_only_layers: List[int] = []  # Layer sizes of the network that only belongs to the value network

        # Iterate through the shared layers and build the shared parts of the network
        last_layer_dim_shared = feature_dim
        for layer in net_arch:
            if isinstance(layer, int):  # the shared layers
                shared_net.append(KANLayer(last_layer_dim_shared, layer))
                last_layer_dim_shared = layer
            else:
                assert isinstance(layer, dict), "Error: the net_arch list can only contain ints and dicts"
                if "pi" in layer:
                    assert isinstance(layer["pi"], list), "Error: net_arch[-1]['pi'] must be List[int]"
                    policy_only_layers = layer["pi"]
                if "vf" in layer:
                    assert isinstance(layer["vf"], list), "Error: net_arch[-1]['vf'] must be List[int]"
                    value_only_layers = layer["vf"]
                break  # Stop after the dict with separate layers

        last_layer_dim_pi = last_layer_dim_shared
        last_layer_dim_vf = last_layer_dim_shared

        # Build the non-shared part of the network
        from itertools import zip_longest
        for pi_layer_size, vf_layer_size in zip_longest(policy_only_layers, value_only_layers):
            if pi_layer_size is not None:
                policy_net.append(KANLayer(last_layer_dim_pi, pi_layer_size))
                last_layer_dim_pi = pi_layer_size
            if vf_layer_size is not None:
                value_net.append(KANLayer(last_layer_dim_vf, vf_layer_size))
                last_layer_dim_vf = vf_layer_size

        # Save dim, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Create networks
        self.shared_net = nn.Sequential(*shared_net).to(device)
        self.policy_net = nn.Sequential(*policy_net).to(device)
        self.value_net = nn.Sequential(*value_net).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent), self.value_net(shared_latent)
    # Add the following methods to the KANMlpExtractor class in stable_baselines/a2c_pinn/policies_pinn.py after the forward method
    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        shared_latent = self.shared_net(features)
        return self.policy_net(shared_latent)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        shared_latent = self.shared_net(features)
        return self.value_net(shared_latent)
    
class KANActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy using enhanced KAN for the MLP extractor.
    """
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = KANMlpExtractor(self.features_dim, self.net_arch, self.activation_fn, self.device)

MlpPolicy = KANActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy  # Can extend if needed
MultiInputPolicy = MultiInputActorCriticPolicy  # Can extend if needed