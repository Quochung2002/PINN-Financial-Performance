# This file is here just to define MlpPolicy/CnnPolicy
# that work for PPO
from stable_baselines3.common.torch_layers import MlpExtractor
from stable_baselines3.common.policies import ActorCriticPolicy, ActorCriticCnnPolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.utils import get_device
import torch.nn as nn
import torch as th
from typing import Union

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer for PIKAN.
    Each edge applies a learnable B-spline activation to univariate inputs.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid = nn.Parameter(th.linspace(-1, 1, grid_size + 2 * spline_order + 1).unsqueeze(0).unsqueeze(0), requires_grad=False)
        self.coefficients = nn.Parameter(th.zeros(out_features, in_features, grid_size + spline_order))
        self.reset_parameters()

    def reset_parameters(self):
        with th.no_grad():
            nn.init.orthogonal_(self.coefficients, gain=1.0)

    def b_spline_basis(self, x: th.Tensor, grid: th.Tensor, order: int) -> th.Tensor:
        num_knots = grid.shape[-1]
        if order == 0:
            return ((x >= grid[..., 0 : num_knots - 1]) & (x < grid[..., 1 : num_knots])).float()
        basis_lower = self.b_spline_basis(x, grid, order - 1)
        left_numer = x - grid[..., 0 : num_knots - (order + 1)]
        left_denom = grid[..., order : num_knots - 1] - grid[..., 0 : num_knots - (order + 1)]
        left = left_numer / left_denom * basis_lower[..., 0 : basis_lower.shape[-1] - 1]
        right_numer = grid[..., order + 1 : ] - x
        right_denom = grid[..., order + 1 : ] - grid[..., 1 : num_knots - order]
        right = right_numer / right_denom * basis_lower[..., 1 : ]
        value = left + right
        return th.nan_to_num(value)

    def forward(self, x: th.Tensor) -> th.Tensor:
        grid = self.grid.expand(x.size(0), self.in_features, -1)
        x = x.unsqueeze(-1)  # Unsqueeze here once, outside recursion
        basis = self.b_spline_basis(x, grid, self.spline_order)
        activations = th.einsum('bfd,ofd->bo', basis, self.coefficients)
        return activations

class KANMlpExtractor(MlpExtractor):
    """
    Custom MlpExtractor using KAN layers instead of standard MLP.
    """
    def __init__(
        self,
        feature_dim: int,
        net_arch: Union[list[int], dict[str, list[int]]],
        activation_fn: type[nn.Module],  # Unused for KAN
        device: Union[th.device, str] = "auto",
    ):
        super().__init__(feature_dim, net_arch, activation_fn, device)
        self.device = get_device(device)  # Explicitly set device

        # Define function to build KAN sequential
        def create_kan_net(input_dim: int, arch: list[int]) -> nn.Sequential:
            layers = []
            current_dim = input_dim
            for dim in arch:
                layers.append(KANLayer(current_dim, dim))
                current_dim = dim
            return nn.Sequential(*layers)

        shared_net: nn.Sequential = nn.Sequential()
        policy_net: nn.Sequential = nn.Sequential()
        value_net: nn.Sequential = nn.Sequential()

        if isinstance(net_arch, dict):
            if "layers_common" in net_arch:
                shared_net = create_kan_net(feature_dim, net_arch["layers_common"])
            if "layers_policy" in net_arch:
                policy_net = create_kan_net(shared_net[-1].out_features if len(shared_net) > 0 else feature_dim, net_arch["layers_policy"])
            if "layers_value" in net_arch:
                value_net = create_kan_net(shared_net[-1].out_features if len(shared_net) > 0 else feature_dim, net_arch["layers_value"])
            self.latent_dim_pi = net_arch["layers_policy"][-1] if "layers_policy" in net_arch and net_arch["layers_policy"] else (shared_net[-1].out_features if len(shared_net) > 0 else feature_dim)
            self.latent_dim_vf = net_arch["layers_value"][-1] if "layers_value" in net_arch and net_arch["layers_value"] else (shared_net[-1].out_features if len(shared_net) > 0 else feature_dim)
        else:
            shared_net = create_kan_net(feature_dim, net_arch)
            self.latent_dim_pi = net_arch[-1] if net_arch else feature_dim
            self.latent_dim_vf = self.latent_dim_pi

        self.shared_net = shared_net.to(self.device)
        self.policy_net = policy_net.to(self.device)
        self.value_net = value_net.to(self.device)

class KANActorCriticPolicy(ActorCriticPolicy):
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = KANMlpExtractor(
            self.features_dim,
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
            device=self.device,
        )

# Update aliases
MlpPolicy = KANActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy  # Can subclass similarly if needed
MultiInputPolicy = MultiInputActorCriticPolicy  # Can subclass if needed