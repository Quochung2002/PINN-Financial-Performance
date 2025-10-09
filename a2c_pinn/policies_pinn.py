from typing import Dict, List, Type, Union

import torch as th
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.preprocessing import get_action_dim

from stable_baselines.common.policies import ActorCriticCnnPolicy, MultiInputActorCriticPolicy

class KANLayer(nn.Module):
    """
    Kolmogorov-Arnold Network (KAN) Layer with Fourier features for better handling high-frequency data.
    """
    def __init__(self, in_features: int, out_features: int, grid_size: int = 5, spline_order: int = 3, fourier_k: int = 5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.fourier_k = fourier_k  # Number of Fourier frequencies

        self.grid = nn.Parameter(th.linspace(-1, 1, grid_size + 2 * spline_order + 1).unsqueeze(0).unsqueeze(0), requires_grad=False)

        self.coefficients = nn.Parameter(th.zeros(out_features, in_features, grid_size + spline_order))

        # Learnable scale for adaptive activation
        self.scale = nn.Parameter(th.ones(1))

        self.reset_parameters()

    def reset_parameters(self):
        with th.no_grad():
            for i in range(self.out_features):
                for j in range(self.in_features):
                    self.coefficients[i, j] = th.sin(th.linspace(0, 2 * th.pi, self.coefficients.shape[-1]))

    def b_spline_basis(self, x: th.Tensor, grid: th.Tensor, order: int) -> th.Tensor:
        x = x.unsqueeze(-1)
        if order == 0:
            return ((x >= grid[:, :-1]) & (x < grid[:, 1:])).float()

        basis_lower = self.b_spline_basis(x, grid, order - 1)
        left = (x - grid[:, :-order]) / (grid[:, order:] - grid[:, :-order]) * basis_lower[:, :, :-1]
        right = (grid[:, order + 1 :] - x) / (grid[:, order + 1 :] - grid[:, 1:-(order - 1)]) * basis_lower[:, :, 1:]
        return left + right

    def forward(self, x: th.Tensor) -> th.Tensor:
        # Add Fourier features
        if self.fourier_k > 0:
            freqs = th.arange(1, self.fourier_k + 1, device=x.device).unsqueeze(0).unsqueeze(2) * th.pi
            fourier_sin = th.sin(freqs * x.unsqueeze(1))
            fourier_cos = th.cos(freqs * x.unsqueeze(1))
            x = th.cat([x, fourier_sin.flatten(1), fourier_cos.flatten(1)], dim=-1)

        basis = self.b_spline_basis(x, self.grid, self.spline_order)
        activations = th.einsum('bid,oid->bo', basis, self.coefficients) * self.scale
        return activations

class KANMlpExtractor(nn.Module):
    """
    KAN-based MLP extractor for actor-critic policies with enhanced depth.
    """
    def __init__(
        self,
        features_dim: int,
        net_arch: Union[List[int], Dict[str, List[int]]] = [256, 256],  # Wider default
        activation_fn: Type[nn.Module] = nn.ReLU,  # Ignored for KAN
        device: Union[th.device, str] = "auto",
    ):
        super().__init__()
        policy_net: List[nn.Module] = []
        value_net: List[nn.Module] = []
        last_layer_dim_shared = features_dim

        if isinstance(net_arch, dict):
            shared_net = net_arch.get("shared_layers", [])
            pi_net = net_arch.get("pi", [])
            vf_net = net_arch.get("vf", [])
        else:
            shared_net = net_arch
            pi_net = []
            vf_net = []

        for layer_dim in shared_net:
            policy_net.append(KANLayer(last_layer_dim_shared, layer_dim))
            value_net.append(KANLayer(last_layer_dim_shared, layer_dim))
            last_layer_dim_shared = layer_dim

        last_layer_dim_pi = last_layer_dim_shared
        for layer_dim in pi_net:
            policy_net.append(KANLayer(last_layer_dim_pi, layer_dim))
            last_layer_dim_pi = layer_dim

        last_layer_dim_vf = last_layer_dim_shared
        for layer_dim in vf_net:
            value_net.append(KANLayer(last_layer_dim_vf, layer_dim))
            last_layer_dim_vf = layer_dim

        self.policy_net = nn.Sequential(*policy_net)
        self.value_net = nn.Sequential(*value_net)
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)

class KANActorCriticPolicy(ActorCriticPolicy):
    """
    Actor-critic policy using enhanced KAN for the MLP extractor.
    """
    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = KANMlpExtractor(self.features_dim, self.net_arch, self.activation_fn, self.device)

MlpPolicy = KANActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy  # Can extend if needed
MultiInputPolicy = MultiInputActorCriticPolicy  # Can extend if needed