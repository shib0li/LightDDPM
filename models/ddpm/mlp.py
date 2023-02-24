import torch
import torch.nn as nn


class MLP(nn.Module):
    """
    Simple fully connected neural network

    Args:
        in_dim
        hidden_dim:
        hidden_layers:
        out_dim:
        act:
    """

    def __init__(
            self,
            in_dim: int,
            hidden_dim: int,
            hidden_layers: int,
            out_dim: int,
            act: str,
    ):
        super().__init__()

        layer_configs = [in_dim] + [hidden_dim] * hidden_layers + [out_dim]

        if act == 'tanh':
            act_fn = nn.Tanh()
        elif act == 'relu':
            act_fn = nn.ReLU()
        elif act == 'elu':
            act_fn = nn.ELU()
        #

        layers = []
        for i in range(len(layer_configs) - 2):
            layers.append(nn.Linear(layer_configs[i], layer_configs[i + 1]))
            nn.init.xavier_normal_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            layers.append(act_fn)
        #

        layers.append(nn.Linear(layer_configs[-2], layer_configs[-1]))
        nn.init.xavier_normal_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        self.net = nn.Sequential(*layers)

    def forward(
            self,
            x: torch.Tensor,
    ) -> torch.Tensor:
        return self.net(x)

class MLPDiffusion(nn.Module):
    def __init__(
            self,
            s_dim,
            hidden_dim,
            hidden_layers,
            n_steps,
    ):
        super().__init__()

        self.s_dim = s_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.n_steps = n_steps

        layer_configs = [s_dim] + [hidden_dim] * hidden_layers + [s_dim]

        layers = []
        for i in range(len(layer_configs) - 2):
            layers.append(nn.Linear(layer_configs[i], layer_configs[i + 1]))
            nn.init.xavier_normal_(layers[-1].weight)
            nn.init.zeros_(layers[-1].bias)
            layers.append(nn.ReLU())
        #

        layers.append(nn.Linear(layer_configs[-2], layer_configs[-1]))
        nn.init.xavier_normal_(layers[-1].weight)
        nn.init.zeros_(layers[-1].bias)

        self.net = nn.ModuleList(layers)
        self.t_embeds = nn.ModuleList([nn.Embedding(n_steps, hidden_dim) for l in range(hidden_layers)])

    def forward(self, x, t):

        for i in range(self.hidden_layers):
            nn_layer = self.net[2 * i]
            t_embed_layer = self.t_embeds[i]
            act_layer = self.net[2 * i + 1]

            x = nn_layer(x)
            x += t_embed_layer(t)
            x = act_layer(x)
        #

        nn_layer = self.net[-1]

        x = nn_layer(x)

        return x
