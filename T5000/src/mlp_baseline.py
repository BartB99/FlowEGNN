import torch
import torch.nn as nn

from utils.embedding import SinusoidalThetaEmbedding


class MLPBaseline(nn.Module):
    """
    Drop-in replacement for EGNN. Same forward signature, same output shape.
    Ignores graph structure entirely — just maps per-node features to velocity.
    """
    def __init__(self, t_embed_dim, input_node_d, input_theta_d, theta_param_embd_dim,
                 hidden_nf=256, n_mlp_layers=4, activation=nn.SiLU(),
                 latent_nf=None, theta_nf=None, n_layers=None, mlp_layers=None,
                 single_layer=None, recurrent=None, norm=None, attention=None,
                 scale_pred=None, coords_weight=None, norm_diff=None):
        super().__init__()

        self.input_theta_d = input_theta_d
        self.theta_param_embd_dim = theta_param_embd_dim
        self.theta_embed = SinusoidalThetaEmbedding(theta_param_embd_dim)

        # Input: x(3) + h(input_node_d) + t_embed(t_embed_dim) + theta_embed(input_theta_d * theta_param_embd_dim)
        theta_emb_dim = input_theta_d * theta_param_embd_dim
        in_dim = 3 + input_node_d + t_embed_dim + theta_emb_dim

        layers = []
        for i in range(n_mlp_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_nf, hidden_nf))
            layers.append(activation)
        layers.append(nn.Linear(hidden_nf, 3))

        self.net = nn.Sequential(*layers)

    def forward(self, h, x, t_embed, batch, edge_index, edge_attr=None, theta=None):
        """
        Same signature as EGNN.forward(). 
        Returns predicted velocity [N, 3].
        """
        self.theta_embed.to(x.device)
        theta_embd = self.theta_embed(theta) if theta is not None else None
        t_per_node = t_embed[batch]  # [N, t_embed_dim]

        components = [x, h]

        if t_per_node is not None:
            components.append(t_per_node)
        if theta_embd is not None:
            theta_per_node = theta_embd[batch]  
            components.append(theta_per_node)

        inp = torch.cat(components, dim=1)  

        vel = self.net(inp) 

        return vel