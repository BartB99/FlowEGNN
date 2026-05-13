import torch
import torch.nn as nn

from egcl import EGCL
from mlp import MLP
from utils.embedding import SinusoidalThetaEmbedding

class EGNN(nn.Module):
    def __init__(self, t_embed_dim, input_node_d, input_theta_d, theta_param_embd_dim, hidden_nf, 
                 latent_nf, theta_nf, n_layers=4, mlp_layers=2, single_layer=False,
                 recurrent=True, activation=nn.SiLU(), norm="layer", attention=True, 
                 scale_pred=True, coords_weight=1.0, norm_diff=True):
        super(EGNN, self).__init__()

        self.n_layers = n_layers
        self.scale_pred = scale_pred        

        self.h_embed = MLP(input_node_d, hidden_nf, latent_nf, mlp_layers, activation, single_layer)
        self.theta_embed = SinusoidalThetaEmbedding(theta_param_embd_dim)
        self.global_attr_projection = nn.Linear(t_embed_dim + input_theta_d * theta_param_embd_dim, theta_nf)

        self.mass_readout = MLP(latent_nf, hidden_nf, 1, mlp_layers, activation, single_layer=False)

        last_layer = False
        self.layers = nn.ModuleList()
        self.theta_predictor = nn.Linear(latent_nf, input_theta_d)
        self.input_theta_d = input_theta_d
        self.theta_param_embd_dim = theta_param_embd_dim

        for i in range(n_layers):
            # last_layer = (i == n_layers - 1)

            layer = EGCL(
                t_embed_dim=t_embed_dim,
                input_nf=latent_nf,
                hidden_nf=hidden_nf,
                global_in_nf=theta_nf,
                mlp_layers=mlp_layers,
                activation=activation,
                recurrent=recurrent,
                norm=norm,
                attention=attention,
                coords_weight=coords_weight,
                norm_diff=norm_diff,
                last_mp_layer=last_layer
            )

            self.layers.append(layer)

    def forward(self, h, x, t_embed, batch, edge_index, edge_attr=None, theta=None):
        """
        h: node features, e.g. mass. [N, in_features]
        x: node coordinates, 3D positions. [n_nodes, coord_features]
        batch: graph indices. [n_nodes]
        """
    
        self.theta_embed.to(x.device)
        theta_embd = self.theta_embed(theta) if theta is not None else None

        x_input = x.clone()

        if theta_embd is None:
            global_attr = t_embed
        elif t_embed is None:
            global_attr = theta_embd
        else:
            global_attr = torch.cat([t_embed, theta_embd], dim=1)  

        global_attr_embd = self.global_attr_projection(global_attr) if global_attr is not None else None
        
        h = self.h_embed(h)

        vel = torch.zeros_like(x)

        for i, layer in enumerate(self.layers):
            # print(f"Layer {i}: h input magnitude = {h.norm(dim=-1).mean():.6f}")
            h, v, edge_feat = layer(h, edge_index, x, t_embed, edge_attr=edge_attr, 
                            global_attr=global_attr_embd[batch] if global_attr_embd is not None else None,
                            )
            # print(f"Layer {i}: h output magnitude = {h.norm(dim=-1).mean():.6f}")
            # print(f"Layer {i}: v magnitude = {v.norm(dim=-1).mean():.6f}")
            # x = wrap(x + v, **BOX)
            vel += v 
                
        # vel = min_image(x - x_input, **BOX)
        # vel = self.vel_mlp(h)
        vel_m = self.mass_readout(h)

        return vel, vel_m