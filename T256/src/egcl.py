import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch_geometric.utils import softmax

from mlp import MLP
from pbc_config import BOX, min_image
from utils import unsorted_segment_mean, unsorted_segment_sum

class EGCL(nn.Module):
    def __init__(self,
                 t_embed_dim,
                 input_nf,
                 hidden_nf,
                 global_in_nf=0,
                 mlp_layers=2,
                 activation=nn.SiLU(),
                 recurrent=True,
                 norm="layer",
                 attention=True,
                 coords_weight=1.0,
                 norm_diff=False,
                 last_mp_layer=False
                 ):
        super(EGCL, self).__init__()
        
        self.activation = activation
        self.recurrent = recurrent
        self.norm = norm
        self.attention = attention
        self.coords_weight = coords_weight
        self.norm_diff = norm_diff
        self.last_mp_layer = last_mp_layer

        edges_in_d =  3     # coordinate difference 
        input_edge_nf = input_nf
        
        self.edge_mlp = MLP(2*input_edge_nf + edges_in_d + global_in_nf + 1, hidden_nf, input_nf, mlp_layers, activation)
        # self.edge_mlp = MLP(input_edge_nf + edges_in_d + global_in_nf + 1, hidden_nf, input_nf, mlp_layers, activation)

        if self.last_mp_layer:
            self.node_mlp = None
        else:
            self.node_mlp = MLP(input_nf + input_nf + global_in_nf, hidden_nf, input_nf, mlp_layers, activation)   
        
        if norm == "layer" and not self.last_mp_layer:
            self.node_norm = nn.LayerNorm(input_nf)
        if norm == "weight" and not self.last_mp_layer:
            for i, module in enumerate(self.node_mlp.mlp):
                if isinstance(module, nn.Linear):
                    self.node_mlp.mlp[i] = weight_norm(module)

        layer = nn.Linear(hidden_nf, 1, bias=False)
        torch.nn.init.xavier_uniform_(layer.weight)#, gain=0.01)
        coord_mlp = []
        coord_mlp.append(nn.Linear(2*input_nf, hidden_nf))
        coord_mlp.append(activation)
        coord_mlp.append(layer)
        self.coord_mlp = nn.Sequential(*coord_mlp)

        if self.attention:
            self.attn_mlp = nn.Sequential(
                nn.Linear(input_nf, hidden_nf),
                activation,
                nn.Linear(hidden_nf, 1),
                )
            
    def edge_model(self, source, target, radial, row, edge_attr=None, global_attr=None, num_nodes=None):

        # print(f"Source shape: {source.shape}, Target shape: {target.shape}, Radial shape: {radial.shape}")
        # components = [source - target, radial]
        # components = [source - target]
        components = [source, target, radial]


        if edge_attr is not None:
            components.append(edge_attr)
        if global_attr is not None:
            components.append(global_attr)

        out = torch.cat(components, dim=1)
        out = self.edge_mlp(out)

        if self.attention:
            att_val = self.attn_mlp(out)
            # Normalize across all neighboring edges ensuring they sum up to 1 for each node
            # alpha = softmax(att_val, row, num_nodes=torch.unique(row).shape[0])
            alpha = softmax(att_val, row, num_nodes=num_nodes)           
            out = out * alpha  # Weighted edge features

        return out

    def node_model(self, h, edge_index, edge_attr, global_attr=None):
        row, _ = edge_index
        agg = unsorted_segment_sum(edge_attr, row, num_segments=h.size(0))
        
        components = [h, agg]
        out = torch.cat(components, dim=1)

        if global_attr is not None:
            out = torch.cat([out, global_attr], dim=1)

        out = self.node_mlp(out)

        if self.recurrent:
            out += h
        
        # print(f"pre-norm: {out.norm(dim=-1).mean():.4f}, std: {out.std(dim=-1).mean():.4f}")

        # Normalization after residual connection ensures stable gradients and smoother training
        if self.norm == "layer":
            out = self.node_norm(out)

        # print(f"post-norm: {out.norm(dim=-1).mean():.4f}, std: {out.std(dim=-1).mean():.4f}")

        return out

    def coord_model(self, coord, h, edge_index, coord_diff, edge_feat):
        row, _ = edge_index
        trans = coord_diff * self.coord_mlp(torch.cat([edge_feat, h[row]], dim=1))
        agg = unsorted_segment_mean(trans, row, num_segments=coord.size(0))
        vel = agg*self.coords_weight
        # print(f"coord_diff magnitude: {coord_diff.norm(dim=-1).mean():.6f}")
        # print(f"coord_mlp output: {self.coord_mlp(torch.cat([edge_feat, h[row]], dim=1)).abs().mean():.6f}")
        # print(f"per-edge trans magnitude: {trans.norm(dim=-1).mean():.6f}")
        # print(f"agg magnitude: {agg.norm(dim=-1).mean():.6f}")
        return vel
    
    def coord_2_scalar_distance(self, edge_index, coord, velocities=None):
        row, col = edge_index                       # two tensors of shape [E] containing the source and target node indices for each edge
        coord_diff = coord[row] - coord[col]        # calculate distances between all connected nodes
        coord_diff = min_image(coord_diff, **BOX)   # PBC minimum image convention
        scalar = torch.sum((coord_diff)**2, 1).unsqueeze(1)  

        if self.norm_diff:
            norm = torch.sqrt(scalar).clamp(min=1e-6)       # Calculate scalar distance between all connected nodes
            coord_diff = coord_diff/(norm)

        # return scalar, coord_diff * 100
        return scalar, coord_diff
      
    def forward(self, h, edge_index, coord, t_embed, edge_attr=None, global_attr=None):
        
        row, col = edge_index  
        scalar, coord_diff = self.coord_2_scalar_distance(edge_index, coord)
        
        edge_feat = self.edge_model(h[row], h[col], scalar, row, coord_diff, 
                                    global_attr[row] if global_attr is not None else None,
                                    num_nodes=h.size(0))
        vel = self.coord_model(coord, h, edge_index, coord_diff, edge_feat)
        if not self.last_mp_layer:
            h = self.node_model(h, edge_index, edge_feat, global_attr)

        return h, vel, edge_feat        