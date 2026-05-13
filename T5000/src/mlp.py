import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2, act_fn=nn.SiLU(), single_layer=False):
        super().__init__()
        self.single_layer = single_layer

        if single_layer:
            self.layer = nn.Linear(input_dim, output_dim)
        else:
            layers = [nn.Linear(input_dim, hidden_dim), act_fn]
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                layers.append(act_fn)
            layers.append(nn.Linear(hidden_dim, output_dim))
            self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        if self.single_layer:
            return self.layer(x)
        return self.mlp(x)