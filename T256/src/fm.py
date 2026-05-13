import torch
import torch.nn as nn

from pbc_config import BOX, min_image, wrap
from prior import scaled_log10_gaussian_mass_prior, uniform_prior, wrapped_gaussian_prior
from utils.data import ot_alignment_variable
from utils.embedding import SinusoidalTimeEmbedding
from utils.graph import compute_local_density, radius_graph_pbc_batch
from utils.scale import scale_thetas

class FlowMatching(nn.Module):
    def __init__(self,
                 sigma_0,
                 sigma_sched,
                 t_embed_dim,
                 version,
                 vnet,
                 batch_size,
                 prior,
                 k,
                 r,
                 optimal_transport,
                 lambda_mass=1.0,
                 t_embed = "linear",
                 dim=3
                 ):
        super().__init__()

        print(f'''
        Initializing Flow Matching model with:
        - sigma_0: {sigma_0}
        - sigma_sched: {sigma_sched}
        - t_embed_dim: {t_embed_dim}
        - version: {version}
        - batch_size: {batch_size}
        - prior: {prior}
        - k: {k}
        - r: {r}
        - optimal_transport: {optimal_transport}
        - lambda_mass: {lambda_mass}
        - t_embed: {t_embed}
        - dim: {dim}
              ''')
        
        self.sigma_0 = sigma_0
        self.sigma_sched = sigma_sched
        self.t_embed_dim = t_embed_dim
        self.t_embed = t_embed
        self.version = version
        self.vnet = vnet
        self.batch_size = batch_size
        self.prior = prior
        self.k = k
        self.r = r
        self.optimal_transport = optimal_transport
        self.lambda_mass = lambda_mass
        self.dim = dim
        
        self.time_mlp = nn.Linear(1, self.t_embed_dim)
        self.sinusoidal_time_embedding = SinusoidalTimeEmbedding(self.t_embed_dim)
        
    def time_embedding(self, t):
        '''
        Time embedding function.
        
        :param t: tensor for FM time in U[0, 1], shape (batch_size, 1)
        '''
        if self.t_embed == "linear":
            self.time_mlp.to(t.device)
            embedding = self.time_mlp(t)
            return embedding
        elif self.t_embed == "sinusoidal":
            self.sinusoidal_time_embedding.to(t.device)
            return self.sinusoidal_time_embedding(t)
        else:
            raise ValueError(f"Unknown time embedding: {self.t_embed}")

    def sigma_const(self, t):
        '''
        Function which returns value of sigma_t at time t.
        
        :param t: tensor for FM time in U[0, 1], shape (batch_size, 1, 1)
        '''
        if self.sigma_sched:
            return self.sigma_0 * (1 - t)
        else:
            return self.sigma_0

    def sample_prior(self, x1):
        '''
        Sample from prior distribution.
        
        :param x1: tensor for data points, shape (batch_size, n_halos, dim)
        '''
        if self.prior == "uniform":
            prior = uniform_prior(x1)
        elif self.prior == "gaussian":
            prior = wrapped_gaussian_prior(x1)
        else:
            raise ValueError(f"Unknown prior: {self.prior}")
        return prior
    
    def sample_xt(self, x0, x1, t, batch):
        '''
        Sample x_t at time t from probability path p_t.
        
        :param x0: tensor for prior points, shape (batch_size, n_halos, dim)
        :param x1: tensor for data points, shape (batch_size, n_halos, dim)
        :param t: tensor for FM time in U[0, 1], shape (batch_size, 1, 1)
        '''
        t = t[batch]

        if self.version == "icfm":
            delta = min_image(x1 - x0, **BOX).to(x1.device)
            mu_t = x0+t*delta
            sigma_t = self.sigma_const(t)
        elif self.version == "fm":
            mu_t = wrap(t * x1, **BOX)
            sigma_t = 1 - (1 - self.sigma_const(t))*t
        else:
            raise ValueError(f"Unknown version: {self.version}")

        # torch.manual_seed(99)
        x = mu_t + sigma_t * torch.randn_like(x1)

        return wrap(x, **BOX)
    
    def conditional_vector_field(self, x, x0, x1, t, batch):
        '''
        Compute the conditional vector field u_t(x) = E[u | x_t = x].
        
        :param x: tensor for current points, shape (batch_size, n_halos, dim)
        :param x0: tensor for prior points, shape (batch_size, n_halos, dim)
        :param x1: tensor for data points, shape (batch_size, n_halos, dim)
        :param t: tensor for FM time in U[0, 1], shape (batch_size, 1, 1)
        '''
        t = t[batch]

        if self.version == "icfm":
            u_t = x1 - x0
            u_t_pbc = min_image(u_t, **BOX)
        elif self.version == "fm":
            d = min_image(x1 - x, **BOX)
            t = torch.clamp(t, min=1e-5)
            x1_aligned = x + d
            nom = x1_aligned - (1 - self.sigma_const(t))*x
            denom = 1 - (1 - self.sigma_const(t))*t
            u_t_pbc = nom/denom
        else:
            raise ValueError(f"Unknown version: {self.version}")
        
        return u_t_pbc

    def forward(self, samples, reduction="mean", x_t=None):
        '''
        Forward pass of Flow Matching model.
        
        :param x1: tensor for data points, shape (batch_size, n_halos, dim)
        :param lambda_reg: regularization parameter
        :param reduction: reduction method for loss computation, "mean" or "sum"
        '''

        # Construct prior
        x0 = self.sample_prior(samples.x)

        # Unpack training data samples and construct graph
        x1 = samples.x
        batch = samples.batch
        m1 = samples.mass.view(-1, 1)
        vel = samples.vel
        theta = scale_thetas(samples.theta.view(-1, 5))
        node_counts_per_graph = torch.bincount(batch)
        assert x1.shape[0] == torch.sum(node_counts_per_graph), "Number of nodes in x is different than number of halos per graph."

        if self.optimal_transport:
            x0 = ot_alignment_variable(x0, x1, batch, self.batch_size)       

        # Sample time and embed
        t = torch.rand(size=(self.batch_size, 1), device=x1.device)
        t_embedded = self.time_embedding(t)
    
        # create x_t 
        x_t = self.sample_xt(x0, x1, t, batch) if x_t is None else x_t  # x_t can be passed directly for testing/ ablation purposes, otherwise sample as usual during training

        # build edge index on x_t 
        edge_index = radius_graph_pbc_batch(x_t, self.r, batch, x_t.device)

        # Construct conditional vector field for x_t
        target_vel = self.conditional_vector_field(x_t, x0, x1, t, batch)

        # Construct m_t and target_vel_m
        m0 = scaled_log10_gaussian_mass_prior(m1)
        t_nodes = t[batch]
        m_t = t_nodes * m1 + (1 - t_nodes) * m0
        target_vel_m = m1 - m0

        # concatenate m_t with local density to form node features for vnet
        density = compute_local_density(edge_index, x_t)
        h_in = torch.cat([m_t, density], dim=1)  

        # Predict velocity using vnet
        pred_vel, pred_vel_m = self.vnet(h_in, x_t, t_embedded, batch, edge_index, theta=theta)   

        if reduction == "mean":
            pos_loss = nn.MSELoss(reduction="mean")(pred_vel, target_vel)
            mass_loss = nn.MSELoss(reduction="mean")(pred_vel_m, target_vel_m)
        elif reduction == "sum":
            pos_loss = nn.MSELoss(reduction="sum")(pred_vel, target_vel)
            mass_loss = nn.MSELoss(reduction="sum")(pred_vel_m, target_vel_m)
        else:
            raise ValueError(f"Unknown reduction method: {reduction}")
        
        total_loss = pos_loss + self.lambda_mass * mass_loss
        
        return total_loss, pos_loss.item(), mass_loss.item()