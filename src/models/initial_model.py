from pyro import sample, plate


from scipy.linalg import null_space, lstsq
from torch.nn.modules import Linear, Module, Softplus, ReLU, Sigmoid
from torch.nn.modules.container import Sequential

from src.data.datasets import SiteDataset
import torch
from torch.distributions import constraints
import seaborn as sns
import pyro
from src.data.datasets import SiteDataset
from pyro import distributions as dist

site_data = SiteDataset("5a0546857ecc773753327266")
floor = site_data.floors[0]
height, width = floor.info["map_info"]["height"], floor.info["map_info"]["width"]
floor_uniform = dist.Uniform(
    low=torch.tensor([0.0, 0.0]), high=torch.tensor([height, width])
).to_event(1)


class TraceGuide(Module):
    def __init__(self, layer_sizes=None, loc_bias=None, time_min_max=None):

        super().__init__()

        if layer_sizes is None:
            layer_sizes = [30]

        layers_x = []
        layers_y = []

        in_size = 1
        for out_size in layer_sizes:
            layers_x.append(Linear(in_size, out_size))
            layers_y.append(Linear(in_size, out_size))
            layers_x.append(Softplus())
            layers_y.append(Softplus())
            in_size = out_size

        layers_x.append(Linear(in_size, 1))
        layers_y.append(Linear(in_size, 1))

        if loc_bias is not None:
            layers_x[-1].bias = torch.nn.Parameter(loc_bias[0])
            layers_y[-1].bias = torch.nn.Parameter(loc_bias[1])

        if time_min_max is not None:
            linspace = torch.linspace(time_min_max[0], time_min_max[1], layer_sizes[0])
            layers_x[0].bias = torch.nn.Parameter(-linspace)
            layers_y[0].bias = torch.nn.Parameter(-linspace)
            # layers_x[0].bias.requires_grad = False
            # layers_y[0].bias.requires_grad = False

        self._forward_x = Sequential(*layers_x)
        self._forward_y = Sequential(*layers_y)

        self.register_parameter("log_scale", torch.nn.Parameter(torch.tensor(0.)))

    def forward(self, x):

        location = torch.cat([self._forward_x(x), self._forward_y(x)], dim=1)
        scale = self.log_scale.exp()

        return location, scale


# class Spline(torch.nn.Module):
#     def __init__(self, knots, values, order=4):

#         super().__init__()

#         self.knots = knots
#         self.values = values
#         self.order = order

#         X = self.basis(knots)

#         coeffs, _, _, _ = lstsq(X, values)
#         n_space = null_space(X)

#         self.register_buffer("coeffs", torch.tensor(coeffs))
#         self.register_buffer("null_space", torch.tensor(n_space))

#         self.register_parameter(
#             "free_parameters",
#             torch.nn.Parameter(torch.zeros((self.null_space.shape[1]))),
#         )

#         self.to(dtype=torch.float)

#     def basis(self, x):

#         return torch.stack(
#             [x.pow(i) for i in range(self.order)]
#             + [torch.relu(x - knot).pow(self.order - 1) for knot in self.knots[1:-1]]
#         ).T

#     def forward(self, x):

#         return self.basis(x) @ (self.coeffs + self.null_space @ self.free_parameters)


class InitialModel(torch.nn.Module):
    def __init__(self, trace_data):

        super().__init__()

        matrices = trace_data.matrices

        self.position = torch.tensor(matrices["position"], dtype=torch.float32)
        self.wifi = torch.tensor(matrices["wifi"], dtype=torch.float32)
        self.time = torch.tensor(matrices["time"], dtype=torch.float32)

        self.T = self.wifi.shape[0]
        self.K = self.wifi.shape[1]

        self.position_is_observed = (
            (~torch.isnan(self.position[:, 0])).nonzero().flatten()
        )
        self.position_is_missing = (
            (torch.isnan(self.position[:, 0])).nonzero().flatten()
        )

        self.wifi_is_observed = (
            ~torch.isnan(self.wifi)
        )

        self.trace_guide = TraceGuide(
            loc_bias=self.position[0],
            time_min_max = (self.time[0], self.time[-1]),
            )
        self.wifi_location = None
        self.loc = None

    def model(self):

        pyro.module("initial_model", self)

        # Normal walking tempo is 1,4 m/s
        sigma_eps = 0.2 #std [m/100ms]  
        sigma = 0.01 #std of noise measurement [m] 

        #Wifi signal strength priors
        mu_omega_0 = -45
        sigma_omega_0 = 10 # signal stregth uncertainty
        sigma_omega = 0.1 #How presis is the measured signal stregth

        x = torch.zeros((self.T, 2))
        x[0, :] = sample("x_0", floor_uniform)

        for t in pyro.markov(range(1, self.T)):
            x[t, :] = sample(f"x_{t}", dist.Normal(x[t - 1], sigma_eps).to_event(1))

        x_hat = sample(
            "x_hat",
            dist.Normal(x[self.position_is_observed], sigma).to_event(2),
            obs=self.position[self.position_is_observed],
        )

        with plate("wifis", self.K) as ind:
            omega_0 = sample("omega_0", dist.Normal(mu_omega_0, sigma_omega_0).to_event(0))
            self.wifi_location = sample("wifi_location", floor_uniform)
        
        distance = torch.cdist(x,self.wifi_location, p=2) 
        signal_strength= omega_0 - 2*torch.log(distance)

        omega = sample(
            name="omega",
            fn=dist.Normal(loc=signal_strength[self.wifi_is_observed], scale=sigma_omega).to_event(1),
            obs=self.wifi[self.wifi_is_observed]
        )

    def guide(self, annealing_factor=1.0):

        pyro.module("initial_model", self)

        location, scale = self.trace_guide(self.time.view(-1, 1))

        with pyro.poutine.scale(None, annealing_factor):
            for t in pyro.markov(range(0, self.T)):
                sample(
                    f"x_{t}",
                    dist.Normal(location[t, :], scale).to_event(1),
                )

        #mu_q = pyro.param("mu_q", lambda: torch.full((self.K,1), 50, dtype = torch.float32),event_dim=1,
        #                 constraint=constraints.positive)
        #sigma_q = pyro.param("beta_q", lambda: torch.full((self.K,1), 0.1,dtype = torch.float32),event_dim=1,
        #                constraint=constraints.positive)

        mu_q = pyro.param("mu_q", lambda: torch.full((1,1), 50, dtype = torch.float32),event_dim=1,
                         constraint=constraints.positive)
        sigma_q = pyro.param("beta_q", lambda: torch.full((1,1), 0.1,dtype = torch.float32),event_dim=1,
                        constraint=constraints.positive)


        with plate("wifis", self.K):
            omega_0 = sample("omega_0", dist.Normal(-mu_q, sigma_q).to_event(0))
            self.loc = pyro.param("loc",lambda: torch.full((self.K, 2), 50, dtype = torch.float32),event_dim=1)
            #loc2 = pyro.param("loc2",event_dim=1)
            #cov = pyro.param("cov",lambda: torch.full((self.K,2), 0.01),
            #             constraint=constraints.positive)

            self.wifi_location = sample("wifi_location", dist.MultivariateNormal(self.loc, torch.tensor([[0.5,0],[0,0.5]])))

if __name__ == "__main__":

    trace = floor.traces[18]

    matrices = trace.matrices
    wifi = torch.tensor(matrices["wifi"])
    position = torch.tensor(matrices["position"])
    time = torch.tensor(matrices["time"])

    height, width = floor.info["map_info"]["height"], floor.info["map_info"]["width"]

    model = InitialModel(trace)

    from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
    from pyro.optim import Adam, ClippedAdam

    # Reset parameter values
    pyro.clear_param_store()

    # Define the number of optimization steps
    n_steps = 12000

    # Setup the optimizer
    adam_params = {"lr": 0.01}
    optimizer = ClippedAdam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=3)
    svi = SVI(model.model, model.guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step()

        print("[%d] ELBO: %.1f" % (step, elbo))
