
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
            layer_sizes = [10]

        layers_x = []
        layers_y = []

        in_size = 1
        for out_size in layer_sizes:
            layers_x.append(Linear(in_size, out_size))
            layers_y.append(Linear(in_size, out_size))
            layers_x.append(Softplus(beta=1.))
            layers_y.append(Softplus(beta=1.))
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
            layers_x[0].weight = torch.nn.Parameter(torch.ones_like(layers_x[0].weight))
            layers_y[0].weight = torch.nn.Parameter(torch.ones_like(layers_x[0].weight))
            layers_x[0].weight.requires_grad = False
            layers_y[0].weight.requires_grad = False

        self._forward_x = Sequential(*layers_x)
        self._forward_y = Sequential(*layers_y)

        self.register_parameter("log_scale", torch.nn.Parameter(torch.tensor(0.)))

    def forward(self, x):

        location = torch.cat([self._forward_x(x), self._forward_y(x)], dim=1)
        scale = self.log_scale.exp()

        return location, scale

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

        self.wifi_mask = ~torch.isnan(self.wifi)
        self.wifi_is_observed = self.wifi_mask.any(axis=1)

        self.wifi[~self.wifi_mask] = 0.

        self.trace_guide = TraceGuide(
            loc_bias=self.position[0],
            time_min_max = (self.time[0], self.time[-1]),
            )

        self.register_parameter("mu_q", torch.nn.Parameter(torch.full((self.K,), -45.)))
        self.register_parameter("log_sigma_q", torch.nn.Parameter(torch.full((self.K,), 0.)))
        self.register_parameter("wifi_location_q", torch.nn.Parameter(torch.tile(floor_uniform.mean, (self.K, 1))))
        self.register_parameter("wifi_location_log_sigma_q", torch.nn.Parameter(torch.full((self.K, 2), 0.)))


    def model(self, annealing_factor=1.):

        pyro.module("initial_model", self)

        relaxed_floor_dist = dist.Normal(
            floor_uniform.mean,
            floor_uniform.stddev
        ).to_event(1)

        # Normal walking tempo is 1,4 m/s
        sigma_eps = 0.2 #std [m/100ms]  
        sigma = 0.01 #std of noise measurement [m] 

        #Wifi signal strength priors
        mu_omega_0 = -45
        sigma_omega_0 = 10 # signal stregth uncertainty
        sigma_omega = 0.1 #How presis is the measured signal stregth

        x = torch.zeros((self.T, 2))
        x[0, :] = sample("x_0", relaxed_floor_dist)

        for t in pyro.markov(range(1, self.T)):
            x[t, :] = sample(f"x_{t}", dist.Normal(x[t - 1], sigma_eps).to_event(1))

        with pyro.plate("x_observed", len(self.position_is_observed)):
            sample(
                "x_hat",
                dist.Normal(x[self.position_is_observed], sigma).to_event(1),
                obs=self.position[self.position_is_observed],
            )

        with plate("wifis", self.K):
            omega_0 = sample("omega_0", dist.Normal(mu_omega_0, sigma_omega_0))
            wifi_location = sample("wifi_location", relaxed_floor_dist)
            distance = torch.cdist(x[self.wifi_is_observed], wifi_location)
            with plate("wifi_is_observed", sum(self.wifi_is_observed)):
                signal_strength= omega_0 - 2*torch.log(distance)
                omega = sample(
                    "omega",
                    dist.Normal(signal_strength, sigma_omega).mask(self.wifi_mask[self.wifi_is_observed, :]),
                    obs=self.wifi[self.wifi_is_observed, :]
                )


        # with plate("wifis", self.K) as ind:
        #     omega_0 = sample("omega_0", dist.Normal(mu_omega_0, sigma_omega_0).to_event(0))
        #     self.wifi_location = sample("wifi_location", floor_uniform)
        
        # distance = torch.cdist(x,self.wifi_location, p=2) 
        # signal_strength= omega_0 - 2*torch.log(distance)

        # omega = sample(
        #     name="omega",
        #     fn=dist.Normal(loc=signal_strength[self.wifi_is_observed], scale=sigma_omega).to_event(1),
        #     obs=self.wifi[self.wifi_is_observed]
        # )

    def guide(self, annealing_factor=1.0):

        pyro.module("initial_model", self)

        location, scale = self.trace_guide(self.time.view(-1, 1))

        with pyro.poutine.scale(None, annealing_factor):
            for t in pyro.markov(range(0, self.T)):
                sample(
                    f"x_{t}",
                    dist.Normal(location[t, :], scale).to_event(1),
                )

        with plate("wifis", self.K):
            omega_0 = sample("omega_0", dist.Normal(self.mu_q, self.log_sigma_q.exp()))
            wifi_location = sample(
                "wifi_location", 
                dist.Normal(self.wifi_location_q, self.wifi_location_log_sigma_q.exp()).to_event(1)
            )

        # with plate("wifis", self.K):
        #     omega_0 = sample("omega_0", dist.Normal(-mu_q, sigma_q))
        #     self.loc = pyro.param("loc",lambda: torch.full((self.K, 2), 50, dtype = torch.float32),event_dim=1)
        #     #loc2 = pyro.param("loc2",event_dim=1)
        #     #cov = pyro.param("cov",lambda: torch.full((self.K,2), 0.01),
        #     #             constraint=constraints.positive)

        #     self.wifi_location = sample("wifi_location", dist.MultivariateNormal(self.loc, torch.tensor([[0.5,0],[0,0.5]])))

if __name__ == "__main__":

    trace = floor.traces[18]

    model = InitialModel(trace)
    model.model()

    from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO, JitTrace_ELBO
    from pyro.optim import Adam, ClippedAdam


    # Reset parameter values
    pyro.clear_param_store()

    model.guide()

    # Define the number of optimization steps
    n_steps = 300

    # Setup the optimizer
    adam_params = {"lr": 0.01}
    optimizer = ClippedAdam(adam_params)

    # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=4)
    svi = SVI(model.model, model.guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step()

        print("[%d] ELBO: %.1f" % (step, elbo))