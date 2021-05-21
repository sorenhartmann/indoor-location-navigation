from pyro import sample, plate

from scipy.linalg import null_space, lstsq
from zmq import device
from pathlib import Path

from src.data.datasets import SiteDataset,get_loader
import torch
import seaborn as sns
import pyro
from src.data.datasets import SiteDataset
from src.models.initial_model import InitialModel
from src.models.trace_guide import TraceGuide
from pyro import distributions as dist

from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
from pyro.optim import Adam, ClippedAdam

from torch.nn.utils.rnn import pad_sequence

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

project_dir = Path(__file__).resolve().parents[2]
checkpoint_dir = project_dir / "models" / "checkpoints"




class BatchedModel(torch.nn.Module):
    def __init__(self, floor_data):

        super().__init__()

        trace_guides = []
        K = floor_data.K

        for trace in floor_data.traces:

            time, position, _ = trace[0]

            pos_is_obs = ~position.isnan().any(-1)
            loc_bias = position[pos_is_obs].mean(axis=0)

            time_min_max = (
                time[pos_is_obs][0],
                time[pos_is_obs][-1],
            )

            trace_guides.append(
                TraceGuide(loc_bias=loc_bias, time_min_max=time_min_max)
            )

        self.trace_guides = torch.nn.ModuleList(trace_guides)
        self.trace_guides.to(dtype=torch.float32)

        self.register_parameter("mu_q", torch.nn.Parameter(torch.full((K,), -45.0)))
        self.register_parameter(
            "log_sigma_q", torch.nn.Parameter(torch.full((K,), 0.0))
        )
        self.register_parameter(
            "wifi_location_q",
            torch.nn.Parameter(torch.tile(floor_uniform.mean, (K, 1))),
        )
        self.register_parameter(
            "wifi_location_log_sigma_q", torch.nn.Parameter(torch.full((K, 2), 0.0))
        )

    def model(
        self,
        mini_batch_index,
        mini_batch_length,
        mini_batch_time,
        mini_batch_position,
        mini_batch_position_mask,
        mini_batch_wifi,
        mini_batch_wifi_mask,
        annealing_factor=1.0,
    ):

        pyro.module("initial_model", self)

        T_max = mini_batch_time.shape[-1]
        K = mini_batch_wifi.shape[-1]

        relaxed_floor_dist = dist.Normal(
            floor_uniform.mean, floor_uniform.stddev
        ).to_event(1)

        # Normal walking tempo is 1,4 m/s
        sigma_eps = torch.tensor(0.2, device=device) # std [m/100ms]
        sigma = torch.tensor(3, device=device) # std of noise measurement [m]
        
        # Wifi signal strength priors
        mu_omega_0 = torch.tensor(-45.0, device=device)
        sigma_omega_0 = torch.tensor(10.0, device=device) # grundsignal styrke spredning. 
        sigma_omega = torch.tensor(10., device=device) # How presis is the measured signal stregth

        with pyro.plate("mini_batch", len(mini_batch_index)):

            x_0 = sample("x_0", relaxed_floor_dist)
            x = torch.zeros(
                x_0.shape[:-1] + (T_max,) + x_0.shape[-1:],  # Batch dims, time, x/y
                dtype=mini_batch_position.dtype,
                device=device
            )
            x[..., 0, :] = x_0

            for t in pyro.markov(range(1, T_max)):
                x[..., t, :] = sample(
                    f"x_{t}",
                    dist.Normal(x[..., t - 1, :], sigma_eps)
                    .to_event(1)
                    .mask(t < mini_batch_length),
                )

        with pyro.plate("x_observed", mini_batch_position_mask.sum()):
            sample(
                "x_hat",
                dist.Normal(x[..., mini_batch_position_mask, :], sigma).to_event(1),
                obs=mini_batch_position[mini_batch_position_mask],
            )

        any_wifi_is_observed = mini_batch_wifi_mask.any(dim=-1)

        with plate("wifis", K):
            omega_0 = sample("omega_0", dist.Normal(mu_omega_0, sigma_omega_0))
            wifi_location = sample("wifi_location", relaxed_floor_dist)
            distance = torch.cdist(x[..., any_wifi_is_observed, :], wifi_location)
            with plate("wifi_is_observed", any_wifi_is_observed.sum()):
                signal_strength = omega_0 - 2 * torch.log(distance)
                omega = sample(
                    "omega",
                    dist.Normal(signal_strength, sigma_omega).mask(
                        mini_batch_wifi_mask[any_wifi_is_observed]
                    ),
                    obs=mini_batch_wifi[any_wifi_is_observed],
                )

    def guide(
        self,
        mini_batch_index,
        mini_batch_length,
        mini_batch_time,
        mini_batch_position,
        mini_batch_position_mask,
        mini_batch_wifi,
        mini_batch_wifi_mask,
        annealing_factor=1.0,
    ):

        pyro.module("initial_model", self)

        T_max = mini_batch_time.shape[-1]
        K = mini_batch_wifi.shape[-1]

        location = torch.zeros((len(mini_batch_index), T_max, 2), device = device)
        log_scale = torch.zeros((len(mini_batch_index),), device = device)


        for i, (index, length) in enumerate(zip(mini_batch_index, mini_batch_length)):
            l, s = self.trace_guides[index](
                mini_batch_time[i, :length].unsqueeze(1)
            )
            location[i, :length, :] = l
            log_scale[i] = s

        with pyro.plate("mini_batch", len(mini_batch_index)):

            for t in pyro.markov(range(0, T_max)):
                sample(
                    f"x_{t}",
                    dist.Normal(location[:, t, :], log_scale.exp().view(-1, 1))
                    .to_event(1)
                    .mask(t < mini_batch_length),
                )

        with plate("wifis", K):
            sample("omega_0", dist.Normal(self.mu_q, self.log_sigma_q.exp()))
            sample(
                "wifi_location",
                dist.Normal(
                    self.wifi_location_q, self.wifi_location_log_sigma_q.exp()
                ).to_event(1),
            )

if __name__ == "__main__":
    site_data = SiteDataset("5a0546857ecc773753327266", wifi_threshold=200, sampling_interval = 200)
    floor = site_data.floors[0]
    height, width = floor.info["map_info"]["height"], floor.info["map_info"]["width"]
    floor_uniform = dist.Uniform(
        low=torch.tensor([0.0, 0.0],device=device), high=torch.tensor([height, width],device=device)
    ).to_event(1) 



    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    dataloader = get_loader(floor, batch_size = 20)
    model = BatchedModel(floor)
    
    # Reset parameter values
    pyro.clear_param_store()

    # Define the number of Epochs
    n_epochs = 2000

    # Setup the optimizer
    adam_params = {"lr": 0.0001}
    optimizer = ClippedAdam(adam_params)

    # # Setup the inference algorithm
    svi = SVI(model.model, model.guide, optimizer, loss=Trace_ELBO(num_particles=10, vectorize_particles=True))
    #if torch.cuda.is_available():
    #    model.to(device = torch.device("cuda"))
    mini_batch = floor[torch.arange(0,16)]
    #with torch.autograd.detect_anomaly():
    for epoch in range(n_epochs):
        #elbo = 0
        #for mini_batch in dataloader:
        #    print(mini_batch[0])
        #    elbo = elbo +  svi.step(*mini_batch)
        elbo = svi.step(*mini_batch)
        if epoch> 0 and epoch%100 == 0:
            torch.save(model.state_dict(), checkpoint_dir/f'BatchedModel_single_lr{adam_params["lr"]}_epochs{epoch}.pt')
        print("epoch[%d] ELBO: %.1f" % (epoch, elbo))
