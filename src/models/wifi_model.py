from src.models.model_trainer import ModelTrainer
from src.data.datasets import SiteDataset
from src.utils import object_to_markdown
from pyro import sample, plate

from zmq import device

import torch
import pyro
from src.models.trace_guide import TraceGuide
from pyro import distributions as dist

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class WifiModel(torch.nn.Module):
    def __init__(self, floor_data):

        super().__init__()

        height, width = (
            floor_data.info["map_info"]["height"],
            floor_data.info["map_info"]["width"],
        )
        self.floor_uniform = dist.Uniform(
            low=torch.tensor([0.0, 0.0], dtype=torch.float64),
            high=torch.tensor([height, width], dtype=torch.float64),
        ).to_event(1)

        trace_guides = []
        self.K = floor_data.K

        for trace in floor_data.traces:

            time, position, _, _ = trace[0]

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

        self.register_parameter("mu_q", torch.nn.Parameter(torch.full((self.K,), -45.0)))
        self.register_parameter(
            "log_sigma_q", torch.nn.Parameter(torch.full((self.K,), 0.0))
        )
        self.register_parameter(
            "wifi_location_q",
            torch.nn.Parameter(torch.tile(self.floor_uniform.mean, (self.K, 1))),
        )
        self.register_parameter(
            "wifi_location_log_sigma_q", torch.nn.Parameter(torch.full((self.K, 2), 0.0))
        )

        self.to(dtype=torch.float64)

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
            self.floor_uniform.mean, self.floor_uniform.stddev
        ).to_event(1)

        # Normal walking tempo is 1,4 m/s
        sigma_eps = torch.tensor(0.25, device=device)  # std [m/100ms]
        sigma = torch.tensor(0.1, device=device)  # std of noise measurement [m]

        # Wifi signal strength priors
        mu_omega_0 = torch.tensor(-45.0, device=device)
        sigma_omega_0 = torch.tensor(
            10.0, device=device
        )  # grundsignal styrke spredning.
        sigma_omega = torch.tensor(
            5.0, device=device
        )  # How accurate is the measured signal stregth

        with pyro.plate("mini_batch", len(mini_batch_index)):

            x_0 = sample("x_0", relaxed_floor_dist)
            x = torch.zeros(
                x_0.shape[:-1] + (T_max,) + x_0.shape[-1:],  # Batch dims, time, x/y
                dtype=mini_batch_position.dtype,
                device=device,
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

        return x, wifi_location

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

        location = torch.zeros((len(mini_batch_index), T_max, 2), device=device)
        scale = torch.zeros((len(mini_batch_index),), device=device)

        for i, (index, length) in enumerate(zip(mini_batch_index, mini_batch_length)):
            l, s = self.trace_guides[index](mini_batch_time[i, :length].unsqueeze(1))
            location[i, :length, :] = l
            scale[i] = s

        with pyro.plate("mini_batch", len(mini_batch_index)):

            for t in pyro.markov(range(0, T_max)):
                sample(
                    f"x_{t}",
                    dist.Normal(location[:, t, :], scale.view(-1, 1))
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
        return location, scale


def train_model():

    torch.manual_seed(123456789)

    # Load data
    site_data = SiteDataset(
        "5a0546857ecc773753327266", wifi_threshold=200, sampling_interval=100
    )
    floor = site_data.floors[0]

    # Setup model
    model = WifiModel(floor)

    # Setup the optimizer
    adam_params = {"lr": 1e-2}  # ., "betas":(0.95, 0.999)}
    optimizer = torch.optim.Adam(model.parameters(), **adam_params)

    # Setup model training
    n_epochs = 1000
    batch_size = 16
    mt = ModelTrainer(
        model_label="wifi_model",
        model=model,
        optimizer=optimizer,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # Train the model
    mt.train(floor)


if __name__ == "__main__":

    train_model()
