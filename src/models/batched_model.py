from pyro import sample, plate

from scipy.linalg import null_space, lstsq

from src.data.datasets import SiteDataset
import torch
import seaborn as sns
import pyro
from src.data.datasets import SiteDataset
from src.models.initial_model import InitialModel, TraceGuide
from pyro import distributions as dist

from torch.nn.utils.rnn import pad_sequence

site_data = SiteDataset("5a0546857ecc773753327266")
floor = site_data.floors[0]
height, width = floor.info["map_info"]["height"], floor.info["map_info"]["width"]
floor_uniform = dist.Uniform(
    low=torch.tensor([0.0, 0.0]), high=torch.tensor([height, width])
).to_event(1)


class BatchedModel(torch.nn.Module):
    def __init__(self, floor_data, K, n_max=12):

        super().__init__()

        trace_guides = []
        for trace in floor_data.traces[:n_max]:

            loc_bias = torch.tensor(trace.data["TYPE_WAYPOINT"].iloc[0].values)
            time_min_max = (
                trace.data["TYPE_WAYPOINT"].index[0].total_seconds(),
                trace.data["TYPE_WAYPOINT"].index[-1].total_seconds(),
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
        sigma_eps = 0.2  # std [m/100ms]
        sigma = 0.01  # std of noise measurement [m]

        # Wifi signal strength priors
        mu_omega_0 = -45.0
        sigma_omega_0 = 10.0  # signal stregth uncertainty
        sigma_omega = 0.1  # How presis is the measured signal stregth

        with pyro.plate("mini_batch", len(mini_batch_index)):

            x_0 = sample("x_0", relaxed_floor_dist)
            x = torch.zeros(
                x_0.shape[:-1] + (T_max,) + x_0.shape[-1:], # Batch dims, time, x/y
                dtype=mini_batch_position.dtype,
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

        location = torch.zeros((len(mini_batch_index), T_max, 2))
        log_scale = torch.zeros((len(mini_batch_index),))

        for i, (index, length) in enumerate(zip(mini_batch_index, mini_batch_length)):
            l, s = self.trace_guides[index](
                mini_batch_time[index, :length].unsqueeze(1)
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

    site_data = SiteDataset("5a0546857ecc773753327266")
    floor = site_data.floors[0]
    height, width = floor.info["map_info"]["height"], floor.info["map_info"]["width"]
    floor_uniform = dist.Uniform(
        low=torch.tensor([0.0, 0.0]), high=torch.tensor([height, width])
    ).to_event(1)

    batch_size = 12
    traces = [trace for trace in floor.traces[:batch_size]]

    mini_batch_index = torch.arange(batch_size)
    mini_batch_length = torch.tensor([len(t.matrices["time"]) for t in traces])

    mini_batch_time = mini_batch_time = pad_sequence(
        [torch.tensor(t.matrices["time"], dtype=torch.float32) for t in traces],
        batch_first=True,
    )
    mini_batch_position = pad_sequence(
        [torch.tensor(t.matrices["position"], dtype=torch.float32) for t in traces],
        batch_first=True,
    )
    mini_batch_position_mask = ~mini_batch_position.isnan().any(dim=-1)
    for i, length in enumerate(mini_batch_length):
        mini_batch_position_mask[i, length:] = False
    mini_batch_position[~mini_batch_position_mask] = 0

    bssids = set()
    for t in traces:
        bssids.update(set(t.data["TYPE_WIFI"]["bssid"].unique()))

    mini_batch_wifi_unpadded = []
    for t in traces:
        wifi = t._get_matrices(bssids=bssids)["wifi"]
        mini_batch_wifi_unpadded.append(torch.tensor(wifi, dtype=torch.float32))

    mini_batch_wifi = pad_sequence(mini_batch_wifi_unpadded, batch_first=True)
    mini_batch_wifi_mask = ~mini_batch_wifi.isnan()
    for i, length in enumerate(mini_batch_length):
        mini_batch_wifi_mask[i, length:, :] = False
    mini_batch_wifi[~mini_batch_wifi_mask] = 0

    _, T, K = mini_batch_wifi.shape

    model = BatchedModel(floor, K)

    # model.model(
    #     mini_batch_index=mini_batch_index,
    #     mini_batch_length=mini_batch_length,
    #     mini_batch_time=mini_batch_time,
    #     mini_batch_position=mini_batch_position,
    #     mini_batch_position_mask=mini_batch_position_mask,
    #     mini_batch_wifi=mini_batch_wifi,
    #     mini_batch_wifi_mask=mini_batch_wifi_mask,
    # )

    # model.guide(
    #     mini_batch_index=mini_batch_index,
    #     mini_batch_length=mini_batch_length,
    #     mini_batch_time=mini_batch_time,
    #     mini_batch_position=mini_batch_position,
    #     mini_batch_position_mask=mini_batch_position_mask,
    #     mini_batch_wifi=mini_batch_wifi,
    #     mini_batch_wifi_mask=mini_batch_wifi_mask,
    # )

    from pyro.infer import MCMC, NUTS, HMC, SVI, Trace_ELBO
    from pyro.optim import Adam, ClippedAdam

    # Reset parameter values
    pyro.clear_param_store()

    # Define the number of optimization steps
    n_steps = 1000

    # Setup the optimizer
    adam_params = {"lr": 0.01}
    optimizer = Adam(adam_params)

    # # Setup the inference algorithm
    elbo = Trace_ELBO(num_particles=10, vectorize_particles=True)
    svi = SVI(model.model, model.guide, optimizer, loss=elbo)

    # Do gradient steps
    for step in range(n_steps):
        elbo = svi.step(
            mini_batch_index=mini_batch_index,
            mini_batch_length=mini_batch_length,
            mini_batch_time=mini_batch_time,
            mini_batch_position=mini_batch_position,
            mini_batch_position_mask=mini_batch_position_mask,
            mini_batch_wifi=mini_batch_wifi,
            mini_batch_wifi_mask=mini_batch_wifi_mask,
        )

        print("[%d] ELBO: %.1f" % (step, elbo))
