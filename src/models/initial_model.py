from src.models.model_trainer import ModelTrainer
from src.data.datasets import FloorDataset, SiteDataset
from src.models.prior_params import prior_params as prior_params_default
from src.utils import object_to_markdown
from pyro import poutine, sample, plate

import torch
import pyro
from src.models.trace_guide import TraceGuide
from pyro import distributions as dist

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class InitialModel(torch.nn.Module):
    def __init__(self, floor_data, prior_params=None):

        super().__init__()

        self.prior_params = prior_params_default
        if prior_params is not None:
            self.prior_params.update(prior_params)

        height, width = (
            floor_data.info["map_info"]["height"],
            floor_data.info["map_info"]["width"],
        )
        print(height, width)
        self.floor_uniform = dist.Uniform(
            low=torch.tensor([0.0, 0.0], dtype=torch.float64, device=device),
            high=torch.tensor([width, height], dtype=torch.float64, device=device),
        ).to_event(1)

        trace_guides = []

        for i, trace in enumerate(floor_data.traces):

            time, position, _, _ = trace[0]

            n_basis_functions = int(min(max(time[-1] // 4, 3), 40))

            if floor_data.validation_mask[i] or floor_data.test_mask[i]:
                loc_bias = self.floor_uniform.mean

                time_min_max = (
                    time[0],
                    time[-1],
                )
            else:
                pos_is_obs = ~position.isnan().any(-1)
                loc_bias = position[pos_is_obs].mean(axis=0)

                time_min_max = (
                    time[pos_is_obs][0],
                    time[pos_is_obs][-1],
                )

            trace_guides.append(
                TraceGuide(
                    n=n_basis_functions, loc_bias=loc_bias, time_min_max=time_min_max
                )
            )

        self.trace_guides = torch.nn.ModuleList(trace_guides)
        self.to(dtype=torch.float64, device=device)

    def model(
        self,
        mini_batch_index,
        mini_batch_length,
        mini_batch_time,
        mini_batch_position,
        mini_batch_position_mask,
        annealing_factor=1.0,
    ):
        
        pyro.module("initial_model", self)

        T_max = mini_batch_time.shape[-1]

        relaxed_floor_dist = dist.Normal(
            self.floor_uniform.mean, self.floor_uniform.stddev
        ).to_event(1)

        sigma_eps = torch.tensor(self.prior_params["sigma_eps"], device=device)
        sigma = torch.tensor(self.prior_params["sigma"], device=device)

        with poutine.scale(None, annealing_factor):

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

        return x

    def guide(
        self,
        mini_batch_index,
        mini_batch_length,
        mini_batch_time,
        mini_batch_position,
        mini_batch_position_mask,
        annealing_factor=1.0,
    ):

        pyro.module("initial_model", self)

        T_max = mini_batch_time.shape[-1]

        location = torch.zeros((len(mini_batch_index), T_max, 2), device=device)
        scale = torch.zeros((len(mini_batch_index),), device=device)

        for i, (index, length) in enumerate(zip(mini_batch_index, mini_batch_length)):
            l, s = self.trace_guides[index](mini_batch_time[i, :length].unsqueeze(1))
            location[i, :length, :] = l
            scale[i] = s

        with poutine.scale(None, annealing_factor):

            with pyro.plate("mini_batch", len(mini_batch_index)):

                for t in pyro.markov(range(0, T_max)):
                    sample(
                        f"x_{t}",
                        dist.Normal(location[:, t, :], scale.view(-1, 1))
                        .to_event(1)
                        .mask(t < mini_batch_length),
                    )

        return location, scale
