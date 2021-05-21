from pyro import sample, plate

from scipy.linalg import null_space, lstsq
from zmq import device
from pathlib import Path

from src.data.datasets import SiteDataset, get_loader
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

        height, width = (
            floor_data.info["map_info"]["height"],
            floor_data.info["map_info"]["width"],
        )
        self.floor_uniform = dist.Uniform(
            low=torch.tensor([0.0, 0.0]), high=torch.tensor([height, width])
        ).to_event(1)

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
            torch.nn.Parameter(torch.tile(self.floor_uniform.mean, (K, 1))),
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
            self.floor_uniform.mean, self.floor_uniform.stddev
        ).to_event(1)

        # Normal walking tempo is 1,4 m/s
        sigma_eps = torch.tensor(0.2, device=device)  # std [m/100ms]
        sigma = torch.tensor(3, device=device)  # std of noise measurement [m]

        # Wifi signal strength priors
        mu_omega_0 = torch.tensor(-45.0, device=device)
        sigma_omega_0 = torch.tensor(
            10.0, device=device
        )  # grundsignal styrke spredning.
        sigma_omega = torch.tensor(
            10.0, device=device
        )  # How presis is the measured signal stregth

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
        log_scale = torch.zeros((len(mini_batch_index),), device=device)

        for i, (index, length) in enumerate(zip(mini_batch_index, mini_batch_length)):
            l, s = self.trace_guides[index](mini_batch_time[i, :length].unsqueeze(1))
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


class ModelTrainer:
    def __init__(
        self, model_label, model, optimizer, n_epochs=1000, batch_size=16, save_every=5
    ):

        self.model_label = model_label

        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.save_every = save_every

        self.checkpoint_path = (checkpoint_dir / model_label).with_suffix(".pt")
        self.load_checkpoint()

    # saves the model and optimizer states to disk
    def save_checkpoint(self):

        checkpoint = {
            "current_epoch": self.current_epoch,
            "loss_history": self.loss_history,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "rng_state": torch.get_rng_state(),
        }

        torch.save(checkpoint, self.checkpoint_path)

    # loads the model and optimizer states from disk
    def load_checkpoint(self):

        if not self.checkpoint_path.exists():
            self.current_epoch = 0
            self.loss_history = []
            return

        checkpoint = torch.load(self.checkpoint_path)
        self.current_epoch = checkpoint["current_epoch"]
        self.loss_history = checkpoint["loss_history"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        torch.set_rng_state(checkpoint["rng_state"])

    def train(self, dataset):

        # Reset parameter values
        pyro.clear_param_store()

        num_particles = 10
        loss_fn = Trace_ELBO(
            num_particles=num_particles, vectorize_particles=True
        ).differentiable_loss

        
        self.data_loader = get_loader(dataset=dataset, batch_size=self.batch_size)

        while self.current_epoch < self.n_epochs:

            elbo = 0
            for mini_batch in self.data_loader:

                loss = loss_fn(self.model.model, self.model.guide, *mini_batch)
                loss.backward()
                self.optimizer.step()
                optimizer.zero_grad()
                elbo = elbo + loss

            print("epoch[%d] ELBO: %.1f" % (self.current_epoch, elbo))

            self.current_epoch += 1
            self.loss_history.append(float(elbo))

            if self.current_epoch % self.save_every == 0:
                self.save_checkpoint()


if __name__ == "__main__":

    torch.manual_seed(123456789)

    # Load data
    site_data = SiteDataset(
        "5a0546857ecc773753327266", wifi_threshold=200, sampling_interval=100
    )
    floor = site_data.floors[0]

    # Setup model
    model = BatchedModel(floor)
    model.to(dtype=torch.float64)

    # Setup the optimizer
    adam_params = {"lr": 1e-3}  # ., "betas":(0.95, 0.999)}
    optimizer = torch.optim.Adam(model.parameters(), **adam_params)

    # Setup model training
    n_epochs = 2000
    batch_size = 16
    mt = ModelTrainer(
        model_label="initial_model",
        model=model,
        optimizer=optimizer,
        n_epochs=n_epochs,
        batch_size=batch_size,
    )

    # Train the model
    mt.train(floor)
