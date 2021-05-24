from os import major

import optuna
from src.data.datasets import FloorDataset
from src.models.initial_model import InitialModel
from src.models.wifi_model import WifiModel
from src.models.beacon_model import BeaconModel
import pyro
import click

from pyexpat import model
from time import time
import pyro
import torch
from pyro.infer import Trace_ELBO, SVI
from src.data.datasets import get_loader
from pathlib import Path
from tqdm import tqdm

def get_mse(model, floor_data_full, test_mask):


    mini_batch = floor_data_full[test_mask.nonzero().flatten()]

    mini_batch_position = mini_batch[3]
    mini_batch_position_mask = mini_batch[4]

    with torch.no_grad():
        loc_q, _ = model.guide(*mini_batch)
        x_est = loc_q[mini_batch_position_mask, :]

    x_hat = mini_batch_position[mini_batch_position_mask, :]

    return ((x_est - x_hat) ** 2).sum(-1).mean()

project_dir = Path(__file__).resolve().parents[1]
checkpoint_dir = project_dir / "checkpoints"


if torch.cuda.is_available():
    device = torch.device("cuda")
    pin_memory = True
else:
    device = torch.device("cpu")
    pin_memory = False


class BetaFunction:
    def __init__(self, beta_0, n_epochs, start=0.25, end=0.75) -> None:

        self.beta_0 = beta_0
        self.n_epochs = n_epochs
        self.start = start * n_epochs
        self.end = end * n_epochs

        self.a = 2 * (1 - beta_0) / self.n_epochs
        self.b = beta_0 - self.a * self.n_epochs / 4

        self.elbo_valid_after = self.end

    def __call__(self, i):

        if i < self.start:
            return self.beta_0
        elif i >= self.start and i < self.end:
            return self.a * i + self.b
        else:
            return 1


class ModelTrainer:
    def __init__(
        self,
        model,
        optimizer,
        model_label=None,
        n_epochs=1000,
        batch_size=16,
        beta_0=1.0,
        save_every=5,
        verbosity=1,
        post_epoch_callback=None,
    ):

        self.post_epoch_callback = post_epoch_callback

        self.model_label = model_label

        self.model = model
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.beta_0 = beta_0
        self.beta_function = BetaFunction(self.beta_0, self.n_epochs)

        self.save_every = save_every

        self.verbosity = verbosity

        self.current_epoch = 0
        self.loss_history = []

        if self.model_label is not None:
            self.checkpoint_path = (checkpoint_dir / model_label).with_suffix(".pt")
            if self.checkpoint_path.exists():
                self.load_checkpoint()
        else:
            self.checkpoint_path = None

    # saves the model and optimizer states to disk
    def save_checkpoint(self):

        checkpoint = {
            "current_epoch": self.current_epoch,
            "loss_history": self.loss_history,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state": self.optimizer.get_state(),
            "rng_state": torch.get_rng_state(),
        }

        torch.save(checkpoint, self.checkpoint_path)

    # loads the model and optimizer states from disk
    def load_checkpoint(self):

        checkpoint = torch.load(self.checkpoint_path, map_location=device)

        self.current_epoch = checkpoint["current_epoch"]
        self.loss_history = checkpoint["loss_history"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.set_state(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["rng_state"])

    def train(self, dataset):

        # Reset parameter values
        pyro.clear_param_store()

        num_particles = 10
        svi = SVI(
            self.model.model,
            self.model.guide,
            self.optimizer,
            loss=Trace_ELBO(num_particles=num_particles, vectorize_particles=True),
        )

        self.data_loader = get_loader(dataset=dataset, batch_size=self.batch_size)

        while self.current_epoch < self.n_epochs:

            elbo = 0
            annealing_factor = self.beta_function(self.current_epoch)

            batch_iter = (
                tqdm(self.data_loader) if self.verbosity > 1 else self.data_loader
            )

            for mini_batch in batch_iter:
                loss = svi.step(*mini_batch, annealing_factor)
                elbo = elbo + loss

            if self.verbosity > 0:
                print("epoch[%d] ELBO: %.1f" % (self.current_epoch, elbo))

            self.current_epoch += 1
            self.loss_history.append(float(elbo))

            if self.current_epoch % self.save_every == 0:
                if self.checkpoint_path is not None:
                    self.save_checkpoint()
                if self.post_epoch_callback is not None:
                    self.post_epoch_callback(self.current_epoch)

        self.save_checkpoint()


@click.command()
@click.argument("model-type", default="initial")
@click.argument("experiment-name", default="unnamed")
@click.option("--n-epochs", type=int, default=500)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--beta_0", type=float, default=0.1)
@click.option("--lr", type=float, default=1e-2)
@click.option("--clip", type=int, default=0)
@click.option("--verbosity", type=int, default=2)
def main(
    model_type, experiment_name, n_epochs, batch_size, beta_0, lr, clip, verbosity
):

    if model_type == "initial":
        ModelClass = InitialModel
        include_wifi = False
        include_beacon = False
        validation_percent = None
        test_percent = None

    elif model_type == "wifi":
        ModelClass = WifiModel
        include_wifi = True
        include_beacon = False
        validation_percent = None
        test_percent = 0.15

    elif model_type == "beacon":
        ModelClass = BeaconModel
        include_wifi = True
        include_beacon = True
        validation_percent = None
        test_percent = 0.15
    else:
        print(f"Cannot read model type: {model_type}")

    floor_data = FloorDataset(
        site_id="5a0546857ecc773753327266",
        floor_id="B1",
        wifi_threshold=200,
        include_wifi=include_wifi,
        include_beacon=include_beacon,
        validation_percent=validation_percent,
        test_percent=test_percent,
    )
    # Setup model
    model = ModelClass(floor_data)

    # Setup the optimizer
    adam_params = {"lr": lr}

    if clip:
        optimizer = pyro.optim.ClippedAdam(adam_params)
    else:
        optimizer = pyro.optim.Adam(adam_params)

    if experiment_name == "unnamed":
        experiment_name = f"{model_type}-model"

    mt = ModelTrainer(
        model=model,
        optimizer=optimizer,
        model_label=experiment_name,
        n_epochs=n_epochs,
        batch_size=batch_size,
        beta_0=beta_0,
        verbosity=verbosity,
    )
    # Train the model
    mt.train(floor_data)


if __name__ == "__main__":

    main()
