import os
from pathlib import Path

from optuna import study
import pyro
from src.train_model import ModelTrainer
from src.data.datasets import FloorDataset
from src.models.initial_model import InitialModel
from src.models.wifi_model import WifiModel
from src.models.beacon_model import BeaconModel

import click
import optuna
import torch


def get_mse(model, floor_data_full, test_mask):

    mini_batch = floor_data_full[test_mask.nonzero().flatten()]

    mini_batch_position = mini_batch[3]
    mini_batch_position_mask = mini_batch[4]

    with torch.no_grad():
        loc_q, _ = model.guide(*mini_batch)
        x_est = loc_q[mini_batch_position_mask, :]

    x_hat = mini_batch_position[mini_batch_position_mask, :]

    return ((x_est - x_hat) ** 2).sum(-1).mean()


class Objective:
    def __init__(
        self,
        model_type,
        site_id=None,
        floor_id=None,
        n_epochs=500,
        batch_size=16,
        use_elbo=True,
    ):

        self.use_elbo = use_elbo

        if site_id is None:
            site_id = "5a0546857ecc773753327266"

        if floor_id is None:
            floor_id = "B1"

        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.model_type = model_type

        if self.model_type == "initial":
            self.ModelClass = InitialModel
            include_wifi = False
            include_beacon = False
            validation_percent = None
            test_percent = None
        elif self.model_type == "wifi":
            self.ModelClass = WifiModel
            include_wifi = True
            include_beacon = False
            validation_percent = None
            test_percent = 0.15
        elif self.model_type == "beacon":
            self.ModelClass = BeaconModel
            include_wifi = True
            include_beacon = True
            validation_percent = None
            test_percent = 0.15

        self.floor_data = FloorDataset(
            site_id=site_id,
            floor_id=floor_id,
            wifi_threshold=200,
            include_wifi=include_wifi,
            include_beacon=include_beacon,
            validation_percent=validation_percent,
            test_percent=test_percent,
        )

        self.floor_data_full = FloorDataset(
            site_id=site_id,
            floor_id=floor_id,
            wifi_threshold=200,
            include_wifi=include_wifi,
            include_beacon=include_beacon,
            validation_percent=None,
            test_percent=None,
        )

    def get_loss(self):

        if self.use_elbo:
            return self.mt.loss_history[-1]
        else:
            return float(
                get_mse(
                    self.mt.model,
                    self.floor_data_full,
                    self.floor_data.test_mask,
                )
            )

    def __call__(self, trial: optuna.trial.Trial):

        beta_0 = trial.suggest_loguniform("beta_0", 1e-4, 1)
        lr = trial.suggest_loguniform("lr", 1e-3, 5e-2)
        sigma_eps = trial.suggest_uniform("sigma_eps", 0.1, 0.5)

        use_clip = trial.suggest_categorical("clip", [True, False])

        model = self.ModelClass(self.floor_data, prior_params={"sigma_eps": sigma_eps})

        # Setup the optimizer
        if use_clip:
            gamma = trial.suggest_uniform("gamma", 0.1, 1.0)
            lrd = gamma ** (1 / self.n_epochs)
            adam_params = {"lr": lr, "lrd": lrd}
            optimizer = pyro.optim.ClippedAdam(adam_params)
        else:
            adam_params = {"lr": lr}
            optimizer = pyro.optim.Adam(adam_params)

        n_epochs = self.n_epochs
        batch_size = self.batch_size

        def callback(epoch):
            trial.report(self.get_loss(), epoch)

        self.mt = ModelTrainer(
            model=model,
            optimizer=optimizer,
            model_label=f"{self.model_type}_hparam_{trial.number:03}",
            n_epochs=n_epochs,
            batch_size=batch_size,
            beta_0=beta_0,
            verbosity=0,
            save_every=5,
            post_epoch_callback=callback,
        )
        self.mt.train(self.floor_data)

        return self.get_loss()


@click.command()
@click.argument("model-type", default="initial")
@click.option("--n-trials", type=int, default=100, show_default=True)
@click.option("--n-epochs", type=int, default=500, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--name", type=str)
@click.option("--use-elbo", type=int, default=1)
@click.option("--seed", type=int, default=None)
def main(
    model_type,
    n_trials,
    n_epochs,
    batch_size,
    name,
    use_elbo,
    seed,
):

    if seed is not None:
        torch.manual_seed(seed)

    root_dir = (Path(__file__).parents[1]).resolve()

    if name is None:
        study_name = f"{model_type}"
    else:
        study_name = name

    storage_name = (
        f"sqlite:///{(root_dir / 'optuna-storage.db').relative_to(os.getcwd())}"
    )

    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        direction="minimize",
        load_if_exists=True,
    )

    objective = Objective(
        model_type, n_epochs=n_epochs, batch_size=batch_size, use_elbo=use_elbo
    )

    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":

    main()
