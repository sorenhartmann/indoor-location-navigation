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


class Objective:
    def __init__(
        self, model_type, site_id=None, floor_id=None, n_epochs=500, batch_size=16
    ):

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
            validation_percent=None
            test_percent=None
        elif self.model_type == "wifi":
            self.ModelClass = WifiModel
            include_wifi = True
            include_beacon = False
            validation_percent=0.3
            test_percent=0.2
        elif self.model_type == "beacon":
            self.ModelClass = BeaconModel
            include_wifi = True
            include_beacon = True
            validation_percent=0.3
            test_percent=0.2

        self.floor_data = FloorDataset(
            site_id=site_id,
            floor_id=floor_id,
            wifi_threshold=400,
            include_wifi=include_wifi,
            include_beacon=include_beacon,
            validation_percent=validation_percent,
            test_percent=test_percent,
        )

    def __call__(self, trial: optuna.trial.Trial):

        beta_0 = trial.suggest_loguniform("beta_0", 1e-4, 1)
        lr = trial.suggest_loguniform("lr", 1e-3, 5e-2)
        sigma_eps = trial.suggest_uniform("sigma_eps", 0.1, 0.5)

        model = self.ModelClass(self.floor_data, prior_params={"sigma_eps": sigma_eps})

        # Setup the optimizer
        adam_params = {"lr": lr}
        optimizer = pyro.optim.Adam(adam_params)

        n_epochs = self.n_epochs
        batch_size = self.batch_size

        mt = ModelTrainer(
            model=model,
            optimizer=optimizer,
            model_label=f"{self.model_type}_hparam_{trial.number:03}",
            n_epochs=n_epochs,
            batch_size=batch_size,
            beta_0=beta_0,
            verbosity=0,
        )
        mt.train(self.floor_data)

        return mt.loss_history[-1]


@click.command()
@click.argument("model-type", default="initial")
@click.option("--n-trials", type=int, default=100, show_default=True)
@click.option("--n-epochs", type=int, default=500, show_default=True)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--name", type=str)
@click.option("--seed", type=int, default=10)
def main(
    model_type,
    n_trials,
    n_epochs,
    batch_size,
    name,
    seed,
):

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

    objective = Objective(model_type, n_epochs=n_epochs, batch_size=batch_size)
    
    study.optimize(objective, n_trials=n_trials)


if __name__ == "__main__":
    main()
    
