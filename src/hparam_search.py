from src.models.model_trainer import ModelTrainer
from src.data.datasets import FloorDataset
from src.models.initial_model import InitialModel
from src.models.wifi_model import WifiModel
from src.models.beacon_model import BeaconModel

import optuna
import torch

class Objective:

    def __init__(self, model_type, site_id=None, floor_id=None):

        if site_id is None:
            site_id = "5a0546857ecc773753327266"

        if floor_id is None:
            floor_id = "B1"

        self.model_type = model_type

        if self.model_type == "initial":
            self.ModelClass = InitialModel
            include_wifi = False
            include_beacon = False
        elif self.model_type == "wifi":
            self.ModelClass = WifiModel
            include_wifi = True
            include_beacon = False
        elif self.model_type == "beacon":
            self.ModelClass = BeaconModel
            include_wifi = True
            include_beacon = True
        self.floor_data = FloorDataset(
            site_id=site_id,
            floor_id=floor_id,
            include_wifi=include_wifi,
            include_beacon=include_beacon,
        )

    def __call__(self, trial: optuna.trial.Trial):

        beta_0 = trial.suggest_loguniform(1e-4, 1)
        lr = trial.suggest_loguniform(1e-4, 5e-2)
        sigma_eps = trial.suggest_uniform(0.1, 0.5)
        n_basis_functions = trial.suggest_int(5, 30)

        model = self.ModelClass(self.floor_data, prior_params={"sigma_eps": sigma_eps})

        # Setup the optimizer
        adam_params = {"lr": lr}
        optimizer = torch.optim.Adam(model.parameters(), **adam_params)

        n_epochs = 1000
        batch_size = 16

        mt = ModelTrainer(
            model=model,
            optimizer=optimizer,
            model_label=f"{self.model_type}_{trial.number:03}",
            n_epochs=n_epochs,
            batch_size=batch_size,
            beta_0=beta_0,
        )
        mt.train()

        return mt.loss_history[-1]

objective = Objective("initial")
