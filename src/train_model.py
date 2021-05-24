from src.models.model_trainer import ModelTrainer
from src.data.datasets import FloorDataset
from src.models.initial_model import InitialModel
from src.models.wifi_model import WifiModel
from src.models.beacon_model import BeaconModel
import pyro
import click

@click.command()
@click.argument("model-type", default="initial")
@click.argument("experiment-name", default="unnamed")
@click.option("--n-epochs", type=int, default=500)
@click.option("--batch-size", type=int, default=16, show_default=True)
@click.option("--beta_0", type=float, default=0.1)
@click.option("--lr", type=float, default=1e-2)
@click.option("--verbosity", type=int, default=2)
def main(model_type, experiment_name, n_epochs, batch_size, beta_0, lr, verbosity):

    if model_type == "initial":
        ModelClass = InitialModel
        include_wifi = False
        include_beacon = False
        validation_percent=None
        test_percent=None

    elif model_type == "wifi":
        ModelClass = WifiModel
        include_wifi = True
        include_beacon = False
        validation_percent=0.3
        test_percent=0.2

    elif model_type == "beacon":
        ModelClass = BeaconModel
        include_wifi = True
        include_beacon = True
        validation_percent=0.3
        test_percent=0.2
    else:
        print(f"Cannot read model type: {model_type}")

    floor_data = FloorDataset(
        site_id="5d2709b303f801723c327472",
        floor_id="1F",
        wifi_threshold=400,
        include_wifi=include_wifi,
        include_beacon=include_beacon,
        validation_percent=validation_percent,
        test_percent=test_percent,
    )
    # Setup model 
    model = ModelClass(floor_data)

    # Setup the optimizer
    adam_params = {"lr": lr, "betas":(0.95, 0.999)}
    optimizer = pyro.optim.ClippedAdam(adam_params)

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
