from pyexpat import model
from time import time
import pyro
import torch
from pyro.infer import Trace_ELBO, SVI
from src.data.datasets import get_loader
from pathlib import Path
from tqdm import tqdm

project_dir = Path(__file__).resolve().parents[2]
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
    ):

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

        checkpoint = torch.load(self.checkpoint_path)
        self.current_epoch = checkpoint["current_epoch"]
        self.loss_history = checkpoint["loss_history"]
        self.model.load_state_dict(checkpoint["model_state_dict"])
        #self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.optimizer.load_state(checkpoint["optimizer_state"])
        torch.set_rng_state(checkpoint["rng_state"])

    def train(self, dataset):

        # Reset parameter values
        pyro.clear_param_store()

        num_particles = 10
        svi = SVI(self.model.model, self.model.guide,self.optimizer, loss = Trace_ELBO(num_particles=num_particles, vectorize_particles=True))

        #loss_fn = Trace_ELBO(
        #    num_particles=num_particles, vectorize_particles=True
        #).differentiable_loss

        self.data_loader = get_loader(dataset=dataset, batch_size=self.batch_size)#, pin_memory = pin_memory)

        while self.current_epoch < self.n_epochs:
            
            elbo = 0
            annealing_factor = self.beta_function(self.current_epoch)

            batch_iter = (
                tqdm(self.data_loader) if self.verbosity > 1 else self.data_loader
            )

            for mini_batch in batch_iter:
                loss = svi.step(*mini_batch)
                #loss = loss_fn(
                #    self.model.model, self.model.guide, *mini_batch, annealing_factor
                #)
                #loss.backward()
                #self.optimizer.step()
                #self.optimizer.zero_grad()
                elbo = elbo + loss

            if self.verbosity > 0:
                print("epoch[%d] ELBO: %.1f" % (self.current_epoch, elbo))

            self.current_epoch += 1
            self.loss_history.append(float(elbo))

            if (
                self.checkpoint_path is not None
                and self.current_epoch % self.save_every == 0
            ):
                self.save_checkpoint()
