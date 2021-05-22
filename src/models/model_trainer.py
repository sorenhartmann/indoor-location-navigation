
import pyro
import torch
from pyro.infer import Trace_ELBO
from src.data.datasets import get_loader
from pathlib import Path

project_dir = Path(__file__).resolve().parents[2]
checkpoint_dir = project_dir / "checkpoints"

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
                self.optimizer.zero_grad()
                elbo = elbo + loss

            print("epoch[%d] ELBO: %.1f" % (self.current_epoch, elbo))

            self.current_epoch += 1
            self.loss_history.append(float(elbo))

            if self.current_epoch % self.save_every == 0:
                self.save_checkpoint()
