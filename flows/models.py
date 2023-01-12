from __future__ import annotations
import torch
import pytorch_lightning as pl
from flows.layers import CouplingLayer, GlobalRescalingLayer, UpsamplingLayer
import flows.utils as utils
import flows.phi_four as phi_four

class MultilevelFlow(pl.LightningModule):
    # NOTE: this model is hardcoded to have beta-lambda parametrization. Fix.
    def __init__(
        self,
        *,
        #m_sq: float,
        beta: float,
        lam: NonNegativeFloat,
        model_spec: list[dict | str],
        # layers: list[Module],
        # train_dataloader: Prior
    ):
        super().__init__()

        layers = []
        n_upsampling = 0
        for layer_spec in reversed(model_spec):
            if layer_spec == "upsampling":
                layers.insert(0, UpsamplingLayer(use_batch_dimension=True))
                n_upsampling += 1
            elif layer_spec == "rescaling":
                layers.insert(0, GlobalRescalingLayer())
            else:
                transform = layer_spec["transform"](**layer_spec["transform_spec"])
                layer = CouplingLayer(transform, layer_spec["net_spec"])
                layers.insert(0, layer)

        # NOTE: this is an in-place operation...
        self.flow = utils.Flow(*layers)
        self.n_upsampling = n_upsampling
        self.action = phi_four.PhiFourActionBeta(beta, lam)
        # self.action = phi_four.PhiUpInterpFourActionBeta(beta, lam)
        # self.action = lambda state: phi_four.PhiFourActionBeta(beta, lam)(self.upscale_interp(state))

        self.curr_iter = 0

        self.upsampling_layer = UpsamplingLayer(use_batch_dimension=True)

        self.save_hyperparameters()

        # self.train_dataloader = train_dataloader

    def _reshape_z(self, z):
        for level in range(self.n_upsampling):
            z, _ = self.upsampling_layer.inverse(z, torch.zeros([1]))
        return z

    def upscale_interp(self, samples):
        sample_size = samples.size()
        upsamples = torch.zeros([*sample_size[:-2]] + [2*sample_size[-1], 2*sample_size[-1]], device=samples.device)
        upsamples[..., ::2, ::2] = samples[...,:,:]
        upsamples[..., ::2, 1::2] = 1.0/2.0 * (samples + samples.roll(-1,-1))
        upsamples[..., 1::2, ::2] = 1.0/2.0 * (samples + samples.roll(-1,-2))
        upsamples[..., 1::2, 1::2] = 1.0/4.0 * (samples + samples.roll(-1,-2) +
                samples.roll(-1,-1) + samples.roll((-1,-1),(-2,-1)) )
        return upsamples

    def log_state(self, phi):
        self.logger.experiment.add_histogram("phi", phi.flatten(), self.curr_iter)
        self.logger.experiment.add_histogram(
            "action", self.action(phi).flatten(), self.curr_iter
        )

    # NOTE: this is not handy for the flow HMC. It would be a more general
    # approach if I could get the flowed batch AND jac det just doing
    # `model.forward(batch)`, and equivalently for the inverse, instead of
    # having to do `model.flow.inverse(batch)`. Computation of the weights can
    # be done in `training step`.
    # NOTE upsampling: this function downsampled the input before applying the
    # network
    # def forward(self, batch):
    #     z, log_prob_z = batch
    #     z = self._reshape_z(z)
    #     phi, log_det_jacob = self.flow(z)
    #     weights = log_prob_z - log_det_jacob + self.action(phi)

    #     # self.curr_iter += 1
    #     # if self.curr_iter % 1000 == 0:
    #     #     self.log_state(phi)

    #     return phi, weights

    def forward(self, batch):
        z, log_prob_z = batch
        # z = self._reshape_z(z)
        phi, log_det_jacob = self.flow(z)
        if self.n_upsampling == 0:
            weights = log_prob_z - log_det_jacob + self.action(phi)
            # weights = log_prob_z - log_det_jacob - self.train_dataloader.log_prob(phi)
        elif self.n_upsampling == 1:
            weights = log_prob_z.view(-1,4).sum(dim=1) - log_det_jacob + self.action(phi)
        else:
            raise NotImplementedError("Models with more than 1 upsampling layers not supported")

        # self.curr_iter += 1
        # if self.curr_iter % 10 == 0:
        #     self.log_state(phi)
        #     print(z)
        #     print(phi)


        return phi, weights

    def training_step(self, batch, batch_idx):
        _, weights = self.forward(batch)
        loss = weights.mean()
        self.log("loss", loss, logger=True)
        self.lr_schedulers().step()
        return loss

    def validation_step(self, batch, batch_idx):
        phi, weights = self.forward(batch)
        loss = weights.mean()
        acceptance = utils.metropolis_acceptance(weights)
        metrics = dict(loss=loss, acceptance=acceptance)
        self.log_dict(metrics, prog_bar=False, logger=True)
        return loss

    def test_step(self, batch, batch_idx):
        phi, weights = self.forward(batch)
        loss = weights.mean()
        acceptance = utils.metropolis_acceptance(weights)
        metrics = dict(loss=loss, acceptance=acceptance)
        self.log_dict(metrics, prog_bar=False, logger=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.flow.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.trainer.max_steps
        )
        return [optimizer], [scheduler]

    @torch.no_grad()
    def sample(self, prior: IterableDataset, n_iter: PositiveInt = 1):
        phi, weights = self.forward(next(prior))
        for _ in range(n_iter - 1):
            _phi, _weights = self.forward(next(prior))
            phi = torch.cat((phi, _phi), dim=0)
            weights = torch.cat((weights, _weights), dim=0)
        return phi, weights
