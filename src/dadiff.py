"""DADiff: domain-adaptive conditional diffusion for cross-modal translation.

The diffusion is trained to predict the clean image of one modality directly
from
* the image of the *other* modality (the condition), and
* a noisy version of the target modality.

Unlike the common parameterisation that predicts the injected noise, the network
outputs an estimate of ``x_0`` (see Eq. 7-10 of the paper).  Domain adaptation is
obtained by aligning the multi-scale features of the conditional encoder with
those of a frozen diffusion model trained on the source domain, using the
dynamic MMD / CORAL fusion described in Sec. III-B2.
"""

from functools import partial

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .losses import MultiScaleAlignmentLoss
from .models.diffusion_backbone import DADiffBackbone


def make_beta_schedule(n_timestep, linear_start=1e-4, linear_end=2e-2):
    """Linear noise schedule, from ``linear_start`` to ``linear_end``."""
    return np.linspace(linear_start, linear_end, n_timestep, dtype=np.float64)


class GaussianDiffusion(nn.Module):
    """Reverse process of the conditional diffusion model.

    Args:
        model: the ``DADiffBackbone``.
        loss_type: ``'l2'`` or ``'l1'``.
        timesteps: total number of diffusion steps ``T``.
        sample_steps: number of denoising iterations used at inference time; the
            loop runs over the last ``sample_steps`` time steps.
    """

    def __init__(self, model, loss_type='l2', timesteps=2000, sample_steps=100,
                 linear_start=1e-4, linear_end=2e-2, device=None):
        super().__init__()
        self.model = model.to(device)
        self.loss_type = loss_type
        self.device = device
        self.timesteps = timesteps
        self.sample_steps = sample_steps
        self.linear_start = linear_start
        self.linear_end = linear_end
        self.set_new_noise_schedule()
        self.set_loss()

    def set_loss(self):
        if self.loss_type == 'l1':
            self.loss_func = nn.L1Loss(reduction='mean').to(self.device)
        elif self.loss_type == 'l2':
            self.loss_func = nn.MSELoss(reduction='mean').to(self.device)
        else:
            raise NotImplementedError(f"Unknown loss type: {self.loss_type}")

    def set_new_noise_schedule(self):
        to_torch = partial(torch.tensor, dtype=torch.float32, device=self.device)

        betas = make_beta_schedule(self.timesteps, self.linear_start, self.linear_end)
        alphas = 1. - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1., alphas_cumprod[:-1])

        self.num_timesteps = int(betas.shape[0])

        self.register_buffer('betas', to_torch(betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(alphas_cumprod_prev))

        # The network is conditioned on sqrt(alpha_bar) rather than on t itself.
        self.sqrt_alphas_cumprod_prev = np.sqrt(np.append(1., alphas_cumprod))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        self.register_buffer('posterior_variance', to_torch(posterior_variance))
        self.register_buffer('posterior_log_variance_clipped',
                             to_torch(np.log(np.maximum(posterior_variance, 1e-20))))
        self.register_buffer('posterior_mean_coef1',
                             to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)))
        self.register_buffer('posterior_mean_coef2',
                             to_torch((1. - alphas_cumprod_prev) * np.sqrt(alphas) / (1. - alphas_cumprod)))

        # Not used by the direct x_0 parameterisation, but registered so that
        # checkpoints saved by the reference implementation load without
        # `strict=False`.
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod - 1)))

    def q_sample(self, x_start, continuous_sqrt_alpha_cumprod, noise=None):
        """Forward diffusion: ``x_t`` from ``x_0`` at the given noise level."""
        if noise is None:
            noise = torch.randn_like(x_start)
        return (continuous_sqrt_alpha_cumprod * x_start
                + (1 - continuous_sqrt_alpha_cumprod ** 2).sqrt() * noise)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = self.posterior_mean_coef1[t] * x_start + self.posterior_mean_coef2[t] * x_t
        return posterior_mean, self.posterior_log_variance_clipped[t]

    def p_mean_variance(self, x_recon, x_t, t):
        return self.q_posterior(x_start=x_recon, x_t=x_t, t=t)

    @torch.no_grad()
    def p_sample(self, x_recon, t, x_t):
        model_mean, model_log_variance = self.p_mean_variance(x_recon=x_recon, t=t, x_t=x_t)
        noise = torch.randn_like(x_recon) if t > 0 else torch.zeros_like(x_recon)
        return model_mean + noise * (0.5 * model_log_variance).exp()

    @torch.no_grad()
    def p_sample_loop(self, condition, target):
        """Iterative denoising used at inference time."""
        b, _, _, _ = condition.shape
        self.model = self.model.to(self.device)

        img = torch.randn_like(target).to(self.device)

        for i in tqdm(reversed(range(self.num_timesteps - self.sample_steps, self.num_timesteps)),
                      desc='sampling', total=self.sample_steps):
            timesteps = torch.FloatTensor(
                np.repeat(self.sqrt_alphas_cumprod_prev[i], b)).view(b, 1).to(self.device)
            x_recon, _ = self.model([condition, img], timesteps)
            img = self.p_sample(x_recon=x_recon, x_t=img, t=i)

        # Final prediction at the smallest noise level.
        x_recon, _ = self.model([condition, img], timesteps=torch.zeros([b, 1]).to(self.device))
        return x_recon

    def p_losses(self, condition, target):
        """Single training step on a batch of (condition, target) pairs."""
        x_start = target
        b, c, h, w = x_start.shape

        t = np.random.randint(1, self.num_timesteps)
        noise = torch.randn(b, c, h, w).to(self.device)

        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.repeat(self.sqrt_alphas_cumprod_prev[t], b)).to(self.device).view(b, 1)
        x_noisy = self.q_sample(x_start=x_start,
                                continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
                                noise=noise)

        x_recon, features = self.model([condition, x_noisy], continuous_sqrt_alpha_cumprod)
        return self.loss_func(x_recon, x_start), features

    def forward(self, condition, target):
        return self.p_losses(condition, target)


class FrozenDiffusion(GaussianDiffusion):
    """Reference model of the source domain; only provides features."""

    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad = False

    @torch.no_grad()
    def features(self, condition, target):
        """Features of the conditional encoder at the current noise level."""
        x_start = target
        b, c, h, w = x_start.shape

        t = np.random.randint(1, self.num_timesteps)
        noise = torch.randn(b, c, h, w).to(self.device)
        continuous_sqrt_alpha_cumprod = torch.FloatTensor(
            np.repeat(self.sqrt_alphas_cumprod_prev[t], b)).to(self.device).view(b, 1)
        x_noisy = self.q_sample(x_start=x_start,
                                continuous_sqrt_alpha_cumprod=continuous_sqrt_alpha_cumprod.view(-1, 1, 1, 1),
                                noise=noise)

        _, features = self.model([condition, x_noisy], continuous_sqrt_alpha_cumprod)
        return features


class DADiff:
    """Domain-adaptive conditional diffusion for one translation direction.

    Args:
        direction: ``'rgb'`` to generate RGB from the SAR image, or ``'sar'``
            for the opposite direction.
        in_chans: number of channels of the images (three for RGB and SAR).
        lr: learning rate of the Adam optimiser.
        alphas: MMD weight of each encoder scale.
    """

    def __init__(self, direction, in_chans, device,
                 timesteps=2000, sample_steps=100, loss_type='l2', lr=1e-4,
                 depths=(2, 2, 2), depths_decoder=(2, 2, 2),
                 attn_drop_rate=0.2, drop_path_rate=0.2,
                 alphas=(0.3, 0.5, 0.7), use_alignment=True):
        self.direction = direction
        self.in_chans = in_chans
        self.device = device
        self.use_alignment = use_alignment

        backbone = DADiffBackbone(in_chans=in_chans,
                                  depths=depths,
                                  depths_decoder=depths_decoder,
                                  attn_drop_rate=attn_drop_rate,
                                  drop_path_rate=drop_path_rate)

        diffusion_kwargs = dict(loss_type=loss_type, timesteps=timesteps,
                                sample_steps=sample_steps, device=device)
        self.netG = GaussianDiffusion(backbone, **diffusion_kwargs)

        # The frozen reference model only exists when the target domain is
        # adapted to a source domain; the source-domain pre-training has no
        # reference domain to align with.
        if use_alignment:
            source_backbone = DADiffBackbone(in_chans=in_chans,
                                             depths=depths,
                                             depths_decoder=depths_decoder,
                                             attn_drop_rate=attn_drop_rate,
                                             drop_path_rate=drop_path_rate)
            self.source_netG = FrozenDiffusion(source_backbone, **diffusion_kwargs)
        else:
            self.source_netG = None

        self.alignment = MultiScaleAlignmentLoss(alphas=alphas)
        self.optimizer = torch.optim.Adam(self.netG.parameters(), lr=lr)

    def optimize_parameters(self, condition, target, source_condition=None, source_target=None):
        """One optimisation step: diffusion loss plus feature alignment."""
        self.netG.train()
        self.optimizer.zero_grad()

        target_loss, target_features = self.netG(condition, target)

        if not self.use_alignment:
            target_loss.backward()
            self.optimizer.step()
            return target_loss, target_loss, 0.0, 0.0, 0.0

        source_features = self.source_netG.features(source_condition, source_target)
        coral, mmd, alignment_loss = self.alignment(source_features, target_features)

        total_loss = target_loss + alignment_loss
        total_loss.backward()
        self.optimizer.step()

        return total_loss, target_loss, alignment_loss, mmd, coral

    @torch.no_grad()
    def generate(self, condition, target):
        """Translate ``condition`` into the target modality."""
        self.netG.eval()
        return self.netG.p_sample_loop(condition, target)

    def state_dict(self):
        state = {'netG': self.netG.state_dict()}
        if self.source_netG is not None:
            state['source_netG'] = self.source_netG.state_dict()
        return state

    def load_state_dict(self, state, load_source=True):
        self.netG.load_state_dict(state['netG'])
        if load_source and 'source_netG' in state and self.source_netG is not None:
            self.source_netG.load_state_dict(state['source_netG'])
        if load_source and 'source_netG' in state and self.source_netG is None:
            raise ValueError("This instance was built without a reference model")

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, load_source=True):
        self.load_state_dict(torch.load(path, map_location=self.device), load_source=load_source)
