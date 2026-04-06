import torch
import torch.nn.functional as F


class RLFundusEnv:
    """RL environment that searches latent space for GAN image realism and diversity."""

    def __init__(
        self,
        generator,
        discriminator,
        latent_dim=256,
        device=None,
        num_chunks=4,
        step_size=0.15,
        diversity_weight=0.05,
        max_steps=30,
    ):
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator = generator.to(self.device).eval()
        self.discriminator = discriminator.to(self.device).eval()

        self.latent_dim = latent_dim
        self.num_chunks = min(num_chunks, latent_dim)
        self.chunk_size = max(1, latent_dim // self.num_chunks)
        self.step_size = step_size
        self.diversity_weight = diversity_weight
        self.max_steps = max_steps

        self.action_space = list(range(8))
        self.n_actions = len(self.action_space)

        self.reset()

    def reset(self):
        self.state = torch.randn(self.latent_dim, device=self.device)
        self.prev_image = None
        self.steps = 0
        z = self.state.unsqueeze(0)
        self.prev_image = self._generate_image(z)
        return self.state.clone()

    def _generate_image(self, z_batch):
        with torch.no_grad():
            img = self.generator(z_batch)
            # clip to [-1,1] if generator output may drift
            img = torch.clamp(img, -1.0, 1.0)
        return img

    def _discriminator_score(self, img):
        with torch.no_grad():
            score = self.discriminator(img)
            # assume 0..1
            return float(score.mean().item())

    def _diversity_bonus(self, img):
        if self.prev_image is None:
            return 0.0
        return float(F.l1_loss(img, self.prev_image, reduction="mean").item())

    def step(self, action: int):
        if action not in self.action_space:
            raise ValueError(f"Action {action} out of bounds")

        chunk_id = action // 2
        direction = 1.0 if (action % 2 == 0) else -1.0
        start = min(chunk_id * self.chunk_size, self.latent_dim - 1)
        end = (chunk_id + 1) * self.chunk_size if chunk_id < self.num_chunks - 1 else self.latent_dim

        self.state = self.state.clone()
        self.state[start:end] = self.state[start:end] + direction * self.step_size

        z = self.state.unsqueeze(0)
        generated_image = self._generate_image(z)

        disc_score = self._discriminator_score(generated_image)
        diversity = self._diversity_bonus(generated_image)

        reward = disc_score + self.diversity_weight * diversity

        self.prev_image = generated_image.detach()
        self.steps += 1

        done = self.steps >= self.max_steps
        return self.state.clone(), reward, done, {"generated_image": generated_image.detach(), "disc_score": disc_score, "diversity": diversity}
