import os
import argparse

import torch
import torchvision.utils as vutils

from models.gan import Generator, Discriminator
from rl.rl_env import RLFundusEnv
from rl.rl_agent import DQNAgent


def parse_args():
    parser = argparse.ArgumentParser(description="Train RL agent to search GAN latent space")
    parser.add_argument("--generator-path", type=str, required=True)
    parser.add_argument("--discriminator-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="rl/results")
    parser.add_argument("--latent-dim", type=int, default=256)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--max-steps", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", type=str, default=None)
    return parser.parse_args()


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def save_image(tensor, path):
    # tensor expected range [-1,1], convert to [0,1] for saving
    vutils.save_image((tensor.clamp(-1, 1) + 1) / 2.0, path)


def main():
    args = parse_args()
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    generator = Generator(latent_dim=args.latent_dim).to(device)
    discriminator = Discriminator().to(device)

    generator.load_state_dict(torch.load(args.generator_path, map_location=device))
    discriminator.load_state_dict(torch.load(args.discriminator_path, map_location=device))

    generator.eval()
    discriminator.eval()

    env = RLFundusEnv(
        generator=generator,
        discriminator=discriminator,
        latent_dim=args.latent_dim,
        device=device,
        num_chunks=4,
        step_size=0.12,
        diversity_weight=0.08,
        max_steps=args.max_steps,
    )

    agent = DQNAgent(
        latent_dim=args.latent_dim,
        n_actions=env.n_actions,
        device=device,
        lr=2e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_final=0.02,
        epsilon_decay=5000,
        buffer_capacity=40000,
        target_update=100,
    )

    ensure_dir(args.output_dir)
    best_eval = -1e9
    best_latents = []
    best_images = []

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        total_reward = 0.0

        for step in range(args.max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.store_transition(state, action, reward, next_state, done)
            _ = agent.optimize(batch_size=args.batch_size)

            state = next_state
            total_reward += reward

            if done:
                break

        if total_reward > best_eval:
            best_eval = total_reward
            best_latents = env.state.detach().cpu().numpy().copy()
            best_images = info.get("generated_image").detach().cpu().clone()

        if episode % 10 == 0 or episode == 1:
            print(f"Episode {episode}/{args.episodes}, total_reward={total_reward:.3f}, best={best_eval:.3f}, epsilon={agent.epsilon:.3f}")

    # Save best results
    latent_path = os.path.join(args.output_dir, "best_latent.npy")
    image_path = os.path.join(args.output_dir, "best_image.png")
    torch.save(torch.tensor(best_latents), latent_path)
    save_image(best_images, image_path)

    # optional agent checkpoint
    torch.save(agent.policy_net.state_dict(), os.path.join(args.output_dir, "dqn_policy_net.pth"))
    print(f"RL training complete. Best reward {best_eval:.3f}")
    print(f"Saved latent: {latent_path}")
    print(f"Saved image: {image_path}")


if __name__ == "__main__":
    main()
