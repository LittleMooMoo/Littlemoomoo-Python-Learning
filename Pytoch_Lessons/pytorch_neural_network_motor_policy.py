"""Train a small neural network to imitate a three-joint PD controller.

This is supervised learning: the PD controller creates the correct answers,
and the neural network learns to predict those answers from robot observations.
"""

import torch
from torch import nn


torch.manual_seed(7)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

sample_count = 5_000
joint_count = 3

# Each row represents one robot observation at one moment in time.
joint_positions = 2.0 * torch.rand(sample_count, joint_count) - 1.0
joint_velocities = 2.0 * torch.rand(sample_count, joint_count) - 1.0
observations = torch.cat((joint_positions, joint_velocities), dim=1)

# Create training targets with a simple PD controller.
desired_positions = torch.tensor([0.0, 0.5, -0.8])
desired_velocities = torch.zeros(joint_count)
kp = torch.tensor([1.0, 1.2, 0.8])
kd = torch.tensor([0.10, 0.12, 0.08])

position_errors = desired_positions - joint_positions
velocity_errors = desired_velocities - joint_velocities
target_actions = kp * position_errors + kd * velocity_errors
target_actions = torch.clamp(target_actions, -1.0, 1.0)

observations = observations.to(device)
target_actions = target_actions.to(device)


class MotorPolicy(nn.Module):
    """Map six observations to three normalized motor commands."""

    def __init__(self) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 3),
            nn.Tanh(),
        )

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.network(observation)


def main() -> None:
    policy = MotorPolicy().to(device)
    loss_function = nn.MSELoss()
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.003)

    print(f"Using device: {device}")
    print(f"Observation shape: {observations.shape}")
    print(f"Target-action shape: {target_actions.shape}")

    for training_step in range(1_001):
        predicted_actions = policy(observations)
        loss = loss_function(predicted_actions, target_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if training_step % 100 == 0:
            print(f"step={training_step:4d}  loss={loss.item():.6f}")

    test_observation = torch.tensor(
        [[0.20, 0.10, -0.40, 0.05, -0.10, 0.20]],
        device=device,
    )

    policy.eval()
    with torch.no_grad():
        predicted_command = policy(test_observation)

    print("\nTest observation:", test_observation.cpu())
    print("Predicted motor command:", predicted_command.cpu())


if __name__ == "__main__":
    main()
