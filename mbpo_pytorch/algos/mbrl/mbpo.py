from __future__ import annotations

from itertools import count
from operator import itemgetter
from typing import Dict, List, TYPE_CHECKING

import numpy as np
import torch

from mbpo_pytorch.misc import logger

if TYPE_CHECKING:
    from mbpo_pytorch.storages import SimpleUniversalBuffer as Buffer
    from mbpo_pytorch.models.dynamics import BaseDynamics
    from mbpo_pytorch.envs.virtual_env import VecVirtualEnv


# noinspection DuplicatedCode
def split_model_buffer(buffer: Buffer, ratio: float):
    full_indices = np.arange(buffer.size)
    np.random.shuffle(full_indices)
    train_indices = full_indices[:int(ratio * buffer.size)]
    val_indices = full_indices[int(ratio * buffer.size):]
    return train_indices, val_indices


class MBPO:
    def __init__(self, dynamics: BaseDynamics, batch_size: int, max_num_epochs: int,
                 rollout_schedule: List[int], l2_loss_coefs: List[float], lr, max_grad_norm=2, verbose=0):
        """

        @param dynamics: pytorch neural network
        @param batch_size:
        @param max_num_epochs:
        @param rollout_schedule:
        @param l2_loss_coefs: a list of l2_loss_coefs used to avoid over-fitting
        @param lr:
        @param max_grad_norm:
        @param verbose:
        """
        self.dynamics = dynamics
        self.epoch = 0

        self.num_networks = self.dynamics.num_networks

        self.max_num_epochs = max_num_epochs
        self.num_rollout_steps = 0
        self.rollout_schedule = rollout_schedule
        self.batch_size = batch_size
        self.l2_loss_coefs = l2_loss_coefs
        self.max_grad_norm = max_grad_norm
        self.training_ratio = 0.8

        self.dynamics_optimizer = torch.optim.Adam(self.dynamics.parameters(), lr)
        self.elite_dynamics_indices = []
        self.verbose = verbose

    @staticmethod
    def check_buffer(buffer):
        assert {'states', 'actions', 'rewards', 'masks', 'next_states'}.issubset(buffer.entry_infos.keys())

    def get_ensemble_samples(self, samples: Dict[str, torch.Tensor]):
        attrs = ['states', 'actions', 'next_states', 'rewards', 'masks']
        batch_size = samples[attrs[0]].shape[0]
        idxes = np.random.randint(batch_size, size=[batch_size, self.num_networks])

        # size: (e.g. states)
        # batch-size * ensemble-size * state-dim
        return [samples[attr][idxes] for attr in attrs]

    def get_repeat_samples(self, samples: Dict[str, torch.Tensor]):
        attrs = ['states', 'actions', 'next_states', 'rewards', 'masks']
        value_list = []
        for attr in attrs:
            value = samples[attr]
            value = torch.unsqueeze(value, 1)
            value_list.append(value.repeat(1, self.num_networks, 1))
        return value_list

    def compute_loss(self, samples: Dict[str, torch.Tensor], use_var_loss=True, use_l2_loss=True, ensemble=True):
        if ensemble:
            states, actions, next_states, rewards, masks = self.get_ensemble_samples(samples)
        else:
            states, actions, next_states, rewards, masks = self.get_repeat_samples(samples)

        batch_size = states.shape[0]
        # forward use dynamics
        diff_state_means, diff_state_logvars, reward_means, reward_logvars = \
            itemgetter('diff_state_means', 'diff_state_logvars', 'reward_means', 'reward_logvars') \
                (self.dynamics.forward(states, actions))

        means, logvars = torch.cat([diff_state_means, reward_means], dim=-1), \
                         torch.cat([diff_state_logvars, reward_logvars], dim=-1)
        # size: batch-size * ensemble-size * (state-dim + reward-dim)
        targets = torch.cat([next_states - states, rewards], dim=-1)
        # size: ensemble-size * batch-size * (state-dim + reward-dim)
        targets = targets.transpose(0, 1).contiguous()
        masks = masks.transpose(0, 1).contiguous()

        if use_var_loss:
            inv_vars = torch.exp(-logvars)
            mse_losses = torch.mean(((means - targets) ** 2) * inv_vars * masks, dim=[-2, -1])
            var_losses = torch.mean(logvars * masks, dim=[-2, -1])
            model_losses = mse_losses + var_losses
        else:
            mse_losses = torch.mean(((means - targets) ** 2) * masks, dim=[-2, -1])
            model_losses = mse_losses

        if use_l2_loss:
            l2_losses = self.dynamics.compute_l2_loss(self.l2_loss_coefs)
            return model_losses, l2_losses
        else:
            return model_losses, None

    # train model use real-buffer
    def update(self, model_buffer: Buffer) -> Dict[str, float]:
        model_loss_epoch = 0.
        l2_loss_epoch = 0.

        if self.max_num_epochs:
            epoch_iter = range(self.max_num_epochs)
        else:
            epoch_iter = count()

        train_indices, val_indices = split_model_buffer(model_buffer, self.training_ratio)

        num_epoch_after_update = 0
        num_updates = 0
        epoch = 0

        self.dynamics.reset_best_snapshots()

        for epoch in epoch_iter:
            train_gen = model_buffer.get_batch_generator_epoch(self.batch_size, train_indices)
            val_gen = model_buffer.get_batch_generator_epoch(None, val_indices)

            for samples in train_gen:
                train_model_loss, train_l2_loss = self.compute_loss(samples, True, True, True)
                train_model_loss, train_l2_loss = train_model_loss.sum(), train_l2_loss.sum()
                train_model_loss += \
                    0.01 * (torch.sum(self.dynamics.max_diff_state_logvar) + torch.sum(self.dynamics.max_reward_logvar) -
                            torch.sum(self.dynamics.min_diff_state_logvar) - torch.sum(self.dynamics.min_reward_logvar))

                model_loss_epoch += train_model_loss.item()
                l2_loss_epoch += train_l2_loss.item()

                self.dynamics_optimizer.zero_grad()
                (train_l2_loss + train_model_loss).backward()
                self.dynamics_optimizer.step()

                num_updates += 1

            with torch.no_grad():
                val_model_loss, _ = self.compute_loss(next(val_gen), False, False, False)
            updated = self.dynamics.update_best_snapshots(val_model_loss, epoch)

            # updated == True, means training is useful.
            if updated:
                num_epoch_after_update = 0
            else:
                num_epoch_after_update += 1
            # if training is useless for 5 epoch, stop training.
            if num_epoch_after_update > 5:
                break

        model_loss_epoch /= num_updates
        l2_loss_epoch /= num_updates

        val_gen = model_buffer.get_batch_generator_epoch(None, val_indices)
        # load best snapshots, which is evaluated by validation set.
        best_epochs = self.dynamics.load_best_snapshots()
        with torch.no_grad():
            val_model_loss, _ = self.compute_loss(next(val_gen), False, False, False)
        self.dynamics.update_elite_indices(val_model_loss)

        if self.verbose > 0:
            logger.log('[ Model Training ] Converge at epoch {}'.format(epoch))
            logger.log('[ Model Training ] Load best state_dict from epoch {}'.format(best_epochs))
            logger.log('[ Model Training ] Validation Model loss of elite networks: {}'.
                       format(val_model_loss.cpu().numpy()[self.dynamics.elite_indices]))

        return {'model_loss': model_loss_epoch, 'l2_loss': l2_loss_epoch}

    def update_rollout_length(self, epoch: int):
        # update the rollout_length by the value of epoch
        min_epoch, max_epoch, min_length, max_length = self.rollout_schedule
        if epoch <= min_epoch:
            y = min_length
        else:
            dx = (epoch - min_epoch) / (max_epoch - min_epoch)
            dx = min(dx, 1)
            y = dx * (max_length - min_length) + min_length
        y = int(y)
        if self.verbose > 0 and self.num_rollout_steps != y:
            logger.log('[ Model Rollout ] Max rollout length {} -> {} '.format(self.num_rollout_steps, y))
        self.num_rollout_steps = y

    def generate_data(self, virtual_envs: VecVirtualEnv, policy_buffer: Buffer, initial_states: torch.Tensor, actor):
        states = initial_states
        batch_size = initial_states.shape[0]
        num_total_samples = 0
        for step in range(self.num_rollout_steps):
            with torch.no_grad():
                actions = actor.act(states)['actions']
            next_states, rewards, dones, _ = virtual_envs.step_with_states(states, actions)
            masks = torch.tensor([[0.0] if done else [1.0] for done in dones], dtype=torch.float32)
            policy_buffer.insert(states=states, actions=actions, masks=masks, rewards=rewards,
                                 next_states=next_states)
            num_total_samples += next_states.shape[0]
            # states which are not done
            states = next_states[torch.where(torch.gt(masks, 0.5))[0], :]
            if states.shape[0] == 0:
                logger.warn('[ Model Rollout ] Breaking early: {}'.format(step))
                break
        if self.verbose:
            logger.log('[ Model Rollout ] {} samples with average rollout length {:.2f}'.
                       format(num_total_samples, num_total_samples / batch_size))


def test():
    from mbpo_pytorch.models.dynamics import ParallelEnsembleDynamics
    import gym
    env = gym.make("CartPole-v1")
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    reward_dim = 1
    hidden_dims = [200, 200, 200, 200]
    ensemble_size = 7
    elite_size = 5

    parallel_dynamics = ParallelEnsembleDynamics(state_dim, action_dim, reward_dim, hidden_dims, ensemble_size,
                                                 elite_size)

    dynamics_batch_size = 256
    rollout_schedule = [20, 150, 1, 15]
    lr = 1.0e-3
    l2_loss_coefs = [0.000025, 0.00005, 0.000075, 0.000075, 0.0001]
    max_num_epochs = 100

    model = MBPO(parallel_dynamics, dynamics_batch_size, rollout_schedule=rollout_schedule, verbose=1,
                 lr=lr, l2_loss_coefs=l2_loss_coefs, max_num_epochs=max_num_epochs)

    model.get_ensemble_samples(None)


if __name__ == "__main__":
    test()
