import numpy as np
import torch
from torch import nn as nn
from torch.distributions.dirichlet import Dirichlet
import random

from rlkit.torch.sac.policies import TanhGaussianPolicy
from rlkit.policies.base import Policy
from rlkit.torch.core import eval_np
from rlkit.torch.distributions import TanhNormal
from rlkit.policies.base import ExplorationPolicy
from rlkit.torch.networks import SplitNetworkSimple,SplitNetworkShared, SplitNetworkAttention

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

class SkillTanhGaussianPolicy(TanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            skill_dim=10,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            **kwargs
        )
        self.skill_dim = skill_dim
        self.skill = 0

    def get_action(self, obs_np, deterministic=False):
        # generate (iters, skill_dim) matrix that stacks one-hot skill vectors
        # online reinforcement learning
        skill_vec = np.zeros(self.skill_dim)
        skill_vec[self.skill] += 1
        obs_np = np.concatenate((obs_np, skill_vec), axis=0)
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {"skill": skill_vec}

    def get_actions(self, obs_np, deterministic=False):
        return eval_np(self, obs_np, deterministic=deterministic)[0]

    def skill_reset(self):
        self.skill = random.randint(0, self.skill_dim-1)

    def forward(
            self,
            obs,
            skill_vec=None,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        if skill_vec is None:
            h = obs
        else:
            h = torch.cat((obs, skill_vec), dim=1)

        for i, fc in enumerate(self.fcs):
            h = self.hidden_activation(fc(h))
        mean = self.last_fc(h)
        if self.std is None:
            log_std = self.last_fc_log_std(h)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
        else:
            std = self.std
            log_std = self.log_std

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class SimpleSplitSkillGaussianPolicy(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            skill_dim=4,
            use_shared=True,
            starter_hiddens=[512],
            continuous=False,
            use_attention = False
    ):
        self.continuous = continuous
        super().__init__()
        use_std = False
        if std is None:
            use_std = True
        if use_attention:
            self.network = SplitNetworkAttention(
                hidden_sizes=hidden_sizes,
                input_x_size=obs_dim,
                output_size=action_dim,
                num_heads=skill_dim,
                starter_hiddens=starter_hiddens
            )
        elif use_shared:
            self.network = SplitNetworkShared(
                hidden_sizes=hidden_sizes,
                input_x_size = obs_dim,
                output_size=action_dim,
                num_heads=skill_dim,
                use_std=True,
                starter_hiddens=starter_hiddens
                )
        else:
            self.network = SplitNetworkSimple(
                hidden_sizes=hidden_sizes,
                input_x_size = obs_dim,
                output_size=action_dim,
                num_heads=skill_dim,
                use_std=True
                )
        self.action_dim = action_dim
        self.skill_dim = skill_dim
        self.skill = 0
        self.skill_reset()

        self.log_std = None
        self.std = std
        # if std is None:
        #     last_hidden_size = obs_dim
        #     if len(hidden_sizes) > 0:
        #         last_hidden_size = hidden_sizes[-1] * skill_dim
        #         print("last hidden size: " + str(last_hidden_size))
        #     self.last_fc_log_std = nn.Linear(last_hidden_size, action_dim)
        #     self.last_fc_log_std.weight.data.uniform_(-init_w, init_w)
        #     self.last_fc_log_std.bias.data.uniform_(-init_w, init_w)
        # else:
        if not (std is None):
            self.log_std = np.log(std)
            assert LOG_SIG_MIN <= self.log_std <= LOG_SIG_MAX

    def get_action(self, obs_np, deterministic=False):
        # generate (iters, skill_dim) matrix that stacks one-hot skill vectors
        # online reinforcement learning
        if self.continuous:
            skill_vec = self.skill
        else:
            skill_vec = np.zeros(self.skill_dim)
            skill_vec[self.skill] += 1
        #obs_np = np.concatenate((obs_np, skill_vec), axis=0)
        actions = self.get_actions(obs_np[None], skill_vec[None],deterministic=deterministic)
        return actions[0, :], {"skill": skill_vec}

    def get_actions(self, obs_np, skill, deterministic=False):
        return eval_np(self, obs_np, skill, deterministic=deterministic)[0]

    def skill_reset(self):
        if self.continuous:
            self.skill = np.random.vonmises(0,2,self.skill_dim)
        else:
            self.skill = random.randint(0, self.skill_dim-1)

    def parameters(self):
        return self.network.parameters()

    def forward(
            self,
            obs,
            skill_vec=None,
            reparameterize=True,
            deterministic=False,
            return_log_prob=False,
    ):
        """
        :param obs: Observation
        :param deterministic: If True, do not sample
        :param return_log_prob: If True, return a sample and its log probability
        """
        if type(skill_vec)==type(None):
            raise ValueError("You always need a skill vector")
        #REQUIRES SKILL VEC
        if self.std is None:
            out,log_std, std = self.network(obs,skill_vec,True)
        else:
            out = self.network(obs,skill_vec,True)
            std = self.std
            log_std = self.log_std

        mean = out

        log_prob = None
        entropy = None
        mean_action_log_prob = None
        pre_tanh_value = None
        if deterministic:
            action = torch.tanh(mean)
        else:
            tanh_normal = TanhNormal(mean, std)
            if return_log_prob:
                if reparameterize is True:
                    action, pre_tanh_value = tanh_normal.rsample(
                        return_pretanh_value=True
                    )
                else:
                    action, pre_tanh_value = tanh_normal.sample(
                        return_pretanh_value=True
                    )
                log_prob = tanh_normal.log_prob(
                    action,
                    pre_tanh_value=pre_tanh_value
                )
                log_prob = log_prob.sum(dim=1, keepdim=True)
            else:
                if reparameterize is True:
                    action = tanh_normal.rsample()
                else:
                    action = tanh_normal.sample()

        return (
            action, mean, log_std, log_prob, entropy, std,
            mean_action_log_prob, pre_tanh_value,
        )

class DirichletSkillTanhGaussianPolicy(SkillTanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            skill_dim=10,
            gamma=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            skill_dim=skill_dim,
            **kwargs
        )
        self.gamma = gamma
        self.skill_space = Dirichlet(torch.ones(self.skill_dim))
        self.skill = self.skill_space.sample().cpu().numpy()

    def get_action(self, obs_np, deterministic=False):
        # generate (skill_dim, ) matrix that stacks one-hot skill vectors
        # online reinforcement learning
        obs_np = np.concatenate((obs_np, self.skill), axis=0)
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {"skill": self.skill}

    def skill_reset(self):
        self.skill = self.skill_space.sample().cpu().numpy()

    def alpha_update(self, epoch, tau):
        d_alpha = min(self.gamma+(1-self.gamma)*epoch/tau, 1) * torch.ones(self.skill_dim).cpu()
        self.skill_space = Dirichlet(torch.tensor(d_alpha))

    def alpha_reset(self):
        self.skill_space = Dirichlet(torch.ones(self.skill_dim))

class RandomSkillTanhGaussianPolicy(SkillTanhGaussianPolicy):
    def __init__(
            self,
            hidden_sizes,
            obs_dim,
            action_dim,
            std=None,
            init_w=1e-3,
            skill_dim=10,
            gamma=1e-3,
            **kwargs
    ):
        super().__init__(
            hidden_sizes=hidden_sizes,
            obs_dim=obs_dim,
            action_dim=action_dim,
            std=std,
            init_w=init_w,
            skill_dim=skill_dim,
            **kwargs
        )

    def get_action(self, obs_np, deterministic=False):
        # generate (skill_dim, ) matrix that stacks one-hot skill vectors
        # online reinforcement learning
        obs_np = np.concatenate((obs_np, self.skill), axis=0)
        actions = self.get_actions(obs_np[None], deterministic=deterministic)
        return actions[0, :], {"skill": self.skill}

    def skill_reset(self):
        pass

class MakeDeterministic(Policy):
    def __init__(self, stochastic_policy):
        self.stochastic_policy = stochastic_policy

    def get_action(self, observation):
        return self.stochastic_policy.get_action(observation,
                                                 deterministic=True)
