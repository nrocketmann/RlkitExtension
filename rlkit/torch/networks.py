"""
General networks for pytorch.

Algorithm-specific networks should go else-where.
"""
import torch
from torch import nn as nn
from torch.nn import functional as F

from rlkit.policies.base import Policy
from rlkit.torch import pytorch_util as ptu
from rlkit.torch.core import eval_np
from rlkit.torch.data_management.normalizer import TorchFixedNormalizer
from rlkit.torch.modules import LayerNorm

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

def identity(x):
    return x


class Mlp(nn.Module):
    def __init__(
            self,
            hidden_sizes,
            output_size,
            input_size,
            init_w=3e-3,
            hidden_activation=F.relu,
            output_activation=identity,
            hidden_init=ptu.fanin_init,
            b_init_value=0.1,
            layer_norm=False,
            layer_norm_kwargs=None,
    ):
        super().__init__()

        if layer_norm_kwargs is None:
            layer_norm_kwargs = dict()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.layer_norm = layer_norm
        self.fcs = []
        self.layer_norms = []
        in_size = input_size

        for i, next_size in enumerate(hidden_sizes):
            fc = nn.Linear(in_size, next_size)
            in_size = next_size
            hidden_init(fc.weight)
            fc.bias.data.fill_(b_init_value)
            self.__setattr__("fc{}".format(i), fc)
            self.fcs.append(fc)

            if self.layer_norm:
                ln = LayerNorm(next_size)
                self.__setattr__("layer_norm{}".format(i), ln)
                self.layer_norms.append(ln)

        self.last_fc = nn.Linear(in_size, output_size)
        self.last_fc.weight.data.uniform_(-init_w, init_w)
        self.last_fc.bias.data.uniform_(-init_w, init_w)

    def forward(self, input, return_preactivations=False, return_last_hidden = False):
        h = input
        for i, fc in enumerate(self.fcs):
            h = fc(h)
            if self.layer_norm and i < len(self.fcs) - 1:
                h = self.layer_norms[i](h)
            h = self.hidden_activation(h)
        preactivation = self.last_fc(h)
        output = self.output_activation(preactivation)
        if return_last_hidden:
            return output, h
        if return_preactivations:
            return output, preactivation
        else:
            return output


class FlattenMlp(Mlp):
    """
    Flatten inputs along dimension 1 and then pass through MLP.
    """

    def forward(self, *inputs, **kwargs):
        flat_inputs = torch.cat(inputs, dim=1)
        return super().forward(flat_inputs, **kwargs)


class MlpPolicy(Mlp, Policy):
    """
    A simpler interface for creating policies.
    """

    def __init__(
            self,
            *args,
            obs_normalizer: TorchFixedNormalizer = None,
            **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.obs_normalizer = obs_normalizer

    def forward(self, obs, **kwargs):
        if self.obs_normalizer:
            obs = self.obs_normalizer.normalize(obs)
        return super().forward(obs, **kwargs)

    def get_action(self, obs_np):
        actions = self.get_actions(obs_np[None])
        return actions[0, :], {}

    def get_actions(self, obs):
        return eval_np(self, obs)


class TanhMlpPolicy(MlpPolicy):
    """
    A helper class since most policies have a tanh output activation.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, output_activation=torch.tanh, **kwargs)


#Can be used for multiple independent policies, or multiple independent Q functions
#Takes skill vector and X inputs
#for a Q function, X would be action and state put together, for a policy it would be just state
class SplitNetworkSimple(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 input_x_size,
                 num_heads,
                 use_std=False
                 ):
        super(SplitNetworkSimple,self).__init__()
        self.num_heads = num_heads
        self.mlps = []
        self.output_size = output_size
        if use_std:
            for i in range(2*num_heads):
                self.mlps.append(FlattenMlp(hidden_sizes,output_size,input_x_size))
        else:
            for i in range(num_heads):
                self.mlps.append(FlattenMlp(hidden_sizes,output_size,input_x_size))
        self.use_std= use_std

    def forward(self,input_x, input_skill, return_hidden = False):
        #input_x shape: batch x inp_dim
        #skill shape: batch x skill_dim=num_heads

        mlp_results = torch.stack([mlp(input_x) for mlp in self.mlps],dim=0).transpose(1,0) #shape batch x num_heads x output_dim
        if self.use_std:
            skill_stretch = input_skill.unsqueeze(-1)
            mean_results = mlp_results[:,:self.num_heads,:]
            std_results = mlp_results[:, self.num_heads:, :]
            composition_mean = (skill_stretch * mean_results).sum(1)  # shape batch x output_dim
            composition_mean = composition_mean / skill_stretch.sum(1)
            composition_std = (skill_stretch * std_results).sum(1)  # shape batch x output_dim
            log_std = composition_std / skill_stretch.sum(1)
            log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
            std = torch.exp(log_std)
            return composition_mean, log_std, std
        else:
            skill_stretch = input_skill.unsqueeze(-1) # shape batch x heads x 1
            composition = (skill_stretch * mlp_results).sum(1)  # shape batch x output_dim
            composition = composition / skill_stretch.sum(1)
            return composition

    def parameters(self):
        for mlp in self.mlps:
            for w in mlp.parameters():
                yield w

#Can be used for multiple independent policies, or multiple independent Q functions
#Takes skill vector and X inputs
#for a Q function, X would be action and state put together, for a policy it would be just state
class SplitNetworkShared(SplitNetworkSimple):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 input_x_size,
                 num_heads,
                 use_std=False,
                 starter_hiddens=[512]
                 ):
        super(SplitNetworkShared,self).__init__(hidden_sizes,output_size,starter_hiddens[-1],num_heads,use_std)
        self.shared_layers = FlattenMlp(starter_hiddens[:-1],starter_hiddens[-1],input_x_size)

    def forward(self,input_x, input_skill, return_hidden = False):
        hidden_rep = self.shared_layers(input_x)
        outputs = super(SplitNetworkShared,self).forward(hidden_rep,input_skill)
        return outputs

#https://math.stackexchange.com/questions/1246358/the-product-of-multiple-univariate-gaussians
#Split network with attention composition
#product of gaussians equation given above
#REQUIRES STD
class SplitNetworkAttention(nn.Module):
    def __init__(self,
                 hidden_sizes,
                 output_size,
                 input_x_size,
                 num_heads,
                 starter_hiddens=[512],
                 attention_dim = 128
                 ):
        super(SplitNetworkAttention,self).__init__()
        self.num_heads = num_heads
        self.mlps = []
        self.output_size = output_size
        self.shared_layers = FlattenMlp(starter_hiddens[:-1], starter_hiddens[-1], input_x_size)
        for i in range(num_heads):
            self.mlps.append(FlattenMlp(hidden_sizes,output_size*2,starter_hiddens[-1]).to('cuda:0'))

        self.key_nn = nn.Linear(num_heads,attention_dim)
        self.query_nn = nn.Linear(hidden_sizes[-1], attention_dim)

    def forward(self,input_x, input_skill, return_hidden = False):
        #input_x shape: batch x inp_dim
        #skill shape: batch x skill_dim=num_heads
        hidden_rep = self.shared_layers(input_x)
        mlp_outputs = [mlp(hidden_rep, return_last_hidden=True) for mlp in self.mlps]
        mlp_results = torch.stack([x[0] for x in mlp_outputs],dim=0).transpose(1,0) #shape batch x num_heads*2 x output_dim
        mlp_hiddens = torch.stack([x[1] for x in mlp_outputs],dim=0).transpose(1,0)

        mean_results = mlp_results[:,:,:self.output_size]#get all means (batch x num_heads x output_dim)
        std_results = mlp_results[:, :, self.output_size:]

        attention_queries = nn.functional.normalize(self.query_nn(mlp_hiddens)) #shape batch x num_heads x attention_dim
        attention_keys = nn.functional.normalize(self.key_nn(input_skill)).unsqueeze(-1) #shape batch x num_heads x 1
        attention_weights = torch.softmax(torch.matmul(attention_queries, attention_keys),dim=1) #batch x num_heads
        modified_stds = std_results/torch.sqrt(attention_weights) #(batch x num_heads x output_dim)


        # https://math.stackexchange.com/questions/1246358/the-product-of-multiple-univariate-gaussians
        final_stds = torch.reciprocal(torch.sqrt(torch.sum(torch.reciprocal(torch.pow(modified_stds,2)),dim=1))) #sum over all heads #batch_size x output_dim
        final_means = torch.pow(final_stds,2) * torch.sum(torch.reciprocal(torch.pow(modified_stds,2)) * mean_results) #same shape

        return final_means, torch.log(final_stds), final_stds

    def parameters(self):
        for mlp in self.mlps:
            for w in mlp.parameters():
                yield w
        for w in self.shared_layers.parameters():
            yield w
        for w in self.key_nn.parameters():
            yield w
        for w in self.query_nn.parameters():
            yield w