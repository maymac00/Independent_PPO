from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

from ActionSelection import *
from utils.misc import *

ACTIONS = [0, 1, 2, 3, 6, 8, 9]


def Linear(input_dim, output_dim, act_fn='leaky_relu', init_weight_uniform=True):
    """
    Creat a linear layer.

    Parameters
    ----------
    input_dim : int
        The input dimension.
    output_dim : int
        The output dimension.
    act_fn : str
        The activation function.
    init_weight_uniform : bool
        Whether uniformly sample initial weights.
    """
    gain = th.nn.init.calculate_gain(act_fn)
    fc = th.nn.Linear(input_dim, output_dim)
    if init_weight_uniform:
        nn.init.xavier_uniform_(fc.weight, gain=gain)
    else:
        nn.init.xavier_normal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc


class SoftmaxActor(nn.Module):
    action_selection = no_filter

    def __init__(self, o_size: int, a_size: int, h_size: int, h_layers: int, eval=False):
        super().__init__()
        self.hidden = [None] * h_layers
        self.hidden[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(h_size, h_size, act_fn='tanh')

        self.hidden = nn.ModuleList(self.hidden)
        self.output = nn.Linear(h_size, a_size)
        self.eval_mode = eval

    def forward(self, x):
        for i in range(len(self.hidden)):
            l = self.hidden[i]
            x = th.tanh(l(x))
        return F.softmax(self.output(x), dim=-1)

    def get_action(self, x, action=None):
        prob = self.forward(x)
        env_action, action, logprob, entropy = self.get_action_data(prob, action)
        if self.eval_mode:
            return env_action, action, logprob[0][action], entropy
        else:
            return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy

    def get_action_data(self, prob, action=None):
        env_action = None
        if action is None:
            if not self.eval_mode:
                action = th.multinomial(prob, 1)
            else:
                action = SoftmaxActor.action_selection(np.array(prob, dtype='float64').squeeze())
            env_action = ACTIONS[action]

        logprob = th.log(prob)
        entropy = -(prob * logprob).sum(-1)
        return env_action, action, logprob, entropy


class Critic(nn.Module):
    def __init__(self, o_size: int, h_size: int, h_layers: int):
        super().__init__()
        self.hidden = [None] * h_layers
        self.hidden[0] = Linear(o_size, h_size, act_fn='tanh')
        for i in range(1, len(self.hidden)):
            self.hidden[i] = Linear(h_size, h_size, act_fn='tanh')
        self.hidden = nn.ModuleList(self.hidden)
        self.output = Linear(h_size, 1, act_fn='linear')

    def forward(self, x):
        for i in range(len(self.hidden)):
            l = self.hidden[i]
            x = F.leaky_relu(l(x))
        return self.output(x)


class RecSoftmaxActor(SoftmaxActor):
    def __init__(self, o_size: int, a_size: int, h_size: int):
        super().__init__(o_size, a_size, h_size)

        self.gru = nn.GRU(h_size, h_size, batch_first=True)

    def forward(self, x, h):
        x = F.tanh(self.hidden_1(x))
        x, h_ = self.gru(x, h)
        x = F.tanh(self.hidden_2(x))
        return F.softmax(self.output(x), dim=-1), h_

    def get_action(self, x, h=None, action=None):
        prob, h_ = self.forward(x, h)

        env_action, action, logprob, entropy = self.get_action_data(prob, action)

        return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy, h_


# https://ojs.aaai.org/index.php/AAAI/article/view/21171
class RecCritic(Critic):
    def __init__(self, o_size: int, h_size: int):
        super().__init__(o_size, h_size)

        self.gru = nn.GRU(h_size, h_size, batch_first=True)

    def forward(self, x, h=None):
        x = F.leaky_relu(self.hidden_1(x))
        x, h_ = self.gru(x, h)
        x = F.leaky_relu(self.hidden_2(x))
        return self.output(x), h_
