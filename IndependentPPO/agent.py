import torch as th
from torch import optim
from torch.distributions import Normal
import torch.nn as nn
import torch.nn.functional as F

from ActionSelection import *
from utils.misc import *

ACTIONS = [0, 1, 2, 3, 4, 5, 6]


class Agent:
    def __init__(self, actor, critic, actor_lr, critic_lr, filter=None):
        self.actor = actor
        self.critic = critic
        self.a_optimizer = optim.Adam(list(self.actor.parameters()), lr=actor_lr, eps=1e-5)
        self.c_optimizer = optim.Adam(list(self.critic.parameters()), lr=critic_lr, eps=1e-5)
        self.device = next(self.actor.parameters()).device
        assert self.device == next(self.critic.parameters()).device

        self.save_dir = None

    def freeze(self):
        self.actor.freeze()
        self.critic.freeze()

    def unfreeze(self):
        self.actor.unfreeze()
        self.critic.unfreeze()

    def isFrozen(self):
        return not self.actor.parameters().__next__().requires_grad

    def getId(self):
        # Hash the sum of the layers of both actor and critic
        s = sum([self.actor.hidden[i].weight.sum() for i in range(len(self.actor.hidden))])
        s += sum([self.critic.hidden[i].weight.sum() for i in range(len(self.critic.hidden))])
        return hash(float(s))

    def predict(self, x):
        return self.actor.predict(x)

    def __str__(self):
        return str(self.getId())


class LagrAgent(Agent):
    # A Lagrangian Agent is an agent with an additional critic for the cost value function, a lagrangian multiplier for the constraint, and its learning rate
    def __init__(self, actor, critic, critic_cost_1, critic_cost_2, actor_lr, critic_lr, constr_limit_1, constr_limit_2,
                 mult_lr, mult_init=0.5):
        super().__init__(actor, critic, actor_lr, critic_lr)

        # Setting up the cost value functions
        self.critic_cost_1 = critic_cost_1
        self.c_cost_optimizer_1 = optim.Adam(list(self.critic_cost_1.parameters()), lr=critic_lr, eps=1e-5)
        self.critic_cost_2 = critic_cost_2
        self.c_cost_optimizer_2 = optim.Adam(list(self.critic_cost_2.parameters()), lr=critic_lr,
                                             eps=1e-5)  # We usually set the same lr for the critics

        # Setting up the Lagrangian multipliers
        self.lag_mul_1 = th.tensor(mult_init, requires_grad=True, device=self.device)
        self.constr_limit_1 = constr_limit_1
        self.lag_optimizer_1 = optim.Adam([self.lag_mul_1], lr=mult_lr)
        self.lag_mul_2 = th.tensor(mult_init, requires_grad=True, device=self.device)
        self.constr_limit_2 = constr_limit_2
        self.lag_optimizer_2 = optim.Adam([self.lag_mul_2], lr=mult_lr)

    def freeze(self):
        super().freeze()
        self.critic_cost_1.freeze()
        self.critic_cost_2.freeze()

    def unfreeze(self):
        super().unfreeze()
        self.critic_cost_1.unfreeze()
        self.critic_cost_2.unfreeze()


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
        nn.init.orthogonal_(fc.weight, gain=gain)
    nn.init.constant_(fc.bias, 0.00)
    return fc


class SoftmaxActor(nn.Module):
    eval_action_selection = FilterSoftmaxActionSelection(ACTIONS, threshold=0.1)
    action_selection = SoftmaxActionSelection(ACTIONS)

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
        return env_action, action, logprob.gather(-1, action.to(th.int64)).squeeze(), entropy

    def get_action_data(self, prob, action=None):
        env_action = None
        if action is None:
            action, env_action = self.select_action(np.array(prob, dtype='float64').squeeze())
            action = th.tensor(action)

        logprob = th.log(prob)
        entropy = -(prob * logprob).sum(-1)
        return env_action, action, logprob, entropy

    def predict(self, x):
        if not self.eval_mode:
            raise ValueError("Cannot predict in training mode")
        # Check if its a tensor
        if not isinstance(x, th.Tensor):
            x = th.tensor(x, dtype=th.float32)
        with th.no_grad():
            prob = self.forward(x)
            action, env_action = self.select_action(np.array(prob, dtype='float64').squeeze())
        return env_action

    def select_action(self, probs):
        if self.eval_mode:
            return SoftmaxActor.eval_action_selection.action_selection(probs)
        else:
            return SoftmaxActor.action_selection.action_selection(probs)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True


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

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        for param in self.parameters():
            param.requires_grad = True

class LexicCritic(Critic):
    def __init__(self, o_size: int, reward_size: int, h_size: int, h_layers: int):
        super().__init__(o_size, h_size, h_layers)
        self.output = Linear(h_size, reward_size, act_fn='linear')  # We output a vector of rewards


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
