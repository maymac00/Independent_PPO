"""Parser

"""
import argparse
import json
from IndependentPPO.utils.misc import str2bool


class CustomArgumentParser(argparse.ArgumentParser):
    def parse_args(self, args=None, namespace=None):
        result = super().parse_args(args, namespace)
        for key, value in vars(result).copy().items():
            if value is None:
                delattr(result, key)
        return result


def get_parser():
    pass


def parse_ppo_args():
    parser = argparse.ArgumentParser()

    # Logging
    parser.add_argument("--tag", type=str, default='IPPO_clip_gaussian', help="Training tag")

    # Environment
    parser.add_argument("--n-agents", type=int, default=2, help="N° of agents")
    parser.add_argument("--env", type=str, default="CommonsGame-v0", help="Gym environment")
    parser.add_argument("--seed", type=int, default=1, help="seed of the experiment")

    # Experiment
    parser.add_argument("--max-steps", type=int, default=500, help="Max n° of steps per episode")
    parser.add_argument("--n-steps", type=int, default=2500, help="Steps between policy updates")
    parser.add_argument("--tot-steps", type=int, default=50000000, help="Total timesteps of the experiment")
    parser.add_argument("--early-stop", type=float, default=50000000, help="Steps to stop the experiment early")
    parser.add_argument("--save-dir", type=str, default="/Gathering_data",
                        help="Directory to save the model and metrics")

    parser.add_argument("--past-actions-memory", type=int, default=0, help="Number of past actions to remember")

    # Algorithm
    parser.add_argument("--clip", type=float, default=0.2, help="Surrogate clipping coefficient")
    parser.add_argument("--target-kl", type=float, default=None, help="Target KL divergence threshold")
    parser.add_argument("--gamma", type=float, default=0.8, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="Gae lambda")
    parser.add_argument("--ent-coef", type=float, default=0.2, help="Entropy coefficient")
    parser.add_argument("--v-coef", type=float, default=0.5, help="Value function coefficient")
    parser.add_argument("--parallelize", type=str2bool, default=True, help="Parallelize the environment")
    parser.add_argument("--n-envs", type=int, default=5, help="Parallelize the environment")
    parser.add_argument("--clip-vloss", type=str2bool, default=True, help="Clip value loss")

    # Update
    parser.add_argument("--actor-lr", type=float, default=3e-4, help="Actor lr")
    parser.add_argument("--critic-lr", type=float, default=5e-3, help="Critic lr")
    parser.add_argument("--anneal-lr", type=str2bool, default=True, help="Toggles annealing learning rates")
    parser.add_argument("--n-epochs", type=int, default=10, help="N° of update epochs")
    parser.add_argument("--norm-adv", type=str2bool, default=True, help="Toggles advantages normalization")
    parser.add_argument("--max-grad-norm", type=float, default=1., help="Maximum norm for gradient clipping")
    parser.add_argument("--critic-times", type=int, default=2, help="Multiplicator for number of epochs of the critic")

    parser.add_argument("--anneal-entropy", type=str2bool, default=True, help="Toggles annealing entropy coefficient")
    parser.add_argument("--concavity-entropy", type=float, default=3.5, help="Sets concavity of entropy coefficient")

    # Lagrangian PPO
    parser.add_argument("--constr-limit-1", type=float, default=2, help="Sets the limit of the constraint 1")
    parser.add_argument("--constr-limit-2", type=float, default=2, help="Sets the limit of the constraint 2")
    parser.add_argument("--mult-lr", type=float, default=0.035, help="Sets initial value of the learning rate for the lagrangian multiplier")
    parser.add_argument("--mult-init", type=float, default=0.5, help="Sets initial value of the lagrangian multiplier")

    # DNN
    parser.add_argument("--h-size", type=int, default=128, help="Layers size")
    parser.add_argument("--h-layers", type=int, default=2, help="Number of layers")

    # Metrics
    parser.add_argument("--last-n", type=int, default=500, help="Average last n metrics")

    # Torch
    parser.add_argument("--n-cpus", type=int, default=8, help="N° of cpus/max threads for process")
    parser.add_argument("--th-deterministic", type=str2bool, default=True,
                        help="Toggles for `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=str2bool, default=False, help="Toggles cuda")  # check .to(device) for GPUs

    args, unknown = parser.parse_known_args()
    # Unkown arguments
    if unknown:
        print(f"Unknown arguments: {unknown}")

    # Add unknown arguments to the args namespace
    for arg in unknown:
        # Split the argument on '=' sign if it's in key=value format
        if '=' in arg:
            key, value = arg.split('=', 1)
            setattr(args, key.lstrip('-'), value)
        else:
            # For flag-like unknown arguments
            setattr(args, arg.lstrip('-'), True)
    # args.batch_size = int(args.n_envs * args.n_steps)     # if you plan to use parallel envs
    args.batch_size = int(1 * args.n_steps)  # otherwise

    return args


def args_from_json(directory):
    def read_json_file(file_path):
        with open(file_path, "r") as file:
            data = json.load(file)
        return data

    # Read the JSON file and store it in a dictionary
    config_data = read_json_file(directory)

    # Create an empty argparse.Namespace object
    json_args = argparse.Namespace()

    # Update the argparse.Namespace object with the dictionary
    for key, value in config_data.items():
        setattr(json_args, key, value)

    # Parse arguments
    parser = CustomArgumentParser()
    # Logging
    parser.add_argument("--tag", type=str, help="Training tag")

    # Environment
    parser.add_argument("--n-agents", type=int, help="N° of agents")
    parser.add_argument("--env", type=str, help="Gym environment")
    parser.add_argument("--seed", type=int, help="seed of the experiment")

    # Experiment
    parser.add_argument("--max-steps", type=int, help="Max n° of steps per episode")
    parser.add_argument("--n-steps", type=int, help="Steps between policy updates")
    parser.add_argument("--tot-steps", type=int, help="Total timesteps of the experiment")
    parser.add_argument("--early-stop", type=float, help="Steps to stop the experiment early")
    parser.add_argument("--save-dir", type=str,
                        help="Directory to save the model and metrics")

    parser.add_argument("--past-actions-memory", type=int, help="Number of past actions to remember")

    # Algorithm
    parser.add_argument("--clip", type=float, help="Surrogate clipping coefficient")
    parser.add_argument("--target-kl", type=float, help="Target KL divergence threshold")
    parser.add_argument("--gamma", type=float, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, help="Gae lambda")
    parser.add_argument("--ent-coef", type=float, help="Entropy coefficient")
    parser.add_argument("--v-coef", type=float, help="Value function coefficient")
    parser.add_argument("--parallelize", type=str2bool, help="Parallelize the environment")
    parser.add_argument("--n-envs", type=int, help="Parallelize the environment")
    parser.add_argument("--clip-vloss", type=str2bool, default=True, help="Clip value loss")

    # Update
    parser.add_argument("--actor-lr", type=float, help="Actor lr")
    parser.add_argument("--critic-lr", type=float, help="Critic lr")
    parser.add_argument("--anneal-lr", type=str2bool, help="Toggles annealing learning rates")
    parser.add_argument("--n-epochs", type=int, help="N° of update epochs")
    parser.add_argument("--norm-adv", type=str2bool, help="Toggles advantages normalization")
    parser.add_argument("--max-grad-norm", type=float, help="Maximum norm for gradient clipping")
    parser.add_argument("--critic-times", type=int, help="Multiplicator for number of epochs of the critic")

    parser.add_argument("--anneal-entropy", type=str2bool, default=True, help="Toggles annealing entropy coefficient")
    parser.add_argument("--concavity-entropy", type=float, default=3.5, help="Sets concavity of entropy coefficient")

    # Lagrangian PPO
    parser.add_argument("--constr-lim-1", type=float, default=2, help="Sets the limit of the constraint 1")
    parser.add_argument("--constr-lim-2", type=float, default=2, help="Sets the limit of the constraint 2")
    parser.add_argument("--mult-lr", type=float, default=0.035, help="Sets initial value of the learning rate for the lagrangian multiplier")
    parser.add_argument("--mult-init", type=float, default=0.5, help="Sets initial value of the lagrangian multiplier")

    # DNN
    parser.add_argument("--h-size", type=int, help="Layers size")
    parser.add_argument("--h-layers", type=int, help="Number of layers")

    # Metrics
    parser.add_argument("--last-n", type=int, help="Average last n metrics")

    # Torch
    parser.add_argument("--n-cpus", type=int, help="N° of cpus/max threads for process")
    parser.add_argument("--th-deterministic", type=str2bool,
                        help="Toggles for `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=str2bool, help="Toggles cuda")  # check .to(device) for GPUs

    cmd_args, unknown = parser.parse_known_args()

    # compare namespace objects and get cmd_args over json_args
    args = vars(json_args)
    for key, value in vars(cmd_args).items():
        if value is not None:
            args[key] = value

    # Unkown arguments
    if unknown:
        print(f"Unknown arguments: {unknown}")

    # Add unknown arguments to the args namespace
    for arg in unknown:
        # Split the argument on '=' sign if it's in key=value format
        if '=' in arg:
            key, value = arg.split('=', 1)
            setattr(args, key.lstrip('-'), value)
        else:
            # For flag-like unknown arguments
            setattr(args, arg.lstrip('-'), True)

    args = argparse.Namespace(**args)
    args.batch_size = int(1 * args.n_steps)
    return args


def write_to_json(directory, args):
    with open(directory + "/config.json", "w") as f:
        json.dump(vars(args), f, indent=4)
