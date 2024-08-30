from torch.utils.tensorboard import SummaryWriter

from .CallbackI import UpdateCallback

# This is just for the agent. It is not intended to be for logging overall performance of the agent.
class TensorBoardCallback(UpdateCallback):

    def __init__(self, ippo, log_dir, freq=1):
        super().__init__(ippo)
        self.log_dir = log_dir
        self.freq = freq

        self.writer = SummaryWriter(log_dir=log_dir+"/"+ippo.run_name)
        # TODO Fix this for ippo
        """self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % (
                "\n".join([f"|{key}|{value}|" for key, value in vars(self.agent.cmd_args).items()])),
        )"""

    def after_update(self):
        for i, agent in enumerate(self.ippo.agents):
            if agent.run_metrics["update_count"] % self.freq == 0:
                for k, v in agent.actor_metrics.items():
                    self.writer.add_scalar(f"Agent_{i}/"+k, v, agent.run_metrics["update_count"])
                for k, v in agent.critic_metrics.items():
                    self.writer.add_scalar(f"Agent_{i}/"+k, v, agent.run_metrics["update_count"])
                for k, v in agent.global_metrics.items():
                    self.writer.add_scalar(f"Agent_{i}/"+k, v, agent.run_metrics["update_count"])

    def before_update(self):
        pass

