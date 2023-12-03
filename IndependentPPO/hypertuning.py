import logging
import sys
import warnings

import optuna
import abc


class OptunaOptimizer(abc.ABC):
    def __init__(self, direction, study_name=None, save=None, n_trials=1, pruner=None):
        self.study_name = study_name
        self.save = save
        self.n_trials = n_trials
        self.direction = direction
        if isinstance(direction, list):
            raise NotImplementedError("Multi-objective optimization is not implemented in this class")
        self.pruner = pruner
        # optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler(sys.stdout))

        if self.study_name is None:
            self.study_name = "study_noname"
        if self.save is None:
            self.save = self.study_name
            self.storage = f'sqlite:///{self.save}/database.db'
        else:
            self.storage = f'sqlite:///{self.save}/{self.study_name}/database.db'
        self.save += f"/{self.study_name}"
        try:
            # Try to load the existing study.
            self.study = optuna.load_study(study_name=self.study_name, storage=self.storage, pruner=pruner)
            print(f"Loaded existing study '{study_name}' with {len(self.study.trials)} trials.")
        except:
            # If the study does not exist, create a new one.
            import os

            os.makedirs(self.save, exist_ok=True)
            # Create file
            f = open(f"{self.save}/database.db", "w+")
            # close file
            f.close()
            if isinstance(direction, str):
                self.study = optuna.create_study(direction=self.direction, study_name=self.study_name,
                                                 storage=self.storage, pruner=pruner)
                print(f"Created new study '{self.study_name}'.")

    @abc.abstractmethod
    def objective(self, trial):
        raise NotImplementedError("Subclasses should implement this method.")

    def pre_trial_callback(self):
        pass

    def pre_objective_callback(self, trial):
        pass

    def post_trial_callback(self, trial, value):
        pass

    def optimize(self):
        for i in range(self.n_trials):
            self.pre_trial_callback()
            trial = self.study.ask()
            self.pre_objective_callback(trial)
            try:
                value = self.objective(trial)
            except optuna.TrialPruned as e:
                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue
            except Exception as e:
                # Handle other exceptions
                self.study.tell(trial, state=optuna.trial.TrialState.FAIL)
                continue

            self.study.tell(trial, value)
            print(
                f"Trial {i} completed with value {value}. Best value is {self.study.best_value} of trial {self.study.best_trial}.")
            self.post_trial_callback(trial, value)

    def best_params(self):
        return self.study.best_params

    def best_value(self):
        return self.study.best_value


class OptunaOptimizeMultiObjective(abc.ABC):
    def __init__(self, direction, study_name=None, save=None, n_trials=1):
        self.study_name = study_name
        self.save = save
        self.n_trials = n_trials
        self.direction = direction
        if isinstance(direction, str):
            raise NotImplementedError("Single-objective optimization is not implemented in this class")

        if self.study_name is None:
            self.study_name = "study_noname"
        if self.save is None:
            self.save = self.study_name
            self.storage = f'sqlite:///{self.save}/database.db'
        else:
            self.storage = f'sqlite:///{self.save}/{self.study_name}/database.db'
        self.save += f"/{self.study_name}"
        try:
            # Try to load the existing study.
            self.study = optuna.load_study(study_name=self.study_name, storage=self.storage)
            print(f"Loaded existing study '{study_name}' with {len(self.study.trials)} trials.")
        except:
            # If the study does not exist, create a new one.
            import os

            os.makedirs(self.save, exist_ok=True)
            # Create file
            f = open(f"{self.save}/database.db", "w+")
            # close file
            f.close()
            self.study = optuna.create_study(directions=self.direction, study_name=self.study_name, storage=self.storage)
            print(f"Created new study '{self.study_name}'.")

    @abc.abstractmethod
    def objective(self, trial):
        raise NotImplementedError("Subclasses should implement this method.")

    def optimize(self):
        for _ in range(self.n_trials):  # Number of iterations you want to control
            # Run a single trial per iteration

            self.study.optimize(self.objective, n_trials=1)

            last_trial = self.study.trials[-1]

            # Now you can use last_trial for further logic
            # For example, print its number and values
            print(f"Trial number: {last_trial.number}, Values: {last_trial.values}")


