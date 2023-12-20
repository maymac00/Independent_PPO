import logging
import sys
import warnings

import numpy as np
import optuna
import abc


class DecreasingCandidatesTPESampler(optuna.samplers.TPESampler):
    """
    A TPE sampler that decreases the number of candidates for the expected improvement calculation over time. With
    fewer candidates, there's a higher chance that the selected hyperparameters will be close to the ones that have
    already performed well. This is because fewer samples mean less chance of capturing outliers or less common
    values from the distributions. The process leans more towards exploitation, focusing on refining and exploiting
    the known good regions of the hyperparameter space.
    """
    def __init__(self, initial_n_ei_candidates=24, **kwargs):
        super().__init__(**kwargs)
        self.initial_n_ei_candidates = initial_n_ei_candidates
        self.n_ei_candidates = initial_n_ei_candidates
        self.trial_count = 0

    def sample_relative(self, study, trial, search_space):
        # Increase the number of candidates by 1 for each trial
        self.trial_count += 1
        self.n_ei_candidates = max(self.initial_n_ei_candidates - self.trial_count, 1)

        # Temporarily set the sampler's n_ei_candidates to the updated value
        original_n_ei_candidates = self._n_ei_candidates
        self._n_ei_candidates = self.n_ei_candidates

        try:
            # Sample the next set of parameters
            return super().sample_relative(study, trial, search_space)
        finally:
            # Restore the original n_ei_candidates value
            self._n_ei_candidates = original_n_ei_candidates


class OptunaOptimizer(abc.ABC):
    def __init__(self, direction, study_name=None, save=None, n_trials=1, pruner=None, sampler=None, **kwargs):
        self.study_name = study_name
        self.save = save
        self.n_trials = n_trials
        self.direction = direction
        if isinstance(direction, list):
            raise NotImplementedError("Multi-objective optimization is not implemented in this class")
        if pruner is None:
            self.pruner = optuna.pruners.NopPruner()
        self.pruner = pruner
        if sampler is None:
            self.sampler = optuna.samplers.TPESampler()
        self.sampler = sampler
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
            self.study = optuna.load_study(study_name=self.study_name, storage=self.storage, pruner=pruner, sampler=sampler, **kwargs)
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
                                                 storage=self.storage, pruner=pruner, sampler=sampler, **kwargs)
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
                print(f"Trial {i} failed with exception {e}.")
                continue

            # Check that value is scalar
            if isinstance(value, list) or isinstance(value, tuple) or isinstance(value, np.ndarray):
                raise ValueError(f"Objective function returned a list or tuple instead of a scalar value: {value}")

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
            self.study = optuna.create_study(directions=self.direction, study_name=self.study_name,
                                             storage=self.storage)
            print(f"Created new study '{self.study_name}'.")

    @abc.abstractmethod
    def objective(self, trial):
        raise NotImplementedError("Subclasses should implement this method.")

    def optimize(self):
        for _ in range(self.n_trials):  # Number of iterations you want to control
            # Run a single trial per iteration

            self.study.optimize(self.objective, n_trials=1)

            last_trial = self.study.trials[-1]
