import argparse
import os
from itertools import product
from typing import List, Dict, Any
import yaml

# Local imports
from ActiveLearning import ActiveLearningModel
from Subtask import Subtask
from HighLevelSelectionStrategy import (
    EpsilonGreedyStrategy,
    GreedyStrategy,
    ProbabilisticSamplingStrategy
)
from ImportanceCalculationStrategy import (
    EMAStrategy,
    MaxMeanStrategy,
    MaxStdStrategy,
    MaxRatioStrategy,
    MaxProductStrategy
)

def create_selection_strategy(strategy_name: str, params: Dict[str, Any]):
    if strategy_name == 'EpsilonGreedy':
        return EpsilonGreedyStrategy(**params)
    elif strategy_name == 'Greedy':
        return GreedyStrategy()
    elif strategy_name == 'ProbabilisticSampling':
        return ProbabilisticSamplingStrategy()
    else:
        raise ValueError(f"Unknown selection strategy: {strategy_name}")

def create_importance_strategy(strategy_name: str, params: Dict[str, Any]):
    if strategy_name == 'EMA':
        return EMAStrategy(**params)
    elif strategy_name == 'MaxMean':
        return MaxMeanStrategy(**params)
    elif strategy_name == 'MaxStd':
        return MaxStdStrategy(**params)
    elif strategy_name == 'MaxRatio':
        return MaxRatioStrategy(**params)
    elif strategy_name == 'MaxProduct':
        return MaxProductStrategy(**params)
    else:
        raise ValueError(f"Unknown importance strategy: {strategy_name}")

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run Active Learning Model Experiments")
    parser.add_argument('-c', '--config', type=str, required=True, help='Path to the configuration file')
    parser.add_argument('--save_dir', type=str, required=True, help='Directory to save the experiments')
    return parser.parse_args()

def run_experiment(subtasks: List[Subtask], selection_strategy: str, importance_strategy: str, 
                   global_config: Dict[str, Any], save_dir: str):
    selection_strat = create_selection_strategy(
        selection_strategy,
        global_config['selection_strategy_params']
    )
    importance_strat = create_importance_strategy(
        importance_strategy,
        global_config['importance_strategy_params']
    )

    active_learning_model = ActiveLearningModel(
        subtasks=subtasks,
        selection_strategy=selection_strat,
        importance_strategy=importance_strat,
        init_explore_size=global_config['init_explore_size'],
        loss_threshold=global_config['loss_threshold'],
        max_train_steps=global_config['max_train_steps']
    )

    active_learning_model.run()

    experiment_dir = os.path.join(save_dir, f"{selection_strategy}_{importance_strategy}")
    active_learning_model.save(experiment_dir)

    print(f"Experiment completed: {selection_strategy} - {importance_strategy}")

def main():
    args = parse_arguments()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    global_config = config['global']
    Subtask.set_random_state(global_config['random_seed'])

    subtasks = Subtask.build_subtasks_from_config(args.config)

    selection_strategies = global_config['selection_strategies']
    importance_strategies = global_config['importance_strategies']

    for selection_strategy, importance_strategy in product(selection_strategies, importance_strategies):
        run_experiment(subtasks, selection_strategy, importance_strategy, global_config, args.save_dir)

    print("All experiments completed.")

if __name__ == "__main__":
    # python main.py -c debug_config.yaml --save_dir ./debug_experiments
    # python main.py -c subtasks_config.yaml --save_dir ./experiments
    main()