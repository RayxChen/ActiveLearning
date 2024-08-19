from ActiveLearning import ActiveLearningModel, Subtask

from HighLevelSelectionStrategy import EpsilonGreedyStrategy, GreedyStrategy, ProbabilisticSamplingStrategy
from ImportanceCalculationStrategy import RecentErrorStrategy, VarianceReductionStrategy, RelativeErrorStrategy


# Using GreedyStrategy
# selection_strategy = GreedyStrategy()

# # Using ProbabilisticSamplingStrategy
# selection_strategy = ProbabilisticSamplingStrategy()

# Using EpsilonGreedyStrategy with epsilon=0.1
selection_strategy = EpsilonGreedyStrategy(epsilon=0.1)
importance_strategy = RecentErrorStrategy()

train_params = {
    'batch_size': 4,
    'learning_rate': 0.001,
    'epochs': 5
}


# Initialize ActiveLearningModel with the chosen selection strategy
active_learning_model = ActiveLearningModel(
    subtasks=subtasks,
    selection_strategy=selection_strategy,
    importance_strategy=importance_strategy,
    init_explore=10,
    train_params=train_params
)

# Run the active learning process
active_learning_model.run()