from collections.abc import Iterable
from typing import Callable, Optional

from flwr.app import (
    ArrayRecord,
    ConfigRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from fedlab.task import Net
from fedlab.utils.model import load_checkpoint, save_history, save_model


class CustomStrategy(FedAvg):
    def __init__(
        self,
        fraction_train: float = 1,
        fraction_evaluate: float = 1,
        min_train_nodes: int = 2,
        min_evaluate_nodes: int = 2,
        min_available_nodes: int = 2,
        weighted_by_key: str = "num-examples",
        arrayrecord_key: str = "arrays",
        configrecord_key: str = "config",
        train_metrics_aggr_fn: Optional[
            Callable[[list[RecordDict], str], MetricRecord]
        ] = None,
        evaluate_metrics_aggr_fn: Optional[
            Callable[[list[RecordDict], str], MetricRecord]
        ] = None,
        checkpoint: int = 0,
    ) -> None:
        super().__init__(
            fraction_train,
            fraction_evaluate,
            min_train_nodes,
            min_evaluate_nodes,
            min_available_nodes,
            weighted_by_key,
            arrayrecord_key,
            configrecord_key,
            train_metrics_aggr_fn,
            evaluate_metrics_aggr_fn,
        )

        self.checkpoint = checkpoint

    def aggregate_train(
        self, server_round: int, replies: Iterable[Message]
    ) -> tuple[Optional[ArrayRecord], Optional[MetricRecord]]:
        arrays, metrics = super().aggregate_train(server_round, replies)

        server_round += self.checkpoint
        save_model(server_round, model=arrays.to_torch_state_dict())
        save_history(server_round, metrics={k: v for k, v in metrics.items()})

        return arrays, metrics

    def aggregate_evaluate(
        self, server_round: int, replies: Iterable[Message]
    ) -> Optional[MetricRecord]:
        metrics = super().aggregate_evaluate(server_round, replies)

        server_round += self.checkpoint
        save_history(server_round, metrics={k: v for k, v in metrics.items()})

        return metrics


# Create ServerApp
app = ServerApp()


@app.main()
def main(grid: Grid, context: Context) -> None:
    # Read run config
    fraction_train: float = context.run_config["fraction-train"]
    fraction_evaluate: float = context.run_config["fraction-evaluate"]
    num_rounds: int = context.run_config["num-server-rounds"]
    lr: float = context.run_config["lr"]
    weight_decay: float = context.run_config["weight-decay"]

    # Load global model
    global_model = Net()
    checkpoint, _ = load_checkpoint(global_model)
    arrays = ArrayRecord(global_model.state_dict())

    # Initialize FedAvg strategy
    strategy = CustomStrategy(
        fraction_train=fraction_train,
        fraction_evaluate=fraction_evaluate,
        checkpoint=checkpoint,
    )

    # Start strategy, run FedAvg for `num_rounds`
    strategy.start(
        grid=grid,
        initial_arrays=arrays,
        train_config=ConfigRecord(
            {
                "lr": lr,
                "weight_decay": weight_decay,
            }
        ),
        num_rounds=num_rounds,
    )
