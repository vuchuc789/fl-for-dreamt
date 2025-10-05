import torch
from flwr.app import ArrayRecord, Context, Message, MetricRecord, RecordDict
from flwr.clientapp import ClientApp

from fedlab.task import Net, load_data
from fedlab.task import test as test_fn
from fedlab.task import train as train_fn

# Flower ClientApp
app = ClientApp()


@app.train()
def train(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    trainloader, _ = load_data(partition_id, batch_size=32)

    # Call the training function
    metrics = train_fn(
        net=model,
        trainloader=trainloader,
        epochs=context.run_config["local-epochs"],
        lr=msg.content["config"]["lr"],
        weight_decay=msg.content["config"]["weight_decay"],
        device=device,
    )

    # Construct and return reply Message
    model_record = ArrayRecord(model.state_dict())
    metrics |= {
        "num-examples": len(trainloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"arrays": model_record, "metrics": metric_record})
    return Message(content=content, reply_to=msg)


@app.evaluate()
def evaluate(msg: Message, context: Context):
    # Load the model and initialize it with the received weights
    model = Net()
    model.load_state_dict(msg.content["arrays"].to_torch_state_dict())
    device = torch.device(
        torch.accelerator.current_accelerator().type
        if torch.accelerator.is_available()
        else "cpu"
    )
    model.to(device)

    # Load the data
    partition_id = context.node_config["partition-id"]
    # num_partitions = context.node_config["num-partitions"]
    _, valloader = load_data(partition_id, batch_size=32)

    # Call the evaluation function
    metrics = test_fn(
        net=model,
        testloader=valloader,
        device=device,
    )

    # Construct and return reply Message
    metrics |= {
        "num-examples": len(valloader.dataset),
    }
    metric_record = MetricRecord(metrics)
    content = RecordDict({"metrics": metric_record})
    return Message(content=content, reply_to=msg)
