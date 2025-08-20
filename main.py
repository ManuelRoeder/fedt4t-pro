"""
MIT License

Copyright (c) 2025 Manuel Roeder

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import copy
import os
import io
import random
from typing import List, Tuple

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # torch imports
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Normalize, Compose

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

# FedT4T project imports
import util
from model import Net
from ipd_client import FedT4TClient
from ipd_tournament_server import Ipd_TournamentServer, Ipd_ClientManager
from ipd_player import ResourceAwareMemOnePlayer, RandomIPDPlayer
from ipd_tournament_strategy import Ipd_TournamentStrategy

# Flower_datasets framework imports
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr.client import ClientApp
from flwr.common import Context, ndarrays_to_parameters, Metrics
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.client_manager import SimpleClientManager
from flwr.simulation import run_simulation

# Axelrod framework imports
import axelrod as axl
from axelrod.action import Action



SHOW_LABEL_DISTRUBUTION_OVER_CLIENTS = False
strategy_mem_depth = 1
FL_STRATEGY_SUBSAMPLE = 0.5

def sow_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True)
    

def visualise_n_random_examples(trainset_, n: int, verbose: bool = True):
    trainset_data = [
        Image.open(io.BytesIO(entry[0].as_py())) for entry in trainset_.data[0]
    ]
    idx = list(range(len(trainset_data)))
    random.shuffle(idx)
    idx = idx[:n]
    if verbose:
        print(f"will display images with idx: {idx}")

    # construct canvas
    num_cols = 8
    num_rows = int(np.ceil(len(idx) / num_cols))
    fig, axs = plt.subplots(figsize=(16, num_rows * 2), nrows=num_rows, ncols=num_cols)

    # display images on canvas
    for c_i, i in enumerate(idx):
        axs.flat[c_i].imshow(trainset_data[i], cmap="gray")

def get_mnist_dataloaders(mnist_dataset, batch_size: int):
    pytorch_transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

    # Prepare transformation functions
    def apply_transforms(batch):
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    mnist_train = mnist_dataset["train"].with_transform(apply_transforms)
    mnist_test = mnist_dataset["test"].with_transform(apply_transforms)

    # Construct PyTorch dataloaders
    trainloader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
    testloader = DataLoader(mnist_test, batch_size=batch_size)
    return trainloader, testloader

def train(net, trainloader, optimizer, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    net.train()
    for batch in trainloader:
        images, labels = batch["image"], batch["label"]
        optimizer.zero_grad()
        loss = criterion(net(images), labels)
        loss.backward()
        optimizer.step()

def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    net.eval()
    with torch.no_grad():
        for batch in testloader:
            images, labels = batch["image"], batch["label"]
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            correct += (predicted == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

def get_params(model):
    """Extract model parameters as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]

def get_number_of_round_with_avg_meetups(avg, N, subsample_fraction):
    return int((avg * (N - 1) / subsample_fraction))
    

def run_centralised(
    trainloader, testloader, epochs: int, lr: float, momentum: float = 0.9
):
    """A minimal (but complete) training loop"""

    # instantiate the model
    model = Net(num_classes=10)

    # define optimiser with hyperparameters supplied
    optim = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    # train for the specified number of epochs
    for e in range(epochs):
        print(f"Training epoch {e} ...")
        train(model, trainloader, optim, epochs)

    # training is completed, then evaluate model on the test set
    loss, accuracy = test(model, testloader)
    print(f"{loss = }")
    print(f"{accuracy = }")
    
def client_fn(context: Context):
    """Returns a FlowerClient containing its data partition."""
   
    partition_id = int(context.node_config["partition-id"])
    partition = fds.load_partition(partition_id, "train")
    # partition into train/validation
    partition_train_val = partition.train_test_split(test_size=0.1, seed=util.SEED)

    # Let's use the function defined earlier to construct the dataloaders
    # and apply the dataset transformations
    trainloader, testloader = get_mnist_dataloaders(partition_train_val, batch_size=32)
    
    # Pop last element from list and set seed
    client_ipd_strat = client_strategies[partition_id]
    client_ipd_strat.set_seed(util.SEED)

    print("Init client(" + str(partition_id) + ") with strategy " + client_ipd_strat.name)

    return FedT4TClient(trainloader=trainloader,
                        valloader=testloader,
                        ipd_strategy=client_ipd_strat,
                        client_id=partition_id).to_client()

def server_fn(context: Context):
    # instantiate the model
    model = Net(num_classes=10)
    ndarrays = get_params(model)
    # Convert model parameters to flwr.common.Parameters
    global_model_init = ndarrays_to_parameters(ndarrays)

    # Define the strategy
    strategy = Ipd_TournamentStrategy(
        fraction_fit=FL_STRATEGY_SUBSAMPLE,  # 50% clients sampled each round to do fit()
        fraction_evaluate=0.1,  # 10% clients sample each round to do evaluate()
        #min_fit_clients= 16,
        evaluate_metrics_aggregation_fn=weighted_average,  # callback defined earlier
        initial_parameters=global_model_init,  # initialised global model
    )
    
    # calculate the number of rounds based on the subsamüling strategy
    avg = 5
    # num_rounds = get_number_of_round_with_avg_meetups(avg, NUM_PARTITIONS, FL_STRATEGY_SUBSAMPLE)
    num_rounds = 151
    #print("Min. number of rounds to have on average " + str(avg) + " matches with " + str(NUM_PARTITIONS) + " participating clients and a subsampling rate of " + str(FL_STRATEGY_SUBSAMPLE) + " is "  +  str(num_rounds))
    # Iterated Prisoners Dilemma Tournament Server
    ipd_tournament_server= Ipd_TournamentServer(client_manager=Ipd_ClientManager(), strategy=strategy, num_rounds=num_rounds, sampling_strategy=util.ClientSamplingStrategy.MORAN)

    # Construct ServerConfig
    config = ServerConfig(num_rounds=num_rounds)

    # Wrap everything into a `ServerAppComponents` object
    return ServerAppComponents(server=ipd_tournament_server, config=config)

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}

# show distribution of dataset over clients at the start of training
def show_dataset_distribution():
    # pre-load partitions
    partitioner = fds.partitioners["train"]

    fig, ax, df = plot_label_distributions(
        partitioner,
        label_name="label",
        plot_type="bar",
        size_unit="absolute",
        partition_id_axis="x",
        legend=True,
        verbose_labels=True,
        max_num_partitions=30,  # Note we are only showing the first 30 so the plot remains readable
        title="Per Partition Labels Distribution",
    )
    plt.show()

def get_client_strategies(exp_str, mem_depth=1, resource_awareness=False):
    client_strategies = list()
    if exp_str == "axelrod_ordinary":
        # ordinary axelrod set filtered by mem depth 1
        client_strategies = [s() for s  in axl.filtered_strategies(filterset={'memory_depth': mem_depth}, strategies=axl.ordinary_strategies)]
    elif exp_str == "m1_selected":

        # initiate one-by one
        strat1 = axl.GTFT(p=0.0)
        strat1.name = "Tit for Tat"
        strat1.set_seed(util.SEED)
        client_strategies.append(strat1)
        
        strat2 = axl.StochasticWSLS(0)
        strat2.name = "Win Stay - Lose Shift"
        strat2.set_seed(util.SEED)
        client_strategies.append(strat2)
        
        strat3 = axl.SoftJoss(0)
        strat3.name = "Cooperator"
        strat3.set_seed(util.SEED)
        client_strategies.append(strat3)
        
        #strat4 = axl.MemoryOnePlayer((0, 0, 0, 0), Action.D)
        #strat4.name = "Defector"
        #strat4.set_seed(util.SEED)
       # client_strategies.append(strat4)
        
        strat4 = axl.MemoryOnePlayer((0.9, 0.5, 0.5, 0.1), Action.C)
        strat4.name = "Contributor"
        strat4.set_seed(util.SEED)
        client_strategies.append(strat4)
        
        strat5 = axl.GTFT(p=0.33)
        strat5.name = "Generous TFT"
        strat5.set_seed(util.SEED)
        client_strategies.append(strat5)
        
        strat6 = axl.GTFT(p=0.75)
        strat6.name = "Forgiving TFT"
        strat6.set_seed(util.SEED)
        client_strategies.append(strat6)
        
        strat7 = axl.MemoryOnePlayer((1.0, 0.0, 0.0, 0.0), Action.C)
        strat7.name = "Grim"
        strat7.set_seed(util.SEED)
        client_strategies.append(strat7)
        
        strat8 = axl.FirmButFair()
        strat8.name = "Firm But Fair"
        strat8.set_seed(util.SEED)
        client_strategies.append(strat8)

    elif exp_str == "axelrod_stochastic":
        # axelrod set filtered by mem depth 1 and stochastic property
        client_strategies = [s() for s in axl.filtered_strategies(filterset={'memory_depth': mem_depth, 'stochastic': True}, strategies=axl.all_strategies)]
        
    elif exp_str == "convergence_xprobing":
        for i in range(1):
                # initiate one-by one
            strat1 = axl.GTFT(p=0.0)
            strat1.name = "Cooperator" + "_" + str(i)
            strat1.set_seed(util.SEED)
            client_strategies.append(strat1)
            
            strat3 = axl.SoftJoss(0)
            strat3.name = "Cooperator" + "_" + str(i)
            strat3.set_seed(util.SEED)
            client_strategies.append(strat3)
            
            strat3 = axl.SoftJoss(0)
            strat3.name = "Cooperator" + "_" + str(i)
            strat3.set_seed(util.SEED)
            client_strategies.append(strat3)
            
            strat3 = axl.SoftJoss(0)
            strat3.name = "Cooperator" + "_" + str(i)
            strat3.set_seed(util.SEED)
            client_strategies.append(strat3)
            
            #strat4 = axl.MemoryOnePlayer((0, 0, 0, 0), Action.D)
            #strat4.name = "Defector"
            #strat4.set_seed(util.SEED)
        # client_strategies.append(strat4)
            
            strat4 = axl.MemoryOnePlayer((0.9, 0.5, 0.5, 0.1), Action.C)
            strat4.name = "Cooperator" + "_" + str(i)
            strat4.set_seed(util.SEED)
            client_strategies.append(strat4)
            
            strat5 = axl.GTFT(p=0.33)
            strat5.name = "Cooperator" + "_" + str(i)
            strat5.set_seed(util.SEED)
            client_strategies.append(strat5)
            
            strat6 = axl.GTFT(p=0.75)
            strat6.name = "Cooperator" + "_" + str(i)
            strat6.set_seed(util.SEED)
            client_strategies.append(strat6)
            
            
            strat8 = axl.FirmButFair()
            strat8.name = "Cooperator" + "_" + str(i)
            strat8.set_seed(util.SEED)
            client_strategies.append(strat8)
            
        for i in range(8):
            strat2 = axl.StochasticWSLS(0)
            strat2.name = "Defector" + "_" + str(i)
            strat2.set_seed(util.SEED)
            client_strategies.append(strat2)

    
    # extend w random
    random_player = RandomIPDPlayer()
    random_player.set_seed(util.SEED)
    #client_strategies.append(random_player)
    
    if resource_awareness:
        # remove all players that are not derived from MemoryOne
        selected_strategies = list()
        for strategy in client_strategies:
            if isinstance(strategy, axl.MemoryOnePlayer):
                #selected_strategies.append(ResourceAwareMemOnePlayer(copy.deepcopy(strategy), initial_resource_value=util.ResourceLevel.LOW.value))
                #selected_strategies.append(ResourceAwareMemOnePlayer(copy.deepcopy(strategy), initial_resource_value=util.ResourceLevel.MODERATE.value))
                selected_strategies.append(ResourceAwareMemOnePlayer(player_instance=copy.deepcopy(strategy), resource_scaling_func=util.linear_scaling, initial_resource_value=util.ResourceLevel.FULL.value))
        #selected_strategies.append(random_player)
        return selected_strategies
                
    return client_strategies
###################### MAIN TRACK ######################

# initialize strategies with memory_depth eq. 1
client_strategies = get_client_strategies("m1_selected", mem_depth=strategy_mem_depth, resource_awareness=True)

# mix list
random.shuffle(client_strategies)
# create partitions for each FL client
NUM_PARTITIONS = len(client_strategies)
#print(*client_strategies, "\\n")
print("Strategies initialized: " + str(NUM_PARTITIONS))

# initialize data partitions
iid_partitioner = IidPartitioner(num_partitions=NUM_PARTITIONS)
dirichlet_partitioner = DirichletPartitioner(num_partitions=NUM_PARTITIONS, alpha=0.1, partition_by="label", seed=util.SEED, min_partition_size=1)
# Let's partition the "train" split of the MNIST dataset
# The MNIST dataset will be downloaded if it hasn't been already
fds = FederatedDataset(dataset="ylecun/mnist", partitioners={"train": dirichlet_partitioner})

if SHOW_LABEL_DISTRUBUTION_OVER_CLIENTS:
    show_dataset_distribution()
    
def main():
    
    # Concstruct the ClientApp passing the client generation function
    client_app = ClientApp(client_fn=client_fn)

    # Create your ServerApp
    server_app = ServerApp(server_fn=server_fn)

    run_simulation(
        server_app=server_app, client_app=client_app, num_supernodes=NUM_PARTITIONS
    )

if __name__ == '__main__':
    sow_seed(util.SEED)
    main()