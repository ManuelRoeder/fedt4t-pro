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

import random
import copy
import concurrent.futures
from collections import defaultdict
from logging import INFO
from typing import Optional, Union

from flwr.common.logger import log
from flwr.server import Server
from flwr.common import (
    FitRes,
    Parameters,
    Scalar,
    GetPropertiesIns,
    GetPropertiesRes,
    Code,
    Properties
)
from flwr.server.client_manager import SimpleClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.server import fit_clients

# Axelrod framework imports
from axelrod.action import Action

# FedT4T framework imports
from ipd_scoring import  format_ranked_payoffs_for_logging_2, plot_average_score_per_strategy_over_rounds, plot_cumulative_cooperations_over_rounds_with_focus, save_strategy_total_scores_over_rounds_with_focus, plot_cumulative_cooperations_over_rounds, save_strategy_score_differences_matrix2, save_average_score_per_client_over_rounds, save_strategy_total_scores_over_rounds, update_scoreboard, get_ipd_score, format_ranked_payoffs_for_logging, get_clients_score_overview, plot_interaction_graph
from util import generate_hash, actions_to_string, ClientSamplingStrategy, moran_sampling
import util
from ipd_tournament_strategy import Ipd_TournamentStrategy

USE_CANTOR_PAIRING = False # use cantor hashing or free-text transmission of match id

FitResultsAndFailures = tuple[
    list[tuple[ClientProxy, FitRes]],
    list[Union[tuple[ClientProxy, FitRes], BaseException]],
]

class Ipd_ClientManager(SimpleClientManager):
    def __init__(self) -> None:
         super().__init__()
         log(INFO, "Starting Ipd client manager")
    
    def sample_moran(
        self,
        num_clients: int,
        server_round: int,
        min_num_clients: Optional[int] = None,
        scoreboard: list = [],
    ) -> list[ClientProxy]:
        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        log(INFO, "Number of available clients for Moran sampling: " + str(len(available_cids)))
        # collect client list async
        proxy_list = list()
        for client_id, client_proxy in self.clients.items():
            proxy_list.append(client_proxy)
        
        # get client ids from nodes async
        participating_client_lst = get_properties_async(client_instructions=[], client_proxies=proxy_list,
                                                        max_workers=None,
                                                        timeout=None,
                                                        group_id=server_round)
        resolved_clients = list()
        for client in participating_client_lst:
            index = client[0]
            uid = available_cids[index]
            id = client[1]["client_id"]
            resolved_clients.append((uid, id))
        
        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = moran_sampling(scoreboard_list=scoreboard, available_clients=resolved_clients, k=num_clients, round_number=server_round, threshold=50)#random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]     


class Ipd_TournamentServer(Server):
    def __init__(
        self,
        *,
        client_manager: Ipd_ClientManager,
        strategy: Optional[Ipd_TournamentStrategy] = None,
        num_rounds: int,
        sampling_strategy: ClientSamplingStrategy
    ) -> None:
        super().__init__(client_manager=client_manager, strategy=strategy)
        self.matchmaking_dict = dict()
        self.ipd_scoreboard_dict = dict()
        self.num_fl_rounds = num_rounds
        self.sampling_strategy = sampling_strategy
        self.client_resource_tracker = dict()
        
    def fit_round(
        self,
        server_round: int,
        timeout: Optional[float],
    ) -> Optional[
        tuple[Optional[Parameters], dict[str, Scalar], FitResultsAndFailures]
    ]:
        """Perform a single round of federated averaging."""
        if self.sampling_strategy == ClientSamplingStrategy.MORAN:
            # Get clients and their respective instructions from strategy
            client_instructions = self.strategy.configure_fit_moran(
                server_round=server_round,
                parameters=self.parameters,
                client_manager=self._client_manager,
                scoreboard= self.ipd_scoreboard_dict
            )
        else:
            client_instructions = self.strategy.configure_fit(
                server_round=server_round,
                parameters=self.parameters,
                client_manager=self._client_manager,
            )
            
        
        if server_round > 1:
            client_instructions = self.ipd_matchmaking(client_instructions, max_workers=self.max_workers, timeout=timeout, server_round=server_round)

        if not client_instructions:
            log(INFO, "configure_fit: no clients selected, cancel")
            return None
        log(
            INFO,
            "configure_fit: strategy sampled %s clients (out of %s)",
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `fit` results from all clients participating in this round
        results, failures = fit_clients(
            client_instructions=client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
            group_id=server_round,
        )
        log(
            INFO,
            "aggregate_fit: received %s results and %s failures",
            len(results),
            len(failures),
        )
        
        if server_round > 1:
            self.resolve_ipd_matchmaking(results, server_round)
        
        # plot or print FL round statistics
        self.statistics(server_round)
       
        # Aggregate training results
        aggregated_result: tuple[
            Optional[Parameters],
            dict[str, Scalar],
        ] = self.strategy.aggregate_fit(server_round, results, failures)

        parameters_aggregated, metrics_aggregated = aggregated_result
        return parameters_aggregated, metrics_aggregated, (results, failures)
    
    def statistics(self, server_round):
         # check some stats
        ranking_str, _ = format_ranked_payoffs_for_logging(self.ipd_scoreboard_dict)
        log(INFO, ranking_str)
        #log(INFO, format_ranked_payoffs_for_logging_2(self.ipd_scoreboard_dict))
        
        if (len(self.ipd_scoreboard_dict) > 0) and (server_round % 25 == 0):
            #plot_unique_strategy_confusion_matrix(self.ipd_scoreboard_dict)
            scoreboard_str, _ = get_clients_score_overview(self.ipd_scoreboard_dict)
            log(INFO, scoreboard_str)
            #plot_strategy_score_differences_matrix(self.ipd_scoreboard_dict)
            #save_strategy_score_differences_matrix2(self.ipd_scoreboard_dict, plot_directory="plots", filename= str(server_round) + "_confusion_matrix.png")
            custom_colors = {
                "Res.M1 | Forgiving TFT": "#17becf",  # Teal
                "Res.M1 | Firm But Fair": "#9467bd",  # Purple
                "Res.M1 | Win Stay - Lose Shift": "#1f77b4",  # Blue
                "Res.M1 | Generous TFT": "#e377c2",  # Pink
                "Res.M1 | Tit for Tat": "#ff7f0e",  # Orange
                "Res.M1 | Cooperator": "#2ca02c",  # Green
                "Res.M1 | Grim": "#d62728",  # Red
                "Res.M1 | Contributor": "#bcbd22",  # Yellow-Green
            }
            #save_strategy_total_scores_over_rounds_with_focus(self.ipd_scoreboard_dict, plot_directory="plots", filename= str(server_round) + "_scoring_plot_foc.pdf")
            #plot_cumulative_cooperations_over_rounds_with_focus(self.ipd_scoreboard_dict, plot_directory="plots", filename= str(server_round) + "_coop_plot.pdf", vertical_lines=[50, 100, 150], exclude_from_focus=["Res.M1 | Defector"], focus_range=(140, 250), custom_colors=custom_colors)
            #plot_interaction_graph(self.ipd_scoreboard_dict, plot_directory="plots", filename= str(server_round) + "_interaction_graph.png" )
            #save_average_score_per_client_over_rounds(self.ipd_scoreboard_dict, plot_directory="plots", filename= str(server_round) + "_scoring_plot_avg.png")
            #plot_average_score_per_strategy_over_rounds(self.ipd_scoreboard_dict, plot_directory="plots", filename= str(server_round) + "_scoring_plot_avg.png")
            #write_unique_matches_to_file(self.ipd_scoreboard_dict)
            
    
    def ipd_matchmaking(self, client_instructions, max_workers, timeout, server_round):
        # collect client list async
        participating_client_lst = get_properties_async(client_instructions=client_instructions,
                                                        max_workers=max_workers,
                                                        timeout=timeout,
                                                        group_id=server_round)
        # shuffle list and pop last two
        random.shuffle(participating_client_lst)
        
        new_client_instructions = [(x, copy.deepcopy(y)) for x, y in client_instructions]
        
        while len(participating_client_lst) > 1:
            # check client participation in previous round
            player_1 = participating_client_lst.pop()
            player_2 = participating_client_lst.pop()
            # create Key
            id_p1 = int(player_1[1]["client_id"])
            id_p2 = int(player_2[1]["client_id"])
            # calc unique match hash
            sorted_x, sorted_y, hash_key = generate_hash(id_p1, id_p2, USE_CANTOR_PAIRING)
            # obtain history
            log(INFO, "pairing input: %s and %s - output %s", sorted_x, sorted_y, hash_key)
            if hash_key in self.matchmaking_dict.keys():
                matchup_1, matchup_2 = self.matchmaking_dict[hash_key]
                # integrate matchups in FitRes of client A
                {"ipd_history_plays": 0, "ipd_history_coplays": 0}
                # fix flip by sort operation
                if sorted_x == int(player_1[1]["client_id"]):
                    new_client_instructions[player_1[0]][1].config["ipd_history_plays"] = actions_to_string(matchup_1) # convert action list to string
                    new_client_instructions[player_1[0]][1].config["ipd_history_coplays"] = actions_to_string(matchup_2)
                    # integrate matchups in FitRes of client B
                    new_client_instructions[player_2[0]][1].config["ipd_history_plays"] = actions_to_string(matchup_2)
                    new_client_instructions[player_2[0]][1].config["ipd_history_coplays"] = actions_to_string(matchup_1)
                else:
                    new_client_instructions[player_1[0]][1].config["ipd_history_plays"] = actions_to_string(matchup_2)
                    new_client_instructions[player_1[0]][1].config["ipd_history_coplays"] = actions_to_string(matchup_1)
                    # integrate matchups in FitRes of client B
                    new_client_instructions[player_2[0]][1].config["ipd_history_plays"] = actions_to_string(matchup_1)
                    new_client_instructions[player_2[0]][1].config["ipd_history_coplays"] = actions_to_string(matchup_2)
            # attach match_id
            new_client_instructions[player_1[0]][1].config["match_id"] = hash_key
            new_client_instructions[player_2[0]][1].config["match_id"] = hash_key
            # attach server round
            new_client_instructions[player_1[0]][1].config["server_round"] = str(server_round)
            new_client_instructions[player_2[0]][1].config["server_round"] = str(server_round)
            # dynamic resource allocation(tracked by server, reduced on client)
            if id_p1 in self.client_resource_tracker.keys():
                new_client_instructions[player_1[0]][1].config["dynamic_client_resource"] = str(self.client_resource_tracker[id_p1])
                
            else:
                new_client_instructions[player_1[0]][1].config["dynamic_client_resource"] = str(util.ResourceLevel.FULL.value)
                
            if id_p2 in self.client_resource_tracker.keys():
                new_client_instructions[player_2[0]][1].config["dynamic_client_resource"] = str(self.client_resource_tracker[id_p2])
            else:
                new_client_instructions[player_2[0]][1].config["dynamic_client_resource"] = str(util.ResourceLevel.FULL.value)
                
        return new_client_instructions
            

    def resolve_ipd_matchmaking(self, results, server_round):
        log(INFO, "Running IPD matchmaking resolve")
        if not results:
            return None, {}
        
        # Collect results
        matchup_results = [
            (fit_res.metrics, fit_res.num_examples)
            for _, fit_res in results
        ]
        
        # Group dictionaries by match_id
        grouped_data = defaultdict(list)
        for metrics, num_examples in matchup_results:
            grouped_data[metrics["match_id"]].append((metrics, num_examples))
        
        # entry -> (dict[] metrics, int num), metrics -> [client_id, match_id]
        # process double-entries
        for match_id, entries in grouped_data.items():
            if len(entries) > 1: # found more than one match_id entry
                #print(f"Duplicate match_id {match_id} found with entries:")
                log(INFO, "Duplicate match_ids found")
                for i in range(len(entries) - 1):
                    idx_lower_client_id, idx_higher_client_id = self.findMetricIdx(entries)
                    # extract results
                    metrics_1, num_examples_1 = entries[idx_lower_client_id]
                    metrics_2, num_examples_2 = entries[idx_higher_client_id]
                    # fetch server history by match_id
                    match_id = metrics_1["match_id"]
                    c1_id = int(metrics_1["client_id"])
                    c2_id = int(metrics_2["client_id"])
                    if match_id in self.matchmaking_dict.keys():
                         # update entries with match result
                        history_c1, history_c2 = self.matchmaking_dict[match_id]
                    else:
                        # create new entry
                        history_c1 = list()
                        history_c2 = list()
                    
                    # Update scoring action
                    action_c1 = (True if num_examples_1 > 0 else False)
                    action_c2 = (True if num_examples_2 > 0 else False)
                    history_c1.append(Action.C if action_c1 else Action.D)
                    history_c2.append(Action.C if action_c2 else Action.D)
                    # get the score for each client
                    score_c1, score_c2 = get_ipd_score(action_c1, action_c2)
                    # update scoreboard with client data
                    c1_ipd_strategy = metrics_1["ipd_strategy_name"]
                    c2_ipd_strategy = metrics_2["ipd_strategy_name"]
                    c1_res_level = metrics_1["resource_level"]
                    c2_res_level = metrics_2["resource_level"]
                    # update resource tracker
                    self.client_resource_tracker[c1_id] = c1_res_level
                    self.client_resource_tracker[c2_id] = c2_res_level
                    
                    update_scoreboard(self.ipd_scoreboard_dict, match_id, (c1_id, action_c1, score_c1, c1_ipd_strategy, c1_res_level), (c2_id, action_c2, score_c2, c2_ipd_strategy, c2_res_level), server_round)
                    self.matchmaking_dict[match_id] = (history_c1, history_c2)
            else:
                log(INFO, "Single match_id found")


    def findMetricIdx(self, entries):
        """ Returns a tuple of entry indices, with the lower client id in the first spot"""
        if int(entries[0][0]["client_id"]) < int(entries[1][0]["client_id"]):
            return 0, 1
        else:
            return 1, 0
                
# async commuinication handling to get client properties, dont touch anymore!
def get_properties(
    client: ClientProxy, ins: GetPropertiesIns, timeout: Optional[float], group_id: int, idx: int
):
    """Refine parameters on a single client."""
    prop_res = client.get_properties(ins, timeout=timeout, group_id=group_id)
    return client, prop_res, idx

def get_properties_async(
        client_instructions: list[tuple[ClientProxy, FitRes]],
        max_workers: Optional[int],
        timeout: Optional[float],
        group_id: int,
        client_proxies: list[ClientProxy] = [],
    ) -> list:
        """Refine parameters concurrently on all selected clients."""
        conf = {'client_id': 0, 'strategy': ''}
        props = GetPropertiesIns(config=conf)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            if len(client_proxies) == 0:
                submitted_fs = {
                    executor.submit(get_properties, client_proxy, props, timeout, group_id, idx)
                    for idx, (client_proxy, _) in enumerate(client_instructions)
                }
            else:
                submitted_fs = {
                    executor.submit(get_properties, client_proxy, props, timeout, group_id, idx)
                    for idx, client_proxy in enumerate(client_proxies)
                }
            finished_fs, _ = concurrent.futures.wait(
                fs=submitted_fs,
                timeout=None,  # Handled in the respective communication stack
            )

        # Gather results
        results: list[tuple[int, Properties]] = []
        failures: list[Union[tuple[ClientProxy, GetPropertiesRes], BaseException]] = []
        for future in finished_fs:
            _handle_finished_future_after_get_properties(
                future=future, results=results, failures=failures
            )
            
        return results


def _handle_finished_future_after_get_properties(
    future: concurrent.futures.Future,  # type: ignore
    results: list[tuple[int, Properties]],
    failures: list[Union[tuple[ClientProxy, GetPropertiesRes], BaseException]],
) -> None:
    """Convert finished future into either a result or a failure."""
    # Check if there was an exception
    failure = future.exception()
    if failure is not None:
        failures.append(failure)
        return

    # Successfully received a result from a client
    result: tuple[ClientProxy, GetPropertiesRes, int] = future.result()
    _, res, idx = result

    # Check result status code
    if res.status.code == Code.OK:
        results.append((idx, res.properties))
        return

    # Not successful, client returned a result where the status code is not OK
    failures.append(result)
        

    