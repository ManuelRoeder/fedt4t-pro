"""
MIT License

Copyright (c) 2024 Manuel Roeder

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

import os
import util
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import networkx as nx
import seaborn as sns
from collections import defaultdict



# payoff-matrix configuration
R = 3 # CC
S = 0 # CD
T = 5 # DC
P = 1 # DD

def get_ipd_score(a1, a2):
    s1 = s2 = 0
    # depending on the payoff-matrix, assign score - hard code for now
    if a1 and a2:
        s1 = R
        s2 = R
    elif a1 and not a2:
        s1 = S
        s2 = T
    elif not a1 and a2:
        s1 = T
        s2 = S
    elif not a1 and not a2:
        s1 = P
        s2 = P
    return s1, s2
        
        
def update_scoreboard(
        ipd_scoreboard_dict, 
        match_id: int, 
        c1_res_tuple: tuple, 
        c2_res_tuple: tuple, 
        server_round: int
        ):
        """
    Updates the scoreboard with the results of a match round between two clients.

    Parameters:
    - match_id (int): Unique identifier for the match.
    - c1_res_tuple (tuple): A tuple containing client 1's data in the form 
                            (client_id, play, payoff, ipd_strategy, res_level).
    - c2_res_tuple (tuple): A tuple containing client 2's data in the form 
                            (client_id, play, payoff, ipd_strategy, res_level).
    - server_round (int): The current round number in the server game loop.

    Each client's result is appended as a tuple:
    (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level).
    """
        # Ensure each client has an entry in the scoreboard dictionary
        ipd_scoreboard_dict.setdefault(c1_res_tuple[0], [])
        ipd_scoreboard_dict.setdefault(c2_res_tuple[0], [])
        
        # Update data for client 1
        ipd_scoreboard_dict[c1_res_tuple[0]].append(
            (server_round, match_id, c2_res_tuple[0], c1_res_tuple[1], c2_res_tuple[1], c1_res_tuple[2], c1_res_tuple[3], c1_res_tuple[4])
        )
        # Update data for client 2
        ipd_scoreboard_dict[c2_res_tuple[0]].append(
            (server_round, match_id, c1_res_tuple[0], c2_res_tuple[1], c1_res_tuple[1], c2_res_tuple[2], c2_res_tuple[3], c2_res_tuple[4])
        )


def get_ranked_payoffs(ipd_scoreboard_dict):
    """
    Calculates the total payoffs for each client and returns a ranked list.

    Parameters:
        - scoreboard_dict: the scoreboard
        
    Returns:
    - List of tuples, where each tuple is in the format (client_id, total_payoff), 
    sorted by total_payoff in descending order.
    """
    # Dictionary to hold the total payoff for each client
    total_payoffs = {}

    # Calculate total payoffs for each client
    for client_id, rounds in ipd_scoreboard_dict.items():
        # Sum up the payoffs for this client across all rounds
        total_payoffs[client_id] = sum(round[5] for round in rounds)

    # Sort clients by total payoffs in descending order and return as a list of tuples
    ranked_payoffs = sorted(total_payoffs.items(), key=lambda x: x[1], reverse=True)

    return ranked_payoffs


def print_ranked_payoffs(ipd_scoreboard_dict):
    """
    Calculates and prints the total payoffs, strategy, and resource level for each client in a ranked table format.
    """
    # Dictionary to hold the total payoff, strategy, and resource level for each client
    client_info = {}

    # Calculate total payoffs, and get strategy and resource level for each client
    for client_id, rounds in ipd_scoreboard_dict.items():
        # Sum up the payoffs for this client across all rounds
        total_payoff = sum(round[5] for round in rounds)  # payoff is at index 5

        # Extract strategy and resource level from the first entry (since they are constant)
        strategy = rounds[0][6]  # ipd_strategy is at index 6
        resource_level = rounds[0][7]  # res_level is at index 7

        # Store the information
        client_info[client_id] = (total_payoff, strategy, resource_level)

    # Sort clients by total payoffs in descending order
    ranked_clients = sorted(client_info.items(), key=lambda x: x[1][0], reverse=True)

    # Print the header for the ranking table
    print(f"{'Rank':<5} {'Client ID':<10} {'Total Payoff':<15} {'Strategy':<20} {'Resource Level':<15}")
    print("-" * 70)

    # Print each client's rank, ID, total payoff, strategy, and resource level
    for rank, (client_id, (total_payoff, strategy, resource_level)) in enumerate(ranked_clients, start=1):
        print(f"{rank:<5} {client_id:<10} {total_payoff:<15} {strategy:<20} {resource_level:<15.2f}")


def plot_payoffs_over_rounds(ipd_scoreboard_dict):
    """
    Creates a line plot of each client's cumulative payoffs over the number of server rounds,
    labeling each client by their strategy name.
    """
    # Dictionary to store cumulative payoffs over rounds for each client
    cumulative_payoffs = {}

    # Loop through each client to calculate cumulative payoffs by round
    for client_id, rounds in ipd_scoreboard_dict.items():
        cumulative_payoff = 0
        rounds_list = []
        payoffs_list = []
        
        for round_data in rounds:
            # round_data format: (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            server_round = round_data[0]
            payoff = round_data[5]  # payoff is at index 5

            # Update cumulative payoff
            cumulative_payoff += payoff
            rounds_list.append(server_round)
            payoffs_list.append(cumulative_payoff)
        
        # Get the strategy for the client from the first round entry
        strategy_name = rounds[0][6]  # ipd_strategy is at index 6
        cumulative_payoffs[strategy_name] = (rounds_list, payoffs_list)

    # Plot each client's cumulative payoff over rounds with strategy labels
    plt.figure(figsize=(10, 6))
    for strategy_name, (rounds_list, payoffs_list) in cumulative_payoffs.items():
        plt.plot(rounds_list, payoffs_list, label=strategy_name)

    # Add titles and labels
    plt.title("Cumulative Payoffs Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Cumulative Payoff")
    plt.legend(title="Strategy")
    plt.grid(True)
    plt.show()
    
    
    
def format_ranked_payoffs_for_logging(ipd_scoreboard_dict):
    """
    Formats the total payoffs, strategy, resource level, number of games, and average payoff
    for each client in a ranked table format, as a string suitable for logging.

    Returns:
    - A formatted string with total payoffs ranked, including strategy, resource level,
      number of games, and average payoff.
    """
    # Dictionary to hold the total payoff, strategy, resource level, number of games, and average payoff for each client
    client_info = {}

    # Calculate total payoffs, and get strategy, resource level, and number of games for each client
    for client_id, rounds in ipd_scoreboard_dict.items():
        # Sum up the payoffs for this client across all rounds they participated in
        total_payoff = sum(round[5] for round in rounds)  # payoff is at index 5

        # Extract strategy and resource level from the first entry (assuming they are constant)
        strategy = rounds[0][6]  # ipd_strategy is at index 6
        resource_level = str(rounds[0][7]) # res_level is at index 7

        # Calculate the number of games (rounds this client actually played)
        num_games = len(rounds)
        
        # Calculate the average payoff, handling cases where num_games is zero
        average_payoff = total_payoff / num_games if num_games > 0 else 0

        # Store the information
        client_info[client_id] = (total_payoff, strategy, resource_level, num_games, average_payoff)

    # Sort clients by total payoffs in descending order
    ranked_clients = sorted(client_info.items(), key=lambda x: x[1][0], reverse=True)

    # Build the formatted string for logging
    output = []
    output.append(" ")
    output.append(f"{'Rank':<5} {'Client ID':<10} {'Total Payoff':<15} {'Strategy':<35} {'Resource Level':<15} {'Games':<10} {'Avg Payoff':<15}")
    output.append("-" * 105)

    # Append each client's rank, ID, total payoff, strategy, resource level, number of games, and average payoff
    for rank, (client_id, (total_payoff, strategy, resource_level, num_games, average_payoff)) in enumerate(ranked_clients, start=1):
        output.append(f"{rank:<5} {client_id:<10} {total_payoff:<15} {strategy:<35} {resource_level:<15} {num_games:<10} {average_payoff:<15.2f}")
    
    output.append("-" * 105)

    # Join all lines into a single formatted string
    formatted_output = "\n".join(output)
    
    return formatted_output, ranked_clients

def plot_cumulative_cooperations_over_rounds(ipd_scoreboard_dict, plot_directory='plots', filename='cumulative_cooperations_over_rounds.png'):
    """
    Plots the cumulative cooperations over server rounds for each strategy and saves the plot to a file.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            play = round_data[3]  # 'C' or 'D'
            strategy_label = round_data[6]  # Strategy name

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, play))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative cooperations for each strategy
    cumulative_cooperations_per_strategy = {strategy: [] for strategy in strategies}
    cumulative_totals = {strategy: 0 for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, play in data_list:
        data_by_round[server_round].append((strategy_label, play))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Update cumulative totals with cooperations from the current round
        cooperations_in_round = defaultdict(int)
        for strategy_label, play in data_by_round.get(server_round, []):
            if play == True:
                cooperations_in_round[strategy_label] += 1

        # Update cumulative totals and append to the lists
        for strategy in strategies:
            cumulative_totals[strategy] += cooperations_in_round.get(strategy, 0)
            cumulative_cooperations_per_strategy[strategy].append(cumulative_totals[strategy])

    # Plot the cumulative cooperations over rounds for each strategy
    plt.figure(figsize=(12, 8))
    for strategy, cumulative_cooperations in cumulative_cooperations_per_strategy.items():
        plt.plot(sorted_rounds, cumulative_cooperations, label=strategy)

    plt.title("Cumulative Cooperations Over Server Rounds by Strategy")
    plt.xlabel("Server Round")
    plt.ylabel("Cumulative Cooperations")
    plt.legend(title="Strategy")
    plt.grid(False)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to {plot_path}")
    
def plot_cumulative_cooperations_over_rounds_with_focus(
    ipd_scoreboard_dict,
    plot_directory='plots',
    filename='cumulative_cooperations_over_rounds.pdf',
    focus_range=(50, 100),
    vertical_lines=None,
    exclude_from_focus=None,
    custom_colors=None
):
    """
    Plots the cumulative cooperations over server rounds for each strategy, with an additional focus subplot zoomed into a specified range of rounds. Includes optional vertical lines and the ability to exclude strategies from the focus area.

    Parameters:
    - ipd_scoreboard_dict (dict): The scoreboard data to plot.
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.
    - focus_range (tuple): A tuple (a, b) specifying the range of rounds to focus on in the zoomed subplot.
    - vertical_lines (list): List of x-axis positions where vertical lines should be drawn.
    - exclude_from_focus (list): List of strategies to exclude from the focus area.
    - custom_colors (dict): Dictionary mapping strategy names to custom colors.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            play = round_data[3]  # 'C' or 'D'
            strategy_label = round_data[6]  # Strategy name

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, play))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative cooperations for each strategy
    cumulative_cooperations_per_strategy = {strategy: [] for strategy in strategies}
    cumulative_totals = {strategy: 0 for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, play in data_list:
        data_by_round[server_round].append((strategy_label, play))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Update cumulative totals with cooperations from the current round
        cooperations_in_round = defaultdict(int)
        for strategy_label, play in data_by_round.get(server_round, []):
            if play == True:  # Assuming `True` means cooperation
                cooperations_in_round[strategy_label] += 1

        # Update cumulative totals and append to the lists
        for strategy in strategies:
            cumulative_totals[strategy] += cooperations_in_round.get(strategy, 0)
            cumulative_cooperations_per_strategy[strategy].append(cumulative_totals[strategy])

    # Use a high-contrast color palette
    #contrast_colors = sns.color_palette("Set2", len(strategies))
    #strategy_colors = {strategy: contrast_colors[i] for i, strategy in enumerate(strategies)}
    # Use custom colors or fallback to Seaborn color palette
    if not custom_colors:
        custom_colors = sns.color_palette("Set2", len(strategies))
        custom_colors = {strategy: custom_colors[i] for i, strategy in enumerate(strategies)}
        
    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [2, 1]})

    # Full Plot
    for strategy, cumulative_cooperations in cumulative_cooperations_per_strategy.items():
        axes[0].plot(sorted_rounds, cumulative_cooperations, label=strategy, color=custom_colors[strategy])

    if vertical_lines:
        for line_x in vertical_lines:
            axes[0].axvline(x=line_x, color='gray', linestyle='--', linewidth=1)

    axes[0].set_title("Cumulative Cooperations Over Server Rounds by Strategy")
    axes[0].set_xlabel("Server Round")
    axes[0].set_ylabel("Cumulative Cooperations")
    axes[0].legend(title="Strategy", loc="upper left")
    axes[0].grid(False)

    # Focused Subplot
    focus_start, focus_end = focus_range
    exclude_from_focus = exclude_from_focus or []

    # Determine y-axis range for the focus area
    focus_y_min = float('inf')
    focus_y_max = float('-inf')
    for strategy, cumulative_cooperations in cumulative_cooperations_per_strategy.items():
        if strategy in exclude_from_focus:
            continue
        focus_scores = [cumulative_cooperations[i] for i in range(focus_start, focus_end + 1) if i < len(cumulative_cooperations)]
        if focus_scores:
            focus_y_min = min(focus_y_min, min(focus_scores))
            focus_y_max = max(focus_y_max, max(focus_scores))

    if focus_y_min == float('inf') or focus_y_max == float('-inf'):
        # If no valid data in the focus range, set a default range
        focus_y_min = 0
        focus_y_max = 1

    # Add padding to the y-axis limits for better visualization
    y_padding = (focus_y_max - focus_y_min) * 0.1  # 10% padding
    focus_y_min -= y_padding
    focus_y_max += y_padding

    for strategy, cumulative_cooperations in cumulative_cooperations_per_strategy.items():
        if strategy not in exclude_from_focus:
            axes[1].plot(sorted_rounds, cumulative_cooperations, label=strategy, color=custom_colors[strategy])

    if vertical_lines:
        for line_x in vertical_lines:
            if focus_start <= line_x <= focus_end:
                axes[1].axvline(x=line_x, color='gray', linestyle='--', linewidth=1)

    axes[1].set_xlim(focus_start, focus_end)
    axes[1].set_ylim(focus_y_min, focus_y_max)
    axes[1].set_title(f"Zoomed View: Rounds {focus_start} to {focus_end}")
    axes[1].set_xlabel("Server Round")
    axes[1].set_ylabel("Cumulative Cooperations")
    axes[1].grid(False)

    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to {plot_path}")


def format_ranked_payoffs_for_logging_2(ipd_scoreboard_dict):
    """
    Formats the ranked payoffs for logging, excluding the resource level and including the cooperation score in percent.

    Returns:
    - A list of strings, each representing a client's performance, sorted by total payoff.
    """
    from collections import defaultdict

    # Dictionary to hold client statistics
    client_stats = {}

    # Iterate over each client to collect statistics
    for client_id, rounds in ipd_scoreboard_dict.items():
        total_payoff = 0
        total_cooperations = 0
        total_games = 0
        strategy = None

        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            client_action = round_data[3]
            payoff = round_data[5]
            client_strategy = round_data[6]

            # Update client strategy (assumed constant)
            if strategy is None:
                strategy = client_strategy

            total_payoff += payoff
            total_games += 1
            if client_action == 'C':
                total_cooperations += 1

        # Calculate cooperation score in percent
        cooperation_score = (total_cooperations / total_games) * 100 if total_games > 0 else 0

        # Store the statistics for the client
        client_stats[client_id] = {
            'Strategy': strategy,
            'Total Payoff': total_payoff,
            'Total Games': total_games,
            'Cooperation Score': cooperation_score
        }

    # Sort clients by total payoff in descending order
    sorted_clients = sorted(client_stats.items(), key=lambda x: x[1]['Total Payoff'], reverse=True)

    # Format the output for logging
    output_lines = []
    for rank, (client_id, stats) in enumerate(sorted_clients, start=1):
        line = (
            f"Rank {rank}: Client ID: {client_id}, Strategy: {stats['Strategy']}, "
            f"Total Payoff: {stats['Total Payoff']}, Total Games: {stats['Total Games']}, "
            f"Cooperation Score: {stats['Cooperation Score']:.2f}%"
        )
        output_lines.append(line)

    return output_lines


def plot_unique_strategy_confusion_matrix(ipd_scoreboard_dict):
    """
    Plots a confusion matrix showing the frequency of interactions between unique clients,
    labeled by their strategy and resource level, ensuring only one unique interaction per round.
    """
    # Dictionary to store counts of interactions between unique client labels
    interaction_counts = defaultdict(int)

    # Track processed pairs for each round to ensure unique interactions
    processed_pairs = set()  # This will store tuples of (server_round, label_pair)

    # Iterate through each client to count unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format: (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            server_round = round_data[0]
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]

            # Ensure the opponent exists and retrieve opponent data for the same round
            if opponent_id in ipd_scoreboard_dict:
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"

                    # Sort labels alphabetically for a unique pair key
                    label_pair = tuple(sorted([client_label, opponent_label]))

                    # Use (server_round, label_pair) as the unique key for each interaction per round
                    unique_key = (server_round, label_pair)
                    if unique_key not in processed_pairs:
                        processed_pairs.add(unique_key)
                        interaction_counts[label_pair] += 1

    # Extract unique labels and create a matrix
    unique_labels = sorted(set(label for pair in interaction_counts.keys() for label in pair))
    matrix = pd.DataFrame(0, index=unique_labels, columns=unique_labels)

    # Fill the confusion matrix based on the unique interaction counts
    for (label_1, label_2), count in interaction_counts.items():
        matrix.at[label_1, label_2] = count
        if label_1 != label_2:
            matrix.at[label_2, label_1] = count  # Symmetric matrix

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Unique Strategy and Resource Level Confusion Matrix")
    plt.xlabel("Client (Strategy | Resource Level)")
    plt.ylabel("Opponent (Strategy | Resource Level)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
    

def save_strategy_score_differences_matrix2(ipd_scoreboard_dict, plot_directory='plots', filename='strategy_score_differences_matrix.png'):
    """
    Calculates the score differences (sum_score1 - sum_score2) for each strategy pair (excluding resource levels)
    and saves the plot.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Dictionary to store accumulated scores between unique strategy pairs
    interaction_data = {}

    # Iterate through each client to collect scores for unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            match_id = round_data[1]
            client_strategy = round_data[6]
            client_label = f"{client_strategy} | Client {client_id}"
            opponent_id = round_data[2]
            client_payoff = round_data[5]  # Client's payoff for this round

            # Ensure the opponent exists and retrieve opponent data for the same match
            if opponent_id in ipd_scoreboard_dict:
                # Find the opponent's data within the same match
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_strategy = opponent_round[6]
                    opponent_label = f"{opponent_strategy} | Client {opponent_id}"
                    opponent_payoff = opponent_round[5]  # Opponent's payoff for this round

                    # Create a label pair
                    label_pair = (client_label, opponent_label)

                    # Initialize or update the interaction data
                    if label_pair not in interaction_data:
                        interaction_data[label_pair] = [client_payoff, opponent_payoff]
                    else:
                        interaction_data[label_pair][0] += client_payoff
                        interaction_data[label_pair][1] += opponent_payoff

    # Extract unique labels
    unique_labels = sorted(set(label for pair in interaction_data.keys() for label in pair))

    # Create a DataFrame to store the score differences
    matrix = pd.DataFrame(0.0, index=unique_labels, columns=unique_labels)

    # Fill the matrix with the score differences
    for (client_label, opponent_label), (sum_score1, sum_score2) in interaction_data.items():
        score_difference = sum_score1 - sum_score2
        matrix.at[client_label, opponent_label] = score_difference

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Strategy Score Differences Matrix (Client Score - Opponent Score)")
    plt.xlabel("Opponent Strategy")
    plt.ylabel("Client Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to {plot_path}")

def plot_strategy_scores_matrix(ipd_scoreboard_dict):
    """
    Plots a confusion matrix showing the total scores of each client against others,
    formatted as '(sum_score1, sum_score2) (number of interactions)'.
    """
    # Dictionary to store accumulated scores and interaction counts between unique strategy pairs
    # Key: (client_label, opponent_label)
    # Value: [sum_client_scores, sum_opponent_scores, interaction_count]
    interaction_data = {}

    # Iterate through each client to collect scores for unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]
            client_payoff = round_data[5]  # Client's payoff for this round

            # Ensure the opponent exists and retrieve opponent data for the same match
            if opponent_id in ipd_scoreboard_dict:
                # Find the opponent's data within the same match
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"
                    opponent_payoff = opponent_round[5]  # Opponent's payoff for this round

                    # Create a label pair
                    label_pair = (client_label, opponent_label)

                    # Initialize or update the interaction data
                    if label_pair not in interaction_data:
                        interaction_data[label_pair] = [client_payoff, opponent_payoff, 1]
                    else:
                        interaction_data[label_pair][0] += client_payoff
                        interaction_data[label_pair][1] += opponent_payoff
                        interaction_data[label_pair][2] += 1

    # Extract unique labels
    unique_labels = sorted(set(label for pair in interaction_data.keys() for label in pair))

    # Create a DataFrame to store the formatted scores
    matrix = pd.DataFrame("", index=unique_labels, columns=unique_labels)

    # Fill the matrix with the formatted scores
    for (label_1, label_2), (sum_score1, sum_score2, interaction_count) in interaction_data.items():
        matrix.at[label_1, label_2] = f"({sum_score1}, {sum_score2}) ({interaction_count})"

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix.isin(['']).astype(int), annot=matrix.values, fmt='', cmap="Blues", cbar=False)
    plt.title("Strategy Scores Matrix with Total Scores and Interaction Counts")
    plt.xlabel("Opponent (Strategy | Resource Level)")
    plt.ylabel("Client (Strategy | Resource Level)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.show()
    
    
def plot_strategy_total_scores_over_rounds(ipd_scoreboard_dict):
    """
    Plots the cumulative total scores obtained by each strategy over the server rounds.
    """
    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            strategy_label = f"{round_data[6]} | {round_data[7]}"  # Strategy name | Resource level
            payoff = round_data[5]  # Payoff

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, payoff))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative scores and totals for each strategy
    cumulative_scores_per_strategy = {strategy: [] for strategy in strategies}
    cumulative_totals = {strategy: 0 for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, payoff in data_list:
        data_by_round[server_round].append((strategy_label, payoff))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Append current cumulative totals to the lists
        for strategy in strategies:
            cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

        # Update cumulative totals with payoffs from the current round
        for strategy_label, payoff in data_by_round.get(server_round, []):
            cumulative_totals[strategy_label] += payoff

    # Append the final cumulative totals after the last round
    for strategy in strategies:
        cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

    # Extend the rounds list to match the length of cumulative scores lists
    extended_rounds = sorted_rounds + [sorted_rounds[-1] + 1]

    # Plot the cumulative total scores over rounds for each strategy
    plt.figure(figsize=(12, 8))
    for strategy, cumulative_scores in cumulative_scores_per_strategy.items():
        plt.plot(extended_rounds, cumulative_scores, label=strategy)

    plt.title("Cumulative Total Scores of Strategies Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Cumulative Total Score")
    plt.legend(title="Strategy")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    
    
def plot_strategy_score_differences_matrix(ipd_scoreboard_dict):
    """
    Plots a confusion matrix showing the score differences (sum_score1 - sum_score2)
    for each strategy pair.

    Each cell displays the difference in total scores between two strategies across all interactions.
    Positive values indicate that the client strategy performed better, negative values indicate that the opponent strategy performed better.
    """
    # Dictionary to store accumulated scores and interaction counts between unique strategy pairs
    # Key: (client_label, opponent_label)
    # Value: [sum_client_scores, sum_opponent_scores]
    interaction_data = {}

    # Iterate through each client to collect scores for unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]
            client_payoff = round_data[5]  # Client's payoff for this round

            # Ensure the opponent exists and retrieve opponent data for the same match
            if opponent_id in ipd_scoreboard_dict:
                # Find the opponent's data within the same match
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"
                    opponent_payoff = opponent_round[5]  # Opponent's payoff for this round

                    # Create a label pair
                    label_pair = (client_label, opponent_label)

                    # Initialize or update the interaction data
                    if label_pair not in interaction_data:
                        interaction_data[label_pair] = [client_payoff, opponent_payoff]
                    else:
                        interaction_data[label_pair][0] += client_payoff
                        interaction_data[label_pair][1] += opponent_payoff

    # Extract unique labels
    unique_labels = sorted(set(label for pair in interaction_data.keys() for label in pair))

    # Create a DataFrame to store the score differences
    matrix = pd.DataFrame(0.0, index=unique_labels, columns=unique_labels)

    # Fill the matrix with the score differences
    for (client_label, opponent_label), (sum_score1, sum_score2) in interaction_data.items():
        score_difference = sum_score1 - sum_score2
        matrix.at[client_label, opponent_label] = score_difference

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, annot=True, fmt=".1f", cmap="coolwarm", center=0, cbar=True)
    plt.title("Strategy Score Differences Matrix (Client Score - Opponent Score)")
    plt.xlabel("Opponent Strategy")
    plt.ylabel("Client Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()
    
    
    
def save_strategy_score_differences_matrix(ipd_scoreboard_dict, plot_directory='plots', filename='strategy_score_differences_matrix.png'):
    """
    Calculates the score differences (sum_score1 - sum_score2) for each strategy pair and saves the plot.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Dictionary to store accumulated scores between unique strategy pairs
    interaction_data = {}

    # Iterate through each client to collect scores for unique interactions
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            match_id = round_data[1]
            client_label = f"{round_data[6]} | {round_data[7]}"  # Client's strategy and resource level
            opponent_id = round_data[2]
            client_payoff = round_data[5]  # Client's payoff for this round

            # Ensure the opponent exists and retrieve opponent data for the same match
            if opponent_id in ipd_scoreboard_dict:
                # Find the opponent's data within the same match
                opponent_round = next((r for r in ipd_scoreboard_dict[opponent_id] if r[1] == match_id), None)
                if opponent_round:
                    opponent_label = f"{opponent_round[6]} | {opponent_round[7]}"
                    opponent_payoff = opponent_round[5]  # Opponent's payoff for this round

                    # Create a label pair
                    label_pair = (client_label, opponent_label)

                    # Initialize or update the interaction data
                    if label_pair not in interaction_data:
                        interaction_data[label_pair] = [client_payoff, opponent_payoff]
                    else:
                        interaction_data[label_pair][0] += client_payoff
                        interaction_data[label_pair][1] += opponent_payoff

    # Extract unique labels
    unique_labels = sorted(set(label for pair in interaction_data.keys() for label in pair))

    # Create a DataFrame to store the score differences
    matrix = pd.DataFrame(0.0, index=unique_labels, columns=unique_labels)

    # Fill the matrix with the score differences
    for (client_label, opponent_label), (sum_score1, sum_score2) in interaction_data.items():
        score_difference = sum_score1 - sum_score2
        matrix.at[client_label, opponent_label] = score_difference

    # Plot the confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap="coolwarm",
        center=0,
        cbar=True,
        linewidths=0.5,
        linecolor='gray'
    )
    plt.title("Strategy Score Differences Matrix (Client Score - Opponent Score)")
    plt.xlabel("Opponent Strategy")
    plt.ylabel("Client Strategy")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    
    
def save_strategy_total_scores_over_rounds(ipd_scoreboard_dict, plot_directory='plots', filename='strategy_total_scores_over_rounds.png'):
    """
    Plots the cumulative total scores obtained by each strategy over the server rounds and saves the plot to a file.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            #strategy_label = f"{round_data[6]} | {round_data[7]}"  # Strategy name | Resource level
            strategy_label = f"{round_data[6]}"  # Strategy name
            payoff = round_data[5]  # Payoff

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, payoff))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative scores and totals for each strategy
    cumulative_scores_per_strategy = {strategy: [] for strategy in strategies}
    cumulative_totals = {strategy: 0 for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, payoff in data_list:
        data_by_round[server_round].append((strategy_label, payoff))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Append current cumulative totals to the lists
        for strategy in strategies:
            cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

        # Update cumulative totals with payoffs from the current round
        for strategy_label, payoff in data_by_round.get(server_round, []):
            cumulative_totals[strategy_label] += payoff

    # Append the final cumulative totals after the last round
    for strategy in strategies:
        cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

    # Extend the rounds list to match the length of cumulative scores lists
    extended_rounds = sorted_rounds + [sorted_rounds[-1] + 1]

    # Plot the cumulative total scores over rounds for each strategy
    plt.figure(figsize=(12, 8))
    for strategy, cumulative_scores in cumulative_scores_per_strategy.items():
        plt.plot(extended_rounds, cumulative_scores, label=strategy)

    plt.title("Cumulative Total Scores of Strategies Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Cumulative Total Score")
    plt.legend(title="Strategy")
    plt.grid(False)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory
    

def save_strategy_total_scores_over_rounds_with_focus(ipd_scoreboard_dict, plot_directory='plots', filename='strategy_total_scores_over_rounds.png', focus_range=(100, 250)):
    """
    Plots the cumulative total scores obtained by each strategy over the server rounds and includes a focus subplot
    zoomed into a specified range of rounds. Saves the plot to a file.

    Parameters:
    - ipd_scoreboard_dict (dict): The scoreboard data to plot.
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.
    - focus_range (tuple): A tuple (a, b) specifying the range of rounds to focus on in the zoomed subplot.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            strategy_label = f"{round_data[6]}"  # Strategy name
            payoff = round_data[5]  # Payoff

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, payoff))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative scores and totals for each strategy
    cumulative_scores_per_strategy = {strategy: [] for strategy in strategies}
    cumulative_totals = {strategy: 0 for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, payoff in data_list:
        data_by_round[server_round].append((strategy_label, payoff))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Append current cumulative totals to the lists
        for strategy in strategies:
            cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

        # Update cumulative totals with payoffs from the current round
        for strategy_label, payoff in data_by_round.get(server_round, []):
            cumulative_totals[strategy_label] += payoff

    # Append the final cumulative totals after the last round
    for strategy in strategies:
        cumulative_scores_per_strategy[strategy].append(cumulative_totals[strategy])

    # Extend the rounds list to match the length of cumulative scores lists
    extended_rounds = sorted_rounds + [sorted_rounds[-1] + 1]

    # Use a high-contrast color palette
    contrast_colors = sns.color_palette("Set2", len(strategies))
    strategy_colors = {strategy: contrast_colors[i] for i, strategy in enumerate(strategies)}

    # Create side-by-side subplots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), gridspec_kw={'width_ratios': [3, 1]})

    # Full Plot
    for strategy, cumulative_scores in cumulative_scores_per_strategy.items():
        axes[0].plot(extended_rounds, cumulative_scores, label=strategy, color=strategy_colors[strategy])

    axes[0].set_title("Cumulative Total Scores of Strategies Over Server Rounds")
    axes[0].set_xlabel("Server Round")
    axes[0].set_ylabel("Cumulative Total Score")
    axes[0].legend(title="Strategy", loc="upper left")
    axes[0].grid(False)

    # Focused Subplot
    focus_start, focus_end = focus_range

    # Determine y-axis range for the focus area
    focus_y_min = float('inf')
    focus_y_max = float('-inf')
    for strategy, cumulative_scores in cumulative_scores_per_strategy.items():
        focus_scores = [cumulative_scores[i] for i in range(focus_start, focus_end + 1) if i < len(cumulative_scores)]
        if focus_scores:
            focus_y_min = min(focus_y_min, min(focus_scores))
            focus_y_max = max(focus_y_max, max(focus_scores))

    if focus_y_min == float('inf') or focus_y_max == float('-inf'):
        # If no valid data in the focus range, set a default range
        focus_y_min = 0
        focus_y_max = 1

    # Add padding to the y-axis limits for better visualization
    y_padding = (focus_y_max - focus_y_min) * 0.1  # 10% padding
    focus_y_min -= y_padding
    focus_y_max += y_padding

    for strategy, cumulative_scores in cumulative_scores_per_strategy.items():
        axes[1].plot(extended_rounds, cumulative_scores, label=strategy, color=strategy_colors[strategy])
    
    axes[1].set_xlim(focus_start, focus_end)
    axes[1].set_ylim(focus_y_min, focus_y_max)
    axes[1].set_title(f"Zoomed View: Rounds {focus_start} to {focus_end}")
    axes[1].set_xlabel("Server Round")
    axes[1].set_ylabel("Cumulative Total Score")
    axes[1].grid(False)

    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path, format="pdf", bbox_inches="tight")
    plt.close()  # Close the figure to free memory

def write_unique_matches_to_file(ipd_scoreboard_dict, filename='matches.txt'):
    """
    Writes all unique matches with the actions performed to a file, sorted by strategy.

    Each line in the file represents one match and includes:
    - Match ID
    - Server Round
    - Client 1 ID, Strategy, Resource Level, Action, Payoff
    - Client 2 ID, Strategy, Resource Level, Action, Payoff

    The matches are sorted by the strategies of the clients involved.

    Parameters:
    - filename (str): The name of the file to write the matches to.
    
    """
    # Dictionary to store match data
    matches_dict = {}

    # Collect match data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            server_round = round_data[0]
            match_id = round_data[1]
            opponent_id = round_data[2]
            client_play = round_data[3]
            client_payoff = round_data[5]
            client_strategy = round_data[6]
            client_res_level = round_data[7]

            # Initialize match entry if it doesn't exist
            if match_id not in matches_dict:
                matches_dict[match_id] = {
                    'server_round': server_round,
                    'clients': {}
                }

            # Add client data to the match
            matches_dict[match_id]['clients'][client_id] = {
                'client_id': client_id,
                'strategy': client_strategy,
                'res_level': client_res_level,
                'action': client_play,
                'payoff': client_payoff
            }

    # Convert the matches dictionary to a list for sorting
    matches_list = []
    for match_id, match_data in matches_dict.items():
        server_round = match_data['server_round']
        clients = match_data['clients']

        # Ensure we have data for both clients in the match
        if len(clients) != 2:
            continue  # Skip incomplete matches

        # Extract client data and sort them by strategy
        client_data_list = list(clients.values())
        client_data_list.sort(key=lambda x: x['strategy'])

        client1_data = client_data_list[0]
        client2_data = client_data_list[1]

        # Create a sorting key based on strategies
        sort_key = (client1_data['strategy'], client2_data['strategy'])

        # Append the match data and sorting key to the list
        matches_list.append((sort_key, match_id, server_round, client1_data, client2_data))

    # Sort the matches list by the sorting key (strategies)
    matches_list.sort()

    # Write match data to the file
    with open(filename, 'w') as file:
        for sort_key, match_id, server_round, client1_data, client2_data in matches_list:
            # Construct the line to write
            line = (
                f"Match ID: {match_id}, Server Round: {server_round}, "
                f"Client 1 ID: {client1_data['client_id']}, Strategy: {client1_data['strategy']}, "
                f"Resource Level: {client1_data['res_level']}, Action: {client1_data['action']}, "
                f"Payoff: {client1_data['payoff']}, "
                f"Client 2 ID: {client2_data['client_id']}, Strategy: {client2_data['strategy']}, "
                f"Resource Level: {client2_data['res_level']}, Action: {client2_data['action']}, "
                f"Payoff: {client2_data['payoff']}\n"
            )

            # Write the line to the file
            file.write(line)

    print(f"All unique matches have been written to '{filename}', sorted by strategy.")


def get_clients_score_overview(ipd_scoreboard_dict):
    """
    Provides an overview of how each client scored their points and returns it as a string.

    For each client, the following information is included:
    - Total Points
    - Number of Games
    - Average Points per Game
    - Actions Distribution (Cooperate vs. Defect)
    - Points Scored Against Each Opponent Strategy

    Returns:
    - A string containing the clients' score overview.
    """
    from collections import defaultdict

    # String list to collect output lines
    output_lines = []

    # Dictionary to hold client statistics
    client_stats = {}

    # Iterate over each client to collect statistics
    for client_id, rounds in ipd_scoreboard_dict.items():
        total_points = 0
        num_games = 0
        actions_count = defaultdict(int)
        opponent_strategy_points = defaultdict(float)
        opponent_strategy_games = defaultdict(int)
        strategy = None
        res_level = None

        for round_data in rounds:
            # round_data format:
            # (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            opponent_id = round_data[2]
            client_action = round_data[3]
            client_payoff = round_data[5]
            client_strategy = round_data[6]
            client_res_level = round_data[7]

            # Update client strategy and resource level (assumed constant)
            if strategy is None:
                strategy = client_strategy
                res_level = client_res_level

            total_points += client_payoff
            num_games += 1
            actions_count[client_action] += 1

            # Get opponent's strategy
            if opponent_id in ipd_scoreboard_dict:
                opponent_rounds = ipd_scoreboard_dict[opponent_id]
                opponent_round = next((r for r in opponent_rounds if r[1] == round_data[1]), None)
                if opponent_round:
                    opponent_strategy = opponent_round[6]
                    opponent_res_level = opponent_round[7]
                    opponent_label = f"{opponent_strategy} | {opponent_res_level}"

                    # Update points and games against this opponent strategy
                    opponent_strategy_points[opponent_label] += client_payoff
                    opponent_strategy_games[opponent_label] += 1

        average_points = total_points / num_games if num_games > 0 else 0

        # Store the statistics for the client
        client_stats[client_id] = {
            'Strategy': f"{strategy} | {res_level}",
            'Total Points': total_points,
            'Number of Games': num_games,
            'Average Points per Game': average_points,
            'Actions Distribution': dict(actions_count),
            'Points Against Opponent Strategies': dict(opponent_strategy_points),
            'Games Against Opponent Strategies': dict(opponent_strategy_games)
        }
        
    output_lines.append("-" * 80)
    # Build the output string
    for client_id, stats in client_stats.items():
        output_lines.append(f"Client ID: {client_id}")
        output_lines.append(f"  Strategy: {stats['Strategy']}")
        output_lines.append(f"  Total Points: {stats['Total Points']}")
        output_lines.append(f"  Number of Games: {stats['Number of Games']}")
        output_lines.append(f"  Average Points per Game: {stats['Average Points per Game']:.2f}")
        output_lines.append(f"  Actions Distribution:")
        for action, count in stats['Actions Distribution'].items():
            output_lines.append(f"    {action}: {count}")
        output_lines.append(f"  Points Scored Against Opponent Strategies:")
        for opponent_strategy, points in stats['Points Against Opponent Strategies'].items():
            games = stats['Games Against Opponent Strategies'][opponent_strategy]
            avg_points = points / games if games > 0 else 0
            output_lines.append(f"    {opponent_strategy}: Total Points = {points}, Games = {games}, Average Points = {avg_points:.2f}")
        output_lines.append("-" * 80)

    # Join the output lines into a single string
    output_string = "\n".join(output_lines)

    return output_string, client_stats


def plot_interaction_graph(ipd_scoreboard_dict, plot_directory='plots', filename='interaction_graph.png'):
    """
    Constructs and plots a graph of client interactions.
    
    - Nodes represent clients and can be labeled with their strategy and resource level.
    - Edges represent interactions between clients and can be weighted by the number of interactions or total payoff.
    
    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.
    """
    import os
    from collections import defaultdict

    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)
    
    # Create a directed graph
    G = nx.DiGraph()

    # Dictionary to store node attributes (strategy and resource level)
    node_attributes = {}

    # Dictionary to store edge weights (number of interactions and total payoffs)
    edge_weights = defaultdict(lambda: {'interactions': 0, 'total_payoff': 0})

    # Iterate over each client to collect interaction data
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            # Unpack round data
            # Format: (server_round, match_id, opponent_id, play, coplay, payoff, ipd_strategy, res_level)
            opponent_id = round_data[2]
            client_payoff = round_data[5]
            client_strategy = round_data[6]
            client_res_level = round_data[7]

            # Add client node with attributes if not already added
            if client_id not in node_attributes:
                node_attributes[client_id] = {
                    'strategy': client_strategy,
                    'res_level': client_res_level
                }
                G.add_node(client_id, strategy=client_strategy, res_level=client_res_level)

            # Add opponent node with attributes if not already added
            if opponent_id in ipd_scoreboard_dict and opponent_id not in node_attributes:
                opponent_rounds = ipd_scoreboard_dict[opponent_id]
                # Assume opponent's strategy and res_level are constant
                opponent_strategy = opponent_rounds[0][6]
                opponent_res_level = opponent_rounds[0][7]
                node_attributes[opponent_id] = {
                    'strategy': opponent_strategy,
                    'res_level': opponent_res_level
                }
                G.add_node(opponent_id, strategy=opponent_strategy, res_level=opponent_res_level)

            # Update edge weights
            edge_key = (client_id, opponent_id)
            edge_weights[edge_key]['interactions'] += 1
            edge_weights[edge_key]['total_payoff'] += client_payoff

    # Add edges to the graph with weights
    for (client_id, opponent_id), weights in edge_weights.items():
        G.add_edge(
            client_id,
            opponent_id,
            interactions=weights['interactions'],
            total_payoff=weights['total_payoff']
        )

    # Draw the graph
    plt.figure(figsize=(12, 10))

    # Position nodes using spring layout for better visualization
    pos = nx.spring_layout(G, k=0.5, iterations=50)

    # Node labels with strategy and resource level
    node_labels = {
        node: f"{node}\n{data['strategy']} | {data['res_level']}"
        for node, data in G.nodes(data=True)
    }

    # Edge widths based on number of interactions
    edge_widths = [G[u][v]['interactions'] for u, v in G.edges()]
    max_width = max(edge_widths) if edge_widths else 1
    edge_widths = [3 * width / max_width for width in edge_widths]  # Normalize widths

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='skyblue')

    # Draw edges
    nx.draw_networkx_edges(G, pos, arrowstyle='->', arrowsize=15, width=edge_widths, edge_color='gray')

    # Draw labels
    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8)

    plt.title("Client Interaction Graph")
    plt.axis('off')
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    print(f"Interaction graph has been saved to '{plot_path}'.")
    
    
def save_average_score_per_client_over_rounds(ipd_scoreboard_dict, plot_directory='plots', filename='average_score_per_client_over_rounds.png'):
    """
    Plots the average score per client over the server rounds and saves the plot to a file.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Collect all unique server rounds and clients
    all_rounds = set()
    clients = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            payoff = round_data[5]  # Payoff

            all_rounds.add(server_round)
            clients.add(client_id)
            data_list.append((server_round, client_id, payoff))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize cumulative scores and counts for each client
    average_scores_per_client = {client_id: [] for client_id in clients}
    cumulative_scores = {client_id: 0 for client_id in clients}
    cumulative_counts = {client_id: 0 for client_id in clients}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, client_id, payoff in data_list:
        data_by_round[server_round].append((client_id, payoff))

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Append current average scores to the lists
        for client_id in clients:
            if cumulative_counts[client_id] > 0:
                average_score = cumulative_scores[client_id] / cumulative_counts[client_id]
            else:
                average_score = 0
            average_scores_per_client[client_id].append(average_score)

        # Update cumulative scores and counts with payoffs from the current round
        for client_id, payoff in data_by_round.get(server_round, []):
            cumulative_scores[client_id] += payoff
            cumulative_counts[client_id] += 1

    # Append the final average scores after the last round
    for client_id in clients:
        if cumulative_counts[client_id] > 0:
            average_score = cumulative_scores[client_id] / cumulative_counts[client_id]
        else:
            average_score = 0
        average_scores_per_client[client_id].append(average_score)

    # Extend the rounds list to match the length of average scores lists
    extended_rounds = sorted_rounds + [sorted_rounds[-1] + 1]

    # Plot the average scores over rounds for each client
    plt.figure(figsize=(12, 8))
    for client_id, average_scores in average_scores_per_client.items():
        plt.plot(extended_rounds, average_scores, label=f"Client {client_id}")

    plt.title("Average Score per Client Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Average Score")
    plt.legend(title="Client")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to {plot_path}")
    
    
def plot_average_score_per_strategy_over_rounds(ipd_scoreboard_dict, plot_directory='plots', filename='average_score_per_strategy_over_rounds.png'):
    """
    Plots the average score per strategy over the server rounds and saves the plot to a file.

    Parameters:
    - plot_directory (str): The directory where the plot image will be saved.
    - filename (str): The filename for the saved plot image.

    The plot will be saved in the specified directory with the given filename.
    """
    # Ensure the plot directory exists
    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    # Collect all unique server rounds and strategies
    all_rounds = set()
    strategies = set()
    data_list = []

    # Gather data from ipd_scoreboard_dict
    for client_id, rounds in ipd_scoreboard_dict.items():
        for round_data in rounds:
            server_round = round_data[0]
            payoff = round_data[5]  # Payoff
            strategy_label = round_data[6]  # Strategy name

            all_rounds.add(server_round)
            strategies.add(strategy_label)
            data_list.append((server_round, strategy_label, payoff))

    # Sort the server rounds
    sorted_rounds = sorted(all_rounds)

    # Initialize average scores for each strategy
    average_scores_per_strategy = {strategy: [] for strategy in strategies}

    # Group data by server round
    data_by_round = defaultdict(list)
    for server_round, strategy_label, payoff in data_list:
        data_by_round[server_round].append((strategy_label, payoff))

    # Initialize previous averages for strategies
    previous_averages = {strategy: 0 for strategy in strategies}

    # Iterate over each server round in order
    for server_round in sorted_rounds:
        # Collect payoffs per strategy for the current round
        payoffs_per_strategy = defaultdict(list)
        for strategy_label, payoff in data_by_round.get(server_round, []):
            payoffs_per_strategy[strategy_label].append(payoff)

        # Calculate average scores for each strategy
        for strategy in strategies:
            if payoffs_per_strategy[strategy]:
                average_score = sum(payoffs_per_strategy[strategy]) / len(payoffs_per_strategy[strategy])
                previous_averages[strategy] = average_score  # Update previous average
            else:
                average_score = previous_averages[strategy]  # Use previous average if no data in this round
            average_scores_per_strategy[strategy].append(average_score)

    # Extend the rounds list if necessary
    rounds_to_plot = sorted_rounds

    # Plot the average scores over rounds for each strategy
    plt.figure(figsize=(12, 8))
    for strategy, average_scores in average_scores_per_strategy.items():
        plt.plot(rounds_to_plot, average_scores, label=strategy)

    plt.title("Average Score per Strategy Over Server Rounds")
    plt.xlabel("Server Round")
    plt.ylabel("Average Score")
    plt.legend(title="Strategy")
    plt.grid(True)
    plt.tight_layout()

    # Save the plot to the specified directory with the given filename
    plot_path = os.path.join(plot_directory, filename)
    plt.savefig(plot_path)
    plt.close()  # Close the figure to free memory

    print(f"Plot saved to {plot_path}")