import copy
import os
import random
import util
import axelrod as axl
from axelrod.action import Action
from axelrod.strategies import WinStayLoseShift
import numpy as np
import matplotlib.pyplot as plt

from ipd_player import ClientShadowPlayer, ResourceAwareMemOnePlayer, RandomIPDPlayer

NUM_GAMES = 20

random.seed(42)

def visualize_scaling():
    # Define parameters
    E_tilde = np.linspace(0, 1.0, 500)  # E_tilde from 0 to 1.0
    E_low = util.ResourceLevel.LOW.value  # Lower threshold
    

    # Create a function to plot and save the figure
    def plot_and_save(output_folder, file_name):
        plt.figure(figsize=(8, 6))

        # Compute scaling values
        linear = util.linear_scaling(E_tilde)
        exponential = util.exponential_scaling(E_tilde, alpha=3)
        logistic = util.logistic_scaling(E_tilde, gamma=15)
        piecewise = util.piecewise_linear_scaling(E_tilde)
        inverse = util.inverse_scaling(E_tilde, beta=5)
        synergy = util.synergy_threshold_scaling(E_tilde, gamma=8)
        
        plt.plot(E_tilde, linear, label="Linear Scaling", color='crimson')
        plt.plot(E_tilde, exponential, label="Exponential Scaling", color='goldenrod')
        plt.plot(E_tilde, logistic, label="Logistic Scaling", color='darkslategray')
        #plt.plot(E_tilde, piecewise, label="Piecewise Linear Scaling")
        plt.plot(E_tilde, inverse, label="Inverse Scaling", color='skyblue')
        plt.plot(E_tilde, synergy, label="Synergy Threshold", color='magenta')
        
        # Highlight areas with different pastel colors
        plt.axvspan(0, E_low, color='lightcoral', alpha=0.2, label="Low Resources")
        plt.axvspan(E_low, 0.5, color='peachpuff', alpha=0.6, label="Moderate Resources")
        plt.axvspan(0.5, 0.75, color='lightgreen', alpha=0.2, label="High Resources")
        plt.axvspan(0.75, 1.0, color='darkseagreen', alpha=0.4, label="Full Resources")

        # Add custom x-axis marks without numerical values except for 0 and 1.0
        x_ticks = [0, E_low, 0.5, 0.75, 1.0]
        x_labels = ["0", r"$E_{\mathrm{Low}}$", r"$E_{\mathrm{Moderate}}$", r"$E_{\mathrm{High}}$", r"$E_{\mathrm{Max}} = 1.0$"]
        plt.xticks(ticks=x_ticks, labels=x_labels)

        # Add labels, legend, and title
        plt.xlabel(r"normalized resource factor $\tilde{E}_i$")
        plt.ylabel(r"$f_{\mathrm{res}}(\tilde{E}_i)$")
        #plt.title(r"Synergy Threshold Function")
        plt.legend()
        plt.grid(True)

        # Save the plot to the specified folder
        os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
        save_path = os.path.join(output_folder, file_name)
        plt.savefig(save_path, bbox_inches='tight', dpi=1200)  # Save with tight layout
        plt.show()
        print(f"Plot saved to: {save_path}")

    # Example usage
    output_folder = "plots"  # Specify your folder here
    file_name = "highlighted_regions_plot.png"  # Specify your file name here

    # Call the function to plot and save
    plot_and_save(output_folder, file_name)

def test_run(client_player):
    # history list received form server instance
    plays = list()
    coplays = list()
    
    for i in range(NUM_GAMES):
        # init shadow player
        surrogate = ClientShadowPlayer()
        surrogate.set_seed(42)
        
        # reset client strategy
        client_player.reset()
        client_player.set_seed(42)
        
        # inject memory if exists
        if len(plays) > 0:
            surrogate._history.extend(coplays, plays)
            client_player._history.extend(plays, coplays)
        
        # evaluate next move based on given history
        next_action = client_player.strategy(opponent=surrogate)
        plays.append(next_action)
        # assign random action to coplays
        coplays.append(random.choice([Action.C, Action.D]))
        
    print("Finished test run")
    plays_list = "| ".join(play.name for play in plays)
    print("Actions of player one:" + plays_list)
    coplays_list = "| ".join(coplay.name for coplay in coplays)
    print("Actions of player two:" + coplays_list)
    print("----------------")
    

def main():

    # initialize player
    client_player = axl.StochasticWSLS(0)
    #client_player.name = "Random"
    #client_player = axl.GTFT(p=0.0)
    #client_player.name = "T4T"
    client_player.set_seed(42)
    
    res_aware_client_player = ResourceAwareMemOnePlayer(player_instance=copy.deepcopy(client_player),
                                                        resource_scaling_func=util.synergy_threshold_scaling,
                                                        initial_resource_value=util.ResourceLevel.FULL.value)
    
    test_run(res_aware_client_player)
    
    
if __name__ == '__main__':
    visualize_scaling()
    main()