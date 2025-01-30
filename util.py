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

import random
import os, csv
from enum import Enum
from axelrod.action import Action
import numpy as np


TRACK_PROBABILITY = False

# Global dictionary to track sampling probabilities and strategies
sampling_data = {}

# CSV file path
csv_file = "sampling_probabilities.csv"

# Initialize CSV file if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode="w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["Round", "Client ID", "Probability", "Strategy"])  # CSV Headers



# global seed
SEED = 42
random.seed(SEED)

class ClientSamplingStrategy(Enum):
    RANDOM = 1
    MORAN = 2
    

class ResourceLevel(float, Enum):
    NONE = -1.0
    EMPTY = 0.0 + 0.1E-6
    LOW = 0.25
    MODERATE = 0.5
    HIGH = 0.75
    FULL = 1.0
    
    @classmethod
    def from_float(cls, value: float):
        """
        Maps a float to the corresponding ResourceLevel enum member.
        
        Parameters:
        - value (float): The float value to map to a ResourceLevel.
        
        Returns:
        - ResourceLevel: The matching enum member, or raises ValueError if not found.
        """
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"No matching ResourceLevel for value: {value}")
    
    def to_string(self):
            """
            Converts the ResourceLevel enum member to its string name.
            
            Returns:
            - str: The name of the ResourceLevel member.
            """
            return self.name
        
    
def generate_hash(x_id, y_id, use_cantor=False):
    sorted_x, sorted_y = sorted([x_id, y_id])
    if use_cantor:
        return sorted_x, sorted_y, str(cantor_pairing(sorted_x, sorted_y))
    else:
        # simple concat str
        hash_str = str(sorted_x) + "_" + str(sorted_y)
        return sorted_x, sorted_y, hash_str

def decode_hash(hash_str, use_cantor=False):
    if use_cantor:
        x, y = reverse_cantor_pairing(hash_str)
        return str(x), str(y)
    else:
        str_lst = hash_str.split("_")
        return str_lst[0], str_lst[1]

def cantor_pairing(x, y):
    """Combine two non-negative integers into a single hash key using Cantor's pairing function."""
    return (x + y) * (x + y + 1) // 2 + y

def reverse_cantor_pairing(z):
    """Retrieve the original pair of numbers (x, y) from the Cantor's pairing function result."""
    # Solve for x and y from the given z (the hash key)
    w = int(((8 * z + 1)**0.5 - 1) // 2)  # Inverse of the quadratic equation
    t = (w * (w + 1)) // 2
    y = z - t
    x = w - y
    return x, y

def append_bool_to_msb(n, new_bool):
    # Find the number of bits in the integer
    num_bits = n.bit_length()
    
    # Shift the integer left by 1 to make space for the new MSB
    n = n << 1
    
    # If the new boolean is True, set the most significant bit to 1
    if new_bool:
        n += 1 << num_bits  # Add 1 at the MSB position
    
    return n


def actions_to_string(actions):
    # Convert each Action to its name (or value, depending on your preference)
    return ''.join(action.name for action in actions)
        
    
def string_to_actions(action_str):
    # Map each character in the string to the corresponding Action enum
    action_map = {
        'D': Action.D,
        'C': Action.C
    }
    return [action_map[char] for char in action_str]



def linear_scaling(res_lvl):
    """
    Computes a linear scaling function:
    f_res(res_lvl) = (res_lvl - EMPTY) / (FULL - EMPTY) if EMPTY <= res_lvl <= FULL
                     1 if res_lvl > FULL
                     0 if res_lvl < EMPTY

    Parameters:
    - res_lvl: Input energy or resource levels (array or scalar)
    
    Returns:
    - The computed result of the function
    """
    res_lvl = np.asarray(res_lvl)  # Ensure input is an array
    scaled = (res_lvl - ResourceLevel.EMPTY.value) / (ResourceLevel.FULL.value - ResourceLevel.EMPTY.value)
    return np.clip(scaled, 0.0, 1.0)  # Clip values to [0, 1]



def synergy_threshold_scaling(res_lvl, gamma=8):
    """
    Computes the value of the function:
    f_res(res_lvl) = 0.5 * (1 + tanh(gamma * (res_lvl - E_low)))
    
    Parameters:
    - res_lvl: Input energy or value (array or scalar)
    - E_low: Lower energy threshold
    - gamma: Scaling factor controlling the steepness around E_low

    Returns:
    - The computed result of the function
    """
    return 0.5 * (1 + np.tanh(gamma * (res_lvl - ResourceLevel.LOW.value)))


def exponential_scaling(res_lvl, alpha=2.0):
    """
    Computes an exponential scaling function:
    f_res(res_lvl) = 1 - exp(-alpha * (res_lvl - EMPTY)) if res_lvl >= EMPTY
                     0 if res_lvl < EMPTY

    Parameters:
    - res_lvl: Input energy or resource levels (array or scalar)
    - alpha: Scaling factor controlling the steepness of the curve
    
    Returns:
    - The computed result of the function
    """
    res_lvl = np.asarray(res_lvl)
    mask = res_lvl >= ResourceLevel.EMPTY.value
    result = np.zeros_like(res_lvl)
    result[mask] = 1 - np.exp(-alpha * (res_lvl[mask] - ResourceLevel.LOW.value))
    return result


def logistic_scaling(res_lvl, gamma=10):
    """
    Computes a logistic scaling function:
    f_res(res_lvl) = 1 / (1 + exp(-gamma * (res_lvl - MODERATE)))

    Parameters:
    - res_lvl: Input energy or resource levels (array or scalar)
    - gamma: Controls the steepness of the sigmoid curve
    
    Returns:
    - The computed result of the function
    """
    res_lvl = np.asarray(res_lvl)
    return 1 / (1 + np.exp(-gamma * (res_lvl - ResourceLevel.MODERATE.value)))


def piecewise_linear_scaling(res_lvl):
    """
    Computes a piecewise linear scaling function:
    f_res(res_lvl) = 0 if res_lvl < EMPTY
                     (res_lvl - EMPTY) / (MODERATE - EMPTY) if EMPTY <= res_lvl < MODERATE
                     1 if MODERATE <= res_lvl <= HIGH
                     1 - (res_lvl - HIGH) / (FULL - HIGH) if res_lvl > HIGH

    Parameters:
    - res_lvl: Input energy or resource levels (array or scalar)
    
    Returns:
    - The computed result of the function
    """
    res_lvl = np.asarray(res_lvl)
    result = np.zeros_like(res_lvl)

    # EMPTY to MODERATE
    mask1 = (res_lvl >= ResourceLevel.EMPTY.value) & (res_lvl < ResourceLevel.MODERATE.value)
    result[mask1] = (res_lvl[mask1] - ResourceLevel.EMPTY.value) / (ResourceLevel.MODERATE.value - ResourceLevel.EMPTY.value)

    # MODERATE to HIGH
    mask2 = (res_lvl >= ResourceLevel.MODERATE.value) & (res_lvl <= ResourceLevel.HIGH.value)
    result[mask2] = 1.0

    # HIGH to FULL
    mask3 = res_lvl > ResourceLevel.HIGH.value
    result[mask3] = 1 - (res_lvl[mask3] - ResourceLevel.HIGH.value) / (ResourceLevel.FULL.value - ResourceLevel.HIGH.value)

    return result

def inverse_scaling(res_lvl, beta=1.0):
    """
    Computes an inverse scaling function:
    f_res(res_lvl) = 1 / (1 + beta * (FULL - res_lvl))

    Parameters:
    - res_lvl: Input energy or resource levels (array or scalar)
    - beta: Controls how rapidly the function decreases near EMPTY
    
    Returns:
    - The computed result of the function
    """
    res_lvl = np.asarray(res_lvl)
    result = np.zeros_like(res_lvl)
    mask = res_lvl >= ResourceLevel.EMPTY.value
    result[mask] = 1 / (1 + beta * (ResourceLevel.FULL.value - res_lvl[mask]))
    return result


    


def random_action_choice(p: float = 0.5) -> Action:
        """
        Return C with probability `p`, else return D

        No random sample is carried out if p is 0 or 1.

        Parameters
        ----------
        p : float
            The probability of picking C

        Returns
        -------
        axelrod.Action
        """
        if p == 0:
            return Action.D

        if p == 1:
            return Action.C

        r = random.uniform(0.0, 1.0)
        if r < p:
            return Action.C
        return Action.D
    
    
def moran_sampling(scoreboard_list, available_clients, k, weight=1, round_number = 1, threshold=50):

    # TRACKING HERE ->
    global sampling_data  # Track probabilities globally
    # TRACKING HERE ->
    
    if round_number < threshold:
        random_warmup = True
    else:
        random_warmup = False
     
    # Convert scoreboard_list into a dictionary for quick lookup
    #scoreboard_dict = {entry['client_id']: entry for entry in scoreboard_list}
    scoreboard_dict = {entry[0]: entry[1] for entry in scoreboard_list}

    # Extract actual client IDs from available_clients
    client_ids = [tup[1] for tup in available_clients]
    client_uids = [tup[0] for tup in available_clients]

    # Extract scores for available clients (defaulting to None if not in scoreboard)
    # scores = np.array([scoreboard_dict.get(client_id, None)[0] for client_id in client_ids])
     # Extract scores safely (default to 0 if client_id is missing)
    scores = np.array([
        scoreboard_dict[client_id][0] if client_id in scoreboard_dict else None
        for client_id in client_ids
    ])

    # Identify missing clients
    missing_mask = np.array([s is None for s in scores])

    if np.all(missing_mask) or random_warmup:  # If all clients are missing or warmup, assign uniform probabilities
        probabilities = np.ones(len(client_ids)) / len(client_ids)
    else:
        # Replace missing scores with the average score to avoid errors
        avg_score = np.mean([s for s in scores if s is not None])
        scores[missing_mask] = avg_score

        # Normalize scores to compute fitness values
        min_score = np.min(scores)
        max_score = np.max(scores)
        
        if min_score == max_score:
            probabilities = np.ones(len(client_ids)) / len(client_ids)
        else:
            fitness = 1.0 + weight * (scores.astype(float) - min_score) / (max_score - min_score)
            total_fitness = np.sum(fitness)
            probabilities = fitness / total_fitness
    
    if TRACK_PROBABILITY:
        # Track probabilities and strategies
        new_rows = []
        for i, client_id in enumerate(client_ids):
            strategy = scoreboard_dict.get(client_id, ('Unknown', 'Unknown'))[1]
            if client_id not in sampling_data:
                sampling_data[client_id] = {"probabilities": [], "strategy": strategy}
            sampling_data[client_id]["probabilities"].append(probabilities[i])
            sampling_data[client_id]["strategy"] = strategy

            # Append to CSV data
            new_rows.append([round_number, client_id, probabilities[i], strategy])

        # Write to CSV
        with open(csv_file, mode="a", newline="") as file:
            writer = csv.writer(file)
            writer.writerows(new_rows)
        

    # Sample k clients without replacement
    selected_indices = np.random.choice(len(client_ids), size=k, replace=False, p=probabilities)
    selected_clients = [client_uids[i] for i in selected_indices]
    
    return selected_clients