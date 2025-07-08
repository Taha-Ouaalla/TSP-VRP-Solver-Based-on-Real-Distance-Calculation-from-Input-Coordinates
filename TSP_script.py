import numpy as np
import sys
import json
import math

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

def read_places_and_matrix(filename):
    """ Read the place names and the distance matrix from a CSV file. """
    # Load the entire file as strings to manipulate place names easily
    data = np.genfromtxt(filename, delimiter=',', skip_header=1, dtype=str)
    # Extract place names, stripping quotes and trimming spaces
    places = [place.strip().strip('"') for place in data[:, 0]]
    # Convert the rest of the data to floats for distance matrix
    distance_matrix = np.array(data[:, 1:], dtype=float)
    return places, distance_matrix

def calculate_tour_cost(tour, distance_matrix):
    """ Calculate the total cost of a tour based on the distance matrix. """
    total_cost = sum(distance_matrix[tour[i]][tour[(i+1) % len(tour)]] for i in range(len(tour)))
    return total_cost

def nearest_neighbor_heuristic(distance_matrix):
    """ Generate an initial tour using the nearest neighbor heuristic. """
    N = len(distance_matrix)
    start = 0
    visited = [False] * N
    path = [start]
    visited[start] = True

    for _ in range(N - 1):
        last = path[-1]
        next_city = np.argmin([distance_matrix[last][j] if not visited[j] else float('inf') for j in range(N)])
        path.append(next_city)
        visited[next_city] = True

    path.append(start)  # Return to the starting city
    return path

def two_opt(distance_matrix, tour):
    """ Improve the tour by repeatedly applying the 2-opt heuristic. """
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 2, n - 1 + (i > 0)):
                ni = (i + 1) % n
                nj = (j + 1) % n
                cur_length = distance_matrix[tour[i]][tour[ni]] + distance_matrix[tour[j]][tour[nj]]
                new_length = distance_matrix[tour[i]][tour[j]] + distance_matrix[tour[ni]][tour[nj]]
                if new_length < cur_length:
                    tour[i + 1:j + 1] = reversed(tour[i + 1:j + 1])
                    improved = True
    return tour

def simulated_annealing(distance_matrix, initial_tour, initial_temp=1000, cooling_rate=0.999, min_temp=1, num_runs=10):
    """ Optimize the tour using simulated annealing. """
    best_tour = None
    best_cost = float('inf')
    for run in range(num_runs):
        current_tour = initial_tour[:]
        current_cost = calculate_tour_cost(current_tour, distance_matrix)
        temp = initial_temp

        while temp > min_temp:
            i, j = np.random.randint(1, len(current_tour)-1), np.random.randint(1, len(current_tour)-1)
            if i > j:
                i, j = j, i
            new_tour = current_tour[:]
            new_tour[i:j+1] = new_tour[i:j+1][::-1]
            new_cost = calculate_tour_cost(new_tour, distance_matrix)
            if new_cost < current_cost or np.random.rand() <= math.exp((current_cost - new_cost) / temp):
                current_tour = new_tour
                current_cost = new_cost
            temp *= cooling_rate

        if current_cost < best_cost:
            best_tour = current_tour
            best_cost = current_cost

    return best_tour, best_cost

def solve_tsp(distance_matrix):
    """ Solve the TSP by generating an initial tour and optimizing it. """
    initial_tour = nearest_neighbor_heuristic(distance_matrix)
    improved_tour = two_opt(distance_matrix, initial_tour)
    final_tour, final_cost = simulated_annealing(distance_matrix, improved_tour)
    return final_cost, final_tour

def main():
    if len(sys.argv) != 2:
        print("Usage: python TSP_script.py <file_path>", file=sys.stderr)
        sys.exit(1)

    try:
        places, distance_matrix = read_places_and_matrix(sys.argv[1])
        cost, path_indices = solve_tsp(distance_matrix)
        path_names = [places[index] for index in path_indices]
        
        result = {
            "Optimized cost": cost,
            "Optimized path": path_names
        }
        print(json.dumps(result, cls=NumpyEncoder))
    except Exception as e:
        print(f"Failed to process the file: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
