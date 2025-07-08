

import sys
import json
import numpy as np
import random
import math
import copy
from typing import List, Tuple, Dict

class VRPSolver:
    def __init__(self, distance_matrix: np.ndarray, num_vehicles: int, depot: int = 0):
        self.distance_matrix = distance_matrix
        self.num_vehicles = num_vehicles
        self.depot = depot
        self.num_nodes = len(distance_matrix)
        self.customers = list(range(1, self.num_nodes))  # Exclude depot
        
    def calculate_savings(self) -> List[Tuple[int, int, float]]:
        """Calculate savings for all customer pairs using Clarke-Wright formula"""
        savings = []
        
        for i in range(1, self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                # Savings = distance(depot, i) + distance(depot, j) - distance(i, j)
                saving = (self.distance_matrix[self.depot][i] + 
                         self.distance_matrix[self.depot][j] - 
                         self.distance_matrix[i][j])
                savings.append((i, j, saving))
        
        # Sort by savings in descending order
        savings.sort(key=lambda x: x[2], reverse=True)
        return savings
    
    def clarke_wright_algorithm(self) -> List[List[int]]:
        """Implement Clarke-Wright algorithm for initial VRP solution"""
        # Initialize: each customer is in its own route
        routes = [[customer] for customer in self.customers]
        
        # Calculate savings
        savings = self.calculate_savings()
        
        # Process savings in descending order
        for i, j, saving in savings:
            if saving <= 0:
                break
                
            # Find routes containing customers i and j
            route_i = None
            route_j = None
            route_i_idx = None
            route_j_idx = None
            
            for idx, route in enumerate(routes):
                if i in route:
                    route_i = route
                    route_i_idx = idx
                if j in route:
                    route_j = route
                    route_j_idx = idx
            
            # Check if customers are in different routes and can be merged
            if (route_i is not None and route_j is not None and 
                route_i != route_j and len(routes) > self.num_vehicles):
                
                # Check if customers are at the ends of their routes
                i_at_end = (route_i[0] == i or route_i[-1] == i)
                j_at_end = (route_j[0] == j or route_j[-1] == j)
                
                if i_at_end and j_at_end:
                    # Merge routes
                    if route_i[-1] == i and route_j[0] == j:
                        # i is at end of route_i, j is at start of route_j
                        merged_route = route_i + route_j
                    elif route_i[0] == i and route_j[-1] == j:
                        # i is at start of route_i, j is at end of route_j
                        merged_route = route_j + route_i
                    elif route_i[-1] == i and route_j[-1] == j:
                        # Both at ends, reverse one route
                        merged_route = route_i + route_j[::-1]
                    elif route_i[0] == i and route_j[0] == j:
                        # Both at starts, reverse one route
                        merged_route = route_i[::-1] + route_j
                    else:
                        continue
                    
                    # Remove old routes and add merged route
                    routes = [route for idx, route in enumerate(routes) 
                             if idx != route_i_idx and idx != route_j_idx]
                    routes.append(merged_route)
        
        # Ensure we have exactly num_vehicles routes
        while len(routes) < self.num_vehicles:
            # Split the longest route
            longest_route_idx = max(range(len(routes)), key=lambda i: len(routes[i]))
            longest_route = routes[longest_route_idx]
            
            if len(longest_route) >= 2:
                mid = len(longest_route) // 2
                route1 = longest_route[:mid]
                route2 = longest_route[mid:]
                
                routes[longest_route_idx] = route1
                routes.append(route2)
            else:
                # If we can't split anymore, add empty routes
                routes.append([])
        
        # Merge routes if we have too many
        while len(routes) > self.num_vehicles:
            # Find two shortest routes to merge
            route_lengths = [(i, len(route)) for i, route in enumerate(routes)]
            route_lengths.sort(key=lambda x: x[1])
            
            if len(route_lengths) >= 2:
                idx1, idx2 = route_lengths[0][0], route_lengths[1][0]
                merged = routes[idx1] + routes[idx2]
                routes = [route for i, route in enumerate(routes) if i != idx1 and i != idx2]
                routes.append(merged)
        
        return routes
    
    def calculate_route_distance(self, route: List[int]) -> float:
        """Calculate total distance for a route including depot"""
        if not route:
            return 0.0
        
        total_distance = 0.0
        # From depot to first customer
        total_distance += self.distance_matrix[self.depot][route[0]]
        
        # Between customers
        for i in range(len(route) - 1):
            total_distance += self.distance_matrix[route[i]][route[i + 1]]
        
        # From last customer back to depot
        total_distance += self.distance_matrix[route[-1]][self.depot]
        
        return total_distance
    
    def calculate_total_distance(self, routes: List[List[int]]) -> float:
        """Calculate total distance for all routes"""
        return sum(self.calculate_route_distance(route) for route in routes)
    
    def two_opt_improvement(self, route: List[int]) -> List[int]:
        """Apply 2-opt improvement to a single route"""
        if len(route) < 3:
            return route
        
        best_route = route[:]
        best_distance = self.calculate_route_distance(best_route)
        improved = True
        
        while improved:
            improved = False
            for i in range(len(route) - 1):
                for j in range(i + 2, len(route)):
                    # Create new route by reversing the segment between i and j
                    new_route = route[:i+1] + route[i+1:j+1][::-1] + route[j+1:]
                    new_distance = self.calculate_route_distance(new_route)
                    
                    if new_distance < best_distance:
                        best_route = new_route
                        best_distance = new_distance
                        improved = True
            
            route = best_route[:]
        
        return best_route
    
    def apply_two_opt_to_all_routes(self, routes: List[List[int]]) -> List[List[int]]:
        """Apply 2-opt improvement to all routes"""
        improved_routes = []
        for route in routes:
            if len(route) > 2:
                improved_route = self.two_opt_improvement(route)
                improved_routes.append(improved_route)
            else:
                improved_routes.append(route)
        
        return improved_routes
    
    def get_random_neighbor(self, routes: List[List[int]]) -> List[List[int]]:
        """Generate a random neighbor solution"""
        new_routes = copy.deepcopy(routes)
        
        # Choose random operation
        operations = ['relocate', 'swap', 'reverse']
        operation = random.choice(operations)
        
        if operation == 'relocate':
            # Relocate a customer from one route to another
            non_empty_routes = [i for i, route in enumerate(new_routes) if route]
            if len(non_empty_routes) >= 1:
                source_route_idx = random.choice(non_empty_routes)
                target_route_idx = random.randint(0, len(new_routes) - 1)
                
                if new_routes[source_route_idx]:
                    customer = new_routes[source_route_idx].pop(random.randint(0, len(new_routes[source_route_idx]) - 1))
                    target_pos = random.randint(0, len(new_routes[target_route_idx]))
                    new_routes[target_route_idx].insert(target_pos, customer)
        
        elif operation == 'swap':
            # Swap two customers between different routes
            non_empty_routes = [i for i, route in enumerate(new_routes) if route]
            if len(non_empty_routes) >= 2:
                route1_idx, route2_idx = random.sample(non_empty_routes, 2)
                
                if new_routes[route1_idx] and new_routes[route2_idx]:
                    pos1 = random.randint(0, len(new_routes[route1_idx]) - 1)
                    pos2 = random.randint(0, len(new_routes[route2_idx]) - 1)
                    
                    new_routes[route1_idx][pos1], new_routes[route2_idx][pos2] = \
                        new_routes[route2_idx][pos2], new_routes[route1_idx][pos1]
        
        elif operation == 'reverse':
            # Reverse a segment within a route
            non_empty_routes = [i for i, route in enumerate(new_routes) if len(route) > 2]
            if non_empty_routes:
                route_idx = random.choice(non_empty_routes)
                route = new_routes[route_idx]
                
                i = random.randint(0, len(route) - 2)
                j = random.randint(i + 1, len(route) - 1)
                
                new_routes[route_idx] = route[:i] + route[i:j+1][::-1] + route[j+1:]
        
        return new_routes
    
    def simulated_annealing(self, initial_routes: List[List[int]], 
                          initial_temp: float = 10000.0, 
                          cooling_rate: float = 0.95,
                          min_temp: float = 1.0,
                          max_iterations: int = 1000) -> List[List[int]]:
        """Apply simulated annealing to improve the solution"""
        current_routes = copy.deepcopy(initial_routes)
        current_cost = self.calculate_total_distance(current_routes)
        
        best_routes = copy.deepcopy(current_routes)
        best_cost = current_cost
        
        temperature = initial_temp
        iteration = 0
        
        while temperature > min_temp and iteration < max_iterations:
            # Generate neighbor solution
            neighbor_routes = self.get_random_neighbor(current_routes)
            neighbor_cost = self.calculate_total_distance(neighbor_routes)
            
            # Calculate acceptance probability
            if neighbor_cost < current_cost:
                # Accept better solution
                current_routes = neighbor_routes
                current_cost = neighbor_cost
                
                # Update best solution if necessary
                if current_cost < best_cost:
                    best_routes = copy.deepcopy(current_routes)
                    best_cost = current_cost
            else:
                # Accept worse solution with probability
                delta = neighbor_cost - current_cost
                probability = math.exp(-delta / temperature)
                
                if random.random() < probability:
                    current_routes = neighbor_routes
                    current_cost = neighbor_cost
            
            # Cool down
            temperature *= cooling_rate
            iteration += 1
        
        return best_routes
    
    def solve(self) -> Tuple[float, List[List[int]]]:
        """Main solving method combining all algorithms"""
        # Step 1: Initial solution using Clarke-Wright
        print("Applying Clarke-Wright algorithm...", file=sys.stderr)
        routes = self.clarke_wright_algorithm()
        initial_cost = self.calculate_total_distance(routes)
        print(f"Initial cost: {initial_cost:.2f}", file=sys.stderr)
        
        # Step 2: Improve with 2-opt
        print("Applying 2-opt improvement...", file=sys.stderr)
        routes = self.apply_two_opt_to_all_routes(routes)
        two_opt_cost = self.calculate_total_distance(routes)
        print(f"After 2-opt: {two_opt_cost:.2f}", file=sys.stderr)
        
        # Step 3: Further improve with simulated annealing
        print("Applying simulated annealing...", file=sys.stderr)
        routes = self.simulated_annealing(routes)
        final_cost = self.calculate_total_distance(routes)
        print(f"Final cost: {final_cost:.2f}", file=sys.stderr)
        
        # Add depot (0) to the beginning and end of each route for output
        output_routes = []
        for route in routes:
            if route:  # Only add non-empty routes
                output_route = [self.depot] + route + [self.depot]
                output_routes.append(output_route)
        
        return final_cost, output_routes

def main():
    if len(sys.argv) != 3:
        print("Usage: python VRP_script.py <distance_matrix_file> <num_vehicles>", file=sys.stderr)
        sys.exit(1)
    
    distance_matrix_file = sys.argv[1]
    num_vehicles = int(sys.argv[2])
    
    try:
        # Load distance matrix
        distance_matrix = np.loadtxt(distance_matrix_file, delimiter=',')
        
        # Create solver
        solver = VRPSolver(distance_matrix, num_vehicles)
        
        # Solve VRP
        total_cost, vehicle_routes = solver.solve()
        
        # Prepare output
        result = {
            "Total cost": total_cost,
            "Vehicle routes": vehicle_routes
        }
        
        # Output JSON result
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()