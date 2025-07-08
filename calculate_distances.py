import sys
import osmnx as ox
import networkx as nx
import numpy as np

def read_locations():
    """
    Read locations from standard input. Expects lines formatted as: name,latitude,longitude
    """
    locations = []
    for line in sys.stdin:
        parts = line.strip().split(',')
        if len(parts) != 3:
            continue
        name, lat, lon = parts
        try:
            locations.append((name, float(lat), float(lon)))
        except ValueError:
            print(f"Error processing line: {line.strip()}", file=sys.stderr)
            continue
    return locations

def calculate_distances(locations, place):
    """
    Calculate the distance matrix for the given locations within the specified place.
    """
    try:
        graph = ox.graph_from_place(place, network_type='all')
    except Exception as e:
        print(f"Failed to fetch graph for the place {place}: {e}", file=sys.stderr)
        sys.exit(2)  # Exit with an error code indicating graph fetch failure

    n = len(locations)
    distance_matrix = np.zeros((n, n))
    node_ids = []

    # Fetch nearest nodes for each location
    for name, lat, lon in locations:
        try:
            node = ox.nearest_nodes(graph, lon, lat)
            node_ids.append(node)
        except Exception as e:
            print(f"Failed to find nearest node for {name}: {e}", file=sys.stderr)
            sys.exit(3)

    # Calculate shortest path distances using NetworkX
    for i in range(n):
        for j in range(i + 1, n):
            try:
                distance = nx.shortest_path_length(graph, node_ids[i], node_ids[j], weight='length')
                distance_matrix[i, j] = distance_matrix[j, i] = distance
            except nx.NetworkXNoPath:
                print(f"No path found between {locations[i][0]} and {locations[j][0]}", file=sys.stderr)
                distance_matrix[i, j] = distance_matrix[j, i] = float('inf')

    return distance_matrix

def main():
    if len(sys.argv) != 2:
        print("Usage: python calculate_distances.py 'City, Country'", file=sys.stderr)
        sys.exit(1)  # Exit with an error code indicating usage error

    place = sys.argv[1]
    locations = read_locations()
    distance_matrix = calculate_distances(locations, place)
    
    # Print the distance matrix as rows of comma-separated values
    for row in distance_matrix:
        print(','.join(map(str, row)))

if __name__ == "__main__":
    main()
