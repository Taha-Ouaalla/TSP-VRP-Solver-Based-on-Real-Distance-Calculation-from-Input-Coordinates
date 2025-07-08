import streamlit as st
import pandas as pd
import numpy as np
import folium
from streamlit_folium import folium_static
import json
import subprocess
import sys
import tempfile
import os
from io import StringIO
from pathlib import Path

# Configuration de la page
st.set_page_config(
    page_title="TSP/VRP Professional Solver",
    page_icon="🗺️",
    layout="wide"
)

class TSPVRPSolver:
    """Classe principale pour gérer les résolutions TSP/VRP"""
    
    def __init__(self):
        self.locations = []
        self.distance_matrix = None
        self.place_name = ""
        # Get the directory where the current script is located
        self.script_dir = Path(__file__).parent.absolute()
    
    def load_locations_from_file(self, file_content):
        """Charge les lieux depuis le contenu du fichier"""
        locations = []
        lines = file_content.decode('utf-8').strip().split('\n')
        
        for line in lines:
            parts = line.strip().split(',')
            if len(parts) == 3:
                name, lat, lon = parts
                try:
                    locations.append((name.strip(), float(lat), float(lon)))
                except ValueError:
                    st.error(f"Erreur format ligne: {line.strip()}")
        
        self.locations = locations
        return locations
    
    def calculate_distance_matrix(self, place_name):
        """Calcule la matrice des distances en utilisant calculate_distances.py"""
        if not self.locations:
            return None
        
        self.place_name = place_name
        
        # Préparer l'input pour le script calculate_distances
        locations_input = "\n".join([f"{name},{lat},{lon}" for name, lat, lon in self.locations])
        
        # Construct the path to calculate_distances.py
        calculate_distances_path = self.script_dir / "calculate_distances.py"
        
        # Check if the file exists
        if not calculate_distances_path.exists():
            st.error(f"Le fichier calculate_distances.py n'existe pas dans {self.script_dir}")
            st.info("Veuillez vous assurer que calculate_distances.py est dans le même dossier que cette application.")
            return None
        
        try:
            # Exécuter le script calculate_distances.py
            process = subprocess.run(
                [sys.executable, str(calculate_distances_path), place_name],
                input=locations_input,
                text=True,
                capture_output=True,
                timeout=6000,
                cwd=str(self.script_dir)  # Set working directory
            )
            
            if process.returncode != 0:
                st.error(f"Erreur dans calculate_distances.py: {process.stderr}")
                return None
            
            # Parser la sortie pour récupérer la matrice
            output_lines = process.stdout.strip().split('\n')
            distance_matrix = []
            
            for line in output_lines:
                if line.strip():
                    row = [float(x) for x in line.split(',')]
                    distance_matrix.append(row)
            
            self.distance_matrix = np.array(distance_matrix)
            return self.distance_matrix
            
        except subprocess.TimeoutExpired:
            st.error("Timeout lors du calcul des distances")
            return None
        except Exception as e:
            st.error(f"Erreur lors du calcul des distances: {str(e)}")
            return None
    
    def solve_tsp(self):
        """Résout le TSP en utilisant TSP_script.py"""
        if self.distance_matrix is None:
            return None, None
        
        # Construct the path to TSP_script.py
        tsp_script_path = self.script_dir / "TSP_script.py"
        
        # Check if the file exists
        if not tsp_script_path.exists():
            st.error(f"Le fichier TSP_script.py n'existe pas dans {self.script_dir}")
            st.info("Veuillez vous assurer que TSP_script.py est dans le même dossier que cette application.")
            return None, None
        
        # Créer un fichier CSV temporaire avec la matrice
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Écrire l'en-tête
            temp_file.write("Place," + ",".join([f"Dist_{i}" for i in range(len(self.locations))]) + "\n")
            
            # Écrire chaque ligne
            for i, (name, _, _) in enumerate(self.locations):
                distances = ",".join([str(self.distance_matrix[i][j]) for j in range(len(self.locations))])
                temp_file.write(f'"{name}",{distances}\n')
            
            temp_file_path = temp_file.name
        
        try:
            # Exécuter TSP_script.py
            process = subprocess.run(
                [sys.executable, str(tsp_script_path), temp_file_path],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.script_dir)  # Set working directory
            )
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_file_path)
            
            if process.returncode != 0:
                st.error(f"Erreur dans TSP_script.py: {process.stderr}")
                return None, None
            
            # Parser la sortie JSON
            result = json.loads(process.stdout)
            cost = result["Optimized cost"]
            path_names = result["Optimized path"]
            
            # Convertir les noms en indices
            name_to_index = {name: i for i, (name, _, _) in enumerate(self.locations)}
            path_indices = [name_to_index[name] for name in path_names if name in name_to_index]
            
            return cost, path_indices
            
        except subprocess.TimeoutExpired:
            st.error("Timeout lors de la résolution TSP")
            return None, None
        except Exception as e:
            st.error(f"Erreur lors de la résolution TSP: {str(e)}")
            return None, None
    
    def solve_vrp(self, num_vehicles):
        """Résout le VRP en utilisant VRP_script.py"""
        if self.distance_matrix is None:
            return None, None
        
        # Construct the path to VRP_script.py
        vrp_script_path = self.script_dir / "VRP_script.py"
        
        # Check if the file exists
        if not vrp_script_path.exists():
            st.error(f"Le fichier VRP_script.py n'existe pas dans {self.script_dir}")
            st.info("Veuillez vous assurer que VRP_script.py est dans le même dossier que cette application.")
            return None, None
        
        # Créer un fichier temporaire avec la matrice
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as temp_file:
            # Sauvegarder la matrice au format CSV
            np.savetxt(temp_file, self.distance_matrix, delimiter=',')
            temp_file_path = temp_file.name
        
        try:
            # Exécuter VRP_script.py
            process = subprocess.run(
                [sys.executable, str(vrp_script_path), temp_file_path, str(num_vehicles)],
                capture_output=True,
                text=True,
                timeout=120,
                cwd=str(self.script_dir)  # Set working directory
            )
            
            # Nettoyer le fichier temporaire
            os.unlink(temp_file_path)
            
            if process.returncode != 0:
                st.error(f"Erreur dans VRP_script.py: {process.stderr}")
                return None, None
            
            # Parser la sortie JSON
            result = json.loads(process.stdout)
            total_cost = result["Total cost"]
            vehicle_routes = result["Vehicle routes"]
            
            return total_cost, vehicle_routes
            
        except subprocess.TimeoutExpired:
            st.error("Timeout lors de la résolution VRP")
            return None, None
        except FileNotFoundError:
            st.error("VRP_script.py non trouvé. Veuillez créer ce fichier.")
            return None, None
        except Exception as e:
            st.error(f"Erreur lors de la résolution VRP: {str(e)}")
            return None, None

class MapVisualizer:
    """Classe pour gérer la visualisation sur carte"""
    
    @staticmethod
    def create_base_map(locations):
        """Crée une carte de base avec les lieux"""
        if not locations:
            return None
        
        # Calculer le centre
        center_lat = sum(loc[1] for loc in locations) / len(locations)
        center_lon = sum(loc[2] for loc in locations) / len(locations)
        
        m = folium.Map(location=[center_lat, center_lon], zoom_start=12)
        
        # Ajouter les marqueurs
        for i, (name, lat, lon) in enumerate(locations):
            folium.Marker(
                [lat, lon],
                popup=f"<b>{name}</b><br/>Index: {i}<br/>Lat: {lat:.4f}<br/>Lon: {lon:.4f}",
                tooltip=f"{name} (Point {i})",
                icon=folium.Icon(color='blue', icon='map-marker')
            ).add_to(m)
        
        return m
    
    @staticmethod
    def add_tsp_route(map_obj, locations, route_indices):
        """Ajoute la route TSP à la carte"""
        if not route_indices or len(route_indices) < 2:
            return map_obj
        
        # Créer les coordonnées de la route
        route_coords = []
        for idx in route_indices:
            if idx < len(locations):
                route_coords.append([locations[idx][1], locations[idx][2]])
        
        # Ajouter la ligne de route
        folium.PolyLine(
            route_coords,
            color='red',
            weight=4,
            opacity=0.8,
            popup="Itinéraire TSP optimal"
        ).add_to(map_obj)
        
        # Ajouter des numéros d'ordre
        for i, idx in enumerate(route_indices[:-1]):  # Exclure le retour au départ
            if idx < len(locations):
                folium.Marker(
                    [locations[idx][1], locations[idx][2]],
                    icon=folium.DivIcon(
                        html=f'<div style="background-color: red; color: white; border-radius: 50%; width: 25px; height: 25px; text-align: center; line-height: 25px; font-weight: bold; font-size: 12px;">{i+1}</div>',
                        icon_size=(25, 25)
                    )
                ).add_to(map_obj)
        
        return map_obj
    
    @staticmethod
    def add_vrp_routes(map_obj, locations, vehicle_routes):
        """Ajoute les routes VRP à la carte"""
        colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'lightred', 'beige', 'darkblue', 'darkgreen']
        
        for vehicle_id, route in enumerate(vehicle_routes):
            if not route:
                continue
                
            color = colors[vehicle_id % len(colors)]
            
            # Créer les coordonnées de la route
            route_coords = []
            for idx in route:
                if idx < len(locations):
                    route_coords.append([locations[idx][1], locations[idx][2]])
            
            # Ajouter la ligne de route
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3,
                opacity=0.8,
                popup=f"Véhicule {vehicle_id + 1}"
            ).add_to(map_obj)
            
            # Ajouter les numéros d'ordre pour ce véhicule
            for i, idx in enumerate(route):
                if idx < len(locations):
                    folium.Marker(
                        [locations[idx][1], locations[idx][2]],
                        icon=folium.DivIcon(
                            html=f'<div style="background-color: {color}; color: white; border-radius: 50%; width: 20px; height: 20px; text-align: center; line-height: 20px; font-weight: bold; font-size: 10px;">V{vehicle_id+1}</div>',
                            icon_size=(20, 20)
                        )
                    ).add_to(map_obj)
        
        return map_obj

def main():
    st.title("🗺️ Professional TSP/VRP Solver")
    st.markdown("Interface professionnelle utilisant vos scripts optimisés")
    
    # Debug information
    with st.expander("🔧 Debug Information"):
        script_dir = Path(__file__).parent.absolute()
        st.write(f"Script directory: {script_dir}")
        st.write("Files in directory:")
        for file_path in script_dir.glob("*"):
            st.write(f"  - {file_path.name}")
    
    st.markdown("---")
    
    # Initialiser le solver
    if 'solver' not in st.session_state:
        st.session_state.solver = TSPVRPSolver()
    
    solver = st.session_state.solver
    
    # Sidebar pour la configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Upload du fichier
        uploaded_file = st.file_uploader(
            "Fichier de lieux (.txt)",
            type="txt",
            help="Format: nom,latitude,longitude"
        )
        
        # Nom de la ville/région
        place_name = st.text_input(
            "Nom de la ville/région",
            value="Casablanca, Morocco",
            help="Ex: 'Paris, France' ou 'New York, USA'"
        )
        
        st.markdown("---")
        
        # Informations sur les fichiers requis
        st.subheader("📁 Fichiers requis")
        script_dir = Path(__file__).parent.absolute()
        
        # Check file existence
        required_files = ["calculate_distances.py", "TSP_script.py", "VRP_script.py"]
        for file_name in required_files:
            file_path = script_dir / file_name
            if file_path.exists():
                st.write(f"✅ {file_name}")
            else:
                st.write(f"❌ {file_name}")
    
    # Interface principale
    if uploaded_file is not None:
        # Charger les lieux
        locations = solver.load_locations_from_file(uploaded_file.read())
        
        if locations:
            st.success(f"✅ {len(locations)} lieux chargés")
            
            # Afficher les lieux
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.subheader("📍 Lieux importés")
                df_locations = pd.DataFrame(locations, columns=['Nom', 'Latitude', 'Longitude'])
                st.dataframe(df_locations, use_container_width=True)
            
            with col2:
                st.subheader("🗺️ Localisation")
                base_map = MapVisualizer.create_base_map(locations)
                if base_map:
                    folium_static(base_map, width=400, height=300)
            
            st.markdown("---")
            
            # Calculer la matrice des distances
            if st.button("🔄 Calculer matrice des distances", type="secondary"):
                with st.spinner("Calcul des distances réelles..."):
                    distance_matrix = solver.calculate_distance_matrix(place_name)
                    
                    if distance_matrix is not None:
                        st.success("✅ Matrice des distances calculée")
                        
                        # Afficher la matrice
                        st.subheader("📊 Matrice des distances (mètres)")
                        df_distances = pd.DataFrame(
                            distance_matrix,
                            index=[loc[0] for loc in locations],
                            columns=[loc[0] for loc in locations]
                        )
                        st.dataframe(df_distances.round(0), use_container_width=True)
            
            # Si la matrice est calculée, afficher les options de résolution
            if solver.distance_matrix is not None:
                st.markdown("---")
                st.subheader("🎯 Résolution")
                
                # Tabs pour TSP et VRP
                tab1, tab2 = st.tabs(["🛣️ TSP", "🚛 VRP"])
                
                with tab1:
                    st.markdown("**Traveling Salesman Problem**")
                    st.write("Trouve le chemin le plus court visitant tous les lieux une fois")
                    
                    if st.button("🚀 Résoudre TSP", type="primary"):
                        with st.spinner("Résolution TSP en cours..."):
                            cost, path_indices = solver.solve_tsp()
                            
                            if cost is not None and path_indices is not None:
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.metric("Distance totale", f"{cost:.0f} m")
                                    st.write("**Itinéraire optimal:**")
                                    for i, idx in enumerate(path_indices[:-1]):
                                        st.write(f"{i+1}. {locations[idx][0]}")
                                
                                with col2:
                                    # Créer la carte avec la route
                                    map_with_route = MapVisualizer.create_base_map(locations)
                                    map_with_route = MapVisualizer.add_tsp_route(map_with_route, locations, path_indices)
                                    folium_static(map_with_route, width=600, height=400)
                
                with tab2:
                    st.markdown("**Vehicle Routing Problem**")
                    st.write("Optimise les routes pour plusieurs véhicules")
                    
                    num_vehicles = st.slider(
                        "Nombre de véhicules",
                        min_value=1,
                        max_value=min(10, len(locations)-1),
                        value=2
                    )
                    
                    if st.button("🚀 Résoudre VRP", type="primary"):
                        with st.spinner("Résolution VRP en cours..."):
                            total_cost, vehicle_routes = solver.solve_vrp(num_vehicles)
                            
                            if total_cost is not None and vehicle_routes is not None:
                                col1, col2 = st.columns([1, 2])
                                
                                with col1:
                                    st.metric("Coût total", f"{total_cost:.0f} m")
                                    st.write("**Routes par véhicule:**")
                                    for i, route in enumerate(vehicle_routes):
                                        st.write(f"**Véhicule {i+1}:**")
                                        for idx in route:
                                            if idx < len(locations):
                                                st.write(f"  → {locations[idx][0]}")
                                
                                with col2:
                                    # Créer la carte avec les routes VRP
                                    map_with_routes = MapVisualizer.create_base_map(locations)
                                    map_with_routes = MapVisualizer.add_vrp_routes(map_with_routes, locations, vehicle_routes)
                                    folium_static(map_with_routes, width=600, height=400)
        
        else:
            st.error("❌ Aucun lieu valide dans le fichier")
    
    else:
        st.info("📝 Veuillez uploader un fichier de lieux pour commencer")
        st.markdown("""
        **Format du fichier (.txt):**
        ```
        Lieu1,latitude1,longitude1
        Lieu2,latitude2,longitude2
        Lieu3,latitude3,longitude3
        ```
        
        **Exemple pour Casablanca:**
        ```
        Gare Casa-Port,33.6002,-7.6197
        Aéroport Mohammed V,33.3675,-7.5398
        Centre-ville,33.5950,-7.6187
        Mosquée Hassan II,33.6084,-7.6326
        ```
        """)

if __name__ == "__main__":
    main()