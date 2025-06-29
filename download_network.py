#!/usr/bin/env python3
"""
Download and save the walking network for CDMX using OSMnx
"""

import osmnx as ox
import time

def download_cdmx_walking_network():
    """Download the walking network for Mexico City"""
    
    print("Downloading walking network for Mexico City...")
    start_time = time.time()
    
    # Define the area (Mexico City)
    place_name = "Distrito Federal, Mexico"
    
    try:
        # Download the walking network
        print(f"Downloading network for: {place_name}")
        G = ox.graph_from_place(
            place_name,
            network_type='walk',
            simplify=True
        )
        
        download_time = time.time() - start_time
        print(f"Network downloaded in {download_time:.2f}s")
        print(f"Network has {len(G.nodes)} nodes and {len(G.edges)} edges")
        
        # Save the network
        print("Saving network...")
        save_start = time.time()
        ox.save_graphml(G, "cdmx_walking_network.graphml")
        save_time = time.time() - save_start
        
        total_time = time.time() - start_time
        print(f"Network saved in {save_time:.2f}s")
        print(f"Total time: {total_time:.2f}s")
        print("Network saved as 'cdmx_walking_network.graphml'")
        
        return G
        
    except Exception as e:
        print(f"Error downloading network: {e}")
        return None

if __name__ == '__main__':
    download_cdmx_walking_network()
