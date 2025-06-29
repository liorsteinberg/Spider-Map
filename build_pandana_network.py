#!/usr/bin/env python3
"""
Script to build and save Pandana network once, so it doesn't need to be rebuilt every time
"""

import osmnx as ox
import pickle
import time
try:
    import pandana as pdna
    PANDANA_AVAILABLE = True
    print("Pandana is available")
except ImportError:
    PANDANA_AVAILABLE = False
    print("Pandana not available - cannot build network")
    exit(1)

def build_and_save_pandana_network():
    """Build Pandana network from OSMnx graph and save it"""
    
    print("Loading OSMnx network...")
    start_time = time.time()
    
    # Load the OSMnx graph
    G = ox.load_graphml("cdmx_walking_network.graphml")
    load_time = time.time() - start_time
    print(f"OSMnx network loaded in {load_time:.2f}s with {len(G.nodes)} nodes and {len(G.edges)} edges")
    
    print("Converting to Pandana network...")
    convert_start = time.time()
    
    # Convert graph to pandana network
    nodes_df, edges_df = ox.graph_to_gdfs(G)
    
    # Pandana requires specific column names
    edges_df = edges_df.reset_index()
    edges_df['from'] = edges_df['u']
    edges_df['to'] = edges_df['v']
    
    convert_time = time.time() - convert_start
    print(f"Graph conversion completed in {convert_time:.2f}s")
    
    print("Building Pandana network (this may take several minutes)...")
    pandana_start = time.time()
    
    # Create pandana network
    pandana_network = pdna.Network(
        nodes_df['x'], nodes_df['y'], 
        edges_df['from'], edges_df['to'], 
        edges_df[['length']]
    )
    
    pandana_time = time.time() - pandana_start
    print(f"Pandana network built in {pandana_time:.2f}s")
    
    print("Saving Pandana network...")
    save_start = time.time()
    
    # Save the pandana network using its built-in save method
    pandana_network.save_hdf5('cdmx_pandana_network.h5')
    
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    
    print(f"Pandana network saved in {save_time:.2f}s")
    print(f"Total time: {total_time:.2f}s")
    print("Pandana network saved as 'cdmx_pandana_network.h5'")
    
    return pandana_network

if __name__ == '__main__':
    build_and_save_pandana_network()
