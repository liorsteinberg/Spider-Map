#!/usr/bin/env python3
"""
Build NetworKit network for Mexico City metro spider map
This script creates a NetworKit graph from OSM data for efficient network analysis
Based on the working approach from the Jupyter notebook
"""

import time
import logging
import pickle
import os
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def haversine_distance(lat1, lng1, lat2, lng2):
    """Calculate haversine distance between two points in meters"""
    R = 6371000  # Earth's radius in meters
    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlng = np.radians(lng2 - lng1)
    
    a = (np.sin(dlat/2)**2 + 
         np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng/2)**2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    
    return R * c

def main():
    logger.info("Building NetworKit network for Mexico City...")
    logger.info("Using the same approach that worked in the notebook!")
    
    try:
        import networkit as nk
        import osmnx as ox
        logger.info(f"NetworKit version: {nk.__version__}")
        logger.info(f"OSMnx version: {ox.__version__}")
    except ImportError as e:
        logger.error(f"Required dependency missing: {e}")
        logger.error("Please install dependencies: pip install networkit osmnx")
        return False

    start_time = time.time()
    
    # Check if NetworKit PKL already exists
    cache_file = 'cdmx_networkit_graph.pkl'
    if os.path.exists(cache_file):
        logger.info(f"NetworKit graph already exists at {cache_file}")
        logger.info("Delete the PKL file if you want to rebuild from scratch")
        return True
    
    # Download OSM data for Mexico City (same as in your notebook)
    logger.info("Downloading OSM street network for Mexico City...")
    download_start = time.time()
    
    try:
        # Use the exact same approach that worked in your notebook
        logger.info("Using: ox.graph_from_place('Mexico City, Mexico', network_type='walk')")
        G = ox.graph_from_place('Mexico City, Mexico', network_type='walk')
        download_time = time.time() - download_start
        logger.info(f"OSM graph downloaded in {download_time:.2f}s")
        logger.info(f"OSM graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
    except Exception as e:
        logger.error(f"Failed to download OSM data: {e}")
        logger.error("Please check your internet connection and try again")
        return False

    # Convert OSMnx graph to NetworKit (copied exactly from your notebook)
    logger.info("Converting OSMnx graph to NetworKit (ultra-fast conversion)...")
    logger.info(f"OSM graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
    convert_start = time.time()
    
    try:
        # Create NetworKit graph
        networkit_graph = nk.Graph(len(G.nodes), weighted=True, directed=False)
        
        # Create mappings for coordinate lookup
        osm_to_nk = {}
        coordinate_mapping = {}
        node_mapping = {}
        
        # Add nodes with coordinate mapping
        logger.info("Adding nodes with coordinate mapping...")
        for i, (osm_id, data) in enumerate(G.nodes(data=True)):
            osm_to_nk[osm_id] = i
            lat, lng = data['y'], data['x']
            coordinate_mapping[i] = (lat, lng)
            # Create a spatial hash for quick lookup (round to ~10m precision)
            coord_key = (round(lat, 4), round(lng, 4))
            node_mapping[coord_key] = i

        # Add edges with distances
        logger.info("Adding edges with distance weights...")
        edge_count = 0
        for u, v, data in G.edges(data=True):
            nk_u = osm_to_nk[u]
            nk_v = osm_to_nk[v]
            
            # Use length if available, otherwise calculate from coordinates
            if 'length' in data:
                weight = data['length']
            else:
                lat1, lng1 = coordinate_mapping[nk_u]
                lat2, lng2 = coordinate_mapping[nk_v]
                weight = haversine_distance(lat1, lng1, lat2, lng2)
            
            networkit_graph.addEdge(nk_u, nk_v, weight)
            edge_count += 1

        convert_time = time.time() - convert_start
        logger.info(f"NetworKit conversion completed in {convert_time:.2f}s")
        logger.info(f"Created NetworKit graph: {networkit_graph.numberOfNodes()} nodes, {networkit_graph.numberOfEdges()} edges")
        
    except Exception as e:
        logger.error(f"Failed to convert to NetworKit: {e}")
        return False

    # Save NetworKit graph (copied exactly from your notebook)
    logger.info("Saving NetworKit graph for future use...")
    save_start = time.time()
    
    try:
        cache_file = 'cdmx_networkit_graph.pkl'
        cache_data = {
            'graph': networkit_graph,
            'node_mapping': node_mapping,
            'coordinate_mapping': coordinate_mapping,
            'metadata': {
                'created_at': time.time(),
                'osm_nodes': len(G.nodes),
                'osm_edges': len(G.edges),
                'nk_nodes': networkit_graph.numberOfNodes(),
                'nk_edges': networkit_graph.numberOfEdges(),
                'network_type': 'walk'
            }
        }
        
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        
        save_time = time.time() - save_start
        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # MB
        logger.info(f"NetworKit graph saved in {save_time:.2f}s")
        logger.info(f"Cache file: {cache_file} ({file_size:.1f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to save NetworKit graph: {e}")
        return False

    # Performance summary
    total_time = time.time() - start_time
    logger.info("=" * 60)
    logger.info("🚀 NetworKit Network Build Complete!")
    logger.info("=" * 60)
    logger.info(f"Total build time: {total_time:.2f}s")
    logger.info(f"  - OSM download: {download_time:.2f}s")
    logger.info(f"  - NetworKit conversion: {convert_time:.2f}s")
    logger.info(f"  - Cache save: {save_time:.2f}s")
    logger.info(f"Network size: {networkit_graph.numberOfNodes():,} nodes, {networkit_graph.numberOfEdges():,} edges")
    logger.info(f"Cache file: {cache_file} ({file_size:.1f} MB)")
    logger.info("")
    logger.info("✅ NetworKit graph is ready! Expected performance benefits:")
    logger.info("   - Fast network creation and processing")
    logger.info("   - Efficient shortest path calculations")
    logger.info("   - Optimized memory usage")
    logger.info("   - Multi-core utilization: Excellent parallelization")
    logger.info("")
    logger.info("🎯 Next steps:")
    logger.info("   1. Run: python walking_service.py")
    logger.info("   2. Open: http://localhost:8080")
    logger.info("   3. Enjoy ultra-fast distance calculations!")
    
    # Quick test (from your notebook)
    logger.info("\n🔍 Testing shortest path calculation...")
    try:
        start_node = 0
        end_node = min(100, networkit_graph.numberOfNodes() - 1)
        
        dijkstra = nk.distance.Dijkstra(networkit_graph, start_node)
        dijkstra.run()
        distance = dijkstra.distance(end_node)
        path = dijkstra.getPath(end_node)
        
        logger.info(f"Sample path from node {start_node} to {end_node}:")
        logger.info(f"Distance: {distance:.1f} meters")
        logger.info(f"Path length: {len(path)} nodes")
        logger.info("✅ NetworKit graph is ready for ultra-fast distance calculations!")
    except Exception as e:
        logger.warning(f"Path test failed (graph still usable): {e}")
    
    return True

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)
