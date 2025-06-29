#!/usr/bin/env python3
"""
Flask backend service for calculating walking distances using Pandana
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import json
from pathlib import Path
import numpy as np
import time
import logging
import os
import sys

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Spider Map Walking Service...")
logger.info(f"Python version: {sys.version}")
logger.info("Checking dependencies...")

try:
    import pandana as pdna
    logger.info(f"Pandana imported successfully, version: {pdna.__version__}")
except ImportError as e:
    logger.error(f"Failed to import pandana: {e}")
    pdna = None

try:
    import osmnx
    logger.info(f"OSMnx imported successfully, version: {osmnx.__version__}")
except ImportError as e:
    logger.error(f"Failed to import osmnx: {e}")

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Global variable to store the network
pandana_network = None

@app.route('/')
def home():
    """Serve the main map interface"""
    return send_from_directory('.', 'map.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    return jsonify({
        "service": "Spider Map Walking Service",
        "status": "running",
        "version": "1.0",
        "description": "Flask backend service for calculating walking distances using Pandana",
        "endpoints": {
            "/": "Main map interface",
            "/api": "API information (this endpoint)",
            "/health": "Health check endpoint",
            "/walking-distances-batch": "POST - Calculate walking distances from center to multiple stations"
        },
        "pandana_loaded": pandana_network is not None
    })

def load_pandana_network():
    """Load the pre-built Pandana network"""
    global pandana_network
    
    if pandana_network is not None:
        logger.info("Pandana network already loaded, returning cached version")
        return pandana_network
    
    if pdna is None:
        logger.error("Pandana not available - cannot load network")
        return None

    logger.info("Loading pre-built Pandana network...")
    logger.info("Note: Contraction hierarchy building (~20s) is unavoidable even with pre-built network")
    load_start = time.time()
    
    try:
        network_file = 'cdmx_pandana_network.h5'
        logger.info(f"Checking for network file: {network_file}")
        
        if os.path.exists(network_file):
            logger.info(f"Network file found, size: {os.path.getsize(network_file)} bytes")
            logger.info("Starting network loading...")
            pandana_network = pdna.Network.from_hdf5(network_file)
            load_time = time.time() - load_start
            logger.info(f"Pre-built Pandana network loaded successfully in {load_time:.2f}s")
            return pandana_network
        else:
            logger.error(f"Pre-built Pandana network file '{network_file}' not found")
            logger.error("Available files in current directory:")
            for file in os.listdir('.'):
                logger.error(f"  - {file}")
            logger.error("Run 'python build_pandana_network.py' first to create the network file.")
            return None
    except Exception as e:
        load_time = time.time() - load_start
        logger.error(f"Failed to load pre-built Pandana network after {load_time:.2f}s: {e}")
        logger.error("You may need to rebuild it with 'python build_pandana_network.py'")
        return None

@app.route('/walking-distances-batch', methods=['POST'])
def get_walking_distances_batch():
    """Calculate walking distances from a center point to multiple stations using Pandana vectorization"""
    try:
        data = request.json
        logger.info(f"Received walking distances request for {len(data.get('stations', []))} stations")
        
        center_lat = data['center_lat']
        center_lng = data['center_lng']
        stations = data['stations']
        
        # Check if network is loaded, try to load if not
        network = pandana_network
        if network is None:
            logger.warning("Pandana network not loaded, attempting to load now...")
            network = load_pandana_network()
            
        if network is None:
            logger.error("Pandana network unavailable - cannot calculate distances")
            return jsonify({
                'error': 'Pandana network not available', 
                'details': 'Network file may be missing or failed to load',
                'suggestion': 'Check server logs for network loading errors'
            }), 500
        
        # Use Pandana for fast vectorized calculations
        start_time = time.time()
        
        # Find nearest nodes for all points
        node_lookup_start = time.time()
        center_node = pandana_network.get_node_ids([center_lng], [center_lat])[0]
        station_lngs = [s['lng'] for s in stations]
        station_lats = [s['lat'] for s in stations]
        station_nodes = pandana_network.get_node_ids(station_lngs, station_lats)
        node_lookup_time = time.time() - node_lookup_start
        print(f"Pandana node lookup took: {node_lookup_time:.3f}s")
        
        # Calculate all shortest path distances at once (FAST)
        distance_calc_start = time.time()
        distances = pandana_network.shortest_path_lengths([center_node] * len(station_nodes), station_nodes, imp_name='length')
        distance_calc_time = time.time() - distance_calc_start
        print(f"Pandana distance calculation took: {distance_calc_time:.3f}s")
        
        # OPTIMIZATION: Only calculate routes for the closest 10 stations
        route_calc_start = time.time()
        
        # Create list of stations with distances and sort by distance
        station_distances = []
        for i, station in enumerate(stations):
            if i < len(distances) and not np.isinf(distances[i]):
                station_distances.append({
                    'station': station,
                    'station_node': station_nodes[i],
                    'distance': float(distances[i]),
                    'index': i
                })
        
        # Sort by distance and take the closest 10 (to account for walking vs Euclidean differences)
        station_distances.sort(key=lambda x: x['distance'])
        closest_10 = station_distances[:10]
        print(f"Smart optimization: Only calculating routes for {len(closest_10)} closest stations instead of {len(stations)}")
        
        # Calculate routes for the closest 10 stations, then pick the best 5 by walking distance
        results = []
        for station_data in closest_10:
            station = station_data['station']
            station_node = station_data['station_node']
            
            # Get the route path only for this close station
            single_route_start = time.time()
            route_nodes = pandana_network.shortest_paths([center_node], [station_node])[0]
            single_route_time = time.time() - single_route_start
            
            # Convert route nodes to coordinates using Pandana's node coordinates
            coord_convert_start = time.time()
            route_coords = []
            for node_id in route_nodes:
                # Get coordinates from Pandana network
                node_x = pandana_network.nodes_df.loc[node_id, 'x']
                node_y = pandana_network.nodes_df.loc[node_id, 'y']
                route_coords.append([node_y, node_x])  # [lat, lng] format for Leaflet
            coord_convert_time = time.time() - coord_convert_start
            
            print(f"Station {station['name']}: route calc {single_route_time:.3f}s, coord convert {coord_convert_time:.3f}s")
            
            results.append({
                'name': station['name'],
                'lat': station['lat'],
                'lng': station['lng'],
                'distance': station_data['distance'],
                'route': route_coords
            })
        
        route_calc_time = time.time() - route_calc_start
        total_time = time.time() - start_time
        print(f"Pandana route calculation took: {route_calc_time:.3f}s (for {len(closest_10)} stations)")
        print(f"Pandana total processing time: {total_time:.3f}s")
        print(f"Performance improvement: {((len(stations) - len(closest_10)) / len(stations) * 100):.1f}% fewer route calculations")
        
        # Sort results by actual walking distance and return top 5
        results.sort(key=lambda x: x['distance'])
        final_5 = results[:5]
        print(f"Final selection: Top 5 stations by actual walking distance from the {len(closest_10)} candidates")
        
        return jsonify({'stations': final_5})
        
    except Exception as e:
        logger.error(f"Batch calculation error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'pandana_loaded': pandana_network is not None
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'ok', 'pandana_loaded': pandana_network is not None})

if __name__ == '__main__':
    logger.info("Starting walking distance service...")
    
    # Pre-load the Pandana network
    logger.info("Attempting to pre-load Pandana network...")
    network_result = load_pandana_network()
    if network_result:
        logger.info("Pandana network pre-loaded successfully")
    else:
        logger.warning("Pandana network failed to pre-load - will attempt loading on first request")
    
    # Use Heroku's PORT environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    logger.info(f"Service ready at http://localhost:{port}")
    app.run(host='0.0.0.0', port=port, debug=False)
