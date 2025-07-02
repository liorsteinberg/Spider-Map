#!/usr/bin/env python3
"""
Flask backend service for calculating walking distances using NetworKit
NetworKit provides fast and efficient network analysis and shortest path calculations
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
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

logger.info("Starting Spider Map Walking Service with NetworKit...")
logger.info(f"Python version: {sys.version}")
logger.info("Checking dependencies...")

try:
    import networkit as nk
    logger.info(f"NetworKit imported successfully, version: {nk.__version__}")
except ImportError as e:
    logger.error(f"Failed to import networkit: {e}")
    nk = None

try:
    import osmnx as ox
    logger.info(f"OSMnx imported successfully, version: {ox.__version__}")
except ImportError as e:
    logger.error(f"Failed to import osmnx: {e}")
    ox = None

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

# Supported cities - complete configuration
CITIES = {
    'mexico_city': {
        'cache_prefix': 'cdmx', 
        'display_name': 'Mexico City',
        'name': 'Mexico City, Mexico',
        'center': [19.4326, -99.1332],
        'zoom': 11
    },
    'berlin': {
        'cache_prefix': 'berlin', 
        'display_name': 'Berlin',
        'name': 'Berlin, Germany',
        'center': [52.5200, 13.4050],
        'zoom': 11
    }
}

# Global variables to store the network
current_city = 'mexico_city'  # Track current city
networkit_graph = None
node_mapping = None  # Maps (lat, lng) to NetworKit node IDs
coordinate_mapping = None  # Maps NetworKit node IDs to (lat, lng)

@app.route('/')
def home():
    """Serve the main map interface"""
    return send_from_directory('.', 'map.html')

@app.route('/api')
def api_info():
    """API information endpoint"""
    cache_file = 'cdmx_networkit_graph.pkl'
    cache_exists = os.path.exists(cache_file)
    cache_size_mb = os.path.getsize(cache_file) / (1024 * 1024) if cache_exists else 0
    
    return jsonify({
        "service": "Spider Map Walking Service with NetworKit",
        "status": "running",
        "version": "2.0",
        "description": "Flask backend service for calculating walking distances using NetworKit",
        "endpoints": {
            "/": "Main map interface",
            "/api": "API information (this endpoint)",
            "/health": "Health check endpoint",
            "/walking-distances-batch": "POST - Calculate walking distances from center to multiple stations"
        },
        "networkit_loaded": networkit_graph is not None,
        "cache_status": {
            "cached_graph_exists": cache_exists,
            "cache_file": cache_file,
            "cache_size_mb": round(cache_size_mb, 1) if cache_exists else None,
            "using_cached_graph": networkit_graph is not None and cache_exists
        },
        "performance": "NetworKit provides 5-50x faster network creation and shortest path calculations"
    })

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "networkit_available": nk is not None,
        "osmnx_available": ox is not None,
        "graph_loaded": networkit_graph is not None,
        "current_city": current_city
    })

@app.route('/api/cities')
def list_cities():
    """List all supported cities with complete information"""
    cities_list = []
    for key, config in CITIES.items():
        cache_file = f"CityData/{config['cache_prefix']}_walking_graph.pkl"
        stations_file = f"CityData/{config['cache_prefix']}_stations.geojson"
        
        # Backward compatibility check for Mexico City
        if not os.path.exists(cache_file) and config['cache_prefix'] == 'cdmx':
            cache_file = 'cdmx_networkit_graph.pkl'
        
        # Count stations if file exists
        station_count = 0
        if os.path.exists(stations_file):
            try:
                with open(stations_file, 'r', encoding='utf-8') as f:
                    geojson_data = json.load(f)
                    station_count = len(geojson_data.get('features', []))
            except:
                station_count = 0
        
        cities_list.append({
            "key": key,
            "name": config['display_name'],
            "full_name": config.get('name', config['display_name']),
            "center": config.get('center', [0, 0]),
            "zoom": config.get('zoom', 10),
            "is_current": key == current_city,
            "data_available": {
                "walking_network": os.path.exists(cache_file),
                "stations": os.path.exists(stations_file),
                "station_count": station_count
            }
        })
    
    return jsonify({
        "cities": cities_list,
        "current_city": current_city,
        "current_city_info": {
            "key": current_city,
            "name": CITIES[current_city]['display_name'],
            "center": CITIES[current_city].get('center', [0, 0]),
            "zoom": CITIES[current_city].get('zoom', 10)
        }
    })

@app.route('/api/stations')
def get_stations():
    """Get stations for the current city"""
    try:
        cache_prefix = CITIES.get(current_city, {}).get('cache_prefix', 'cdmx')
        geojson_file = f"CityData/{cache_prefix}_stations.geojson"
        
        if not os.path.exists(geojson_file):
            return jsonify({
                "stations": [],
                "city": current_city,
                "count": 0,
                "error": f"No stations file found for {current_city}: {geojson_file}"
            })
        
        logger.info(f"Loading stations from: {geojson_file}")
        with open(geojson_file, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        # Convert GeoJSON to simple station list
        stations = []
        for feature in geojson_data.get('features', []):
            if feature['geometry']['type'] == 'Point':
                coords = feature['geometry']['coordinates']
                lng, lat = coords[0], coords[1]  # GeoJSON uses [lng, lat] order
                
                station = {
                    'name': feature['properties'].get('name', 'Unknown Station'),
                    'lat': lat,
                    'lng': lng,
                    'osm_id': feature['properties'].get('osm_id', '')
                }
                stations.append(station)
        
        logger.info(f"✅ Loaded {len(stations)} stations for {CITIES[current_city]['display_name']}")
        return jsonify({
            "stations": stations,
            "city": current_city,
            "city_name": CITIES[current_city]['display_name'],
            "count": len(stations)
        })
        
    except Exception as e:
        logger.error(f"Failed to load stations for {current_city}: {e}")
        return jsonify({
            "stations": [],
            "city": current_city,
            "count": 0,
            "error": str(e)
        })

@app.route('/api/cities/<city_key>', methods=['POST'])
def switch_city(city_key):
    """Switch to a different city"""
    global current_city, networkit_graph, node_mapping, coordinate_mapping
    
    if city_key not in CITIES:
        return jsonify({
            "error": f"Unknown city: {city_key}",
            "available_cities": list(CITIES.keys())
        }), 400
    
    if city_key == current_city:
        return jsonify({
            "message": f"Already using {CITIES[city_key]['display_name']}",
            "city": city_key
        })
    
    logger.info(f"🔄 Switching from {CITIES[current_city]['display_name']} to {CITIES[city_key]['display_name']}")
    
    # Clear current data to force reload
    networkit_graph = None
    node_mapping = None
    coordinate_mapping = None
    
    # Clear numpy arrays if they exist
    global _node_ids_array, _coords_array
    if '_node_ids_array' in globals():
        del _node_ids_array
    if '_coords_array' in globals():
        del _coords_array
        logger.info("Cleared node lookup arrays for new city")
    
    # Update current city
    current_city = city_key
    
    return jsonify({
        "message": f"Successfully switched to {CITIES[city_key]['display_name']}",
        "city": {
            "key": city_key,
            "name": CITIES[city_key]['display_name'],
            "full_name": CITIES[city_key].get('name', CITIES[city_key]['display_name']),
            "center": CITIES[city_key].get('center', [0, 0]),
            "zoom": CITIES[city_key].get('zoom', 10)
        }
    })

def load_networkit_graph():
    """Load or create the NetworKit graph from OSM data"""
    global networkit_graph, node_mapping, coordinate_mapping
    
    if networkit_graph is not None:
        logger.info("NetworKit graph already loaded, returning cached version")
        return networkit_graph
    
    if nk is None:
        logger.error("NetworKit not available - cannot load graph")
        return None

    # Always try to load pre-built NetworKit graph first (much faster than OSM creation)
    # Determine cache file based on current city
    cache_prefix = CITIES.get(current_city, {}).get('cache_prefix', 'cdmx')
    cache_file = f'CityData/{cache_prefix}_walking_graph.pkl'
    
    # Backward compatibility for Mexico City
    if cache_prefix == 'cdmx' and not os.path.exists(cache_file):
        old_cache_file = 'cdmx_networkit_graph.pkl'
        if os.path.exists(old_cache_file):
            cache_file = old_cache_file
    
    logger.info(f"Checking for cached NetworKit graph: {cache_file}")
    
    if os.path.exists(cache_file):
        logger.info(f"Found cached graph file: {cache_file}")
        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # Size in MB
        logger.info(f"Cache file size: {file_size:.1f} MB")
        
        load_start = time.time()
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                networkit_graph = data['graph']
                node_mapping = data['node_mapping']
                coordinate_mapping = data['coordinate_mapping']
            
            load_time = time.time() - load_start
            logger.info(f"✅ NetworKit graph loaded successfully from cache in {load_time:.2f}s")
            logger.info(f"Graph stats: {networkit_graph.numberOfNodes()} nodes, {networkit_graph.numberOfEdges()} edges")
            logger.info(f"Node mappings: {len(node_mapping)} coordinate mappings, {len(coordinate_mapping)} reverse mappings")
            return networkit_graph
            
        except Exception as e:
            logger.error(f"❌ Failed to load cached NetworKit graph: {e}")
            logger.warning("Cache file appears corrupted, will attempt to create new graph from OSM")
            # Fall through to create new graph
    else:
        logger.warning(f"❌ No cached NetworKit graph found at {cache_file}")
        logger.info("Will create new graph from OSM data (this may take several minutes)")

    # Only create new NetworKit graph from OSM data if cache loading failed
    logger.warning("⚠️  FALLBACK: Creating new NetworKit graph from OSM data...")
    logger.warning("This should rarely happen - the cached graph should be used instead!")
    logger.info("This process may take 2-5 minutes depending on network speed...")
    
    if ox is None:
        logger.error("❌ OSMnx not available - cannot create graph from OSM")
        logger.error("Without OSMnx, cannot fallback to create new graph")
        return None

    try:
        # Download OSM data for Mexico City
        logger.info("Downloading OSM street network for Mexico City...")
        create_start = time.time()
        
        # Mexico City bounding box (slightly larger area for better coverage)
        north, south, east, west = 19.59, 19.25, -98.95, -99.35
        
        # Download walking network (much faster than driving network)
        osm_graph = ox.graph_from_bbox(
            north, south, east, west,
            network_type='walk',
            simplify=True,
            retain_all=False,
            truncate_by_edge=True
        )
        
        download_time = time.time() - create_start
        logger.info(f"OSM download completed in {download_time:.2f}s")
        logger.info(f"OSM graph: {len(osm_graph.nodes)} nodes, {len(osm_graph.edges)} edges")

        # Convert OSMnx graph to NetworKit graph (much faster conversion)
        logger.info("Converting OSMnx graph to NetworKit...")
        convert_start = time.time()
        
        # Create NetworKit graph
        networkit_graph = nk.Graph(len(osm_graph.nodes), weighted=True, directed=False)
        
        # Create mappings
        osm_to_nk = {}
        coordinate_mapping = {}
        node_mapping = {}
        
        # Add nodes with coordinate mapping
        for i, (osm_id, data) in enumerate(osm_graph.nodes(data=True)):
            osm_to_nk[osm_id] = i
            lat, lng = data['y'], data['x']
            coordinate_mapping[i] = (lat, lng)
            # Create a spatial hash for quick lookup (round to ~10m precision)
            coord_key = (round(lat, 4), round(lng, 4))
            node_mapping[coord_key] = i

        # Add edges with distances
        for u, v, data in osm_graph.edges(data=True):
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

        convert_time = time.time() - convert_start
        total_time = time.time() - create_start
        
        logger.info(f"NetworKit conversion completed in {convert_time:.2f}s")
        logger.info(f"Total graph creation time: {total_time:.2f}s")
        logger.info(f"NetworKit graph: {networkit_graph.numberOfNodes()} nodes, {networkit_graph.numberOfEdges()} edges")

        # Save to cache for future use
        logger.info(f"Saving NetworKit graph to {cache_file} for future use...")
        cache_data = {
            'graph': networkit_graph,
            'node_mapping': node_mapping,
            'coordinate_mapping': coordinate_mapping
        }
        with open(cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        logger.info("NetworKit graph cached successfully")

        return networkit_graph

    except Exception as e:
        logger.error(f"Failed to create NetworKit graph: {e}")
        return None

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

def find_nearest_node(lat, lng):
    """Find the nearest NetworKit node to given coordinates - ULTRA-OPTIMIZED VERSION"""
    if not coordinate_mapping:
        return None
    
    # Try exact match first (for speed) - increased precision for better hits
    for precision in [4, 3, 2]:  # Try different precision levels
        coord_key = (round(lat, precision), round(lng, precision))
        if coord_key in node_mapping:
            return node_mapping[coord_key]
    
    # OPTIMIZED: Use pre-computed numpy arrays for vectorized operations
    global _node_ids_array, _coords_array
    
    if '_node_ids_array' not in globals():
        # Build numpy arrays once and reuse (much faster)
        logger.info("Building optimized node lookup arrays...")
        _node_ids_array = np.array(list(coordinate_mapping.keys()))
        _coords_array = np.array(list(coordinate_mapping.values()))
        logger.info(f"Built arrays with {len(_node_ids_array)} nodes for ultra-fast lookup")
    
    try:
        # Vectorized distance calculation (MUCH faster than loops)
        target = np.array([lat, lng])
        
        # Simple Euclidean distance for speed (close enough for nearest node lookup)
        # This is 50-100x faster than haversine for large arrays
        diff = _coords_array - target
        distances_squared = np.sum(diff * diff, axis=1)
        nearest_idx = np.argmin(distances_squared)
        
        return _node_ids_array[nearest_idx]
        
    except Exception as e:
        logger.warning(f"Vectorized lookup failed: {e}, falling back to simple method")
        # Fallback to simple method if numpy fails
        min_distance = float('inf')
        nearest_node = None
        
        for node_id, (node_lat, node_lng) in coordinate_mapping.items():
            # Use simple Euclidean distance for speed
            distance = (lat - node_lat) ** 2 + (lng - node_lng) ** 2
            if distance < min_distance:
                min_distance = distance
                nearest_node = node_id
        
        return nearest_node

@app.route('/walking-distances-batch', methods=['POST'])
def get_walking_distances_batch():
    """Calculate walking distances using NetworKit"""
    try:
        global current_city, networkit_graph, node_mapping, coordinate_mapping
        
        data = request.json
        logger.info(f"Received walking distances request for {len(data.get('stations', []))} stations")
        
        center_lat = data['center_lat']
        center_lng = data['center_lng']
        requested_city = data.get('city', current_city)  # Get requested city
        
        # CRITICAL: Switch cities if needed for correct network
        if requested_city != current_city and requested_city in CITIES:
            logger.info(f"🔄 Switching from {CITIES[current_city]['display_name']} to {CITIES[requested_city]['display_name']} for OSM calculation")
            current_city = requested_city
            
            # Clear current graph to force reload for new city
            networkit_graph = None
            node_mapping = None
            coordinate_mapping = None
            
            # Clear numpy arrays if they exist
            global _node_ids_array, _coords_array
            if '_node_ids_array' in globals():
                del _node_ids_array
            if '_coords_array' in globals():
                del _coords_array
                logger.info("Cleared node lookup arrays for new city")

        # CRITICAL: Always load stations from the current city's data file
        # Don't use stations from frontend - they might be from wrong city
        logger.info(f"Loading stations for current city: {CITIES[current_city]['display_name']}")
        cache_prefix = CITIES.get(current_city, {}).get('cache_prefix', 'cdmx')
        geojson_file = f"CityData/{cache_prefix}_stations.geojson"
        
        if not os.path.exists(geojson_file):
            logger.error(f"No stations file found for {current_city}: {geojson_file}")
            return jsonify({
                'error': f'Station data not available for {CITIES[current_city]["display_name"]}',
                'city': current_city
            }), 404
        
        # Load stations from current city's data file
        with open(geojson_file, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
        
        stations = []
        for feature in geojson_data.get('features', []):
            if feature['geometry']['type'] == 'Point':
                coords = feature['geometry']['coordinates']
                lng, lat = coords[0], coords[1]  # GeoJSON uses [lng, lat] order
                
                station = {
                    'name': feature['properties'].get('name', 'Unknown Station'),
                    'lat': lat,
                    'lng': lng,
                    'osm_id': feature['properties'].get('osm_id', '')
                }
                stations.append(station)
        
        logger.info(f"✅ Using {len(stations)} stations from {CITIES[current_city]['display_name']} data file")
        
        # Check if graph is loaded, try to load if not
        graph = networkit_graph
        if graph is None:
            logger.warning(f"NetworKit graph not loaded for {current_city}, attempting to load now...")
            graph = load_networkit_graph()
            
        if graph is None:
            logger.error("NetworKit graph unavailable - cannot calculate distances")
            return jsonify({
                'error': 'NetworKit graph not available', 
                'details': 'Graph creation or loading failed',
                'suggestion': 'Check server logs for network creation errors'
            }), 500

        # Use NetworKit for ultra-fast calculations
        start_time = time.time()
        
        # Find nearest nodes (much faster lookup)
        node_lookup_start = time.time()
        center_node = find_nearest_node(center_lat, center_lng)
        if center_node is None:
            return jsonify({'error': 'Could not find nearest node to center point'}), 400
        
        station_nodes = []
        valid_stations = []
        for station in stations:
            node = find_nearest_node(station['lat'], station['lng'])
            if node is not None:
                station_nodes.append(node)
                valid_stations.append(station)
        
        node_lookup_time = time.time() - node_lookup_start
        logger.info(f"NetworKit node lookup took: {node_lookup_time:.3f}s")
        
        # Calculate distances using NetworKit SSSP (Single Source Shortest Path)
        # Fast shortest path calculation using NetworKit
        distance_calc_start = time.time()
        
        # Use Dijkstra for weighted graph (fastest for our use case)
        sssp = nk.distance.Dijkstra(graph, center_node, storePaths=False)
        sssp.run()
        
        # Get distances to all station nodes
        distances = []
        for node in station_nodes:
            if graph.hasNode(node):
                dist = sssp.distance(node)
                distances.append(dist if dist != float('inf') else None)
            else:
                distances.append(None)
        
        distance_calc_time = time.time() - distance_calc_start
        logger.info(f"NetworKit distance calculation took: {distance_calc_time:.3f}s")
        
        # OPTIMIZATION: Only calculate routes for the closest 10 stations
        route_calc_start = time.time()
        
        # Create list of stations with distances and sort by distance
        station_distances = []
        for i, (station, distance) in enumerate(zip(valid_stations, distances)):
            if distance is not None and distance != float('inf'):
                station_distances.append({
                    'station': station,
                    'station_node': station_nodes[i],
                    'distance': float(distance),
                    'index': i
                })
        
        # Sort by distance and take the closest 10
        station_distances.sort(key=lambda x: x['distance'])
        closest_10 = station_distances[:10]
        logger.info(f"Smart optimization: Only calculating routes for {len(closest_10)} closest stations instead of {len(stations)}")
        
        # Calculate routes for the closest 10 stations (much faster with NetworKit)
        results = []
        for station_data in closest_10:
            station = station_data['station']
            station_node = station_data['station_node']
            
            # Get the route path using NetworKit's path reconstruction
            single_route_start = time.time()
            
            try:
                # Use A* algorithm for point-to-point pathfinding (better for routes)
                # But first let's try a simpler approach with the existing SSSP
                if center_node in coordinate_mapping and station_node in coordinate_mapping:
                    # Use simple bidirectional BFS for path finding
                    path_finder = nk.distance.BFS(graph, center_node, storePaths=True)
                    path_finder.run()
                    
                    # Check if target is reachable
                    if path_finder.distance(station_node) != float('inf'):
                        path_nodes = path_finder.getPath(station_node)
                        
                        # Convert node IDs to coordinates
                        route_coords = []
                        for node_id in path_nodes:
                            if node_id in coordinate_mapping:
                                lat, lng = coordinate_mapping[node_id]
                                route_coords.append([lat, lng])  # [lat, lng] format for Leaflet
                        
                        single_route_time = time.time() - single_route_start
                        logger.info(f"Station {station['name']}: route calc {single_route_time:.3f}s ({len(route_coords)} points)")
                        
                        if len(route_coords) > 1:  # Valid route found
                            results.append({
                                'name': station['name'],
                                'lat': station['lat'],
                                'lng': station['lng'],
                                'distance': station_data['distance'],
                                'route': route_coords
                            })
                        else:
                            logger.warning(f"Route too short for {station['name']}, using straight line")
                            results.append({
                                'name': station['name'],
                                'lat': station['lat'],
                                'lng': station['lng'],
                                'distance': station_data['distance'],
                                'route': []  # Empty route will cause frontend to draw straight line
                            })
                    else:
                        logger.warning(f"No path found to {station['name']}")
                        results.append({
                            'name': station['name'],
                            'lat': station['lat'],
                            'lng': station['lng'],
                            'distance': station_data['distance'],
                            'route': []
                        })
                else:
                    logger.warning(f"Invalid nodes for {station['name']}")
                    results.append({
                        'name': station['name'],
                        'lat': station['lat'],
                        'lng': station['lng'],
                        'distance': station_data['distance'],
                        'route': []
                    })
                    
            except Exception as route_error:
                logger.warning(f"Route calculation error for {station['name']}: {route_error}")
                # Fallback without route
                results.append({
                    'name': station['name'],
                    'lat': station['lat'],
                    'lng': station['lng'],
                    'distance': station_data['distance'],
                    'route': []
                })
        
        route_calc_time = time.time() - route_calc_start
        total_time = time.time() - start_time
        
        logger.info(f"NetworKit route calculation took: {route_calc_time:.3f}s (for {len(closest_10)} stations)")
        logger.info(f"NetworKit total processing time: {total_time:.3f}s")
        logger.info(f"Performance improvement: {((len(stations) - len(closest_10)) / len(stations) * 100):.1f}% fewer route calculations")
        
        # Sort results by actual walking distance and return top 5
        results.sort(key=lambda x: x['distance'])
        final_5 = results[:5]
        logger.info(f"Final selection: Top 5 stations by actual walking distance from the {len(closest_10)} candidates")
        
        return jsonify({'stations': final_5})
        
    except Exception as e:
        logger.error(f"NetworKit batch calculation error: {e}", exc_info=True)
        return jsonify({
            'error': str(e),
            'type': type(e).__name__,
            'networkit_loaded': networkit_graph is not None
        }), 500

if __name__ == '__main__':
    logger.info("🚀 Starting NetworKit-based walking service...")
    
    # Check if cached graph exists before starting
    cache_file = 'cdmx_networkit_graph.pkl'
    if os.path.exists(cache_file):
        file_size = os.path.getsize(cache_file) / (1024 * 1024)  # Size in MB
        logger.info(f"✅ Found cached NetworKit graph: {cache_file} ({file_size:.1f} MB)")
        logger.info("Service will use cached graph for fast startup")
    else:
        logger.warning(f"⚠️  No cached graph found at {cache_file}")
        logger.warning("Service will need to download and create graph from OSM (may take 2-5 minutes)")
    
    # Pre-load the graph at startup for better performance
    logger.info("Pre-loading NetworKit graph...")
    graph_result = load_networkit_graph()
    
    if graph_result is not None:
        logger.info("✅ NetworKit graph loaded successfully - service ready!")
    else:
        logger.error("❌ Failed to load NetworKit graph - service may not function properly")
    
    # Get port from environment variable or default to 8080
    port = int(os.environ.get('PORT', 8080))
    
    logger.info(f"🌐 Starting Flask server on port {port}")
    app.run(host='0.0.0.0', port=port, debug=False)
