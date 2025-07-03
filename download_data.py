#!/usr/bin/env python3
"""
Download and build NetworKit networks for multiple cities
This script creates NetworKit graphs from OSM data for efficient network analysis
Supports both walking networks and subway/metro station data
"""

import time
import logging
import pickle
import json
import os
import argparse
import numpy as np
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supported cities configuration
CITIES = {
    'mexico_city': {
        'name': 'Mexico City, Mexico',
        'display_name': 'Mexico City',
        'cache_prefix': 'cdmx'
    },
    'berlin': {
        'name': 'Berlin, Germany', 
        'display_name': 'Berlin',
        'cache_prefix': 'berlin'
    },
    'beijing': {
        'name': 'Beijing, China',
        'display_name': 'Beijing',
        'cache_prefix': 'beijing'
    },
    'montreal': {
        'name': 'Montreal, Quebec, Canada',
        'display_name': 'Montreal',
        'cache_prefix': 'montreal'
    }
}

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

def download_walking_network(city_config):
    """Download and process walking network for a city"""
    city_name = city_config['name']
    cache_prefix = city_config['cache_prefix']
    
    try:
        import networkit as nk
        import osmnx as ox
        logger.info(f"NetworKit version: {nk.__version__}")
        logger.info(f"OSMnx version: {ox.__version__}")
    except ImportError as e:
        logger.error(f"Required dependency missing: {e}")
        logger.error("Please install dependencies: pip install networkit osmnx")
        return False

    # Ensure CityData directory exists
    os.makedirs('CityData', exist_ok=True)

    # Check if walking network already exists
    cache_file = f'CityData/{cache_prefix}_walking_graph.pkl'
    if os.path.exists(cache_file):
        logger.info(f"Walking network already exists at {cache_file}")
        return True
    
    logger.info(f"Downloading OSM walking network for {city_name}...")
    download_start = time.time()
    
    try:
        logger.info(f"Using: ox.graph_from_place('{city_name}', network_type='walk')")
        G = ox.graph_from_place(city_name, network_type='walk')
        download_time = time.time() - download_start
        logger.info(f"OSM graph downloaded in {download_time:.2f}s")
        logger.info(f"OSM graph: {len(G.nodes)} nodes, {len(G.edges)} edges")
        
    except Exception as e:
        logger.error(f"Failed to download OSM data: {e}")
        return False

    # Convert OSMnx graph to NetworKit
    logger.info("Converting OSMnx graph to NetworKit...")
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

        convert_time = time.time() - convert_start
        logger.info(f"NetworKit conversion completed in {convert_time:.2f}s")
        logger.info(f"Created NetworKit graph: {networkit_graph.numberOfNodes()} nodes, {networkit_graph.numberOfEdges()} edges")
        
    except Exception as e:
        logger.error(f"Failed to convert to NetworKit: {e}")
        return False

    # Save walking network
    logger.info("Saving walking network...")
    save_start = time.time()
    
    try:
        cache_data = {
            'graph': networkit_graph,
            'node_mapping': node_mapping,
            'coordinate_mapping': coordinate_mapping,
            'metadata': {
                'created_at': time.time(),
                'city': city_name,
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
        logger.info(f"Walking network saved in {save_time:.2f}s")
        logger.info(f"Cache file: {cache_file} ({file_size:.1f} MB)")
        
    except Exception as e:
        logger.error(f"Failed to save walking network: {e}")
        return False

    return True

def download_subway_stations(city_config):
    """Download subway/metro stations for a city and save as GeoJSON"""
    city_name = city_config['name']
    cache_prefix = city_config['cache_prefix']
    
    try:
        import osmnx as ox
    except ImportError as e:
        logger.error(f"Required dependency missing: {e}")
        return False

    # Ensure CityData directory exists
    os.makedirs('CityData', exist_ok=True)

    # Check if subway stations already exist
    geojson_file = f'CityData/{cache_prefix}_stations.geojson'
    if os.path.exists(geojson_file):
        logger.info(f"Subway stations already exist at {geojson_file}")
        return True
    
    logger.info(f"Downloading subway stations for {city_name}...")
    download_start = time.time()
    
    try:
        # Download subway stations using OSMnx
        logger.info(f"Using: ox.features_from_place('{city_name}', tags={{'station': 'subway'}})")
        stations_gdf = ox.features_from_place(city_name, tags={'station': 'subway'})
        
        download_time = time.time() - download_start
        logger.info(f"Subway stations downloaded in {download_time:.2f}s")
        logger.info(f"Found {len(stations_gdf)} subway stations")
        
    except Exception as e:
        logger.error(f"Failed to download subway stations: {e}")
        logger.warning("Continuing without subway stations...")
        return False

    # Process and save subway stations as GeoJSON
    logger.info("Processing subway stations...")
    
    try:
        # Create GeoJSON structure
        geojson_data = {
            "type": "FeatureCollection",
            "features": []
        }
        
        stations_processed = 0
        for idx, row in stations_gdf.iterrows():
            # Get station coordinates
            if hasattr(row.geometry, 'centroid'):
                centroid = row.geometry.centroid
                lat, lng = centroid.y, centroid.x
            elif hasattr(row.geometry, 'y') and hasattr(row.geometry, 'x'):
                lat, lng = row.geometry.y, row.geometry.x
            else:
                continue
            
            # Get station name
            station_name = row.get('name', f'Station {stations_processed + 1}')
            if not station_name or str(station_name).strip() == '' or str(station_name) == 'nan':
                station_name = f'Station {stations_processed + 1}'
            
            # Create GeoJSON feature
            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [lng, lat]  # GeoJSON uses [lng, lat] order
                },
                "properties": {
                    "name": str(station_name),
                    "osm_id": str(idx) if hasattr(idx, '__iter__') else str(idx)
                }
            }
            
            geojson_data["features"].append(feature)
            stations_processed += 1
        
        # Save as GeoJSON
        with open(geojson_file, 'w', encoding='utf-8') as f:
            json.dump(geojson_data, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(geojson_file) / 1024  # KB
        logger.info(f"Subway stations saved as GeoJSON: {geojson_file} ({file_size:.1f} KB)")
        logger.info(f"Processed {stations_processed} subway stations")
        
    except Exception as e:
        logger.error(f"Failed to process subway stations: {e}")
        return False

    return True

def download_metro_lines(city_config, skip_if_exists=True):
    """Download metro line information from Wikidata for stations"""
    cache_prefix = city_config['cache_prefix']
    
    # File paths
    stations_file = f'CityData/{cache_prefix}_stations.geojson'
    station_lines_file = f'CityData/{cache_prefix}_station_lines.json'
    
    # Check if already exists
    if skip_if_exists and os.path.exists(station_lines_file):
        logger.info(f"Metro line data already exists at {station_lines_file}")
        return True
    
    # Check if stations file exists
    if not os.path.exists(stations_file):
        logger.warning(f"Stations file not found: {stations_file}")
        logger.warning("Run download_subway_stations first")
        return False
    
    logger.info(f"Fetching metro line data from Wikidata for {city_config['display_name']}...")
    
    # Load stations
    try:
        with open(stations_file, 'r', encoding='utf-8') as f:
            geojson_data = json.load(f)
    except Exception as e:
        logger.error(f"Failed to load stations file: {e}")
        return False
    
    # Helper function to query Wikidata
    def get_wikidata_lines(wikidata_id):
        """Fetch metro line info from Wikidata using P81 (connecting line)"""
        if not wikidata_id or not wikidata_id.startswith('Q'):
            return []
        
        # SPARQL query to get connecting lines and their colors
        sparql_query = f"""
        SELECT ?line ?lineLabel ?color WHERE {{
          wd:{wikidata_id} wdt:P81 ?line .
          OPTIONAL {{ ?line wdt:P465 ?color . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,fr,de,es,zh". }}
        }}
        """
        
        url = "https://query.wikidata.org/sparql"
        params = {
            'query': sparql_query,
            'format': 'json'
        }
        
        try:
            response = requests.get(url, params=params, headers={'User-Agent': 'Spider-Map/1.0'}, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            lines = []
            for result in data['results']['bindings']:
                line_uri = result['line']['value']
                line_qid = line_uri.split('/')[-1]
                line_label = result.get('lineLabel', {}).get('value', line_qid)
                line_color = result.get('color', {}).get('value', None)
                
                line_info = {
                    'qid': line_qid,
                    'label': line_label
                }
                if line_color:
                    line_info['color'] = f"#{line_color}"  # Wikidata returns RGB without #
                
                lines.append(line_info)
            return lines
        except Exception as e:
            logger.debug(f"Error fetching Wikidata for {wikidata_id}: {e}")
            return []
    
    # Process stations
    station_lines = {}
    stations_processed = 0
    stations_with_wikidata = 0
    stations_with_lines = 0
    
    # First, try to get wikidata IDs from OSM if not in GeoJSON
    try:
        import osmnx as ox
        logger.info("Fetching additional station metadata from OSM...")
        osm_stations = ox.features_from_place(
            city_config['name'],
            tags={'station': 'subway'}
        )
        
        # Create a mapping of station names to wikidata IDs
        wikidata_mapping = {}
        for idx, station in osm_stations.iterrows():
            if 'name' in station and station['name'] and 'wikidata' in station:
                wikidata_id = str(station['wikidata'])
                if wikidata_id and wikidata_id != 'nan':
                    wikidata_mapping[station['name']] = wikidata_id
    except Exception as e:
        logger.warning(f"Could not fetch additional OSM data: {e}")
        wikidata_mapping = {}
    
    # Process each station
    for feature in geojson_data.get('features', []):
        if feature['geometry']['type'] != 'Point':
            continue
        
        station_name = feature['properties'].get('name', '')
        if not station_name:
            continue
        
        stations_processed += 1
        
        # Try to get wikidata ID from mapping
        wikidata_id = wikidata_mapping.get(station_name)
        
        if wikidata_id:
            stations_with_wikidata += 1
            
            # Get line info from Wikidata
            lines = get_wikidata_lines(wikidata_id)
            
            if lines:
                station_lines[station_name] = lines
                stations_with_lines += 1
                logger.debug(f"{station_name}: {len(lines)} lines found")
            
            # Be nice to Wikidata API
            time.sleep(0.1)
    
    # Save results
    try:
        with open(station_lines_file, 'w', encoding='utf-8') as f:
            json.dump(station_lines, f, indent=2, ensure_ascii=False)
        
        file_size = os.path.getsize(station_lines_file) / 1024  # KB
        logger.info(f"Metro line data saved: {station_lines_file} ({file_size:.1f} KB)")
        logger.info(f"Summary: {stations_processed} stations, {stations_with_wikidata} with Wikidata, {stations_with_lines} with line data")
        
        # Check if we need manual color configuration
        lines_with_colors = 0
        all_line_qids = set()
        for station_data in station_lines.values():
            for line in station_data:
                all_line_qids.add(line['qid'])
                if 'color' in line:
                    lines_with_colors += 1
        
        if all_line_qids and lines_with_colors == 0:
            logger.warning("No line colors found in Wikidata!")
            logger.info("Checking manual color configuration...")
            
            # Try to load manual colors and create metro_lines file
            manual_colors_file = 'CityData/metro_line_colors.json'
            if os.path.exists(manual_colors_file):
                try:
                    with open(manual_colors_file, 'r', encoding='utf-8') as f:
                        manual_data = json.load(f)
                    
                    city_colors = manual_data.get('cities', {}).get(cache_prefix, {})
                    if city_colors and not isinstance(city_colors.get('_comment'), str):
                        # Create metro_lines.json file
                        metro_lines_file = f'CityData/{cache_prefix}_metro_lines.json'
                        metro_lines_data = {'lines': city_colors}
                        
                        with open(metro_lines_file, 'w', encoding='utf-8') as f:
                            json.dump(metro_lines_data, f, indent=2, ensure_ascii=False)
                        
                        logger.info(f"Created metro lines file with manual colors: {metro_lines_file}")
                    else:
                        logger.info(f"No manual colors configured for {cache_prefix}")
                        logger.info(f"Add colors to {manual_colors_file} for line QIDs: {sorted(all_line_qids)}")
                except Exception as e:
                    logger.warning(f"Could not load manual colors: {e}")
            else:
                logger.info(f"Manual color file not found: {manual_colors_file}")
                logger.info(f"Line QIDs found: {sorted(all_line_qids)}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to save metro line data: {e}")
        return False

def download_city_data(city_key, include_metro_lines=False):
    """Download all data for a specific city"""
    if city_key not in CITIES:
        logger.error(f"Unknown city: {city_key}")
        logger.error(f"Available cities: {list(CITIES.keys())}")
        return False
    
    city_config = CITIES[city_key]
    logger.info(f"Downloading data for {city_config['display_name']}...")
    
    start_time = time.time()
    
    # Determine number of steps based on options
    total_steps = 2
    if include_metro_lines:
        total_steps = 3
    
    # Download walking network
    logger.info(f"Step 1/{total_steps}: Downloading walking network...")
    walking_success = download_walking_network(city_config)
    
    # Download subway stations
    logger.info(f"Step 2/{total_steps}: Downloading subway stations...")
    subway_success = download_subway_stations(city_config)
    
    # Download metro line data if requested
    metro_lines_success = False
    if include_metro_lines and subway_success:
        logger.info(f"Step 3/{total_steps}: Fetching metro line data...")
        metro_lines_success = download_metro_lines(city_config)
    
    total_time = time.time() - start_time
    
    # Summary
    logger.info("=" * 60)
    logger.info(f"🚀 Data Download Complete for {city_config['display_name']}!")
    logger.info("=" * 60)
    logger.info(f"Total download time: {total_time:.2f}s")
    logger.info(f"Walking network: {'✅ Success' if walking_success else '❌ Failed'}")
    logger.info(f"Subway stations: {'✅ Success' if subway_success else '❌ Failed'}")
    if include_metro_lines:
        logger.info(f"Metro line data: {'✅ Success' if metro_lines_success else '❌ Failed'}")
    
    if walking_success:
        logger.info("🎯 Next steps:")
        logger.info("   1. Run: python walking_service.py")
        logger.info("   2. Open: http://localhost:8080")
        logger.info("   3. Select your city and enjoy!")
    
    return walking_success  # At minimum we need walking network

def main():
    parser = argparse.ArgumentParser(description='Download city data for Spider Map')
    parser.add_argument('--city', choices=list(CITIES.keys()), 
                       help='City to download data for')
    parser.add_argument('--all', action='store_true',
                       help='Download data for all supported cities')
    parser.add_argument('--list-cities', action='store_true',
                       help='List all supported cities')
    parser.add_argument('--metro-lines', action='store_true',
                       help='Also fetch metro line data from Wikidata (optional feature)')
    parser.add_argument('--skip-metro-lines', action='store_true',
                       help='Skip fetching metro line data even if available')
    
    args = parser.parse_args()
    
    # Determine whether to include metro lines
    include_metro_lines = args.metro_lines and not args.skip_metro_lines
    
    if args.list_cities:
        logger.info("Supported cities:")
        for key, config in CITIES.items():
            logger.info(f"  {key}: {config['display_name']}")
        return True
    
    if args.all:
        logger.info("Downloading data for all cities...")
        if include_metro_lines:
            logger.info("Including metro line data...")
        success_count = 0
        for city_key in CITIES.keys():
            if download_city_data(city_key, include_metro_lines=include_metro_lines):
                success_count += 1
        
        logger.info(f"Successfully downloaded data for {success_count}/{len(CITIES)} cities")
        return success_count > 0
    
    if args.city:
        return download_city_data(args.city, include_metro_lines=include_metro_lines)
    
    # Default behavior - download Mexico City for backward compatibility
    logger.info("No city specified, downloading Mexico City (default)")
    return download_city_data('mexico_city', include_metro_lines=include_metro_lines)

if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)
