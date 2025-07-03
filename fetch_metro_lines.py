#!/usr/bin/env python3
"""
Fetch metro line information from OSM + Wikidata
"""
import osmnx as ox
import requests
import json
import time

def get_wikidata_lines(wikidata_id):
    """Fetch metro line info from Wikidata using P81 (connecting line)"""
    # Extract Q-number from wikidata ID
    if wikidata_id.startswith('Q'):
        qid = wikidata_id
    else:
        return []
    
    # SPARQL query to get connecting lines
    sparql_query = f"""
    SELECT ?line ?lineLabel WHERE {{
      wd:{qid} wdt:P81 ?line .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en,fr". }}
    }}
    """
    
    url = "https://query.wikidata.org/sparql"
    params = {
        'query': sparql_query,
        'format': 'json'
    }
    
    try:
        response = requests.get(url, params=params, headers={'User-Agent': 'Spider-Map/1.0'})
        response.raise_for_status()
        data = response.json()
        
        lines = []
        for result in data['results']['bindings']:
            line_uri = result['line']['value']
            line_qid = line_uri.split('/')[-1]
            line_label = result.get('lineLabel', {}).get('value', line_qid)
            lines.append({
                'qid': line_qid,
                'label': line_label
            })
        return lines
    except Exception as e:
        print(f"Error fetching Wikidata for {qid}: {e}")
        return []

def fetch_montreal_station_lines():
    """Fetch Montreal stations with their line information"""
    print("Fetching Montreal metro stations from OSM...")
    
    # Get stations with wikidata tags
    stations = ox.features_from_place(
        "Montreal, Quebec, Canada",
        tags={'station': 'subway'}
    )
    
    print(f"Found {len(stations)} stations")
    
    station_lines = {}
    stations_with_wikidata = 0
    
    for idx, station in stations.iterrows():
        if 'name' not in station or not station['name']:
            continue
            
        station_name = station['name']
        
        # Check for wikidata ID
        if 'wikidata' in station and station['wikidata'] and str(station['wikidata']) != 'nan':
            wikidata_id = str(station['wikidata'])
            stations_with_wikidata += 1
            
            print(f"\n{station_name} ({wikidata_id}):")
            
            # Get line info from Wikidata
            lines = get_wikidata_lines(wikidata_id)
            
            if lines:
                station_lines[station_name] = lines
                for line in lines:
                    print(f"  - {line['label']} ({line['qid']})")
            else:
                print("  - No line data found")
            
            # Be nice to Wikidata API
            time.sleep(0.1)
    
    print(f"\n\nSummary:")
    print(f"Total stations: {len(stations)}")
    print(f"Stations with Wikidata: {stations_with_wikidata}")
    print(f"Stations with line data: {len(station_lines)}")
    
    # Save the results
    output_file = 'CityData/montreal_station_lines.json'
    with open(output_file, 'w') as f:
        json.dump(station_lines, f, indent=2)
    print(f"\nSaved to: {output_file}")
    
    return station_lines

# Known Montreal metro lines from Wikidata
MONTREAL_LINES = {
    'Q952838': {'name': 'Green Line', 'color': '#00A650', 'number': 1},
    'Q952795': {'name': 'Orange Line', 'color': '#FF6319', 'number': 2},
    'Q1131454': {'name': 'Yellow Line', 'color': '#FFD503', 'number': 4},
    'Q804970': {'name': 'Blue Line', 'color': '#0075C9', 'number': 5}
}

if __name__ == "__main__":
    station_lines = fetch_montreal_station_lines()
    
    print("\n\nLine mapping for colors:")
    for line_qid, line_info in MONTREAL_LINES.items():
        print(f"{line_info['name']}: {line_info['color']}")