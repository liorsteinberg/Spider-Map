#!/usr/bin/env python3
"""
Explore metro line data from OSM for Montreal
"""
import osmnx as ox
import pandas as pd
import json

print("Fetching Montreal metro data from OSM...")

# Get metro routes (lines)
try:
    # Try to get subway routes
    routes = ox.features_from_place(
        "Montreal, Quebec, Canada", 
        tags={'route': 'subway'}
    )
    print(f"\nFound {len(routes)} subway routes")
    
    # Display route information
    for idx, route in routes.iterrows():
        print(f"\nRoute {idx}:")
        for key, value in route.items():
            if value is not None and key != 'geometry' and str(value) != 'nan':
                print(f"  {key}: {value}")
except Exception as e:
    print(f"Error fetching routes: {e}")

# Get more detailed station info with relations
print("\n\nFetching detailed station data...")
try:
    # Get stations with more tags
    stations = ox.features_from_place(
        "Montreal, Quebec, Canada",
        tags={'station': 'subway'}
    )
    
    # Check what attributes stations have
    print(f"\nFound {len(stations)} stations")
    print("\nAvailable station attributes:")
    
    # Get all unique non-null attributes across all stations
    all_attrs = set()
    for idx, station in stations.iterrows():
        for key, value in station.items():
            if value is not None and str(value) != 'nan' and key != 'geometry':
                all_attrs.add(key)
    
    print(sorted(all_attrs))
    
    # Show a few example stations with their attributes
    print("\n\nExample stations with attributes:")
    count = 0
    for idx, station in stations.iterrows():
        if count >= 3:
            break
        if 'name' in station and station['name']:
            print(f"\n{station['name']}:")
            for key, value in station.items():
                if value is not None and str(value) != 'nan' and key != 'geometry':
                    print(f"  {key}: {value}")
            count += 1
            
except Exception as e:
    print(f"Error: {e}")

# Try alternative approach - get railway data
print("\n\nTrying railway=station approach...")
try:
    railway_stations = ox.features_from_place(
        "Montreal, Quebec, Canada",
        tags={'railway': 'station', 'subway': 'yes'}
    )
    print(f"Found {len(railway_stations)} railway stations marked as subway")
    
    # Check for line information
    for idx, station in railway_stations.head(3).iterrows():
        if 'name' in station and station['name']:
            print(f"\n{station['name']}:")
            for key, value in station.items():
                if value is not None and str(value) != 'nan' and key != 'geometry':
                    if 'line' in key.lower() or 'route' in key.lower() or 'colour' in key.lower() or 'color' in key.lower():
                        print(f"  {key}: {value}")
except Exception as e:
    print(f"Error: {e}")

# Check for Montreal-specific line colors
print("\n\nMontreal Metro Line Colors (hardcoded knowledge):")
montreal_lines = {
    "Green Line": "#00A650",  # Line 1
    "Orange Line": "#FF6319", # Line 2  
    "Yellow Line": "#FFD503", # Line 4
    "Blue Line": "#0075C9"    # Line 5
}
for line, color in montreal_lines.items():
    print(f"  {line}: {color}")