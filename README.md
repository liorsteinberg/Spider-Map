# CDMX Metro Spider Map

An interactive web map showing the 5 closest Mexico City metro stations to any location, with toggle between Euclidean and actual walking distances using OpenStreetMap routing.

## Features

- **Interactive Map**: Click anywhere to find the 5 closest metro stations
- **Distance Modes**: Toggle between Euclidean (straight-line) and OSM Network (walking) distances  
- **Route Visualization**: See actual walking routes to stations when using OSM Network mode
- **High Performance**: Optimized with Pandana for fast network calculations (~0.1s response time)

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the service:**
   ```bash
   python walking_service.py
   ```

3. **Open the map:**
   Open `map.html` in your browser

## File Structure

### Core Application
- `map.html` - Interactive web interface
- `walking_service.py` - Flask backend with Pandana optimization
- `CDMX-metro-stations-simple.geojson` - Metro station data

### Pre-built Networks (Generated)
- `cdmx_walking_network.graphml` - OSMnx street network (213 MB)
- `cdmx_pandana_network.h5` - Pre-built Pandana network (22 MB)

### Setup & Documentation
- `build_pandana_network.py` - Build optimized Pandana network
- `download_network.py` - Download street network from OpenStreetMap
- `requirements.txt` - Python dependencies
- `PANDANA_SETUP.md` - Detailed setup instructions

## Performance Optimization

The app uses a smart three-stage optimization:

1. **Fast Distance Calculation**: Vectorized distances to all stations (~0.01s)
2. **Focused Route Calculation**: Routes for closest 10 stations only (~0.08s)  
3. **Final Selection**: Top 5 by actual walking distance

**Result**: ~87% performance improvement over naive implementations.

## Requirements

- Python 3.8+
- pandana
- osmnx
- flask
- flask-cors
- networkx
- numpy

## Setup from Scratch

If you need to rebuild the networks:

```bash
# Download street network (one-time, ~1 minute)
python download_network.py

# Build optimized Pandana network (one-time, ~30 seconds)  
python build_pandana_network.py

# Start the service
python walking_service.py
```

The service runs on `http://localhost:8080`
