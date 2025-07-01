# CDMX Metro Spider Map

An interactive web map showing the 5 closest Mexico City metro stations to any location, with toggle between Euclidean and actual walking distances using OpenStreetMap routing.

This project is inspired by [Carlos Enrique Vázquez Juárez's original work](https://carto.mx/webmap/spoke/) and builds upon it to create an open-source, enhanced version with additional features such as pedestrian network routing and high-performance network analysis.

## 🌐 Live Demo

**Try it now:** [https://spidermap.steinberg.nu/](https://spidermap.steinberg.nu/)

## 🚀 Fast NetworKit Backend

**High-performance**: Uses **[NetworKit](https://github.com/networkit/networkit)** for ultra-fast network analysis and distance calculations!

## Features

- **Interactive Map**: Click anywhere to find the 5 closest metro stations
- **Distance Modes**: Toggle between Euclidean (straight-line) and OSM Network (walking) distances  
- **Route Visualization**: See actual walking routes to stations when using OSM Network mode
- **Ultra-High Performance**: NetworKit backend with fast network calculations
- **Optimized Frontend**: Spatial indexing and caching for instant Euclidean distance calculations

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Build NetworKit network:**
   ```bash
   python build_networkit_network.py  # ~10 seconds (downloads from OSM)
   ```

3. **Start the service:**
   ```bash
   python walking_service.py
   ```

4. **Open the map:**
   Visit `http://localhost:8080`

## File Structure

### Core Application
- `map.html` - Interactive web interface with frontend optimizations
- `walking_service.py` - Ultra-fast NetworKit backend service ⭐
- `CDMX-metro-stations-simple.geojson` - Metro station data (165 stations)

### Pre-built Networks
- `cdmx_networkit_graph.pkl` - NetworKit graph cache (~50MB) ⭐

### Setup & Documentation
- `build_networkit_network.py` - Build NetworKit network
- `requirements.txt` - Project dependencies

## Requirements

- Python 3.8+
- networkit
- osmnx
- flask
- flask-cors
- networkx
- numpy

## Setup from Scratch

If you need to rebuild the network:

```bash
# Build NetworKit network (downloads from OSM if needed)
python build_networkit_network.py

# Start the service
python walking_service.py
```

The service runs on `http://localhost:8080`
