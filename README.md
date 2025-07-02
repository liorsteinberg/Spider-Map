# Spider Map - Multi-City Metro Network Explorer

An interactive web map showing the closest metro stations to any location, with adjustable spider legs (1-8 stations) and toggle between Euclidean and actual walking distances using OpenStreetMap routing.

This project supports multiple cities and is inspired by [Carlos Enrique Vázquez Juárez's original work](https://carto.mx/webmap/spoke/) and builds upon it to create an open-source, enhanced version with additional features such as pedestrian network routing and high-performance network analysis.

## 🌐 Live Demo

**Try it now:** [https://spidermap.steinberg.nu/](https://spidermap.steinberg.nu/)

## 🚀 Fast NetworKit Backend

**High-performance**: Uses **[NetworKit](https://github.com/networkit/networkit)** for ultra-fast network analysis and distance calculations!

## Features

- **Multi-City Support**: Explore metro networks in multiple cities worldwide
- **Interactive Map**: Click anywhere to find the closest metro stations
- **Adjustable Spider Legs**: Choose between 1-8 stations using the slider
- **Distance Modes**: Toggle between Euclidean (straight-line) and OSM Network (walking) distances  
- **Route Visualization**: See actual walking routes to stations when using OSM Network mode
- **Ultra-High Performance**: NetworKit backend with fast network calculations
- **Optimized Frontend**: Spatial indexing and caching for instant Euclidean distance calculations

## 🌍 Available Cities

- **Mexico City, Mexico**
- **Berlin, Germany**
- **Beijing, China**

## 👨‍💻 Created By

**Lior Steinberg** - [steinberg.nu](https://steinberg.nu)  
Urban planner and co-founder of [Humankind](https://humankind.city)

## 🚀 Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Download city data:**
   ```bash
   python download_data.py --city mexico_city  # or berlin, beijing
   # Or download all cities: python download_data.py --all
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
- `download_data.py` - Download and build city network data

### City Data
- `CityData/` - Contains network graphs and station data for all cities
  - `{city}_walking_graph.pkl` - NetworKit graph cache for each city
  - `{city}_stations.geojson` - Metro station data for each city

### Setup & Documentation
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

If you need to rebuild the network or add a new city:

```bash
# Download data for a specific city
python download_data.py --city mexico_city

# Download data for all supported cities
python download_data.py --all

# List available cities
python download_data.py --list-cities

# Start the service
python walking_service.py
```

The service runs on `http://localhost:8080`
