# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Spider Map is an interactive web application that visualizes the closest metro stations to any location with both Euclidean and actual walking distances. The project emphasizes performance through NetworKit graph algorithms and spatial indexing.

## Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Download city data (required before first run)
python download_data.py --city mexico_city  # Download specific city
python download_data.py --all               # Download all cities
python download_data.py --list-cities       # List available cities

# Run development server
python walking_service.py                   # Runs on http://localhost:8080

# Production deployment
gunicorn walking_service:app
```

## Architecture Overview

### Backend Service (walking_service.py)

The Flask backend provides high-performance network analysis:

- **NetworKit Integration**: Uses pre-computed NetworKit graphs for ultra-fast Dijkstra calculations (50x faster than NetworkX)
- **Spatial Indexing**: Implements cKDTree for O(log n) station lookups
- **Multi-City Support**: Dynamic city switching with separate graph caches
- **Performance Optimizations**:
  - Single Dijkstra run with path storage
  - Pre-computed graphs cached as pickle files
  - Vectorized operations with NumPy
  - Efficient memory management with graph clearing on city switch

Key API endpoints:
- `/` - Serves the interactive map
- `/walking_distance` - Calculates walking distances to nearest stations
- `/set_city` - Switches between cities
- `/get_stations` - Returns station data for current city
- `/health` - Health check endpoint
- `/api` - API info

### Frontend (map.html)

Single-page application with:
- **Leaflet.js** for interactive mapping
- **Spatial indexing** for instant Euclidean calculations
- **Frontend caching** to minimize API calls
- **Responsive design** with Tailwind CSS
- Click-to-explore functionality with 1-8 adjustable spider legs

### Data Pipeline (download_data.py)

Prepares city data by:
1. Downloading OSM walking networks via OSMnx
2. Converting to NetworKit format for performance
3. Building spatial indexes
4. Saving as compressed pickle files in `CityData/`

## Performance Considerations

1. **Graph Operations**: Always use NetworKit methods, not NetworkX
2. **Spatial Queries**: Use the pre-built cKDTree indexes
3. **City Switching**: Clears previous graph from memory before loading new one
4. **Frontend**: Implements its own spatial indexing to avoid unnecessary API calls

## Adding New Cities

1. Add city configuration to `download_data.py`
2. Run download script for the new city
3. The backend automatically detects new city files
4. Add city to frontend dropdown in `map.html`

## Common Development Tasks

When modifying network analysis:
- Ensure NetworKit graph format is maintained
- Test performance with large click volumes
- Verify path reconstruction works correctly

When updating the frontend:
- Test across different screen sizes
- Ensure map interactions remain responsive
- Verify city switching clears previous data

## Deployment Notes

- Designed for Heroku/cloud deployment (see Procfile)
- Uses environment variable PORT if available
- CORS enabled for API access
- Gunicorn for production WSGI serving