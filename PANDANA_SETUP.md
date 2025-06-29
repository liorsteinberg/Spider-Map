# Pandana Network Setup

This project now uses pre-built Pandana networks to optimize startup time.

## Setup Process

### 1. Download OSMnx Network (one-time)
```bash
python download_network.py
```
This downloads the walking network for Mexico City and saves it as `cdmx_walking_network.graphml`.

### 2. Build Pandana Network (one-time)
```bash
python build_pandana_network.py
```
This converts the OSMnx network to Pandana format and saves it as `cdmx_pandana_network.h5`.

### 3. Run the Service
```bash
python walking_service.py
```
The service will now load the pre-built Pandana network, which is faster than building it from scratch.

## Performance Notes

- **Original approach**: Convert OSMnx → Pandana → Build contraction hierarchies (~30-35s)
- **Optimized approach**: Load pre-built Pandana → Build contraction hierarchies (~20s)
- **Unavoidable**: Contraction hierarchy building (~18-20s) - this happens even with pre-built networks

The contraction hierarchy building is a Pandana requirement for fast shortest path calculations and cannot be avoided.

## Files Created

- `cdmx_walking_network.graphml` (213 MB) - OSMnx network
- `cdmx_pandana_network.h5` (22 MB) - Pre-built Pandana network
- `build_pandana_network.py` - Script to build Pandana network
- `download_network.py` - Script to download OSMnx network
- `test_pandana_load.py` - Test script for Pandana loading

## Timing Results

With the optimized setup, the timing output will show:
- Node lookup: ~0.02s
- Distance calculation: ~0.15s  
- Route calculation: ~0.04s per station
- Total per request: ~0.4s for 5 stations

This is significantly faster than the original NetworkX approach which took ~3-5s per request.
