# Metro Line Colors Feature

This feature adds color-coded metro lines to station markers on the map.

## File Structure

For each city that supports metro line colors, two files are needed in the `CityData/` directory:

1. `{city_prefix}_station_lines.json` - Maps station names to their line IDs
2. `{city_prefix}_metro_lines.json` - Defines colors for each line

## File Formats

### Station Lines File (`{city_prefix}_station_lines.json`)
Maps station names to Wikidata QIDs of their lines:
```json
{
  "Station Name": [
    {
      "qid": "Q123456",
      "label": "Line Name"
    }
  ]
}
```

### Metro Lines File (`{city_prefix}_metro_lines.json`)
Defines the color and name for each line:
```json
{
  "lines": {
    "Q123456": {
      "name": "Line Name",
      "color": "#FF0000",
      "number": 1
    }
  }
}
```

## Adding Support for a New City

1. **Generate Station-Line Mappings**:
   - Modify `fetch_metro_lines.py` to target your city
   - Run the script to query Wikidata for station line assignments
   - This creates `CityData/{city_prefix}_station_lines.json`

2. **Create Line Color Definitions**:
   - Research official metro line colors for the city
   - Create `CityData/{city_prefix}_metro_lines.json`
   - Map Wikidata QIDs to official colors

3. **No Code Changes Required**:
   - The backend automatically detects and loads these files
   - Uses the city's `cache_prefix` from `CITIES` configuration

## Example: Montreal

- `montreal_station_lines.json` - Generated from Wikidata, maps 65 stations to lines
- `montreal_metro_lines.json` - Manually created with official STM colors:
  - Green Line: #00A650
  - Orange Line: #FF6319
  - Yellow Line: #FFD503
  - Blue Line: #0075C9

## Alternative Data Sources

If Wikidata is incomplete for your city:
- GTFS feeds often include route colors
- OpenStreetMap may have `colour` tags on routes
- Transit agency APIs or documentation
- Manual station-to-line mapping

## Frontend Display

Stations automatically display with:
- Primary line color as the main dot color
- Multi-line stations show additional small dots
- Colors appear in both the map markers and station list panel