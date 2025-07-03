# Metro Line Colors Feature

This feature adds color-coded metro lines to station markers on the map. It's fully integrated into the main download workflow and works automatically for any city.

## How It Works

The metro line feature is optional and can be enabled when downloading city data:

```bash
python download_data.py --city montreal --metro-lines
```

This will:
1. Download the walking network and stations (standard)
2. Query Wikidata for station-to-line mappings using property P81
3. Attempt to fetch line colors from Wikidata using property P465
4. Fall back to manual color configuration if needed

## File Structure

For each city, the metro line feature creates:

- `{city_prefix}_station_lines.json` - Auto-generated mapping of stations to line QIDs
- `{city_prefix}_metro_lines.json` - Line color definitions (auto-created if manual colors exist)

## Manual Color Configuration

When Wikidata doesn't have color data (common for many cities), you can add colors manually to `CityData/metro_line_colors.json`:

```json
{
  "cities": {
    "your_city_prefix": {
      "Q123456": {
        "name": "Line 1",
        "color": "#FF0000",
        "number": 1
      },
      "Q789012": {
        "name": "Line 2", 
        "color": "#0000FF",
        "number": 2
      }
    }
  }
}
```

The system will automatically use these colors when generating the metro lines file.

## Adding Support for a New City

1. **Ensure your city is in CITIES configuration** in `download_data.py`

2. **Run the download with metro lines**:
   ```bash
   python download_data.py --city your_city --metro-lines
   ```

3. **Check the output** - it will tell you:
   - How many stations were found with Wikidata IDs
   - Which line QIDs were discovered
   - Whether colors need to be added manually

4. **If colors are missing**, add them to `metro_line_colors.json` using the QIDs reported

## Example: Montreal

Montreal has excellent Wikidata coverage for stations but no color data:

1. Running `python download_data.py --city montreal --metro-lines` finds:
   - 65 stations with Wikidata IDs
   - 4 metro lines: Q1925762, Q967397, Q1597847, Q1726049

2. Manual colors were added to `metro_line_colors.json`:
   - Green Line (Q1925762): #00A650
   - Orange Line (Q967397): #FF6319  
   - Yellow Line (Q1597847): #FFD503
   - Blue Line (Q1726049): #0075C9

3. The system automatically creates `montreal_metro_lines.json` from this configuration

## Troubleshooting

**No stations found with Wikidata IDs**: 
- OpenStreetMap data for your city may lack `wikidata` tags
- Consider contributing to OSM by adding these tags

**No line data found**:
- Stations may not have P81 (connecting line) property in Wikidata
- You can contribute this data to Wikidata

**Colors not showing**:
- Ensure `metro_line_colors.json` has entries for your city's line QIDs
- Check that the city prefix matches exactly

## Frontend Display

When metro line data is available:
- Station markers show the primary line color
- Multi-line stations display additional colored dots
- The station list panel shows colored indicators
- All features work without line data (graceful degradation)