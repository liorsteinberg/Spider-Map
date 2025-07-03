# Spider Map - Multi-City Metro Network Explorer

An interactive web map showing the closest metro stations to any location, with adjustable "spider legs" (1-8 stations) and the ability to toggle between Euclidean (straight-line) and actual walking distances using OpenStreetMap routing.

This project is an open-source, enhanced version of [Carlos Enrique Vázquez Juárez's original work](https://carto.mx/webmap/spoke/). It significantly builds upon the original by introducing **pedestrian network routing** and a **high-performance backend** for ultra-fast network analysis.

---

## 🌐 Live Demo

**Try it now:** [https://spidermap.steinberg.nu/](https://spidermap.steinberg.nu/)

---

## ✨ Key Features

* **Multi-City Support**: Explore metro networks in various cities worldwide.
* **Interactive Map**: Simply click anywhere to instantly find the closest metro stations.
* **Adjustable Spider Legs**: Easily choose to display between 1 to 8 closest stations using a slider.
* **Flexible Distance Modes**: Switch between **Euclidean** (straight-line) and **OSM Network** (actual walking) distances.
* **Route Visualization**: When using OSM Network mode, see the precise walking routes to stations.
* **Ultra-High Performance**: Powered by a **[NetworKit](https://github.com/networkit/networkit)** backend for lightning-fast network analysis and distance calculations.
* **Optimized Frontend**: Features spatial indexing and caching for instant Euclidean distance calculations on the map.

---

## 🌍 Available Cities

* **Mexico City, Mexico**
* **Berlin, Germany**
* **Beijing, China**

---

## 🚀 Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### Prerequisites

* **Python 3.8+**
* All other dependencies are listed in `requirements.txt`.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-repo/spider-map.git](https://github.com/your-repo/spider-map.git) # Replace with your actual repo URL
    cd spider-map
    ```

2.  **Install Python dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Download city data:**
    You can download data for a specific city or all supported cities.
    ```bash
    python download_data.py --city mexico_city  # or berlin, beijing
    # To download all cities:
    # python download_data.py --all
    # To list available cities:
    # python download_data.py --list-cities
    ```

4.  **Start the backend service:**
    ```bash
    python walking_service.py
    ```

5.  **Open the map in your browser:**
    Visit `http://localhost:8080`

---

## 📂 Project Structure

* `map.html` - The interactive web interface with frontend optimizations.
* `walking_service.py` - The ultra-fast NetworKit backend service. ⭐
* `download_data.py` - Script to download and build city network data.
* `requirements.txt` - Lists all project dependencies.
* `CityData/` - Directory containing network graphs and station data for all cities.
    * `{city}_walking_graph.pkl` - NetworKit graph cache for each city.
    * `{city}_stations.geojson` - Metro station data for each city.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

---

## 👨‍💻 Created By

**Lior Steinberg** - [steinberg.nu](https://steinberg.nu)  
Urban planner and co-founder of [Humankind](https://humankind.city)