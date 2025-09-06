import io
import zipfile
import logging
from datetime import datetime, timedelta
from math import radians, cos, sin, asin, sqrt

import geojson
import pandas as pd
import requests
import pytz
from flask import Flask, jsonify, request, Response
from google.transit import gtfs_realtime_pb2

# -------------------------------------------------------------------
# Config & Logging
# -------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

STATIC_GTFS_URL = "https://dartfirststate.com/RiderInfo/Routes/gtfs_data/dartfirststate_de_us.zip"
GTFS_RT_URL = "https://tmc.deldot.gov/gtfs/VehiclePositions.pb"
DEFAULT_SPEED_MS = 6.7
gtfs_data = {}

# -------------------------------------------------------------------
# Geo helpers
# -------------------------------------------------------------------
def haversine(lon1, lat1, lon2, lat2):
    lon1, lat1, lon2, lat2 = map(radians, [lon1, lat1, lon2, lat2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    r = 6371000
    return c * r

def cumulative_distances(coords):
    dists = [0.0]
    for i in range(1, len(coords)):
        d = haversine(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1])
        dists.append(dists[-1] + d)
    return dists

def nearest_shape_index(coords, point):
    plon, plat = point
    best_i, best_d = 0, float("inf")
    for i, (lon, lat) in enumerate(coords):
        d = haversine(lon, lat, plon, plat)
        if d < best_d:
            best_d, best_i = d, i
    return best_i

# -------------------------------------------------------------------
# GTFS processing
# -------------------------------------------------------------------
def process_gtfs_data():
    logging.info("Downloading static GTFS data…")
    r = requests.get(STATIC_GTFS_URL, timeout=60)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as z:
        gtfs_data["routes"] = pd.read_csv(z.open("routes.txt"), dtype={"route_id": str})[["route_id", "route_short_name", "route_long_name"]].to_dict("records")
        trips_df = pd.read_csv(z.open("trips.txt"), dtype={"route_id": str, "trip_id": str, "shape_id": str, "direction_id": "Int64"})
        gtfs_data["stop_times_df"] = pd.read_csv(z.open("stop_times.txt"), dtype={"trip_id": str, "stop_id": str})
        gtfs_data["stops_dict"] = pd.read_csv(z.open("stops.txt"), dtype={"stop_id": str}).set_index("stop_id").to_dict("index")
        gtfs_data["trips_df"] = trips_df.set_index("trip_id")
        shapes_df = pd.read_csv(z.open("shapes.txt"), dtype={"shape_id": str})

    shapes = {}
    for shape_id, group in shapes_df.groupby("shape_id"):
        sorted_group = group.sort_values("shape_pt_sequence")
        coords = list(zip(sorted_group.shape_pt_lon.values, sorted_group.shape_pt_lat.values))
        shapes[shape_id] = {"coords": coords, "cumdist": cumulative_distances(coords)}
    gtfs_data["shapes"] = shapes

    route_info = {}
    trips_with_shape = trips_df.dropna(subset=["shape_id", "direction_id"])
    logging.info("Building route/direction info…")
    for (route_id, direction_id), group in trips_with_shape.groupby(["route_id", "direction_id"]):
        if route_id not in route_info: route_info[route_id] = {}
        mode_shape = group["shape_id"].mode()
        if mode_shape.empty: continue
        shape_id = str(mode_shape.iloc[0])
        if shape_id not in shapes: continue
        hs_mode = group["trip_headsign"].mode()
        headsign = hs_mode.iloc[0] if not hs_mode.empty else "Unknown Destination"
        route_info[route_id][int(direction_id)] = {
            "shape_id": shape_id,
            "headsign": headsign,
            "trip_ids": set(group['trip_id'].unique())
        }
    gtfs_data["route_info"] = route_info
    logging.info("GTFS static loaded.")

# -------------------------------------------------------------------
# Flask + HTML
# -------------------------------------------------------------------
HTML_TEMPLATE = """
<!doctype html>
<html lang="en">
<head>
  <title>DART Delaware Live Bus Tracker</title>
  <meta charset="utf-8" /><meta name="viewport" content="width=device-width, initial-scale=1" />
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css"/>
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script><script src="https://cdn.tailwindcss.com"></script>
  <style>
    body { font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, "Helvetica Neue", Arial, "Noto Sans", "Apple Color Emoji", "Segoe UI Emoji"; }
    .custom-scrollbar::-webkit-scrollbar { width: 6px; } .custom-scrollbar::-webkit-scrollbar-thumb { background: #c5c5c5; border-radius: 3px; }
    .leaflet-popup-content-wrapper { border-radius: 10px; }
  </style>
</head>
<body class="relative overflow-hidden">
  <div id="map" class="w-full h-screen z-0"></div>
  <div id="map-overlay" class="hidden md:hidden fixed inset-0 bg-black bg-opacity-50 z-20 transition-opacity duration-300"></div>
  <button id="menu-toggle" class="md:hidden absolute top-4 left-4 z-10 bg-white p-2.5 rounded-md shadow-lg"><svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M4 6h16M4 12h16m-7 6h7" /></svg></button>
  <div id="sidebar" class="absolute top-0 left-0 h-screen w-full max-w-sm md:w-[380px] bg-white shadow-lg z-30 p-4 flex flex-col transform -translate-x-full md:translate-x-0 transition-transform duration-300 ease-in-out">
    <div class="flex items-center justify-between border-b pb-2 mb-2"><h1 class="text-2xl font-bold">DART Routes</h1><button id="close-sidebar" class="md:hidden p-1"><svg xmlns="http://www.w3.org/2000/svg" class="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12" /></svg></button></div>
    <div id="directions-container" class="border-b pb-2"></div><div id="route-list" class="flex-grow overflow-y-auto mt-3 custom-scrollbar"></div>
  </div>
  <div id="stop-panel" class="absolute top-0 right-0 h-screen w-full max-w-sm md:w-[380px] bg-white shadow-lg z-30 p-4 flex flex-col transform translate-x-full transition-transform duration-300 ease-in-out">
    <div class="flex items-center justify-between border-b pb-2"><h2 id="stop-panel-title" class="text-xl font-semibold">Upcoming Stops</h2><button id="close-stop-panel" class="text-sm px-3 py-1.5 rounded-md bg-gray-200 hover:bg-gray-300">Close</button></div>
    <div id="stop-list" class="flex-grow overflow-y-auto mt-3 custom-scrollbar"></div>
  </div>
  <script>
    const map = L.map('map', { zoomControl: false }).setView([39.15, -75.52], 10);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.tileLayer('https://tiles.stadiamaps.com/tiles/alidade_smooth/{z}/{x}/{y}{r}.png', { maxZoom: 19, attribution: '&copy; Stadia Maps &copy; OpenMapTiles &copy; OpenStreetMap' }).addTo(map);
    let vehiclesLayer = L.layerGroup().addTo(map), routeShapeLayer = L.layerGroup().addTo(map), stopsLayer = L.layerGroup().addTo(map);
    let selectedRouteId, selectedDirectionId, vehicleUpdateInterval;
    const sidebar = document.getElementById('sidebar'), stopPanel = document.getElementById('stop-panel'), mapOverlay = document.getElementById('map-overlay');
    const menuToggleBtn = document.getElementById('menu-toggle'), closeSidebarBtn = document.getElementById('close-sidebar'), closeStopPanelBtn = document.getElementById('close-stop-panel');
    const stopListEl = document.getElementById('stop-list'), stopPanelTitle = document.getElementById('stop-panel-title');
    function isMobile() { return window.innerWidth < 768; }
    function showSidebar() { sidebar.classList.remove('-translate-x-full'); if(isMobile()) mapOverlay.classList.remove('hidden'); }
    function hideSidebar() { sidebar.classList.add('-translate-x-full'); mapOverlay.classList.add('hidden'); }
    function showStopPanel() { stopPanel.classList.remove('translate-x-full'); if(isMobile()) mapOverlay.classList.remove('hidden'); }
    function hideStopPanel() { stopPanel.classList.add('translate-x-full'); mapOverlay.classList.add('hidden'); }
    menuToggleBtn.onclick = showSidebar; closeSidebarBtn.onclick = hideSidebar; closeStopPanelBtn.onclick = hideStopPanel; mapOverlay.onclick = () => { hideSidebar(); hideStopPanel(); };
    
    fetch('/api/routes').then(r => r.json()).then(routes => {
      routes.sort((a,b)=> (parseInt(a.route_short_name,10)||a.route_short_name).toString().localeCompare((parseInt(b.route_short_name,10)||b.route_short_name).toString(), undefined, {numeric: true}));
      const routeList = document.getElementById('route-list');
      routes.forEach(route => {
        const item = document.createElement('div');
        item.className = 'p-3 my-1 cursor-pointer hover:bg-gray-200 rounded-md';
        item.innerHTML = `<span class="font-bold text-lg text-blue-800 w-12 inline-block">${route.route_short_name||''}</span> <span>${route.route_long_name||''}</span>`;
        item.onclick = () => selectRoute(route.route_id, item);
        routeList.appendChild(item);
      });
    });

    function selectRoute(routeId, element) {
      selectedRouteId = routeId; selectedDirectionId = null; hideStopPanel();
      document.querySelectorAll('#route-list div').forEach(el => el.classList.remove('bg-blue-100','font-semibold'));
      element.classList.add('bg-blue-100','font-semibold');
      routeShapeLayer.clearLayers(); stopsLayer.clearLayers(); vehiclesLayer.clearLayers();
      const dirC = document.getElementById('directions-container');
      dirC.innerHTML = '<p class="text-gray-500">Loading…</p>';
      fetch(`/api/route_directions/${routeId}`).then(r=>r.json()).then(directions=>{
        dirC.innerHTML = '';
        if(!directions.length) { dirC.innerHTML = '<p class="text-red-500">No directions found.</p>'; return; }
        directions.forEach(dir=>{
          const b=document.createElement('button');
          b.className='w-full text-left p-2 my-1 rounded-lg bg-gray-100 hover:bg-gray-200 font-medium';
          b.textContent = dir.headsign;
          b.onclick = () => selectDirection(dir.direction_id, b);
          dirC.appendChild(b);
        });
        if (dirC.firstChild) dirC.firstChild.click();
      });
    }

    function selectDirection(directionId, btn) {
        selectedDirectionId = directionId;
        
        document.querySelectorAll('#directions-container button').forEach(el => {
            el.classList.remove('bg-blue-600', 'text-white', 'font-bold');
            el.classList.add('bg-gray-100', 'hover:bg-gray-200', 'font-medium');
        });

        btn.classList.add('bg-blue-600', 'text-white', 'font-bold');
        btn.classList.remove('bg-gray-100', 'hover:bg-gray-200', 'font-medium');

        if (isMobile()) hideSidebar();
        hideStopPanel(); routeShapeLayer.clearLayers(); stopsLayer.clearLayers();
        fetch(`/api/direction_details?route=${selectedRouteId}&direction=${selectedDirectionId}`).then(r => r.json()).then(data => {
            if(data.shape){ const g = L.geoJSON(data.shape, { style:{ color:'#005EB8', weight:5, opacity:0.8 } }).addTo(routeShapeLayer); map.fitBounds(g.getBounds().pad(0.1)); }
        });
        loadVehicleData();
        if (vehicleUpdateInterval) clearInterval(vehicleUpdateInterval);
        vehicleUpdateInterval = setInterval(loadVehicleData, 15000);
    }

    function loadVehicleData() {
      if(!selectedRouteId || selectedDirectionId===null) return;
      fetch(`/api/vehicles?route=${selectedRouteId}&direction=${selectedDirectionId}`).then(r=>r.json()).then(fc=>{
        vehiclesLayer.clearLayers();
        L.geoJSON(fc, { pointToLayer: (feat, latlng) => {
          const bearing = feat.properties.bearing || 0, dir = selectedDirectionId, path = "M12 2 L22 19 L12 14 L2 19 Z";
          const iconHtml = `<svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" style="transform: rotate(${bearing}deg);"><path d="${path}" fill="${dir===0?'#000':'#FFF'}" stroke="${dir===0?'#FFF':'#000'}" stroke-width="1.5"/></svg>`;
          const marker = L.marker(latlng, { icon: L.divIcon({ html: iconHtml, className: '', iconSize: [28, 28], iconAnchor: [14, 14] }) });
          marker.on('click', () => showPredictionsForVehicle(feat.properties));
          return marker;
        }}).addTo(vehiclesLayer);
      }).catch(e=>console.error('Vehicle fetch error:', e));
    }
    
    function showPredictionsForVehicle(props) {
        showStopPanel();
        stopPanelTitle.textContent = `Stops for Vehicle ${props.vehicle_id}`;
        stopListEl.innerHTML = '<p class="py-4">Loading predictions...</p>';
        stopsLayer.clearLayers();
        fetch(`/api/vehicle_predictions?trip_id=${props.trip_id}`).then(r => r.json()).then(preds => {
            stopListEl.innerHTML = '';
            if (!preds.length) { stopListEl.innerHTML = '<p class="py-4">No upcoming stops found.</p>'; return; }
            preds.forEach(item => {
                const row = document.createElement('div');
                row.className = 'flex items-center justify-between p-2.5 border-b cursor-pointer hover:bg-gray-100';
                let sColor = 'text-blue-700', sText = 'On Time';
                if (item.delay_minutes !== null) {
                    if (item.delay_minutes > 1) { sColor = 'text-red-600'; sText = `${item.delay_minutes} min late`; }
                    else if (item.delay_minutes < -1) { sColor = 'text-green-600'; sText = `${-item.delay_minutes} min early`; }
                } else { sText = "N/A"; sColor = "text-gray-500"; }
                row.innerHTML = `<div class="flex-grow pr-2"><div class="font-medium">${item.stop_name}</div><div class="text-sm font-semibold ${sColor}">${sText}</div></div><div class="text-right w-28"><div class="font-bold text-lg ${sColor}">${item.eta_clock}</div><div class="text-xs text-gray-500">Schd. ${item.scheduled_clock}</div><div class="text-sm">${item.eta_text}</div></div>`;
                row.onclick = ()=> { if(item.lat && item.lon) map.setView([item.lat, item.lon], 16); };
                stopListEl.appendChild(row);
                if(item.lat && item.lon) {
                    L.circleMarker([item.lat, item.lon], {radius:5, color:'#333', weight:1.5, fillColor:'#fff', fillOpacity:1})
                     .bindPopup(`<b>${item.stop_name}</b><br>Predicted: <b class="${sColor}">${item.eta_clock}</b><br>Scheduled: ${item.scheduled_clock}<br>Status: <b class="${sColor}">${sText}</b>`).addTo(stopsLayer);
                }
            });
        }).catch(e => { console.error('Prediction error:', e); stopListEl.innerHTML = '<p class="text-red-500 py-4">Could not load.</p>'; });
    }
  </script>
</body>
</html>
"""
app = Flask(__name__)

# -------------------------------------------------------------------
# API Routes
# -------------------------------------------------------------------
@app.route("/")
def index(): return Response(HTML_TEMPLATE, mimetype="text/html")

@app.route("/api/routes")
def api_routes(): return jsonify(gtfs_data.get("routes", []))

@app.route("/api/route_directions/<route_id>")
def api_route_directions(route_id):
    route_info = gtfs_data.get("route_info", {}).get(route_id)
    if not route_info: return jsonify([])
    return jsonify([{"direction_id": int(did), "headsign": d["headsign"]} for did, d in route_info.items()])

@app.route("/api/direction_details")
def api_direction_details():
    route_id = request.args.get("route")
    try: direction_id = int(request.args.get("direction"))
    except (ValueError, TypeError): return jsonify({"error": "Invalid direction"}), 400
    details = gtfs_data.get("route_info", {}).get(route_id, {}).get(direction_id)
    if not details: return jsonify({"error": "Not found"}), 404
    shape_coords = gtfs_data["shapes"][details["shape_id"]]["coords"]
    return jsonify({"shape": geojson.LineString(shape_coords)})

@app.route("/api/vehicles")
def api_vehicles():
    route_id, direction_id_str = request.args.get("route"), request.args.get("direction")
    try: direction_id = int(direction_id_str)
    except (ValueError, TypeError): return jsonify({"error": "Invalid direction"}), 400
    details = gtfs_data.get("route_info", {}).get(route_id, {}).get(direction_id)
    if not details: return jsonify(geojson.FeatureCollection([]))
    valid_trip_ids = details['trip_ids']
    try:
        feed = gtfs_realtime_pb2.FeedMessage()
        r = requests.get(GTFS_RT_URL, timeout=15)
        r.raise_for_status()
        feed.ParseFromString(r.content)
        features = []
        for entity in feed.entity:
            v = entity.vehicle
            if entity.HasField("vehicle") and v.HasField("position") and v.trip.trip_id in valid_trip_ids:
                props = {"trip_id": v.trip.trip_id, "vehicle_id": getattr(v.vehicle, "label", v.vehicle.id), "bearing": v.position.bearing}
                features.append(geojson.Feature(geometry=geojson.Point((v.position.longitude, v.position.latitude)), properties=props))
        return jsonify(geojson.FeatureCollection(features))
    except Exception:
        logging.exception("vehicles endpoint failed")
        return jsonify({"error": "internal error"}), 500

@app.route("/api/vehicle_predictions")
def api_vehicle_predictions():
    trip_id = request.args.get("trip_id")
    if not trip_id: return jsonify([])

    vehicle_pos, vehicle_speed = None, DEFAULT_SPEED_MS
    try:
        feed = gtfs_realtime_pb2.FeedMessage()
        r = requests.get(GTFS_RT_URL, timeout=15)
        r.raise_for_status()
        feed.ParseFromString(r.content)
        for entity in feed.entity:
            if entity.HasField("vehicle") and entity.vehicle.trip.trip_id == trip_id:
                vehicle_pos = (entity.vehicle.position.longitude, entity.vehicle.position.latitude)
                if entity.vehicle.position.HasField("speed"): vehicle_speed = entity.vehicle.position.speed
                break
    except Exception:
        logging.exception("Failed to fetch vehicle for prediction")
        return jsonify([])

    if not vehicle_pos: return jsonify([])

    try:
        trip_details = gtfs_data['trips_df'].loc[trip_id]
        shape_id = trip_details['shape_id']
        shape_info = gtfs_data["shapes"][shape_id]
        trip_stop_times = gtfs_data['stop_times_df'][gtfs_data['stop_times_df'].trip_id == trip_id].sort_values("stop_sequence")
        scheduled_times_map = trip_stop_times.set_index('stop_id')['arrival_time'].to_dict()
        ordered_stops = trip_stop_times["stop_id"].tolist()
    except (KeyError, IndexError):
        logging.error(f"Could not find static details for trip_id {trip_id}")
        return jsonify([])

    shape_coords, shape_cumdist = shape_info["coords"], shape_info["cumdist"]
    vehicle_idx = nearest_shape_index(shape_coords, vehicle_pos)
    vehicle_dist_along_shape = shape_cumdist[vehicle_idx]
    
    results = []
    # --- TIMEZONE FIX STARTS HERE ---
    ET = pytz.timezone('America/New_York')
    now_et = datetime.now(ET)
    today_midnight_et = now_et.replace(hour=0, minute=0, second=0, microsecond=0)
    # --- TIMEZONE FIX ENDS HERE ---
    
    for stop_id in ordered_stops:
        stop_details = gtfs_data["stops_dict"].get(stop_id)
        if not stop_details: continue

        stop_idx = nearest_shape_index(shape_coords, (stop_details["stop_lon"], stop_details["stop_lat"]))
        stop_dist_along_shape = shape_cumdist[stop_idx]

        if stop_dist_along_shape > vehicle_dist_along_shape:
            remaining_dist = stop_dist_along_shape - vehicle_dist_along_shape
            eta_seconds = remaining_dist / (vehicle_speed if vehicle_speed > 1 else DEFAULT_SPEED_MS)
            
            eta_time = now_et + timedelta(seconds=eta_seconds)
            eta_clock = eta_time.strftime("%-I:%M %p")
            minutes = max(1, int(round(eta_seconds / 60.0)))
            eta_text = f"{minutes} min"

            scheduled_clock, delay_minutes = "—", None
            scheduled_time_str = scheduled_times_map.get(stop_id)
            if scheduled_time_str:
                try:
                    h, m, s = map(int, scheduled_time_str.split(':'))
                    scheduled_datetime = today_midnight_et + timedelta(hours=h, minutes=m, seconds=s)
                    delay_seconds = (eta_time - scheduled_datetime).total_seconds()
                    delay_minutes = int(round(delay_seconds / 60.0))
                    scheduled_clock = scheduled_datetime.strftime("%-I:%M %p")
                except (ValueError, TypeError):
                    logging.warning(f"Could not parse scheduled time: {scheduled_time_str}")

            results.append({
                "stop_id": stop_id, "stop_name": stop_details.get("stop_name", "Unknown"),
                "eta_text": eta_text, "eta_clock": eta_clock,
                "scheduled_clock": scheduled_clock, "delay_minutes": delay_minutes,
                "lat": stop_details.get("stop_lat"), "lon": stop_details.get("stop_lon"),
            })
    return jsonify(results)

# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------
# This will run once when the server starts, loading all the GTFS data.
process_gtfs_data()

if __name__ == "__main__":
    # The app.run() is only for local development.
    app.run(debug=True, port=5000)

