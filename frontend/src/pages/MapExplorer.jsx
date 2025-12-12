import React, { useEffect, useState, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, ArcLayer, ColumnLayer } from '@deck.gl/layers';
import { HeatmapLayer, HexagonLayer } from '@deck.gl/aggregation-layers';
import { Map } from 'react-map-gl/maplibre';
import { fetchMapData } from '../api';
import { Layers, Map as MapIcon, Activity, Navigation, Filter, Hexagon } from 'lucide-react';

// Initial View State for NYC (3D Tilted)
const INITIAL_VIEW_STATE = {
    longitude: -74.0060,
    latitude: 40.7128,
    zoom: 12,
    pitch: 50,
    bearing: 20
};

function MapExplorer() {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);
    const [_hoverInfo, setHoverInfo] = useState(null);
    const [selectedStation, setSelectedStation] = useState(null);
    const [pulseRadius, setPulseRadius] = useState(100);

    // Controls - Default to 'points' for individual station visibility
    const [activeLayer, setActiveLayer] = useState('points');
    const [tooltipInfo, setTooltipInfo] = useState(null);
    const [topStations, setTopStations] = useState([]);
    const [minTrips, setMinTrips] = useState(0);
    const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

    // Live Data State
    const [liveMode, setLiveMode] = useState(false);
    const [liveData, setLiveData] = useState([]);

    // API Base URL from environment
    const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://18.218.154.66.nip.io';

    // Base Activity Data Fetch
    useEffect(() => {
        const loadData = async () => {
            try {
                const result = await fetchMapData();
                setData(result.locations);

                // Process top stations
                const sorted = [...result.locations].sort((a, b) => b.trip_count - a.trip_count);
                setTopStations(sorted.slice(0, 50));
            } catch (err) {
                console.error("Failed to load map data", err);
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, []);

    // Live Data Fetch Effect
    useEffect(() => {
        let intervalId;

        const fetchLive = async () => {
            try {
                const res = await fetch(`${API_BASE_URL}/live/stations`);
                if (res.ok) {
                    const json = await res.json();
                    setLiveData(json);
                }
            } catch (error) {
                console.error("Live fetch error:", error);
            }
        };

        if (liveMode) {
            fetchLive(); // Initial fetch
            intervalId = setInterval(fetchLive, 60000); // Poll every 60s
        }

        return () => clearInterval(intervalId);
    }, [liveMode]);

    // Pulsing animation effect
    useEffect(() => {
        if (!selectedStation) return;

        const interval = setInterval(() => {
            setPulseRadius(prev => prev >= 80 ? 30 : prev + 5);
        }, 50);

        return () => clearInterval(interval);
    }, [selectedStation]);

    // Filter Data
    const filteredData = useMemo(() => {
        if (!data) return [];
        return data.filter(d => d.trip_count >= minTrips);
    }, [data, minTrips]);

    // Highlight layer for selected station - SMALL pulsing ring
    const highlightLayer = selectedStation ? new ScatterplotLayer({
        id: 'highlight-layer',
        data: [selectedStation],
        pickable: false,
        opacity: 0.8,
        stroked: true,
        filled: false,
        radiusScale: 1,
        radiusMinPixels: pulseRadius,
        radiusMaxPixels: 100,
        lineWidthMinPixels: 4,
        getPosition: d => [d.lon, d.lat],
        getRadius: 50,
        getFillColor: [59, 130, 246, 0],
        getLineColor: [59, 130, 246, 255]
    }) : null;

    // Bright center marker for selected station
    const centerMarkerLayer = selectedStation ? new ScatterplotLayer({
        id: 'center-marker-layer',
        data: [selectedStation],
        pickable: false,
        opacity: 1,
        stroked: true,
        filled: true,
        radiusScale: 1,
        radiusMinPixels: 10,
        radiusMaxPixels: 20,
        lineWidthMinPixels: 3,
        getPosition: d => [d.lon, d.lat],
        getRadius: 15,
        getFillColor: [255, 255, 255, 255],
        getLineColor: [59, 130, 246]
    }) : null;

    // Bright vertical pillar to mark selected station location
    // This shoots up above the hexagon bars to show which one is selected
    const stationMarkerPillar = selectedStation ? new ColumnLayer({
        id: 'station-marker-pillar',
        data: [selectedStation],
        diskResolution: 6, // Hexagonal shape
        radius: 30, // Thin pillar
        extruded: true,
        elevationScale: 1,
        getPosition: d => [d.lon, d.lat],
        getElevation: 5000, // Very tall - shoots way above all hexagons
        getFillColor: [0, 255, 255, 255], // Bright cyan
        getLineColor: [255, 255, 255, 255], // White outline
        wireframe: false,
        material: {
            ambient: 1.0,
            diffuse: 1.0,
            shininess: 300,
            specularColor: [255, 255, 255]
        }
    }) : null;

    const layers = [
        activeLayer === 'heatmap' && new HeatmapLayer({
            id: 'heatmap-layer',
            data: filteredData,
            pickable: true, // Enable hover detection
            getPosition: d => [d.lon, d.lat],
            getWeight: d => d.trip_count,
            radiusPixels: 40,
            intensity: 2,
            threshold: 0.05,
            colorRange: [
                [255, 255, 178],
                [254, 204, 92],
                [253, 141, 60],
                [240, 59, 32],
                [189, 0, 38]
            ]
        }),
        activeLayer === 'hexagon' && new HexagonLayer({
            id: 'hexagon-layer',
            data: filteredData,
            pickable: true, // Enable hover detection
            getPosition: d => [d.lon, d.lat],
            getElevationWeight: d => d.trip_count,
            elevationScale: 5,
            extruded: true,
            radius: 200,
            opacity: 0.8,
            coverage: 0.9,
            upperPercentile: 99,
            material: {
                ambient: 0.64,
                diffuse: 0.6,
                shininess: 32,
                specularColor: [51, 51, 51]
            },
            transitions: {
                elevationScale: 3000
            }
        }),
        activeLayer === 'points' && new ScatterplotLayer({
            id: 'scatterplot-layer',
            data: filteredData,
            pickable: true,
            opacity: 0.8,
            stroked: true,
            filled: true,
            radiusScale: 6,
            radiusMinPixels: 2,
            radiusMaxPixels: 30,
            lineWidthMinPixels: 1,
            getPosition: d => [d.lon, d.lat],
            getRadius: d => liveMode ? Math.max(5, d.bikes_available * 2) : Math.sqrt(d.trip_count),
            getFillColor: d => liveMode ? (d.bikes_available < 3 ? [252, 165, 165] : [59, 130, 246]) : [59, 130, 246],
            getLineColor: [0, 0, 0],
            onHover: info => setHoverInfo(info)
        }),
        // Add highlight layers on top
        activeLayer === 'hexagon' && stationMarkerPillar, // Only show pillar in Hexagon mode
        highlightLayer,
        centerMarkerLayer
    ].filter(Boolean);

    const flyToStation = (station) => {
        setSelectedStation(station);
        setPulseRadius(100); // Reset pulse
        setViewState({
            ...viewState,
            longitude: Number(station.lon), // Ensure number
            latitude: Number(station.lat),
            zoom: 14, // Zoom 14 is good for individual station
            pitch: 50,
            bearing: 20,
            transitionDuration: 1500
        });

        // Clear selection after 8 seconds
        setTimeout(() => setSelectedStation(null), 8000);
    };

    // Dynamic Max for Slider
    const maxTrips = useMemo(() => {
        if (!data || data.length === 0) return 50000;
        return data.reduce((max, d) => (d.trip_count > max ? d.trip_count : max), 0);
    }, [data]);

    // Live Mode Toggle Handler
    const toggleLiveMode = () => {
        setLiveMode(!liveMode);
        // Reset to points view when entering live mode for clarity
        if (!liveMode) setActiveLayer('points');
    };

    if (loading && !data.length && !liveMode) { // Only show loading if historical data is not loaded and not in live mode
        return (
            <div className="flex justify-center items-center h-96">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
            </div>
        );
    }

    return (
        <div className="flex h-[calc(100vh-120px)] bg-background overflow-hidden">
            {/* --- Sidebar --- */}
            <div className="w-72 border-r border-gray-800 bg-[#0a0b10] p-4 flex flex-col gap-6 overflow-y-auto z-10">
                <div>
                    <h2 className="text-lg font-semibold text-foreground mb-1">Map Explorer</h2>
                    <div className="flex items-center justify-between">
                        <p className="text-xs text-muted-foreground">Geospatial Analysis</p>
                        <button
                            onClick={toggleLiveMode}
                            className={`text-[10px] px-2 py-0.5 rounded border border-primary/50 transition-all ${liveMode ? 'bg-green-500/20 text-green-400 animate-pulse' : 'bg-transparent text-muted-foreground hover:text-foreground'}`}
                        >
                            {liveMode ? '● LIVE' : '○ HISTORICAL'}
                        </button>
                    </div>
                </div>

                {/* Layer Control */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-foreground flex items-center gap-2">
                        <Layers className="w-4 h-4" /> Visualization
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                        <button
                            onClick={() => setActiveLayer('heatmap')}
                            disabled={liveMode}
                            className={`flex flex-col items-center justify-center p-2 rounded border transition-all ${activeLayer === 'heatmap' && !liveMode ? 'bg-primary/20 border-primary text-primary' : 'bg-secondary/30 border-transparent text-muted-foreground hover:bg-secondary/50'} ${liveMode ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            <Activity className="w-5 h-5 mb-1" />
                            <span className="text-[10px]">Heatmap</span>
                        </button>
                        <button
                            onClick={() => setActiveLayer('hexagon')}
                            disabled={liveMode}
                            className={`flex flex-col items-center justify-center p-2 rounded border transition-all ${activeLayer === 'hexagon' && !liveMode ? 'bg-primary/20 border-primary text-primary' : 'bg-secondary/30 border-transparent text-muted-foreground hover:bg-secondary/50'} ${liveMode ? 'opacity-50 cursor-not-allowed' : ''}`}
                        >
                            <Hexagon className="w-5 h-5 mb-1" />
                            <span className="text-[10px]">3D Hex</span>
                        </button>
                        <button
                            onClick={() => setActiveLayer('points')}
                            className={`flex flex-col items-center justify-center p-2 rounded border transition-all ${activeLayer === 'points' ? 'bg-primary/20 border-primary text-primary' : 'bg-secondary/30 border-transparent text-muted-foreground hover:bg-secondary/50'}`}
                        >
                            <MapIcon className="w-5 h-5 mb-1" />
                            <span className="text-[10px]">Points</span>
                        </button>
                    </div>
                </div>

                {/* Filters (Hidden in Live Mode for now to avoid confusion or adapted) */}
                {!liveMode && (
                    <div className="space-y-3">
                        <label className="text-sm font-medium text-foreground flex items-center gap-2">
                            <Filter className="w-4 h-4" /> Filter Stations
                        </label>
                        <div className="px-1">
                            <div className="flex justify-between text-xs text-muted-foreground mb-2">
                                <span>Min Trips</span>
                                <span className="font-mono text-foreground">{minTrips}</span>
                            </div>
                            <input
                                type="range"
                                min="0"
                                max={maxTrips}
                                step={Math.ceil(maxTrips / 100)}
                                value={minTrips}
                                onChange={(e) => setMinTrips(Number(e.target.value))}
                                className="w-full h-1 bg-secondary rounded-lg appearance-none cursor-pointer accent-primary"
                            />
                        </div>
                    </div>
                )}

                {liveMode && (
                    <div className="space-y-3 p-3 bg-green-900/10 border border-green-500/20 rounded font-mono text-xs">
                        <div className="text-green-400 font-bold mb-1">REAL-TIME FEED</div>
                        <div className="flex justify-between">
                            <span>Stations Online:</span>
                            <span>{liveData.length}</span>
                        </div>
                        <div className="flex justify-between text-muted-foreground">
                            <span>Updates:</span>
                            <span>Every 60s</span>
                        </div>
                    </div>
                )}

                {/* Top Stations / Interactions */}
                <div className="flex-1 overflow-hidden flex flex-col">
                    <label className="text-sm font-medium text-foreground flex items-center gap-2 mb-3">
                        <Navigation className="w-4 h-4" /> {liveMode ? 'Station Status' : 'Top Locations'}
                    </label>
                    <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                        {(liveMode ? liveData.slice(0, 50) : topStations).map((station, idx) => (
                            <button
                                key={liveMode ? station.id : idx}
                                onClick={() => flyToStation(station)}
                                className="w-full text-left p-2 rounded bg-secondary/20 hover:bg-secondary/40 border border-transparent hover:border-border transition-all group"
                            >
                                <div className="flex justify-between items-start">
                                    <span className="text-xs font-medium text-foreground line-clamp-1 group-hover:text-primary transition-colors">
                                        {liveMode ? station.name : station.station_name}
                                    </span>
                                    {liveMode ? (
                                        <span className={`text-[10px] font-mono px-1 rounded ${station.bikes_available < 3 ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                                            {station.bikes_available} 🚲
                                        </span>
                                    ) : (
                                        <span className="text-[10px] font-mono text-muted-foreground bg-secondary/50 px-1 rounded">
                                            #{idx + 1}
                                        </span>
                                    )}
                                </div>
                                {!liveMode && (
                                    <div className="text-[10px] text-muted-foreground mt-1">
                                        {station.trip_count.toLocaleString()} trips
                                    </div>
                                )}
                                {liveMode && (
                                    <div className="flex gap-2 text-[9px] text-muted-foreground mt-1">
                                        <span>Docks: {station.docks_available}</span>
                                    </div>
                                )}
                            </button>
                        ))}
                    </div>
                </div>
            </div>

            {/* --- Map Area --- */}
            <div className="flex-1 relative">
                <DeckGL
                    initialViewState={viewState}
                    onViewStateChange={({ viewState }) => setViewState(viewState)}
                    controller={true}
                    layers={layers}
                    onClick={({ object }) => {
                        if (!object) return;
                        if (activeLayer === 'points') {
                            flyToStation(object);
                        }
                    }}
                    getTooltip={({ object, layer }) => {
                        if (!object) return null;

                        if (liveMode) {
                            if (object.name) {
                                return {
                                    html: `<div style="padding: 8px;">
                                        <div style="font-weight: bold; color: #60a5fa; margin-bottom: 4px;">📍 LIVE STATION</div>
                                        <div style="font-size: 13px; font-weight: bold; margin-bottom: 4px;">${object.name}</div>
                                        <div style="font-size: 14px; color: #22c55e;">🚲 ${object.bikes_available} bikes available</div>
                                        <div style="font-size: 12px; color: #9ca3af;">🅿️ ${object.docks_available} docks available</div>
                                    </div>`,
                                    style: {
                                        backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                        borderRadius: '8px',
                                        border: '1px solid #60a5fa',
                                        color: 'white'
                                    }
                                };
                            }
                            return null;
                        }

                        // For hexagon bars - find stations near this hexagon's position
                        if (layer && layer.id === 'hexagon-layer') {
                            const totalTrips = object.elevationValue || object.colorValue || 0;

                            // Try multiple ways to get station names
                            let stationNames = [];

                            // Method 1: Try object.points (newer deck.gl versions)
                            if (object.points && Array.isArray(object.points)) {
                                for (const p of object.points) {
                                    const name = p.source?.station_name || p.station_name;
                                    if (name) stationNames.push(name);
                                }
                            }

                            // Method 2: Check if object itself has position and search in data
                            if (stationNames.length === 0 && object.position) {
                                const [hexLon, hexLat] = object.position;
                                const radius = 0.003;
                                for (const station of (filteredData || [])) {
                                    if (Math.abs(station.lon - hexLon) < radius && Math.abs(station.lat - hexLat) < radius) {
                                        stationNames.push(station.station_name);
                                    }
                                }
                            }

                            // Method 3: Use colorValue.points if available
                            if (stationNames.length === 0 && object.colorValue) {
                                // Find closest station by trip count
                                const targetTrips = totalTrips;
                                let bestMatch = null;
                                let smallestDiff = Infinity;
                                for (const station of (filteredData || [])) {
                                    const diff = Math.abs(station.trip_count - targetTrips);
                                    if (diff < smallestDiff) {
                                        smallestDiff = diff;
                                        bestMatch = station;
                                    }
                                }
                                if (bestMatch && smallestDiff < targetTrips * 0.1) {
                                    stationNames.push(bestMatch.station_name);
                                }
                            }

                            stationNames = [...new Set(stationNames)];

                            // Build tooltip HTML
                            const stationHTML = stationNames.length > 0
                                ? `<div style="font-size: 13px; font-weight: bold; margin-bottom: 6px; color: #fff;">${stationNames.length === 1 ? stationNames[0] : `${stationNames.length} stations:`}</div>
                                   ${stationNames.length > 1 ? `<div style="font-size: 11px; color: #9ca3af; margin-bottom: 6px;">${stationNames.slice(0, 5).map(s => `• ${s}`).join('<br/>')}</div>` : ''}`
                                : `<div style="font-size: 11px; color: #9ca3af; margin-bottom: 4px;">Multiple stations aggregated</div>`;

                            return {
                                html: `<div style="padding: 10px; min-width: 220px;">
                                    <div style="font-weight: bold; color: #60a5fa; margin-bottom: 6px; font-size: 11px; letter-spacing: 0.5px;">📍 STATION INFO</div>
                                    ${stationHTML}
                                    <div style="font-size: 16px; font-weight: bold; color: #22c55e;">🚴 ${totalTrips.toLocaleString()} trips</div>
                                </div>`,
                                style: {
                                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                    borderRadius: '10px',
                                    border: '1px solid #3b82f6',
                                    color: 'white',
                                    boxShadow: '0 4px 20px rgba(59, 130, 246, 0.3)'
                                }
                            };
                        }

                        // For heatmap - show nearest station
                        if (layer && layer.id === 'heatmap-layer') {
                            return {
                                html: `<div style="padding: 8px;">
                                    <div style="font-weight: bold; color: #f97316;">🔥 HEAT ZONE</div>
                                    <div style="font-size: 11px; color: #9ca3af;">High activity area</div>
                                </div>`,
                                style: {
                                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                    borderRadius: '8px',
                                    border: '1px solid #f97316',
                                    color: 'white'
                                }
                            };
                        }

                        // For points layer
                        if (object.station_name) {
                            return {
                                html: `<div style="padding: 8px;">
                                    <div style="font-weight: bold; color: #60a5fa; margin-bottom: 4px;">📍 STATION</div>
                                    <div style="font-size: 13px; font-weight: bold; margin-bottom: 4px;">${object.station_name}</div>
                                    <div style="font-size: 14px; color: #22c55e;">🚴 ${object.trip_count.toLocaleString()} trips</div>
                                </div>`,
                                style: {
                                    backgroundColor: 'rgba(15, 23, 42, 0.95)',
                                    borderRadius: '8px',
                                    border: '1px solid #60a5fa',
                                    color: 'white'
                                }
                            };
                        }

                        return null;
                    }}
                >
                    <Map
                        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    />
                </DeckGL>

                {/* Selected Station Popup - Premium Design */}
                {selectedStation && (
                    <div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-50">
                        <div className="relative">
                            {/* Glowing background effect */}
                            <div className="absolute inset-0 bg-gradient-to-r from-blue-600 to-purple-600 rounded-2xl blur-xl opacity-50 animate-pulse"></div>

                            {/* Main card */}
                            <div className="relative bg-gradient-to-br from-slate-900/95 to-slate-800/95 backdrop-blur-xl px-6 py-4 rounded-2xl shadow-2xl border border-white/20">
                                <div className="flex items-center gap-4">
                                    {/* Cycling bicycle animation */}
                                    <div className="relative w-14 h-14 flex items-center justify-center">
                                        {/* Rotating circle track */}
                                        <div className="absolute inset-0 rounded-full border-2 border-dashed border-blue-400/30 animate-spin" style={{ animationDuration: '3s' }}></div>
                                        {/* Bicycle icon bouncing */}
                                        <div className="animate-bounce" style={{ animationDuration: '0.8s' }}>
                                            <svg className="w-8 h-8 text-blue-400" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                                <circle cx="5.5" cy="17.5" r="3.5" />
                                                <circle cx="18.5" cy="17.5" r="3.5" />
                                                <path d="M15 6a1 1 0 1 0 0-2 1 1 0 0 0 0 2zm-3 11.5V14l-3-3 4-3 2 3h3" />
                                            </svg>
                                        </div>
                                    </div>

                                    {/* Station info */}
                                    <div className="flex-1">
                                        <div className="flex items-center gap-2 mb-1">
                                            <span className="px-2 py-0.5 bg-blue-500/20 border border-blue-400/30 rounded-full text-blue-300 text-[10px] font-bold uppercase tracking-wider">Station</span>
                                        </div>
                                        <div className="text-white font-bold text-lg leading-tight">{selectedStation.station_name}</div>
                                        <div className="flex items-center gap-2 mt-1">
                                            <div className="text-2xl font-black bg-gradient-to-r from-blue-400 to-purple-400 bg-clip-text text-transparent">
                                                {selectedStation.trip_count.toLocaleString()}
                                            </div>
                                            <span className="text-gray-400 text-sm">trips</span>
                                        </div>
                                    </div>

                                    {/* Rank badge */}
                                    <div className="flex flex-col items-center">
                                        <div className="w-10 h-10 rounded-full bg-gradient-to-br from-yellow-400 to-orange-500 flex items-center justify-center shadow-lg">
                                            <span className="text-white font-black text-sm">#{topStations.findIndex(s => s.station_name === selectedStation.station_name) + 1 || '?'}</span>
                                        </div>
                                        <span className="text-gray-500 text-[9px] mt-1">RANK</span>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                )}

                {/* Legend / Info Overlay */}
                <div className="absolute bottom-6 right-6 glass-panel p-4 rounded-xl max-w-xs">
                    <div className="flex items-center gap-2 mb-2">
                        <div className="w-2 h-2 rounded-full bg-blue-500 animate-pulse"></div>
                        <span className="text-xs font-bold text-foreground">
                            {activeLayer === 'hexagon' ? '3D DENSITY' : activeLayer === 'heatmap' ? 'HEATMAP INTENSITY' : 'STATION VOLUME'}
                        </span>
                    </div>
                    <div className="text-[10px] text-muted-foreground">
                        {activeLayer === 'hexagon' && "Height represents aggregated trip volume in 200m radius."}
                        {activeLayer === 'heatmap' && "Warmer colors indicate higher concentration of trip starts."}
                        {activeLayer === 'points' && "Circle size corresponds to total trip volume."}
                    </div>
                </div>
            </div>
        </div>
    );
}

export default MapExplorer;
