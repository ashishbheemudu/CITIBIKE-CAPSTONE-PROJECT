import React, { useEffect, useState, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, ArcLayer } from '@deck.gl/layers';
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

    // Controls
    const [activeLayer, setActiveLayer] = useState('heatmap'); // 'heatmap', 'points', 'hexagon'
    const [minTrips, setMinTrips] = useState(0);
    const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

    // Pulsing animation effect
    useEffect(() => {
        if (!selectedStation) return;

        const interval = setInterval(() => {
            setPulseRadius(prev => prev >= 300 ? 100 : prev + 20);
        }, 50);

        return () => clearInterval(interval);
    }, [selectedStation]);

    useEffect(() => {
        const loadData = async () => {
            try {
                const mapData = await fetchMapData();
                setData(mapData);
            } catch (error) {
                console.error("Failed to fetch map data:", error);
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, []);

    // Filter Data
    const filteredData = useMemo(() => {
        if (!data) return [];
        return data.filter(d => d.trip_count >= minTrips);
    }, [data, minTrips]);

    // Top Stations for Sidebar
    const topStations = useMemo(() => {
        if (!data) return [];
        return [...data].sort((a, b) => b.trip_count - a.trip_count).slice(0, 10);
    }, [data]);

    // Highlight layer for selected station
    const highlightLayer = selectedStation ? new ScatterplotLayer({
        id: 'highlight-layer',
        data: [selectedStation],
        pickable: false,
        opacity: 0.6,
        stroked: true,
        filled: true,
        radiusScale: 1,
        radiusMinPixels: pulseRadius,
        radiusMaxPixels: 500,
        lineWidthMinPixels: 3,
        getPosition: d => [d.lon, d.lat],
        getRadius: 100,
        getFillColor: [59, 130, 246, 100],
        getLineColor: [59, 130, 246, 255]
    }) : null;

    // Center marker for selected station
    const centerMarkerLayer = selectedStation ? new ScatterplotLayer({
        id: 'center-marker-layer',
        data: [selectedStation],
        pickable: false,
        opacity: 1,
        stroked: true,
        filled: true,
        radiusScale: 1,
        radiusMinPixels: 12,
        radiusMaxPixels: 30,
        lineWidthMinPixels: 3,
        getPosition: d => [d.lon, d.lat],
        getRadius: 20,
        getFillColor: [255, 255, 255],
        getLineColor: [59, 130, 246]
    }) : null;

    const layers = [
        activeLayer === 'heatmap' && new HeatmapLayer({
            id: 'heatmap-layer',
            data: filteredData,
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
            getRadius: d => Math.sqrt(d.trip_count),
            getFillColor: [59, 130, 246],
            getLineColor: [0, 0, 0],
            onHover: info => setHoverInfo(info)
        }),
        // Add highlight layers on top
        highlightLayer,
        centerMarkerLayer
    ].filter(Boolean);

    const flyToStation = (station) => {
        setSelectedStation(station);
        setPulseRadius(100); // Reset pulse
        setViewState({
            ...viewState,
            longitude: station.lon,
            latitude: station.lat,
            zoom: 16,
            pitch: 45,
            bearing: 0,
            transitionDuration: 1500
        });

        // Clear selection after 5 seconds
        setTimeout(() => setSelectedStation(null), 5000);
    };

    // Dynamic Max for Slider
    const maxTrips = useMemo(() => {
        if (!data || data.length === 0) return 50000;
        return data.reduce((max, d) => (d.trip_count > max ? d.trip_count : max), 0);
    }, [data]);

    if (loading) {
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
                    <p className="text-xs text-muted-foreground">Geospatial Analysis</p>
                </div>

                {/* Layer Control */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-foreground flex items-center gap-2">
                        <Layers className="w-4 h-4" /> Visualization
                    </label>
                    <div className="grid grid-cols-3 gap-2">
                        <button
                            onClick={() => setActiveLayer('heatmap')}
                            className={`flex flex-col items-center justify-center p-2 rounded border transition-all ${activeLayer === 'heatmap' ? 'bg-primary/20 border-primary text-primary' : 'bg-secondary/30 border-transparent text-muted-foreground hover:bg-secondary/50'}`}
                        >
                            <Activity className="w-5 h-5 mb-1" />
                            <span className="text-[10px]">Heatmap</span>
                        </button>
                        <button
                            onClick={() => setActiveLayer('hexagon')}
                            className={`flex flex-col items-center justify-center p-2 rounded border transition-all ${activeLayer === 'hexagon' ? 'bg-primary/20 border-primary text-primary' : 'bg-secondary/30 border-transparent text-muted-foreground hover:bg-secondary/50'}`}
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

                {/* Filters */}
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

                {/* Top Stations */}
                <div className="flex-1 overflow-hidden flex flex-col">
                    <label className="text-sm font-medium text-foreground flex items-center gap-2 mb-3">
                        <Navigation className="w-4 h-4" /> Top Locations
                    </label>
                    <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                        {topStations.map((station, idx) => (
                            <button
                                key={idx}
                                onClick={() => flyToStation(station)}
                                className="w-full text-left p-2 rounded bg-secondary/20 hover:bg-secondary/40 border border-transparent hover:border-border transition-all group"
                            >
                                <div className="flex justify-between items-start">
                                    <span className="text-xs font-medium text-foreground line-clamp-1 group-hover:text-primary transition-colors">
                                        {station.station_name}
                                    </span>
                                    <span className="text-[10px] font-mono text-muted-foreground bg-secondary/50 px-1 rounded">
                                        #{idx + 1}
                                    </span>
                                </div>
                                <div className="text-[10px] text-muted-foreground mt-1">
                                    {station.trip_count.toLocaleString()} trips
                                </div>
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
                    getTooltip={({ object }) => {
                        if (!object) return null;
                        if (activeLayer === 'hexagon') {
                            const points = object.points || [];
                            const stationNames = [...new Set(points.map(p => p.station_name))];
                            const stationLabel = stationNames.length === 1
                                ? `Station: ${stationNames[0]}`
                                : `Stations: ${stationNames.slice(0, 3).join(', ')}${stationNames.length > 3 ? '...' : ''}`;
                            return `${stationLabel}\nTotal Trips: ${object.elevationValue.toLocaleString()}`;
                        }
                        return `${object.station_name}\nTrips: ${object.trip_count.toLocaleString()}`;
                    }}
                >
                    <Map
                        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    />
                </DeckGL>

                {/* Selected Station Popup */}
                {selectedStation && (
                    <div className="absolute top-6 left-1/2 transform -translate-x-1/2 z-50">
                        <div className="bg-gradient-to-r from-blue-600 to-indigo-600 px-6 py-3 rounded-xl shadow-2xl border border-blue-400/30 animate-pulse">
                            <div className="flex items-center gap-3">
                                <div className="w-4 h-4 bg-white rounded-full animate-ping"></div>
                                <div>
                                    <div className="text-white font-bold text-sm">{selectedStation.station_name}</div>
                                    <div className="text-blue-200 text-xs">{selectedStation.trip_count.toLocaleString()} total trips</div>
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
