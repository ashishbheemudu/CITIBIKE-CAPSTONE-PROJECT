import React, { useEffect, useState, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ArcLayer, ScatterplotLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl/maplibre';
import { fetchRoutes } from '../api';
import { Navigation, Filter, Route as RouteIcon, ArrowRight } from 'lucide-react';

// Initial View State for NYC
const INITIAL_VIEW_STATE = {
    longitude: -74.0060,
    latitude: 40.7128,
    zoom: 12,
    pitch: 45,
    bearing: 0
};

function RouteAnalysis() {
    const [routes, setRoutes] = useState(null);
    const [loading, setLoading] = useState(true);
    const [_hoverInfo, setHoverInfo] = useState(null);
    const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);

    // Filters
    const [minTrips, setMinTrips] = useState(0);

    useEffect(() => {
        const loadData = async () => {
            try {
                const data = await fetchRoutes(200); // Fetch top 200 routes
                setRoutes(data);
            } catch (error) {
                console.error("Failed to fetch routes:", error);
            } finally {
                setLoading(false);
            }
        };
        loadData();
    }, []);

    // Filtered Routes
    const filteredRoutes = useMemo(() => {
        if (!routes) return [];
        return routes.filter(r => r.trip_count >= minTrips);
    }, [routes, minTrips]);

    // Top Routes for Sidebar
    const topRoutesList = useMemo(() => {
        return [...filteredRoutes].sort((a, b) => b.trip_count - a.trip_count).slice(0, 10);
    }, [filteredRoutes]);

    // Layers
    const layers = [
        new ArcLayer({
            id: 'arc-layer',
            data: filteredRoutes,
            getSourcePosition: d => [d.start_lon, d.start_lat],
            getTargetPosition: d => [d.end_lon, d.end_lat],
            getSourceColor: [59, 130, 246], // Blue start
            getTargetColor: [236, 72, 153], // Pink end
            getWidth: 3,
            pickable: true,
            onHover: info => setHoverInfo(info)
        }),
        new ScatterplotLayer({
            id: 'start-points',
            data: filteredRoutes,
            getPosition: d => [d.start_lon, d.start_lat],
            getRadius: 30,
            getFillColor: [59, 130, 246],
            radiusMinPixels: 2
        }),
        new ScatterplotLayer({
            id: 'end-points',
            data: filteredRoutes,
            getPosition: d => [d.end_lon, d.end_lat],
            getRadius: 30,
            getFillColor: [236, 72, 153],
            radiusMinPixels: 2
        })
    ];

    const flyToRoute = (route) => {
        const midLat = (route.start_lat + route.end_lat) / 2;
        const midLon = (route.start_lon + route.end_lon) / 2;

        setViewState({
            ...viewState,
            longitude: midLon,
            latitude: midLat,
            zoom: 13,
            pitch: 50,
            bearing: 30,
            transitionDuration: 1000
        });
    };

    // Dynamic Max for Slider
    const maxTrips = useMemo(() => {
        if (!routes || routes.length === 0) return 5000;
        return routes.reduce((max, r) => (r.trip_count > max ? r.trip_count : max), 0);
    }, [routes]);

    if (loading) {
        return (
            <div className="flex justify-center items-center h-96 text-indigo-500">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-indigo-500"></div>
            </div>
        );
    }

    if (!routes) {
        return (
            <div className="flex justify-center items-center h-96 text-red-500">
                Failed to load route data.
            </div>
        );
    }

    return (
        <div className="flex h-[calc(100vh-120px)] bg-background overflow-hidden">
            {/* --- Sidebar --- */}
            <div className="w-80 border-r border-gray-800 bg-[#0a0b10] p-4 flex flex-col gap-6 overflow-y-auto z-10">
                <div>
                    <h2 className="text-lg font-semibold text-foreground mb-1">Route Analysis</h2>
                    <p className="text-xs text-muted-foreground">Top Commuter Flows</p>
                </div>

                {/* Filters */}
                <div className="space-y-3">
                    <label className="text-sm font-medium text-foreground flex items-center gap-2">
                        <Filter className="w-4 h-4" /> Filter Volume
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

                {/* Top Routes List */}
                <div className="flex-1 overflow-hidden flex flex-col">
                    <label className="text-sm font-medium text-foreground flex items-center gap-2 mb-3">
                        <Navigation className="w-4 h-4" /> Top Routes
                    </label>
                    <div className="flex-1 overflow-y-auto space-y-2 pr-1 custom-scrollbar">
                        {topRoutesList.map((route, idx) => (
                            <button
                                key={idx}
                                onClick={() => flyToRoute(route)}
                                className="w-full text-left p-3 rounded bg-secondary/20 hover:bg-secondary/40 border border-transparent hover:border-border transition-all group"
                            >
                                <div className="flex justify-between items-start mb-1">
                                    <span className="text-[10px] font-mono text-muted-foreground bg-secondary/50 px-1 rounded">
                                        #{idx + 1}
                                    </span>
                                    <span className="text-xs font-bold text-primary">
                                        {route.trip_count.toLocaleString()} trips
                                    </span>
                                </div>
                                <div className="flex items-center gap-2 text-xs text-foreground">
                                    <span className="truncate max-w-[40%]">{route.start_station_name}</span>
                                    <ArrowRight className="w-3 h-3 text-muted-foreground flex-shrink-0" />
                                    <span className="truncate max-w-[40%]">{route.end_station_name}</span>
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
                    getTooltip={({ object }) => object && object.start_station_name && (
                        `${object.start_station_name} ➝ ${object.end_station_name}\nTrips: ${object.trip_count}`
                    )}
                >
                    <Map
                        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    />
                </DeckGL>

                {/* Legend */}
                <div className="absolute bottom-6 right-6 glass-panel p-4 rounded-xl max-w-xs">
                    <div className="flex items-center gap-2 mb-2">
                        <RouteIcon className="w-4 h-4 text-primary" />
                        <span className="text-xs font-bold text-foreground">FLOW DIRECTION</span>
                    </div>
                    <div className="flex items-center justify-between text-[10px] text-muted-foreground">
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                            <span>Start</span>
                        </div>
                        <div className="h-0.5 w-8 bg-gradient-to-r from-blue-500 to-pink-500"></div>
                        <div className="flex items-center gap-1">
                            <div className="w-2 h-2 rounded-full bg-pink-500"></div>
                            <span>End</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    );
}

export default RouteAnalysis;
