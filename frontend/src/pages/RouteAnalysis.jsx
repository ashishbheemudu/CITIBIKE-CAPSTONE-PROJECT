import React, { useEffect, useState, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ArcLayer, ScatterplotLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl/maplibre';
import { fetchRoutes } from '../api';
import { Navigation, Filter, Route as RouteIcon, ArrowRight, X } from 'lucide-react';

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
    const [selectedRoute, setSelectedRoute] = useState(null);
    const [animationTime, setAnimationTime] = useState(0);

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

    // Animation loop for selected route pulsing
    useEffect(() => {
        if (!selectedRoute) return;

        const interval = setInterval(() => {
            setAnimationTime(t => (t + 0.1) % (Math.PI * 2));
        }, 50);

        return () => clearInterval(interval);
    }, [selectedRoute]);

    // Filtered Routes
    const filteredRoutes = useMemo(() => {
        if (!routes) return [];
        return routes.filter(r => r.trip_count >= minTrips);
    }, [routes, minTrips]);

    // Top Routes for Sidebar
    const topRoutesList = useMemo(() => {
        return [...filteredRoutes].sort((a, b) => b.trip_count - a.trip_count).slice(0, 10);
    }, [filteredRoutes]);

    // Check if route matches selected
    const isSelectedRoute = (route, currentSelected) => {
        if (!currentSelected) return false;
        return route.start_station_name === currentSelected.start_station_name &&
            route.end_station_name === currentSelected.end_station_name;
    };

    // Layers with dynamic styling based on selection
    const layers = useMemo(() => {
        const baseLayers = [];

        // Filter routes: non-selected ones for background
        const nonSelectedRoutes = selectedRoute
            ? filteredRoutes.filter(r => !isSelectedRoute(r, selectedRoute))
            : filteredRoutes;

        // Background routes (dimmed when something is selected)
        baseLayers.push(
            new ArcLayer({
                id: 'arc-layer-background',
                data: nonSelectedRoutes,
                getSourcePosition: d => [d.start_lon, d.start_lat],
                getTargetPosition: d => [d.end_lon, d.end_lat],
                getSourceColor: selectedRoute ? [59, 130, 246, 60] : [59, 130, 246, 200],
                getTargetColor: selectedRoute ? [236, 72, 153, 60] : [236, 72, 153, 200],
                getWidth: selectedRoute ? 1 : 3,
                pickable: true,
                onHover: info => setHoverInfo(info),
                onClick: info => {
                    if (info.object) {
                        setSelectedRoute(info.object);
                        flyToRoute(info.object);
                    }
                }
            })
        );

        // Selected route highlight layer (animated)
        if (selectedRoute) {
            const pulseWidth = 6 + Math.sin(animationTime) * 3;
            const glowOpacity = 180 + Math.sin(animationTime * 2) * 75;

            baseLayers.push(
                new ArcLayer({
                    id: 'arc-layer-selected',
                    data: [selectedRoute],
                    getSourcePosition: d => [d.start_lon, d.start_lat],
                    getTargetPosition: d => [d.end_lon, d.end_lat],
                    getSourceColor: [0, 255, 255, glowOpacity], // Cyan glow
                    getTargetColor: [255, 0, 255, glowOpacity], // Magenta glow
                    getWidth: pulseWidth,
                    pickable: false,
                    updateTriggers: {
                        getWidth: [animationTime],
                        getSourceColor: [animationTime],
                        getTargetColor: [animationTime]
                    }
                })
            );

            // Inner bright arc
            baseLayers.push(
                new ArcLayer({
                    id: 'arc-layer-selected-inner',
                    data: [selectedRoute],
                    getSourcePosition: d => [d.start_lon, d.start_lat],
                    getTargetPosition: d => [d.end_lon, d.end_lat],
                    getSourceColor: [100, 255, 255, 255], // Bright cyan
                    getTargetColor: [255, 100, 255, 255], // Bright magenta
                    getWidth: 4,
                    pickable: false
                })
            );

            // Animated endpoint markers
            const markerRadius = 60 + Math.sin(animationTime * 3) * 20;

            baseLayers.push(
                new ScatterplotLayer({
                    id: 'selected-start-pulse',
                    data: [selectedRoute],
                    getPosition: d => [d.start_lon, d.start_lat],
                    getRadius: markerRadius,
                    getFillColor: [0, 255, 255, 100],
                    stroked: true,
                    getLineColor: [0, 255, 255, 255],
                    lineWidthMinPixels: 2,
                    updateTriggers: { getRadius: [animationTime] }
                }),
                new ScatterplotLayer({
                    id: 'selected-end-pulse',
                    data: [selectedRoute],
                    getPosition: d => [d.end_lon, d.end_lat],
                    getRadius: markerRadius,
                    getFillColor: [255, 0, 255, 100],
                    stroked: true,
                    getLineColor: [255, 0, 255, 255],
                    lineWidthMinPixels: 2,
                    updateTriggers: { getRadius: [animationTime] }
                })
            );
        }

        // Base endpoint markers
        baseLayers.push(
            new ScatterplotLayer({
                id: 'start-points',
                data: filteredRoutes,
                getPosition: d => [d.start_lon, d.start_lat],
                getRadius: 30,
                getFillColor: d => isSelectedRoute(d, selectedRoute) ? [0, 255, 255] : [59, 130, 246],
                radiusMinPixels: 2,
                opacity: selectedRoute ? 0.3 : 1
            }),
            new ScatterplotLayer({
                id: 'end-points',
                data: filteredRoutes,
                getPosition: d => [d.end_lon, d.end_lat],
                getRadius: 30,
                getFillColor: d => isSelectedRoute(d, selectedRoute) ? [255, 0, 255] : [236, 72, 153],
                radiusMinPixels: 2,
                opacity: selectedRoute ? 0.3 : 1
            })
        );

        return baseLayers;
    }, [filteredRoutes, selectedRoute, animationTime]);

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

    const handleRouteClick = (route) => {
        setSelectedRoute(route);
        flyToRoute(route);
    };

    const clearSelection = () => {
        setSelectedRoute(null);
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

                {/* Selected Route Info */}
                {selectedRoute && (
                    <div className="bg-gradient-to-r from-cyan-500/20 to-pink-500/20 border border-cyan-500/50 rounded-xl p-4 relative animate-pulse">
                        <button
                            onClick={clearSelection}
                            className="absolute top-2 right-2 p-1 hover:bg-white/10 rounded transition-colors"
                        >
                            <X className="w-4 h-4 text-gray-400" />
                        </button>
                        <div className="text-xs text-cyan-400 font-bold mb-2">SELECTED ROUTE</div>
                        <div className="text-sm text-white font-medium mb-1">{selectedRoute.start_station_name}</div>
                        <div className="flex items-center gap-2 text-xs text-gray-400 mb-1">
                            <ArrowRight className="w-3 h-3" />
                        </div>
                        <div className="text-sm text-white font-medium mb-2">{selectedRoute.end_station_name}</div>
                        <div className="text-lg font-bold text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-pink-400">
                            {selectedRoute.trip_count.toLocaleString()} trips
                        </div>
                    </div>
                )}

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
                                onClick={() => handleRouteClick(route)}
                                className={`w-full text-left p-3 rounded border transition-all group ${isSelectedRoute(route, selectedRoute)
                                    ? 'bg-gradient-to-r from-cyan-500/30 to-pink-500/30 border-cyan-500/50 shadow-[0_0_15px_rgba(0,255,255,0.2)]'
                                    : 'bg-secondary/20 hover:bg-secondary/40 border-transparent hover:border-border'
                                    }`}
                            >
                                <div className="flex justify-between items-start mb-1">
                                    <span className={`text-[10px] font-mono px-1 rounded ${isSelectedRoute(route, selectedRoute)
                                        ? 'text-cyan-400 bg-cyan-500/20'
                                        : 'text-muted-foreground bg-secondary/50'
                                        }`}>
                                        #{idx + 1}
                                    </span>
                                    <span className={`text-xs font-bold ${isSelectedRoute(route, selectedRoute) ? 'text-cyan-400' : 'text-primary'
                                        }`}>
                                        {route.trip_count.toLocaleString()} trips
                                    </span>
                                </div>
                                <div className="flex items-center gap-2 text-xs text-foreground">
                                    <span className="truncate max-w-[40%]">{route.start_station_name}</span>
                                    <ArrowRight className={`w-3 h-3 flex-shrink-0 ${isSelectedRoute(route, selectedRoute) ? 'text-pink-400' : 'text-muted-foreground'
                                        }`} />
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
                        `${object.start_station_name} ➝ ${object.end_station_name}\nTrips: ${object.trip_count.toLocaleString()}`
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
                    {selectedRoute && (
                        <div className="mt-3 pt-3 border-t border-white/10 text-[10px] text-cyan-400">
                            Click anywhere on map or press X to clear selection
                        </div>
                    )}
                </div>

                {/* Clear Selection Button (floating) */}
                {selectedRoute && (
                    <button
                        onClick={clearSelection}
                        className="absolute top-4 right-4 bg-black/60 backdrop-blur-md border border-cyan-500/50 text-cyan-400 px-4 py-2 rounded-lg text-sm hover:bg-cyan-500/20 transition-all flex items-center gap-2"
                    >
                        <X className="w-4 h-4" />
                        Clear Selection
                    </button>
                )}
            </div>
        </div>
    );
}

export default RouteAnalysis;

