import React, { useEffect, useState, useMemo, useCallback } from 'react';
import { fetchStations, fetchStationDetails } from '../api';
import ChartErrorBoundary from '../components/ChartErrorBoundary';
import {
    BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, AreaChart, Area
} from 'recharts';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer } from '@deck.gl/layers';
import { Map } from 'react-map-gl/maplibre';
import { Search, MapPin, Clock, Calendar, Activity } from 'lucide-react';

// Custom Tooltip for Recharts
const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
        return (
            <div className="glass-panel p-3 rounded border border-border/50 shadow-xl">
                <p className="text-sm font-bold text-foreground mb-1">{label}</p>
                <p className="text-xs text-primary">
                    {payload[0].value.toLocaleString()} trips
                </p>
            </div>
        );
    }
    return null;
};

function StationDrilldown() {
    const [stations, setStations] = useState([]);
    const [searchTerm, setSearchTerm] = useState('');
    const [selectedStation, setSelectedStation] = useState(null);
    const [stationData, setStationData] = useState(null);
    const [loadingList, setLoadingList] = useState(true);
    const [loadingDetails, setLoadingDetails] = useState(false);

    // Handle Selection - wrapped in useCallback to stabilize reference
    const handleSelectStation = useCallback(async (stationName) => {
        setSelectedStation(stationName);
        setLoadingDetails(true);
        try {
            const data = await fetchStationDetails(stationName);
            setStationData(data);
        } catch (error) {
            console.error("Failed to load station details:", error);
            setStationData(null);
        } finally {
            setLoadingDetails(false);
        }
    }, []); // No dependencies needed - uses only setters and API call

    // Load Station List
    useEffect(() => {
        const loadStations = async () => {
            try {
                const list = await fetchStations();
                setStations(list.sort());
                // Select first station by default if available
                if (list.length > 0) {
                    handleSelectStation(list[0]);
                }
            } catch (error) {
                console.error("Failed to load stations:", error);
            } finally {
                setLoadingList(false);
            }
        };
        loadStations();
    }, [handleSelectStation]); // Now safe to add as dependency

    // Filter Stations
    const filteredStations = useMemo(() => {
        return stations.filter(s =>
            s.toLowerCase().includes(searchTerm.toLowerCase())
        );
    }, [stations, searchTerm]);



    // Mini Map Layer - optimized to only depend on location coordinates
    const mapLayer = useMemo(() => {
        if (!stationData || !stationData.location) return null;
        return new ScatterplotLayer({
            id: 'station-marker',
            data: [stationData.location],
            getPosition: d => [d.lon, d.lat],
            getRadius: 100,
            getFillColor: [236, 72, 153], // Pink
            stroked: true,
            getLineColor: [255, 255, 255],
            getLineWidth: 2,
            radiusMinPixels: 5,
            radiusMaxPixels: 15
        });
    }, [stationData?.location?.lat, stationData?.location?.lon]); // Only re-create when coordinates change

    return (
        <div className="flex h-[calc(100vh-64px)] bg-background overflow-hidden">
            {/* --- Sidebar (Search & List) --- */}
            <div className="w-80 border-r border-border bg-card/30 flex flex-col backdrop-blur-md z-10">
                <div className="p-4 border-b border-border">
                    <h2 className="text-lg font-semibold text-foreground mb-1">Station Drilldown</h2>
                    <p className="text-xs text-muted-foreground mb-4">Deep dive into station analytics</p>

                    <div className="relative">
                        <Search className="absolute left-2 top-2.5 w-4 h-4 text-muted-foreground" />
                        <input
                            type="text"
                            placeholder="Search stations..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            className="w-full bg-secondary/50 border border-border rounded pl-8 pr-3 py-2 text-sm text-foreground focus:outline-none focus:ring-1 focus:ring-primary placeholder:text-muted-foreground/50"
                        />
                    </div>
                </div>

                <div className="flex-1 overflow-y-auto custom-scrollbar p-2 space-y-1">
                    {loadingList ? (
                        <div className="flex justify-center p-4">
                            <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-primary"></div>
                        </div>
                    ) : (
                        filteredStations.map((station) => (
                            <button
                                key={station}
                                onClick={() => handleSelectStation(station)}
                                className={`w-full text-left px-3 py-2 rounded text-xs transition-colors truncate
                                    ${selectedStation === station
                                        ? 'bg-primary/20 text-primary font-medium border border-primary/30'
                                        : 'text-muted-foreground hover:bg-secondary/50 hover:text-foreground'
                                    }`}
                            >
                                {station}
                            </button>
                        ))
                    )}
                </div>
            </div>

            {/* --- Main Content --- */}
            <div className="flex-1 overflow-y-auto p-6 relative">
                {loadingDetails ? (
                    <div className="absolute inset-0 flex justify-center items-center bg-background/80 backdrop-blur-sm z-20">
                        <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-primary"></div>
                    </div>
                ) : stationData ? (
                    <div className="w-full space-y-6">{/* Changed from max-w-6xl mx-auto to w-full */}
                        {/* Header */}
                        <div className="flex items-end justify-between border-b border-border pb-4">
                            <div>
                                <h1 className="text-2xl font-bold text-foreground flex items-center gap-2">
                                    <MapPin className="w-6 h-6 text-primary" />
                                    {selectedStation}
                                </h1>
                                <p className="text-sm text-muted-foreground mt-1">
                                    Detailed performance metrics and usage patterns.
                                </p>
                            </div>
                        </div>

                        {/* KPI Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                            <div className="glass-panel p-4 rounded-xl border border-border/50">
                                <div className="flex items-center gap-2 text-muted-foreground mb-2">
                                    <Activity className="w-4 h-4" />
                                    <span className="text-xs font-medium uppercase tracking-wider">Total Trips</span>
                                </div>
                                <div className="text-2xl font-bold text-foreground">
                                    {stationData.kpis.total_trips.toLocaleString()}
                                </div>
                            </div>
                            <div className="glass-panel p-4 rounded-xl border border-border/50">
                                <div className="flex items-center gap-2 text-muted-foreground mb-2">
                                    <Calendar className="w-4 h-4" />
                                    <span className="text-xs font-medium uppercase tracking-wider">Avg Daily Trips</span>
                                </div>
                                <div className="text-2xl font-bold text-foreground">
                                    {stationData.kpis.avg_trips_day.toLocaleString()}
                                </div>
                            </div>
                            <div className="glass-panel p-4 rounded-xl border border-border/50">
                                <div className="flex items-center gap-2 text-muted-foreground mb-2">
                                    <Clock className="w-4 h-4" />
                                    <span className="text-xs font-medium uppercase tracking-wider">Peak Hour</span>
                                </div>
                                <div className="text-2xl font-bold text-foreground">
                                    {stationData.kpis.peak_hour}:00
                                </div>
                            </div>
                        </div>

                        {/* Charts Row */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {/* Hourly Profile */}
                            <div className="glass-panel p-6 rounded-xl border border-border/50">
                                <ChartErrorBoundary>
                                    <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
                                        <Clock className="w-4 h-4 text-primary" /> Hourly Demand Profile
                                    </h3>
                                    <div className="h-64">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <AreaChart data={stationData.hourly_profile}>
                                                <defs>
                                                    <linearGradient id="colorHourly" x1="0" y1="0" x2="0" y2="1">
                                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
                                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                                    </linearGradient>
                                                </defs>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                                <XAxis
                                                    dataKey="hour"
                                                    stroke="#666"
                                                    tick={{ fontSize: 10 }}
                                                    tickFormatter={(val) => `${val}:00`}
                                                />
                                                <YAxis stroke="#666" tick={{ fontSize: 10 }} />
                                                <Tooltip content={<CustomTooltip />} cursor={{ stroke: '#3b82f6', strokeWidth: 1 }} />
                                                <Area
                                                    type="monotone"
                                                    dataKey="trip_count"
                                                    stroke="#3b82f6"
                                                    strokeWidth={2}
                                                    fillOpacity={1}
                                                    fill="url(#colorHourly)"
                                                />
                                            </AreaChart>
                                        </ResponsiveContainer>
                                    </div>
                                </ChartErrorBoundary>
                            </div>

                            {/* Daily Profile */}
                            <div className="glass-panel p-6 rounded-xl border border-border/50">
                                <ChartErrorBoundary>
                                    <h3 className="text-sm font-semibold text-foreground mb-4 flex items-center gap-2">
                                        <Calendar className="w-4 h-4 text-emerald-500" /> Weekly Pattern
                                    </h3>
                                    <div className="h-64">
                                        <ResponsiveContainer width="100%" height="100%">
                                            <BarChart data={stationData.daily_profile}>
                                                <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                                <XAxis
                                                    dataKey="day_of_week"
                                                    stroke="#666"
                                                    tick={{ fontSize: 10 }}
                                                    tickFormatter={(val) => val.slice(0, 3)}
                                                />
                                                <YAxis stroke="#666" tick={{ fontSize: 10 }} />
                                                <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.05)' }} />
                                                <Bar dataKey="trip_count" fill="#10b981" radius={[4, 4, 0, 0]} />
                                            </BarChart>
                                        </ResponsiveContainer>
                                    </div>
                                </ChartErrorBoundary>
                            </div>
                        </div>

                        {/* Mini Map */}
                        <div className="glass-panel p-1 rounded-xl border border-border/50 overflow-hidden h-64 relative">
                            <DeckGL
                                initialViewState={{
                                    longitude: stationData.location.lon,
                                    latitude: stationData.location.lat,
                                    zoom: 14,
                                    pitch: 0,
                                    bearing: 0
                                }}
                                controller={true}
                                layers={[mapLayer]}
                            >
                                <Map
                                    mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                                />
                            </DeckGL>
                            <div className="absolute top-4 left-4 bg-black/50 backdrop-blur px-2 py-1 rounded text-xs text-white border border-white/10">
                                Station Location
                            </div>
                        </div>

                    </div>
                ) : (
                    <div className="flex flex-col items-center justify-center h-full text-muted-foreground">
                        <MapPin className="w-12 h-12 mb-4 opacity-20" />
                        <p>Select a station to view details</p>
                    </div>
                )}
            </div>
        </div>
    );
}

export default StationDrilldown;
