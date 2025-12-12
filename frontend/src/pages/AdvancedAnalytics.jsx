import React, { useEffect, useState, useMemo } from 'react';
import DeckGL from '@deck.gl/react';
import { ScatterplotLayer, IconLayer, ColumnLayer } from '@deck.gl/layers';
import { HeatmapLayer } from '@deck.gl/aggregation-layers';
import { Map } from 'react-map-gl/maplibre';
import { Layers, Activity, ShieldAlert, Zap, Thermometer, Users, Cpu, Radio, Globe } from 'lucide-react';

// Initial View State for NYC
const INITIAL_VIEW_STATE = {
    longitude: -74.0060,
    latitude: 40.7128,
    zoom: 12,
    pitch: 50,
    bearing: -10
};

// API Base URL - use environment variable with fallback to production
const API_BASE = import.meta.env.VITE_API_URL || 'https://18.218.154.66.nip.io/api';

function AdvancedAnalytics() {
    const [viewState, setViewState] = useState(INITIAL_VIEW_STATE);
    const [activeMode, setActiveMode] = useState('abm'); // 'abm', 'uhi', 'safety', 'web3'

    // Data States
    const [abmData, setAbmData] = useState(null);
    const [uhiData, setUhiData] = useState(null);
    const [safetyData, setSafetyData] = useState(null);
    const [web3Data, setWeb3Data] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    // Fetch Data based on active mode
    useEffect(() => {
        const fetchData = async () => {
            setLoading(true);
            setError(null);
            try {
                let res;
                if (activeMode === 'abm' && !abmData) {
                    res = await fetch(`${API_BASE}/advanced-analytics/abm`);
                } else if (activeMode === 'uhi' && !uhiData) {
                    res = await fetch(`${API_BASE}/advanced-analytics/uhi`);
                } else if (activeMode === 'safety' && !safetyData) {
                    res = await fetch(`${API_BASE}/advanced-analytics/safety`);
                } else if (activeMode === 'web3' && !web3Data) {
                    res = await fetch(`${API_BASE}/advanced-analytics/web3`);
                }

                if (res) {
                    if (!res.ok) throw new Error(`HTTP Error: ${res.status}`);
                    const data = await res.json();
                    if (activeMode === 'abm') setAbmData(data);
                    if (activeMode === 'uhi') setUhiData(data);
                    if (activeMode === 'safety') setSafetyData(data);
                    if (activeMode === 'web3') setWeb3Data(data);
                }
            } catch (error) {
                console.error("Failed to fetch Advanced Analytics data:", error);
                setError("DATA STREAM INTERRUPTED");
            } finally {
                setLoading(false);
            }
        };
        fetchData();
    }, [activeMode]);

    // Universal Animation Loop
    const [animationTime, setAnimationTime] = useState(0);

    useEffect(() => {
        const interval = setInterval(() => {
            setAnimationTime(t => t + 0.05);

            // ABM Specific Updates (Position)
            if (activeMode === 'abm' && abmData) {
                setAbmData(prevData => {
                    return prevData.map(agent => ({
                        ...agent,
                        lng: agent.lng + (Math.random() - 0.5) * 0.0005,
                        lat: agent.lat + (Math.random() - 0.5) * 0.0005,
                        energy_level: Math.max(0, agent.energy_level - 0.1)
                    }));
                });
            }
        }, 50); // 20 FPS

        return () => clearInterval(interval);
    }, [activeMode, abmData]);

    // Layers Configuration
    const layers = useMemo(() => {
        const layerList = [];

        if (activeMode === 'abm' && abmData) {
            layerList.push(
                new ScatterplotLayer({
                    id: 'abm-agents',
                    data: abmData,
                    getPosition: d => [d.lng, d.lat],
                    getFillColor: d => d.state === 'RIDING' ? [0, 255, 255] : [255, 0, 128], // Cyan / Magenta
                    getRadius: 40,
                    radiusMinPixels: 3,
                    opacity: 0.9,
                    pickable: true,
                    updateTriggers: {
                        getPosition: [abmData]
                    },
                    transitions: {
                        getPosition: 50
                    }
                })
            );
        }

        if (activeMode === 'uhi' && uhiData) {
            const pulse = 1.0 + Math.sin(animationTime) * 0.2;
            layerList.push(
                new HeatmapLayer({
                    id: 'uhi-heatmap',
                    data: uhiData,
                    getPosition: d => [d.lng, d.lat],
                    getWeight: d => d.heat_stress_index,
                    radiusPixels: 60 + Math.sin(animationTime * 2) * 5,
                    intensity: 2.0 * pulse,
                    threshold: 0.1,
                    aggregation: 'SUM',
                    updateTriggers: {
                        radiusPixels: [animationTime],
                        intensity: [animationTime]
                    },
                    colorRange: [
                        [255, 255, 204],
                        [255, 237, 160],
                        [254, 217, 118],
                        [254, 178, 76],
                        [253, 141, 60],
                        [252, 78, 42],
                        [227, 26, 28],
                        [189, 0, 38],
                        [128, 0, 38]
                    ]
                })
            );
        }

        if (activeMode === 'safety' && safetyData) {
            layerList.push(
                new ScatterplotLayer({
                    id: 'safety-incidents',
                    data: safetyData,
                    getPosition: d => [d.lng, d.lat],
                    getFillColor: [255, 50, 50],
                    getRadius: d => (d.severity * 40) + (Math.sin(animationTime * 8) * 15),
                    opacity: 0.7 + Math.sin(animationTime * 5) * 0.3,
                    stroked: true,
                    getLineColor: [255, 255, 255],
                    lineWidthMinPixels: 2,
                    pickable: true,
                    filled: true,
                    updateTriggers: {
                        getRadius: [animationTime],
                        opacity: [animationTime]
                    }
                })
            );
        }

        if (activeMode === 'web3' && web3Data) {
            layerList.push(
                new ColumnLayer({
                    id: 'web3-rewards',
                    data: web3Data,
                    getPosition: d => [d.lng, d.lat],
                    getElevation: d => (d.bikecoin_reward * 30) + (Math.sin(animationTime + d.lng) * 150),
                    getFillColor: d => d.action_type === 'pickup' ? [0, 255, 128] : [255, 165, 0],
                    radius: 50,
                    extruded: true,
                    pickable: true,
                    elevationScale: 1,
                    updateTriggers: {
                        getElevation: [animationTime]
                    },
                    transitions: {
                        getElevation: 100
                    }
                })
            );
        }

        return layerList;
    }, [activeMode, abmData, uhiData, safetyData, web3Data, animationTime]);

    // Tooltip
    const getTooltip = ({ object }) => {
        if (!object) return null;
        if (activeMode === 'abm') {
            return {
                html: `
                    <div style="font-family: monospace; background: rgba(0,0,0,0.9); padding: 10px; border: 1px solid #0ff; color: #fff; border-radius: 4px;">
                        <div style="font-weight: bold; color: #0ff; margin-bottom: 4px;">AGENT: ${object.agent_id}</div>
                        <div>STATE: <span style="color: ${object.state === 'RIDING' ? '#0f0' : '#f0f'}">${object.state}</span></div>
                        <div>ENERGY: ${object.energy_level}%</div>
                        <div>MODE: ${object.decision_factor}</div>
                    </div>
                `
            };
        }
        if (activeMode === 'safety') {
            return {
                html: `
                    <div style="font-family: monospace; background: rgba(50,0,0,0.9); padding: 10px; border: 1px solid #f00; color: #fff; border-radius: 4px;">
                        <div style="font-weight: bold; color: #f00; margin-bottom: 4px;">⚠ SAFETY ALERT</div>
                        <div>TYPE: ${object.event_type}</div>
                        <div>SEVERITY: ${(object.severity * 100).toFixed(0)}%</div>
                    </div>
                `
            };
        }
        if (activeMode === 'web3') {
            return {
                html: `
                    <div style="font-family: monospace; background: rgba(0,50,0,0.9); padding: 10px; border: 1px solid #0f0; color: #fff; border-radius: 4px;">
                        <div style="font-weight: bold; color: #0f0; margin-bottom: 4px;">🪙 BIKECOIN REWARD</div>
                        <div>ACTION: ${object.action_type}</div>
                        <div>VALUE: ${object.bikecoin_reward} BKC</div>
                    </div>
                `
            };
        }
        return null;
    };

    return (
        <div className="flex h-[calc(100vh-64px)] bg-[#02040a] overflow-hidden relative font-mono text-white selection:bg-cyan-500 selection:text-black">
            {/* Cyberpunk Grid Background */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(0,255,255,0.03)_1px,transparent_1px),linear-gradient(90deg,rgba(0,255,255,0.03)_1px,transparent_1px)] bg-[size:40px_40px] pointer-events-none" />
            <div className="absolute inset-0 bg-gradient-to-t from-[#02040a] via-transparent to-transparent pointer-events-none" />

            {/* Top Command Bar */}
            <div className="absolute top-0 left-0 right-0 h-16 bg-black/40 backdrop-blur-md border-b border-white/10 flex items-center justify-between px-8 z-30">
                <div className="flex items-center gap-4">
                    <div className="w-3 h-3 bg-cyan-500 rounded-full animate-pulse shadow-[0_0_10px_#0ff]" />
                    <h1 className="text-xl font-bold tracking-widest text-transparent bg-clip-text bg-gradient-to-r from-cyan-400 to-blue-600">
                        CITY_OS <span className="text-xs text-gray-500 ml-2">KERNEL v9.0.1</span>
                    </h1>
                </div>

                <div className="flex items-center gap-8 text-xs text-gray-400">
                    <div className="flex items-center gap-2">
                        <Cpu size={14} className="text-cyan-500" />
                        <span>CPU: 12%</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Radio size={14} className="text-green-500" />
                        <span>NET: CONNECTED</span>
                    </div>
                    <div className="flex items-center gap-2">
                        <Globe size={14} className="text-purple-500" />
                        <span>NODES: 2,451</span>
                    </div>
                </div>
            </div>

            {/* Sidebar Controls */}
            <div className="absolute top-24 left-8 z-20 w-72 flex flex-col gap-4">
                <div className="backdrop-blur-xl bg-black/60 border border-white/10 p-1 rounded-2xl shadow-[0_0_30px_rgba(0,0,0,0.5)]">
                    <div className="p-4 border-b border-white/5 mb-2">
                        <h2 className="text-xs font-bold text-gray-500 uppercase tracking-[0.2em]">Active Protocol</h2>
                    </div>
                    <div className="flex flex-col gap-2 p-2">
                        <ModeButton
                            active={activeMode === 'abm'}
                            onClick={() => setActiveMode('abm')}
                            icon={<Users size={16} />}
                            label="AGENT SIMULATION"
                            sub="Autonomous Swarm Logic"
                            color="cyan"
                        />
                        <ModeButton
                            active={activeMode === 'uhi'}
                            onClick={() => setActiveMode('uhi')}
                            icon={<Thermometer size={16} />}
                            label="THERMAL VISION"
                            sub="Landsat LST Analysis"
                            color="red"
                        />
                        <ModeButton
                            active={activeMode === 'safety'}
                            onClick={() => setActiveMode('safety')}
                            icon={<ShieldAlert size={16} />}
                            label="SAFETY SENTINEL"
                            sub="Computer Vision Events"
                            color="orange"
                        />
                        <ModeButton
                            active={activeMode === 'web3'}
                            onClick={() => setActiveMode('web3')}
                            icon={<Zap size={16} />}
                            label="TOKENOMICS"
                            sub="BikeCoin Incentive Net"
                            color="emerald"
                        />
                    </div>
                </div>

                {/* System Status Panel */}
                <div className="backdrop-blur-xl bg-black/60 border border-white/10 p-4 rounded-2xl">
                    <div className="flex justify-between items-center mb-2">
                        <span className="text-xs text-gray-500">SYSTEM LOAD</span>
                        <span className="text-xs text-cyan-400">OPTIMAL</span>
                    </div>
                    <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-cyan-500 to-blue-500 w-[35%] animate-pulse" />
                    </div>

                    <div className="flex justify-between items-center mt-4 mb-2">
                        <span className="text-xs text-gray-500">DATA INTEGRITY</span>
                        <span className="text-xs text-green-400">VERIFIED</span>
                    </div>
                    <div className="w-full h-1 bg-gray-800 rounded-full overflow-hidden">
                        <div className="h-full bg-gradient-to-r from-green-500 to-emerald-500 w-[98%]" />
                    </div>
                </div>
            </div>

            {/* Error Alert */}
            {error && (
                <div className="absolute top-24 right-8 z-50 animate-bounce">
                    <div className="bg-red-500/10 border border-red-500 text-red-500 px-6 py-4 rounded-xl backdrop-blur-md flex items-center gap-3 shadow-[0_0_20px_rgba(255,0,0,0.2)]">
                        <ShieldAlert size={24} />
                        <div>
                            <div className="font-bold">SYSTEM ALERT</div>
                            <div className="text-xs opacity-80">{error}</div>
                        </div>
                    </div>
                </div>
            )}

            {/* Loading Overlay */}
            {loading && (
                <div className="absolute inset-0 z-40 bg-black/20 backdrop-blur-[2px] flex items-center justify-center">
                    <div className="bg-black/80 border border-cyan-500/30 px-8 py-6 rounded-2xl flex flex-col items-center gap-4 shadow-[0_0_50px_rgba(0,255,255,0.1)]">
                        <div className="w-12 h-12 border-4 border-cyan-500/30 border-t-cyan-500 rounded-full animate-spin" />
                        <div className="text-cyan-500 text-xs tracking-[0.2em] animate-pulse">ESTABLISHING UPLINK...</div>
                    </div>
                </div>
            )}

            {/* Map */}
            <div className="w-full h-full">
                <DeckGL
                    initialViewState={viewState}
                    onViewStateChange={({ viewState }) => setViewState(viewState)}
                    controller={true}
                    layers={layers}
                    getTooltip={getTooltip}
                    style={{ background: '#000' }}
                >
                    <Map
                        mapStyle="https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json"
                    />
                </DeckGL>
            </div>

            {/* HUD Legend */}
            <div className="absolute bottom-8 right-8 z-20 max-w-md">
                <div className="backdrop-blur-xl bg-black/80 border-l-2 border-cyan-500 p-6 rounded-r-xl shadow-2xl relative overflow-hidden group">
                    <div className="absolute top-0 right-0 p-2 opacity-20 group-hover:opacity-100 transition-opacity">
                        <Activity size={100} className="text-cyan-500 -mt-8 -mr-8" />
                    </div>

                    <h3 className="text-white font-bold text-lg mb-1 flex items-center gap-3 relative z-10">
                        {activeMode === 'abm' && <Users className="text-cyan-400" size={20} />}
                        {activeMode === 'uhi' && <Thermometer className="text-red-400" size={20} />}
                        {activeMode === 'safety' && <ShieldAlert className="text-orange-400" size={20} />}
                        {activeMode === 'web3' && <Zap className="text-emerald-400" size={20} />}

                        <span className="tracking-widest">
                            {activeMode === 'abm' && "AGENT_SWARM"}
                            {activeMode === 'uhi' && "MICRO_CLIMATE"}
                            {activeMode === 'safety' && "CV_SAFETY_NET"}
                            {activeMode === 'web3' && "TOKEN_ECONOMY"}
                        </span>
                    </h3>

                    <div className="h-px w-full bg-gradient-to-r from-cyan-500/50 to-transparent my-3" />

                    <p className="text-gray-400 text-xs leading-relaxed font-light relative z-10">
                        {activeMode === 'abm' && "Real-time tracking of 50 autonomous agents. Behavior logic driven by energy constraints and weather variables. [CYAN: Active] [MAGENTA: Idle/Charging]"}
                        {activeMode === 'uhi' && "Thermal satellite imagery proxy. Visualizing urban heat islands with high spatial resolution. Pulsing zones indicate critical heat stress requiring mitigation."}
                        {activeMode === 'safety' && "Computer Vision anomaly detection. Aggregating near-miss events from traffic feeds. [RED ZONES] indicate high-probability collision vectors."}
                        {activeMode === 'web3' && "Decentralized incentive layer. Visualizing BikeCoin (BKC) distribution. [GREEN: Pickup Bonus] [ORANGE: Dropoff Bonus]. Height indicates reward magnitude."}
                    </p>
                </div>
            </div>
        </div>
    );
}

// Cyberpunk Mode Button
const ModeButton = ({ active, onClick, icon, label, sub, color }) => {
    const colors = {
        cyan: 'hover:border-cyan-500/50 hover:bg-cyan-500/10 text-cyan-400',
        red: 'hover:border-red-500/50 hover:bg-red-500/10 text-red-400',
        orange: 'hover:border-orange-500/50 hover:bg-orange-500/10 text-orange-400',
        emerald: 'hover:border-emerald-500/50 hover:bg-emerald-500/10 text-emerald-400',
    };

    const activeColors = {
        cyan: 'border-cyan-500 bg-cyan-500/20 text-cyan-300 shadow-[0_0_15px_rgba(0,255,255,0.3)]',
        red: 'border-red-500 bg-red-500/20 text-red-300 shadow-[0_0_15px_rgba(255,0,0,0.3)]',
        orange: 'border-orange-500 bg-orange-500/20 text-orange-300 shadow-[0_0_15px_rgba(255,165,0,0.3)]',
        emerald: 'border-emerald-500 bg-emerald-500/20 text-emerald-300 shadow-[0_0_15px_rgba(16,185,129,0.3)]',
    };

    return (
        <button
            onClick={onClick}
            className={`
                group relative flex items-center gap-4 px-4 py-3 rounded-xl transition-all duration-300 border
                ${active
                    ? activeColors[color]
                    : `border-transparent bg-transparent ${colors[color]} opacity-60 hover:opacity-100`
                }
            `}
        >
            <div className={`
                p-2 rounded-lg transition-all duration-300
                ${active ? 'bg-black/40' : 'bg-white/5 group-hover:bg-white/10'}
            `}>
                {icon}
            </div>
            <div className="text-left">
                <span className="block font-bold text-sm tracking-wider">{label}</span>
                <span className="text-[10px] opacity-70 uppercase tracking-widest">{sub}</span>
            </div>

            {/* Active Indicator */}
            {active && (
                <div className={`absolute right-3 w-1.5 h-1.5 rounded-full animate-pulse bg-current shadow-[0_0_10px_currentColor]`} />
            )}
        </button>
    );
};

export default AdvancedAnalytics;
