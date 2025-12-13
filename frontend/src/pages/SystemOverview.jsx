import React, { useState, useEffect, useMemo, useCallback } from 'react';
import DateSelector from '../components/DateSelector';
import ChartErrorBoundary from '../components/ChartErrorBoundary';


import {
    AreaChart, Area, LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    ComposedChart, Bar, Legend
} from 'recharts';
import { fetchSystemOverview } from '../api';
import {
    Calendar, Filter, Activity, TrendingUp, Users, CloudRain, Sun,
    Droplets, ArrowRight, Zap, GripHorizontal
} from 'lucide-react';

// --- Premium UI Components ---

const GlassCard = ({ children, className = "" }) => (
    <div className={`backdrop-blur-xl bg-black/40 border border-white/10 rounded-2xl shadow-[0_0_15px_rgba(0,0,0,0.3)] transition-all duration-300 hover:border-white/20 hover:bg-black/50 ${className}`}>
        {children}
    </div>
);

const MetricCard = ({ title, value, subtext, icon: Icon, trend, color = "blue" }) => {
    const colors = {
        blue: "text-blue-400 bg-blue-500/10 border-blue-500/20",
        emerald: "text-emerald-400 bg-emerald-500/10 border-emerald-500/20",
        purple: "text-purple-400 bg-purple-500/10 border-purple-500/20",
        orange: "text-orange-400 bg-orange-500/10 border-orange-500/20",
    };

    return (
        <GlassCard className="p-4 md:p-6 flex flex-col justify-between relative overflow-hidden group">
            {/* Ambient Glow */}
            <div className={`absolute -right-6 -top-6 w-24 h-24 rounded-full blur-[50px] opacity-20 transition-opacity group-hover:opacity-40 ${color === 'blue' ? 'bg-blue-500' : color === 'emerald' ? 'bg-emerald-500' : color === 'purple' ? 'bg-purple-500' : 'bg-orange-500'}`} />

            <div className="flex justify-between items-start mb-4 relative z-10">
                <span className="text-gray-400 text-xs font-bold uppercase tracking-widest">{title}</span>
                <div className={`p-2 rounded-lg ${colors[color]}`}>
                    {Icon && <Icon className="w-4 h-4" />}
                </div>
            </div>

            <div className="relative z-10">
                <div className="text-2xl md:text-4xl font-bold font-sans tracking-tighter text-white mb-2 shadow-black drop-shadow-lg">
                    {value}
                </div>
                {subtext && (
                    <div className="flex items-center gap-2">
                        <span className={`text-xs font-medium px-2 py-0.5 rounded-full border ${trend === 'up' ? 'text-emerald-400 border-emerald-500/30 bg-emerald-500/10' : trend === 'down' ? 'text-rose-400 border-rose-500/30 bg-rose-500/10' : 'text-gray-400 border-gray-700 bg-gray-800/50'}`}>
                            {subtext}
                        </span>
                    </div>
                )}
            </div>
        </GlassCard>
    );
};

// --- Main Component ---

const SystemOverview = () => {
    const [data, setData] = useState(null);
    const [loading, setLoading] = useState(true);

    // Filters
    const [dateRange, setDateRange] = useState({ start: '2020-01-01', end: '2020-02-01' });
    const [selectedDays, setSelectedDays] = useState(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']);

    // Chart Controls
    const [activeOverlays, setActiveOverlays] = useState({
        temp: true,
        precip: false,
        humidity: false // New
    });
    const [smoothing, setSmoothing] = useState(0);

    const loadData = useCallback(async () => {
        setLoading(true);
        try {
            const result = await fetchSystemOverview({
                startDate: dateRange.start,
                endDate: dateRange.end,
                daysOfWeek: selectedDays
            });
            setData(result);
        } catch (error) {
            console.error("Failed to load data:", error);
        } finally {
            setLoading(false);
        }
    }, [dateRange.start, dateRange.end, selectedDays]);

    useEffect(() => {
        const timer = setTimeout(() => {
            loadData();
        }, 500);
        return () => clearTimeout(timer);
    }, [loadData]);

    const toggleOverlay = (key) => {
        setActiveOverlays(prev => ({ ...prev, [key]: !prev[key] }));
    };

    // Process Chart Data with Smoothing
    const chartData = useMemo(() => {
        if (!data?.time_series) return [];
        let processed = [...data.time_series];

        if (smoothing > 0) {
            processed = processed.map((item, index, arr) => {
                const start = Math.max(0, index - smoothing);
                const end = Math.min(arr.length, index + smoothing + 1);
                const subset = arr.slice(start, end);
                const avg = subset.reduce((sum, curr) => sum + curr.trip_count, 0) / subset.length;
                return { ...item, trip_count_smooth: Math.round(avg) };
            });
        } else {
            processed = processed.map(item => ({ ...item, trip_count_smooth: item.trip_count }));
        }
        return processed;
    }, [data, smoothing]);

    if (loading) {
        return (
            <div className="flex justify-center items-center h-[calc(100vh-100px)]">
                <div className="relative">
                    <div className="w-16 h-16 border-4 border-blue-500/30 border-t-blue-500 rounded-full animate-spin" />
                    <div className="absolute inset-0 flex items-center justify-center">
                        <Zap className="w-6 h-6 text-blue-500 animate-pulse" />
                    </div>
                </div>
            </div>
        );
    }

    if (!data) {
        return (
            <div className="flex justify-center items-center h-96 flex-col gap-6">
                <div className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-red-400 to-orange-500">
                    System Connectivity Error
                </div>
                <button
                    onClick={loadData}
                    className="px-6 py-2 bg-white/5 border border-white/10 rounded-lg hover:bg-white/10 transition-all text-white font-mono text-sm"
                >
                    RETRY_CONNECTION
                </button>
            </div>
        );
    }

    return (
        <div className="p-4 md:p-8 space-y-6 md:space-y-8 min-h-screen bg-[#050505] text-white selection:bg-blue-500/30">
            {/* Header */}
            <div className="flex flex-col md:flex-row md:items-end justify-between border-b border-white/5 pb-4 md:pb-6 gap-4">
                <div>
                    <h1 className="text-2xl md:text-4xl font-bold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-white via-gray-200 to-gray-500 mb-1 md:mb-2">
                        System Overview
                    </h1>
                    <p className="text-gray-500 font-mono text-[10px] md:text-xs tracking-widest uppercase">
                        Real-time Fleet Telemetry
                    </p>
                </div>
                <div className="flex items-center gap-3 bg-emerald-500/5 border border-emerald-500/20 px-3 md:px-4 py-1.5 md:py-2 rounded-full self-start md:self-auto">
                    <span className="relative flex h-2 w-2">
                        <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75"></span>
                        <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500"></span>
                    </span>
                    <span className="text-xs font-bold text-emerald-500 tracking-wider">LIVE FEED ACTIVE</span>
                </div>
            </div>

            {/* Filter Toolbar */}
            <GlassCard className="px-6 py-4 flex items-center justify-between flex-wrap gap-4">
                {/* Date and Days */}
                <div className="flex items-center gap-6">
                    <div className="flex items-center gap-3 bg-black/30 p-1.5 rounded-xl border border-white/5">
                        <div className="p-2 bg-white/5 rounded-lg">
                            <Calendar className="w-4 h-4 text-blue-400" />
                        </div>
                        <div className="flex items-center gap-2 px-2">
                            <DateSelector
                                value={dateRange.start}
                                onChange={(val) => setDateRange(prev => ({ ...prev, start: val }))}
                                className="bg-transparent border-none text-sm focus:ring-0 text-gray-300 font-mono"
                            />
                            <ArrowRight className="w-3 h-3 text-gray-600" />
                            <DateSelector
                                value={dateRange.end}
                                onChange={(val) => setDateRange(prev => ({ ...prev, end: val }))}
                                className="bg-transparent border-none text-sm focus:ring-0 text-gray-300 font-mono"
                            />
                        </div>
                    </div>

                    <div className="h-8 w-px bg-white/10" />

                    <div className="flex items-center gap-2">
                        {['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'].map((day, idx) => {
                            const fullDay = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'][idx];
                            const isActive = selectedDays.includes(fullDay);
                            return (
                                <button
                                    key={day}
                                    onClick={() => {
                                        if (isActive) setSelectedDays(prev => prev.filter(d => d !== fullDay));
                                        else setSelectedDays(prev => [...prev, fullDay]);
                                    }}
                                    className={`text-[10px] font-bold px-3 py-1.5 rounded-lg transition-all duration-300 uppercase tracking-wide
                                        ${isActive
                                            ? 'bg-blue-500 text-white shadow-[0_0_10px_rgba(59,130,246,0.5)]'
                                            : 'bg-white/5 text-gray-500 hover:bg-white/10'
                                        }`}
                                >
                                    {day}
                                </button>
                            );
                        })}
                    </div>
                </div>
            </GlassCard>

            {/* KPIs */}
            {data.message ? (
                <div className="p-12 text-center border border-dashed border-gray-800 rounded-2xl">
                    <p className="text-gray-500 mb-2">{data.message}</p>
                    <p className="text-xs text-blue-500 cursor-pointer hover:underline" onClick={() => setDateRange({ start: '2019-01-01', end: '2019-12-31' })}>
                        Reset to default range
                    </p>
                </div>
            ) : (
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    <MetricCard
                        title="Network Traffic"
                        value={data?.kpis?.total_trips?.toLocaleString() ?? '0'}
                        subtext="Total Trips"
                        icon={Activity}
                        color="blue"
                    />
                    <MetricCard
                        title="Weekend Load"
                        value={`${Math.round(data?.kpis?.weekend_share ?? 0)}%`}
                        subtext={data?.kpis?.weekend_share > 30 ? "Leisure Mode" : "Commute Mode"}
                        icon={Users}
                        trend={data?.kpis?.weekend_share > 30 ? 'up' : 'down'}
                        color="purple"
                    />
                    <MetricCard
                        title="Peak Usage"
                        value={`${data?.kpis?.peak_hour ?? 0}:00`}
                        subtext="High Traffic Window"
                        icon={TrendingUp}
                        color="orange"
                    />
                    <MetricCard
                        title="Wknd Impact"
                        value={data?.kpis?.weekend_effect ?? 0}
                        subtext="Cohen's d Score"
                        icon={Filter}
                        color="emerald"
                    />
                </div>
            )}

            {/* Main Chart */}
            <GlassCard className="p-8">
                <ChartErrorBoundary>
                    <div className="flex justify-between items-center mb-8">
                        <div>
                            <h3 className="text-xl font-bold text-white mb-1">Demand & Weather Correlation</h3>
                            <p className="text-xs text-gray-500 uppercase tracking-widest">Multi-Variate Analysis</p>
                        </div>

                        <div className="flex items-center gap-6">
                            {/* Weather Toggles */}
                            <div className="flex bg-black/40 p-1 rounded-xl border border-white/5">
                                <button
                                    onClick={() => toggleOverlay('temp')}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${activeOverlays.temp ? 'bg-orange-500/20 text-orange-400' : 'text-gray-500 hover:text-white'}`}
                                >
                                    <Sun size={14} /> Temp
                                </button>
                                <button
                                    onClick={() => toggleOverlay('precip')}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${activeOverlays.precip ? 'bg-blue-500/20 text-blue-400' : 'text-gray-500 hover:text-white'}`}
                                >
                                    <CloudRain size={14} /> Precip
                                </button>
                                <button
                                    onClick={() => toggleOverlay('humidity')}
                                    className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-xs font-medium transition-all ${activeOverlays.humidity ? 'bg-cyan-500/20 text-cyan-400' : 'text-gray-500 hover:text-white'}`}
                                >
                                    <Droplets size={14} /> Humidity
                                </button>
                            </div>

                            {/* Smoothing */}
                            <div className="flex items-center gap-3 pl-6 border-l border-white/10">
                                <GripHorizontal size={16} className="text-gray-500" />
                                <input
                                    type="range" min="0" max="10" step="1"
                                    value={smoothing}
                                    onChange={(e) => setSmoothing(parseInt(e.target.value))}
                                    className="w-24 h-1.5 bg-gray-800 rounded-full appearance-none cursor-pointer accent-blue-500"
                                />
                            </div>
                        </div>
                    </div>

                    <div style={{ height: 400 }}>
                        <ResponsiveContainer width="100%" height="100%">
                            <ComposedChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                                <defs>
                                    <linearGradient id="colorTrips" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.4} />
                                        <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
                                    </linearGradient>
                                    <linearGradient id="colorTemp" x1="0" y1="0" x2="0" y2="1">
                                        <stop offset="5%" stopColor="#f97316" stopOpacity={0.2} />
                                        <stop offset="95%" stopColor="#f97316" stopOpacity={0} />
                                    </linearGradient>
                                </defs>
                                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                <XAxis
                                    dataKey="date"
                                    stroke="rgba(255,255,255,0.3)"
                                    tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.3)' }}
                                    tickLine={false}
                                    axisLine={false}
                                    minTickGap={30}
                                />
                                <YAxis
                                    yAxisId="left"
                                    stroke="rgba(255,255,255,0.3)"
                                    tick={{ fontSize: 10, fill: 'rgba(255,255,255,0.3)' }}
                                    tickLine={false}
                                    axisLine={false}
                                    label={{ value: 'Trips', angle: -90, position: 'insideLeft', fill: '#3b82f6', fontSize: 10 }}
                                />

                                {/* Right Axis for Weather */}
                                <YAxis
                                    yAxisId="right"
                                    orientation="right"
                                    stroke="rgba(255,255,255,0.1)"
                                    tick={false}
                                    tickLine={false}
                                    axisLine={false}
                                    domain={['auto', 'auto']} // Auto-scale to match data range
                                    allowDataOverflow={false}
                                />

                                <Tooltip
                                    contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '12px', boxShadow: '0 10px 40px rgba(0,0,0,0.5)' }}
                                    itemStyle={{ fontSize: '12px', fontWeight: 500 }}
                                    labelStyle={{ color: '#9ca3af', marginBottom: '8px', fontSize: '10px', textTransform: 'uppercase', letterSpacing: '1px' }}
                                    formatter={(value, name) => {
                                        if (name === 'Trips') return [value.toLocaleString(), 'Trips'];
                                        if (name.includes('Temp')) return [`${value.toFixed(1)}°C`, 'Temperature'];
                                        if (name.includes('Precip')) return [`${value.toFixed(1)}mm`, 'Precipitation'];
                                        if (name.includes('Humidity')) return [`${value.toFixed(0)}%`, 'Humidity'];
                                        return [value, name];
                                    }}
                                />

                                <Area
                                    yAxisId="left"
                                    type="monotone"
                                    dataKey="trip_count_smooth"
                                    name="Trips"
                                    stroke="#3b82f6"
                                    strokeWidth={3}
                                    fill="url(#colorTrips)"
                                    activeDot={{ r: 6, strokeWidth: 0, fill: '#fff', boxShadow: '0 0 10px #3b82f6' }}
                                />

                                {activeOverlays.temp && (
                                    <Line
                                        yAxisId="right"
                                        type="monotone"
                                        dataKey="temp"
                                        name="Temp"
                                        stroke="#f97316"
                                        strokeWidth={2}
                                        dot={false}
                                        strokeDasharray="5 5"
                                    />
                                )}
                                {activeOverlays.humidity && (
                                    <Line
                                        yAxisId="right"
                                        type="monotone"
                                        dataKey="humidity"
                                        name="Humidity"
                                        stroke="#06b6d4"
                                        strokeWidth={2}
                                        dot={false}
                                    />
                                )}
                                {activeOverlays.precip && (
                                    <Bar
                                        yAxisId="right"
                                        dataKey="precip"
                                        name="Precip"
                                        fill="#3b82f6"
                                        opacity={0.3}
                                        barSize={10}
                                    />
                                )}
                            </ComposedChart>
                        </ResponsiveContainer>
                    </div>
                </ChartErrorBoundary>
            </GlassCard>

            {/* Bottom Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                {/* Trend Chart */}
                <GlassCard className="p-6">
                    <ChartErrorBoundary>
                        <h4 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6">Volatility & Trend</h4>
                        <div style={{ height: 200 }}>
                            <ResponsiveContainer width="100%" height="100%">
                                <AreaChart data={data?.trend}>
                                    <defs>
                                        <linearGradient id="trendGradient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.3} />
                                            <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
                                    <XAxis dataKey="date" hide />
                                    <YAxis hide />
                                    <Tooltip
                                        contentStyle={{ backgroundColor: 'rgba(0,0,0,0.8)', border: '1px solid rgba(255,255,255,0.1)' }}
                                    />
                                    <Area type="monotone" dataKey="upper_ci" stroke="none" fill="#8b5cf6" fillOpacity={0.05} />
                                    <Area type="monotone" dataKey="lower_ci" stroke="none" fill="#000" fillOpacity={0.5} />
                                    <Area type="monotone" dataKey="7_day_ma" stroke="#8b5cf6" strokeWidth={2} fill="url(#trendGradient)" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </div>
                    </ChartErrorBoundary>
                </GlassCard>

                {/* Anomalies Table */}
                <GlassCard className="p-6 flex flex-col">
                    <h4 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6">Anomalies Detected</h4>
                    <div className="flex-1 overflow-y-auto pr-2 space-y-3 custom-scrollbar" style={{ maxHeight: 200 }}>
                        {data?.anomalies?.map((anomaly, i) => (
                            <div key={i} className="flex justify-between items-center p-3 rounded-xl bg-white/5 hover:bg-white/10 transition-colors border border-white/5">
                                <span className="text-xs text-gray-400 font-mono">{anomaly.date}</span>
                                <div className="flex items-center gap-4">
                                    <span className="font-bold text-white text-sm">{anomaly.trip_count.toLocaleString()}</span>
                                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${anomaly.z_score > 0 ? 'bg-emerald-500/20 text-emerald-400' : 'bg-rose-500/20 text-rose-400'}`}>
                                        {anomaly.z_score > 0 ? '+' : ''}{anomaly.z_score}σ
                                    </span>
                                </div>
                            </div>
                        ))}
                    </div>
                </GlassCard>

                {/* Rider Heatmap */}
                <GlassCard className="p-6">
                    <h4 className="text-sm font-bold text-gray-400 uppercase tracking-widest mb-6">Member vs Casual</h4>
                    <div className="h-[200px] flex flex-col justify-center">
                        {data?.heatmap && (
                            <div className="space-y-2">
                                {/* Label Row */}
                                <div className="flex justify-between text-[10px] text-gray-500 uppercase tracking-wider px-10 mb-2">
                                    <span>Casual Riders</span>
                                    <span>Members</span>
                                </div>
                                {data.heatmap.map((row, i) => (
                                    <div key={i} className="flex items-center gap-3">
                                        <span className="text-[10px] font-bold text-gray-500 w-8 text-right">{row.day_of_week?.slice(0, 3)}</span>
                                        <div className="flex-1 flex gap-1 h-3">
                                            {/* Casual Bar (Left, Aligned Right) */}
                                            <div className="flex-1 flex justify-end">
                                                <div
                                                    className="h-full bg-gradient-to-l from-blue-500 to-blue-900 rounded-l-sm transition-all duration-1000"
                                                    style={{ width: `${Math.min(100, (row.casual / 50000) * 100)}%` }}
                                                />
                                            </div>
                                            {/* Member Bar (Right, Aligned Left) */}
                                            <div className="flex-1 flex justify-start">
                                                <div
                                                    className="h-full bg-gradient-to-r from-purple-500 to-purple-900 rounded-r-sm transition-all duration-1000"
                                                    style={{ width: `${Math.min(100, (row.member / 200000) * 100)}%` }}
                                                />
                                            </div>
                                        </div>
                                    </div>
                                ))}
                            </div>
                        )}
                    </div>
                </GlassCard>
            </div>
        </div>
    );
};

export default SystemOverview;
