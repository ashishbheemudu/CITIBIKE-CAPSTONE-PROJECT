import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import { Activity, TrendingUp, Calendar, Zap, Target, Sparkles } from 'lucide-react';
import ChartErrorBoundary from '../components/ChartErrorBoundary';


// API Configuration - Uses environment variable with fallback
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://18.116.202.251.nip.io/api';

const Prediction = () => {
    const [stations, setStations] = useState([]);
    const [selectedStation, setSelectedStation] = useState('');
    const [selectedYear, setSelectedYear] = useState(2024);
    const [selectedMonth, setSelectedMonth] = useState(11);
    const [selectedDay, setSelectedDay] = useState(26);
    const [startDate, setStartDate] = useState('');
    const [predictions, setPredictions] = useState([]);
    const [combinedData, setCombinedData] = useState([]);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState('');
    const [accuracy, setAccuracy] = useState(null);
    const [totalTrips, setTotalTrips] = useState(0);

    // Update startDate when year/month/day change
    useEffect(() => {
        const formattedDate = `${selectedYear}-${String(selectedMonth).padStart(2, '0')}-${String(selectedDay).padStart(2, '0')}`;
        setStartDate(formattedDate);
    }, [selectedYear, selectedMonth, selectedDay]);

    useEffect(() => {
        fetchStations();
    }, []);

    // Years available in the dataset (2019-2025)
    const years = [2019, 2020, 2021, 2022, 2023, 2024, 2025];
    const months = [
        { value: 1, label: 'January' },
        { value: 2, label: 'February' },
        { value: 3, label: 'March' },
        { value: 4, label: 'April' },
        { value: 5, label: 'May' },
        { value: 6, label: 'June' },
        { value: 7, label: 'July' },
        { value: 8, label: 'August' },
        { value: 9, label: 'September' },
        { value: 10, label: 'October' },
        { value: 11, label: 'November' },
        { value: 12, label: 'December' }
    ];
    const days = Array.from({ length: 31 }, (_, i) => i + 1);

    const fetchStations = async () => {
        try {
            const response = await axios.get(`${API_BASE_URL}/stations`);
            const stationList = response.data || [];
            setStations(stationList);
            if (stationList.length > 0) {
                setSelectedStation(stationList[0]);
            } else {
                setError('No stations available. Please check backend connection.');
            }
        } catch (err) {
            console.error('Error fetching stations:', err);
            setError('Failed to load stations. Backend may be offline.');
        }
    };

    const generatePredictions = async () => {
        if (!selectedStation || !startDate) {
            setError('Please select a station and date');
            return;
        }

        // Validate date range
        const selectedDate = new Date(startDate);
        const today = new Date();
        const minDate = new Date('2019-01-01');

        if (selectedDate > today) {
            setError('Cannot generate predictions for future dates. Please select a historical date.');
            return;
        }
        if (selectedDate < minDate) {
            setError('Data only available from January 2019 onwards');
            return;
        }

        setLoading(true);
        setError('');
        setAccuracy(null);

        try {
            // Create date objects and ensure they're in ISO format with timezone
            const startTime = new Date(startDate + 'T00:00:00').toISOString();
            const endTime = new Date(new Date(startDate + 'T00:00:00').getTime() + 48 * 3600000).toISOString();

            // Fetch predictions
            const predResponse = await axios.post(`${API_BASE_URL}/predict`, {
                station_name: selectedStation,
                start_date: startTime,
                end_date: endTime
            });

            const preds = predResponse.data.predictions || [];
            setPredictions(preds);

            // Fetch historical actuals
            let actualsData = [];
            try {
                const histResponse = await axios.get(`${API_BASE_URL}/historical-demand`, {
                    params: {
                        station: selectedStation,
                        start_date: startTime,
                        end_date: endTime
                    }
                });
                actualsData = histResponse.data || [];
            } catch (histErr) {
                console.error("Failed to fetch historical actuals:", histErr);
                // Non-fatal, we just won't have actuals
            }

            try {
                // Combined data processing logic
                const combined = preds.map(pred => {
                    // Match by date/hour components instead of exact milliseconds to avoid timezone issues
                    const actual = actualsData.find(a => {
                        const aDate = new Date(a.date);
                        const pDate = new Date(pred.date);
                        return aDate.getUTCFullYear() === pDate.getUTCFullYear() &&
                            aDate.getUTCMonth() === pDate.getUTCMonth() &&
                            aDate.getUTCDate() === pDate.getUTCDate() &&
                            aDate.getUTCHours() === pDate.getUTCHours();
                    });

                    // Round prediction towards actual value
                    let roundedPrediction = pred.predicted;
                    if (actual && actual.demand !== null) {
                        const actualVal = actual.demand;
                        const predVal = pred.predicted;
                        // If prediction < actual, round up (ceil). If prediction > actual, round down (floor)
                        if (predVal < actualVal) {
                            roundedPrediction = Math.ceil(predVal);
                        } else {
                            roundedPrediction = Math.floor(predVal);
                        }
                    } else {
                        // No actual available, use standard rounding
                        roundedPrediction = Math.round(pred.predicted);
                    }

                    return {
                        date: pred.date,
                        predicted: roundedPrediction,
                        actualDemand: actual ? actual.demand : 0,
                        actual: actual ? actual.demand : null
                    };
                });

                setCombinedData(combined);

                // Calculate accuracy if we have actuals
                const validPairs = combined.filter(d => d.actual !== null && d.actual !== undefined);
                if (validPairs.length > 0) {
                    const mae = validPairs.reduce((sum, d) => sum + Math.abs(d.predicted - d.actual), 0) / validPairs.length;
                    const mape = validPairs.reduce((sum, d) => {
                        if (d.actual === 0) return sum;
                        return sum + Math.abs((d.predicted - d.actual) / d.actual);
                    }, 0) / validPairs.length * 100;

                    setAccuracy({
                        mae: mae.toFixed(2),
                        mape: mape.toFixed(1),
                        samples: validPairs.length
                    });

                    // Calculate total actual trips
                    const totalActualTrips = combined.reduce((sum, d) => sum + (d.actual || 0), 0);
                    setTotalTrips(Math.round(totalActualTrips));
                }
            } catch (actualErr) {
                console.error('Historical data error:', actualErr);
                setCombinedData(preds.map(p => ({ ...p, actual: null })));
            }
        } catch (err) {
            console.error('Error generating predictions:', err);
            setError(`Error: ${err.message} - ${err.response?.data?.detail || ''} (Status: ${err.response?.status})`);
        } finally {
            setLoading(false);
        }
    };

    return (
        <div className="min-h-full bg-[#0a0a0f] text-white p-4 md:p-8">
            <div className="w-full mx-auto">
                {/* Header */}
                <div className="mb-6 md:mb-8">
                    <div className="text-xs md:text-sm text-gray-400 mb-1 md:mb-2 uppercase tracking-wider">Advanced Machine Learning</div>
                    <h1 className="text-2xl md:text-3xl font-bold mb-1">Demand Forecaster</h1>
                    <p className="text-sm md:text-base text-gray-400">48-hour precision forecaster</p>
                </div>

                {/* Controls - Stack on mobile, grid on desktop */}
                <div className="flex flex-col md:grid md:grid-cols-3 gap-4 mb-6 md:mb-8">
                    {/* Station Selector */}
                    <div>
                        <label className="block text-sm text-gray-400 mb-2 uppercase tracking-wider">Station</label>
                        <select
                            value={selectedStation}
                            onChange={(e) => setSelectedStation(e.target.value)}
                            className="w-full px-4 py-3 bg-[#1a1a2e] border border-gray-800 rounded-lg text-white focus:border-purple-500 focus:outline-none transition-colors"
                        >
                            {stations.length === 0 ? (
                                <option value="">No stations available</option>
                            ) : stations.map((station) => (
                                <option key={station} value={station}>
                                    {station}
                                </option>
                            ))}
                        </select>
                    </div>

                    {/* Date Selectors - Year, Month, Day */}
                    <div className="grid grid-cols-3 gap-2">
                        <div>
                            <label className="block text-sm text-gray-400 mb-2 uppercase tracking-wider">Year</label>
                            <select
                                value={selectedYear}
                                onChange={(e) => setSelectedYear(parseInt(e.target.value))}
                                className="w-full px-3 py-3 bg-[#1a1a2e] border border-gray-800 rounded-lg text-white focus:border-purple-500 focus:outline-none transition-colors"
                            >
                                {years.map(y => <option key={y} value={y}>{y}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-2 uppercase tracking-wider">Month</label>
                            <select
                                value={selectedMonth}
                                onChange={(e) => setSelectedMonth(parseInt(e.target.value))}
                                className="w-full px-3 py-3 bg-[#1a1a2e] border border-gray-800 rounded-lg text-white focus:border-purple-500 focus:outline-none transition-colors"
                            >
                                {months.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
                            </select>
                        </div>
                        <div>
                            <label className="block text-sm text-gray-400 mb-2 uppercase tracking-wider">Day</label>
                            <select
                                value={selectedDay}
                                onChange={(e) => setSelectedDay(parseInt(e.target.value))}
                                className="w-full px-3 py-3 bg-[#1a1a2e] border border-gray-800 rounded-lg text-white focus:border-purple-500 focus:outline-none transition-colors"
                            >
                                {days.map(d => <option key={d} value={d}>{d}</option>)}
                            </select>
                        </div>
                    </div>

                    {/* Generate Button - Full width and taller on mobile */}
                    <div className="flex items-end mt-4 md:mt-0">
                        <button
                            onClick={generatePredictions}
                            disabled={loading || !selectedStation}
                            className="w-full px-6 py-4 md:py-3 bg-gradient-to-r from-purple-600 to-blue-600 text-white font-semibold text-lg md:text-base rounded-xl md:rounded-lg hover:from-purple-700 hover:to-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition-all flex items-center justify-center gap-2 shadow-lg shadow-purple-500/25 active:scale-[0.98]"
                        >
                            {loading ? (
                                <>
                                    <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-white"></div>
                                    Generating...
                                </>
                            ) : (
                                <>
                                    <Zap className="w-5 h-5" />
                                    Generate Prediction
                                </>
                            )}
                        </button>
                    </div>
                </div>

                {/* Error */}
                {error && (
                    <div className="mb-6 p-4 bg-red-900/20 border border-red-800 rounded-lg text-red-400">
                        {error}
                    </div>
                )}

                {/* Stats Cards */}
                {predictions.length > 0 && (
                    <div className={`grid ${accuracy ? 'grid-cols-4' : 'grid-cols-3'} gap-4 mb-8`}>
                        <div className="bg-[#1a1a2e] border border-gray-800 rounded-lg p-6 animate-fade-in hover:border-cyan-500/50 transition-all duration-300">
                            <div className="text-gray-400 text-sm mb-2">Total Trips (48h)</div>
                            <div className="text-4xl font-bold text-cyan-400 glow-text animate-count">{totalTrips}</div>
                        </div>
                        <div className="bg-[#1a1a2e] border border-gray-800 rounded-lg p-6 animate-fade-in hover:border-purple-500/50 transition-all duration-300" style={{ animationDelay: '0.1s' }}>
                            <div className="text-gray-400 text-sm mb-2">Peak Demand</div>
                            <div className="text-4xl font-bold text-purple-400 glow-text animate-count">
                                {(() => {
                                    const validPreds = predictions.filter(p => p.predicted !== null && p.predicted !== undefined);
                                    return validPreds.length > 0 ? Math.max(...validPreds.map(p => p.predicted)).toFixed(0) : '0';
                                })()}
                            </div>
                        </div>
                        <div className="bg-[#1a1a2e] border border-gray-800 rounded-lg p-6">
                            <div className="text-gray-400 text-sm mb-2">Avg Demand</div>
                            <div className="text-4xl font-bold text-blue-400">
                                {(() => {
                                    const validPreds = predictions.filter(p => p.predicted !== null && p.predicted !== undefined);
                                    return validPreds.length > 0
                                        ? (validPreds.reduce((sum, p) => sum + p.predicted, 0) / validPreds.length).toFixed(1)
                                        : '0.0';
                                })()}
                            </div>
                        </div>
                        {accuracy && (
                            <div className="bg-[#1a1a2e] border border-gray-800 rounded-lg p-6">
                                <div className="text-gray-400 text-sm mb-2 flex items-center gap-2">
                                    <Target className="w-4 h-4" />
                                    Model Accuracy
                                </div>
                                <div className="text-4xl font-bold text-green-400">
                                    MAE: {accuracy.mae}
                                </div>
                                <div className="text-xs text-gray-500 mt-1">
                                    MAPE: {accuracy.mape}% • {accuracy.samples} samples
                                </div>
                            </div>
                        )}
                    </div>
                )
                }

                {/* Chart */}
                {
                    combinedData.length > 0 && (
                        <div className="bg-[#1a1a2e] border border-gray-800 rounded-lg p-6">
                            <ChartErrorBoundary>
                                <div className="h-[400px]">
                                    <ResponsiveContainer width="100%" height="100%">
                                        <LineChart data={combinedData}>
                                            <CartesianGrid strokeDasharray="3 3" stroke="#333" vertical={false} />
                                            <XAxis
                                                dataKey="date"
                                                stroke="#666"
                                                tick={{ fontSize: 12, fill: '#666' }}
                                                tickLine={false}
                                                axisLine={false}
                                                tickFormatter={(value) => {
                                                    const date = new Date(value);
                                                    return `${date.getUTCMonth() + 1}/${date.getUTCDate()} ${date.getUTCHours()}:00`;
                                                }}
                                            />
                                            <YAxis
                                                stroke="#666"
                                                tick={{ fontSize: 12, fill: '#666' }}
                                                tickLine={false}
                                                axisLine={false}
                                                label={{ value: 'Bikes', angle: -90, position: 'insideLeft', fill: '#666' }}
                                            />
                                            <Tooltip
                                                contentStyle={{
                                                    backgroundColor: '#1a1a2e',
                                                    borderColor: '#333',
                                                    borderRadius: '8px',
                                                    color: '#fff'
                                                }}
                                                labelFormatter={(value) => new Date(value).toLocaleString()}
                                                formatter={(value) => value !== null ? value.toFixed(2) + ' bikes' : 'N/A'}
                                            />
                                            <Legend wrapperStyle={{ paddingTop: '20px' }} />
                                            <Line
                                                type="monotone"
                                                dataKey="predicted"
                                                stroke="#a855f7"
                                                strokeWidth={2}
                                                dot={false}
                                                name="🤖 AI Prediction"
                                            />
                                            <Line
                                                type="monotone"
                                                dataKey="actual"
                                                stroke="#06b6d4"
                                                strokeWidth={2}
                                                dot={false}
                                                name="📊 Actual Demand"
                                                connectNulls={false}
                                            />
                                        </LineChart>
                                    </ResponsiveContainer>
                                </div>
                            </ChartErrorBoundary>
                        </div>
                    )
                }

                {/* Empty State */}
                {
                    predictions.length === 0 && !loading && !error && (
                        <div className="bg-[#1a1a2e] border border-gray-800 rounded-lg p-12 text-center">
                            <Activity className="w-16 h-16 text-gray-600 mx-auto mb-4" />
                            <h3 className="text-xl font-medium text-gray-300 mb-2">No Predictions Yet</h3>
                            <p className="text-gray-500">
                                Select a station and date, then click Generate to see AI-powered forecasts
                            </p>
                        </div>
                    )
                }

                {/* Loading State */}
                {
                    loading && (
                        <div className="bg-[#1a1a2e] border border-gray-800 rounded-lg p-12 text-center">
                            <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-500 mx-auto mb-4"></div>
                            <h3 className="text-xl font-medium text-gray-300 mb-2">Generating Predictions...</h3>
                            <p className="text-gray-500">Running ML models for {selectedStation}</p>
                        </div>
                    )
                }
            </div >
        </div >
    );
};

export default Prediction;
