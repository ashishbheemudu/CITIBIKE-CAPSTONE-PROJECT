import React, { useState, useEffect, useCallback } from 'react';
import {
    Box,
    Typography,
    Paper,
    Grid,
    Chip,
    Button,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    LinearProgress,
    Snackbar,
    Alert,
    IconButton,
    Tooltip
} from '@mui/material';
import LocalShippingIcon from '@mui/icons-material/LocalShipping';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';
import RefreshIcon from '@mui/icons-material/Refresh';
import DirectionsBikeIcon from '@mui/icons-material/DirectionsBike';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';

// API Configuration
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://18.116.202.251.nip.io/api';

const RebalancingDashboard = () => {
    const [actions, setActions] = useState([]);
    const [loading, setLoading] = useState(true);
    const [error, setError] = useState(null);
    const [lastUpdated, setLastUpdated] = useState(null);
    const [dispatchedStations, setDispatchedStations] = useState(new Set());
    const [snackbar, setSnackbar] = useState({ open: false, message: '', severity: 'success' });

    // Calculate rebalancing needs from live station data
    const calculateRebalancingNeeds = useCallback((stations) => {
        const needs = [];

        stations.forEach(station => {
            const capacity = station.capacity || (station.num_bikes_available + station.num_docks_available) || 30;
            const available = station.num_bikes_available || 0;
            const docks = station.num_docks_available || 0;
            const fillRate = capacity > 0 ? available / capacity : 0.5;

            let action = 'HOLD';
            let quantity = 0;
            let priority = 'LOW';

            // Station needs bikes (low availability)
            if (fillRate < 0.15) {
                action = 'RESTOCK';
                quantity = Math.ceil(capacity * 0.4 - available);
                priority = fillRate < 0.05 ? 'CRITICAL' : 'HIGH';
            }
            // Station too full (needs bikes picked up)
            else if (fillRate > 0.85) {
                action = 'PICKUP';
                quantity = Math.ceil(available - capacity * 0.6);
                priority = fillRate > 0.95 ? 'CRITICAL' : 'HIGH';
            }
            // Moderate imbalance
            else if (fillRate < 0.25) {
                action = 'RESTOCK';
                quantity = Math.ceil(capacity * 0.35 - available);
                priority = 'MEDIUM';
            }
            else if (fillRate > 0.75) {
                action = 'PICKUP';
                quantity = Math.ceil(available - capacity * 0.65);
                priority = 'MEDIUM';
            }

            if (action !== 'HOLD' && quantity > 0) {
                needs.push({
                    station: station.name || station.station_name || `Station ${station.station_id}`,
                    station_id: station.station_id,
                    action,
                    quantity: Math.min(quantity, 15), // Cap at 15 per trip
                    priority,
                    current_bikes: available,
                    current_docks: docks,
                    capacity,
                    fill_rate: Math.round(fillRate * 100)
                });
            }
        });

        // Sort by priority (CRITICAL first, then HIGH, etc.)
        const priorityOrder = { 'CRITICAL': 0, 'HIGH': 1, 'MEDIUM': 2, 'LOW': 3 };
        needs.sort((a, b) => priorityOrder[a.priority] - priorityOrder[b.priority]);

        return needs.slice(0, 50); // Return top 50
    }, []);

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError(null);

        try {
            // Try the new rebalancing API endpoint
            const response = await fetch(`${API_BASE_URL}/rebalancing`);

            if (!response.ok) {
                throw new Error(`API Error: ${response.status}`);
            }

            const rebalancingNeeds = await response.json();

            setActions(rebalancingNeeds);
            setLastUpdated(new Date());
            setLoading(false);
        } catch (err) {
            console.error("Failed to fetch live data, using computed fallback", err);

            // Generate realistic fallback data with variety
            const fallbackData = generateRealisticFallback();
            setActions(fallbackData);
            setError("Using simulated data - live feed unavailable");
            setLastUpdated(new Date());
            setLoading(false);
        }
    }, []);

    // Generate realistic fallback data with variety
    const generateRealisticFallback = () => {
        const stations = [
            "W 21 St & 6 Ave", "Broadway & W 58 St", "Central Park S & 6 Ave",
            "E 47 St & 2 Ave", "University Pl & E 14 St", "West St & Chambers St",
            "Lafayette St & E 8 St", "Columbus Ave & W 72 St", "1 Ave & E 68 St",
            "8 Ave & W 31 St", "Broadway & E 14 St", "Greenwich Ave & 8 Ave",
            "E 17 St & Broadway", "W 41 St & 8 Ave", "Canal St & Rutgers St",
            "Metropolitan Ave & Bedford Ave", "Pier 40 - Hudson River Park",
            "W 4 St & 7 Ave S", "Amsterdam Ave & W 73 St", "6 Ave & W 34 St"
        ];

        const priorities = ['CRITICAL', 'HIGH', 'HIGH', 'MEDIUM', 'MEDIUM', 'MEDIUM', 'LOW'];
        const actions = ['RESTOCK', 'PICKUP'];

        return stations.map((station, idx) => ({
            station,
            station_id: `STA${idx + 1}`,
            action: actions[idx % 2],
            quantity: Math.floor(Math.random() * 12) + 3, // 3-15 bikes
            priority: priorities[idx % priorities.length],
            current_bikes: Math.floor(Math.random() * 25),
            current_docks: Math.floor(Math.random() * 20),
            capacity: 35,
            fill_rate: Math.floor(Math.random() * 100)
        }));
    };

    useEffect(() => {
        fetchData();
        // Refresh every 5 minutes
        const interval = setInterval(fetchData, 5 * 60 * 1000);
        return () => clearInterval(interval);
    }, [fetchData]);

    const handleDispatch = (station) => {
        // Add to dispatched set
        setDispatchedStations(prev => new Set([...prev, station]));

        // Show success message
        setSnackbar({
            open: true,
            message: `Dispatch order sent for ${station}`,
            severity: 'success'
        });

        // Remove from list after 2 seconds (simulating completion)
        setTimeout(() => {
            setActions(prev => prev.filter(a => a.station !== station));
        }, 2000);
    };

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'CRITICAL': return 'error';
            case 'HIGH': return 'warning';
            case 'MEDIUM': return 'info';
            case 'LOW': return 'success';
            default: return 'default';
        }
    };

    const systemHealth = actions.length > 0
        ? Math.round(100 - (actions.filter(a => a.priority === 'CRITICAL').length * 5 + actions.filter(a => a.priority === 'HIGH').length * 2))
        : 100;

    return (
        <Box sx={{ p: 4, backgroundColor: '#0a0b1e', minHeight: '100%', color: 'white' }}>
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 4 }}>
                <Box>
                    <Typography variant="h4" sx={{ fontWeight: 'bold', background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                        <LocalShippingIcon sx={{ mr: 2, verticalAlign: 'bottom', color: '#FF8E53' }} />
                        Fleet Command Center
                    </Typography>
                    {lastUpdated && (
                        <Typography variant="caption" color="gray">
                            Last updated: {lastUpdated.toLocaleTimeString()}
                        </Typography>
                    )}
                </Box>
                <Tooltip title="Refresh data">
                    <IconButton onClick={fetchData} sx={{ color: '#FF8E53' }}>
                        <RefreshIcon />
                    </IconButton>
                </Tooltip>
            </Box>

            {error && (
                <Alert severity="warning" sx={{ mb: 3, backgroundColor: 'rgba(255, 152, 0, 0.1)' }}>
                    {error}
                </Alert>
            )}

            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">Pending Moves</Typography>
                        <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#FF8E53' }}>
                            {actions.filter(a => a.action !== 'HOLD').length}
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">Critical Shortages</Typography>
                        <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#f44336' }}>
                            {actions.filter(a => a.priority === 'CRITICAL').length}
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">Bikes to Move</Typography>
                        <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#2196f3' }}>
                            {actions.reduce((sum, a) => sum + a.quantity, 0)}
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} md={3}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">System Health</Typography>
                        <Typography variant="h3" sx={{ fontWeight: 'bold', color: systemHealth > 90 ? '#4caf50' : systemHealth > 70 ? '#ff9800' : '#f44336' }}>
                            {systemHealth}%
                        </Typography>
                    </Paper>
                </Grid>
            </Grid>

            <Paper sx={{ width: '100%', overflow: 'hidden', background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                <TableContainer sx={{ maxHeight: 600 }}>
                    <Table stickyHeader aria-label="rebalancing table">
                        <TableHead>
                            <TableRow>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Station</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Action</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Quantity</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Current Status</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Priority</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Action</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {loading ? (
                                <TableRow>
                                    <TableCell colSpan={6}>
                                        <LinearProgress sx={{ backgroundColor: 'rgba(255,255,255,0.1)', '& .MuiLinearProgress-bar': { backgroundColor: '#FF8E53' } }} />
                                    </TableCell>
                                </TableRow>
                            ) : actions.length === 0 ? (
                                <TableRow>
                                    <TableCell colSpan={6} sx={{ color: 'gray', textAlign: 'center', py: 4 }}>
                                        <CheckCircleIcon sx={{ fontSize: 48, color: '#4caf50', mb: 2 }} />
                                        <Typography>All stations are balanced! No actions needed.</Typography>
                                    </TableCell>
                                </TableRow>
                            ) : (
                                actions.map((row, index) => (
                                    <TableRow
                                        hover
                                        key={index}
                                        sx={{
                                            '&:hover': { backgroundColor: 'rgba(255,255,255,0.05)' },
                                            opacity: dispatchedStations.has(row.station) ? 0.5 : 1,
                                            transition: 'opacity 0.3s'
                                        }}
                                    >
                                        <TableCell sx={{ color: 'white' }}>{row.station}</TableCell>
                                        <TableCell>
                                            <Chip
                                                icon={row.action === 'RESTOCK' ? <DirectionsBikeIcon /> : <WarningAmberIcon />}
                                                label={row.action}
                                                size="small"
                                                sx={{
                                                    backgroundColor: row.action === 'RESTOCK' ? 'rgba(76, 175, 80, 0.2)' : 'rgba(244, 67, 54, 0.2)',
                                                    color: row.action === 'RESTOCK' ? '#4caf50' : '#f44336',
                                                    fontWeight: 'bold'
                                                }}
                                            />
                                        </TableCell>
                                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>
                                            {row.quantity} bikes
                                        </TableCell>
                                        <TableCell sx={{ color: 'gray' }}>
                                            <Tooltip title={`${row.current_bikes} bikes / ${row.current_docks} docks`}>
                                                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                                                    <LinearProgress
                                                        variant="determinate"
                                                        value={row.fill_rate}
                                                        sx={{
                                                            width: 60,
                                                            height: 8,
                                                            borderRadius: 4,
                                                            backgroundColor: 'rgba(255,255,255,0.1)',
                                                            '& .MuiLinearProgress-bar': {
                                                                backgroundColor: row.fill_rate < 20 ? '#f44336' : row.fill_rate > 80 ? '#ff9800' : '#4caf50'
                                                            }
                                                        }}
                                                    />
                                                    <Typography variant="caption">{row.fill_rate}%</Typography>
                                                </Box>
                                            </Tooltip>
                                        </TableCell>
                                        <TableCell>
                                            <Chip
                                                label={row.priority}
                                                color={getPriorityColor(row.priority)}
                                                variant="outlined"
                                                size="small"
                                            />
                                        </TableCell>
                                        <TableCell>
                                            <Button
                                                variant="contained"
                                                size="small"
                                                startIcon={<CheckCircleIcon />}
                                                disabled={dispatchedStations.has(row.station)}
                                                onClick={() => handleDispatch(row.station)}
                                                sx={{
                                                    backgroundColor: '#FF8E53',
                                                    '&:hover': { backgroundColor: '#FE6B8B' },
                                                    '&:disabled': { backgroundColor: 'rgba(255,255,255,0.1)' }
                                                }}
                                            >
                                                {dispatchedStations.has(row.station) ? 'Dispatched' : 'Dispatch'}
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Paper>

            <Snackbar
                open={snackbar.open}
                autoHideDuration={4000}
                onClose={() => setSnackbar({ ...snackbar, open: false })}
                anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
            >
                <Alert severity={snackbar.severity} sx={{ width: '100%' }}>
                    {snackbar.message}
                </Alert>
            </Snackbar>
        </Box>
    );
};

export default RebalancingDashboard;
