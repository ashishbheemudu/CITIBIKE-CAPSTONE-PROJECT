import React, { useState, useEffect } from 'react';
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
    LinearProgress
} from '@mui/material';
import LocalShippingIcon from '@mui/icons-material/LocalShipping';
import WarningIcon from '@mui/icons-material/Warning';
import CheckCircleIcon from '@mui/icons-material/CheckCircle';

const RebalancingDashboard = () => {
    const [actions, setActions] = useState([]);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('/data/dashboard_rebalancing.json');
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setActions(data);
                setLoading(false);
            } catch (error) {
                console.error("Failed to fetch rebalancing data", error);
                // Fallback to empty or error state, but don't use mock data anymore
                setLoading(false);
            }
        };

        fetchData();
    }, []);

    const getPriorityColor = (priority) => {
        switch (priority) {
            case 'CRITICAL': return 'error';
            case 'HIGH': return 'warning';
            case 'MEDIUM': return 'info';
            case 'LOW': return 'success';
            default: return 'default';
        }
    };

    return (
        <Box sx={{ p: 4, backgroundColor: '#0a0b1e', minHeight: '100%', color: 'white' }}>
            <Typography variant="h4" sx={{ mb: 4, fontWeight: 'bold', background: 'linear-gradient(45deg, #FE6B8B 30%, #FF8E53 90%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                <LocalShippingIcon sx={{ mr: 2, verticalAlign: 'bottom', color: '#FF8E53' }} />
                Fleet Command Center
            </Typography>

            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">Pending Moves</Typography>
                        <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#FF8E53' }}>{actions.filter(a => a.action !== 'HOLD').length}</Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">Critical Shortages</Typography>
                        <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#f44336' }}>
                            {actions.filter(a => a.priority === 'CRITICAL').length}
                        </Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">System Health</Typography>
                        <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#4caf50' }}>98%</Typography>
                    </Paper>
                </Grid>
            </Grid>

            <Paper sx={{ width: '100%', overflow: 'hidden', background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                <TableContainer sx={{ maxHeight: 600 }}>
                    <Table stickyHeader aria-label="sticky table">
                        <TableHead>
                            <TableRow>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Station</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Action</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Quantity</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Priority</TableCell>
                                <TableCell sx={{ backgroundColor: '#1a1b2e', color: 'gray' }}>Status</TableCell>
                            </TableRow>
                        </TableHead>
                        <TableBody>
                            {loading ? (
                                <TableRow>
                                    <TableCell colSpan={5}>
                                        <LinearProgress sx={{ backgroundColor: 'rgba(255,255,255,0.1)', '& .MuiLinearProgress-bar': { backgroundColor: '#FF8E53' } }} />
                                    </TableCell>
                                </TableRow>
                            ) : (
                                actions.map((row, index) => (
                                    <TableRow hover role="checkbox" tabIndex={-1} key={index} sx={{ '&:hover': { backgroundColor: 'rgba(255,255,255,0.05)' } }}>
                                        <TableCell sx={{ color: 'white' }}>{row.station}</TableCell>
                                        <TableCell sx={{ color: 'white' }}>
                                            <Chip
                                                label={row.action}
                                                size="small"
                                                sx={{
                                                    backgroundColor: row.action === 'RESTOCK' ? 'rgba(76, 175, 80, 0.2)' : row.action === 'PICKUP' ? 'rgba(244, 67, 54, 0.2)' : 'rgba(255,255,255,0.1)',
                                                    color: row.action === 'RESTOCK' ? '#4caf50' : row.action === 'PICKUP' ? '#f44336' : 'gray',
                                                    fontWeight: 'bold'
                                                }}
                                            />
                                        </TableCell>
                                        <TableCell sx={{ color: 'white', fontWeight: 'bold' }}>{row.quantity}</TableCell>
                                        <TableCell sx={{ color: 'white' }}>
                                            <Chip
                                                label={row.priority}
                                                color={getPriorityColor(row.priority)}
                                                variant="outlined"
                                                size="small"
                                            />
                                        </TableCell>
                                        <TableCell sx={{ color: 'white' }}>
                                            <Button variant="outlined" size="small" startIcon={<CheckCircleIcon />} sx={{ borderColor: 'rgba(255,255,255,0.3)', color: 'rgba(255,255,255,0.7)' }}>
                                                Dispatch
                                            </Button>
                                        </TableCell>
                                    </TableRow>
                                ))
                            )}
                        </TableBody>
                    </Table>
                </TableContainer>
            </Paper>
        </Box>
    );
};

export default RebalancingDashboard;
