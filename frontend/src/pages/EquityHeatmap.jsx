import React, { useState, useEffect } from 'react';
import {
    Box,
    Typography,
    Paper,
    Grid,
    LinearProgress,
    Card,
    CardContent,
    Tooltip,
    IconButton
} from '@mui/material';
import PublicIcon from '@mui/icons-material/Public';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import Diversity1Icon from '@mui/icons-material/Diversity1';
import TrendingUpIcon from '@mui/icons-material/TrendingUp';

const EquityHeatmap = () => {
    const [scores, setScores] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const response = await fetch('/data/dashboard_equity.json');
                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }
                const data = await response.json();
                setScores(data);
                setLoading(false);
            } catch (error) {
                console.error("Failed to fetch equity data", error);
                setLoading(false);
            }
        };
        fetchData();
    }, []);

    const getScoreColor = (score) => {
        if (score >= 0.9) return '#4caf50'; // Green
        if (score >= 0.8) return '#ff9800'; // Orange
        return '#f44336'; // Red
    };

    return (
        <Box sx={{ p: 4, backgroundColor: '#0a0b1e', minHeight: '100%', color: 'white' }}>
            <Typography variant="h4" sx={{ mb: 1, fontWeight: 'bold', background: 'linear-gradient(45deg, #00C853 30%, #B2FF59 90%)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                <PublicIcon sx={{ mr: 2, verticalAlign: 'bottom', color: '#B2FF59' }} />
                Social Equity Monitor
            </Typography>
            <Typography variant="subtitle1" sx={{ mb: 4, color: 'gray' }}>
                Real-time tracking of service availability across all boroughs.
            </Typography>

            <Grid container spacing={3} sx={{ mb: 4 }}>
                <Grid item xs={12} md={4}>
                    <Paper sx={{ p: 3, background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)' }}>
                        <Typography variant="h6" color="gray">System Gini Coefficient</Typography>
                        <Box sx={{ display: 'flex', alignItems: 'baseline', gap: 1 }}>
                            <Typography variant="h3" sx={{ fontWeight: 'bold', color: '#00C853' }}>0.12</Typography>
                            <Typography variant="body2" color="success.main" sx={{ display: 'flex', alignItems: 'center' }}>
                                <TrendingUpIcon fontSize="small" sx={{ mr: 0.5 }} /> Excellent
                            </Typography>
                        </Box>
                        <Typography variant="caption" color="gray">Lower is better (0 = Perfect Equality)</Typography>
                    </Paper>
                </Grid>
                <Grid item xs={12} md={8}>
                    <Paper sx={{ p: 3, height: '100%', background: 'rgba(255,255,255,0.05)', backdropFilter: 'blur(10px)', border: '1px solid rgba(255,255,255,0.1)', display: 'flex', alignItems: 'center' }}>
                        <Diversity1Icon sx={{ fontSize: 60, color: 'rgba(255,255,255,0.1)', mr: 3 }} />
                        <Box>
                            <Typography variant="h6" color="white">Why this matters?</Typography>
                            <Typography variant="body2" color="gray">
                                Our "God Protocol" doesn't just optimize for profit. It actively rebalances bikes to underserved areas (Bronx, Queens) to ensure
                                fair access to public transit for all New Yorkers, regardless of neighborhood density.
                            </Typography>
                        </Box>
                    </Paper>
                </Grid>
            </Grid>

            <Typography variant="h5" sx={{ mb: 3, fontWeight: 'bold' }}>Borough Service Levels</Typography>

            <Grid container spacing={3}>
                {loading ? (
                    <Grid item xs={12}><LinearProgress /></Grid>
                ) : (
                    Object.entries(scores).map(([borough, score]) => (
                        <Grid item xs={12} md={6} lg={4} key={borough}>
                            <Card sx={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.05)' }}>
                                <CardContent>
                                    <Box sx={{ display: 'flex', justifyContent: 'space-between', mb: 2 }}>
                                        <Typography variant="h6" color="white">{borough}</Typography>
                                        <Tooltip title="Service Availability Score (0-1)">
                                            <IconButton size="small"><InfoOutlinedIcon sx={{ color: 'gray' }} /></IconButton>
                                        </Tooltip>
                                    </Box>

                                    <Box sx={{ display: 'flex', alignItems: 'center', mb: 1 }}>
                                        <Box sx={{ flexGrow: 1, mr: 2 }}>
                                            <LinearProgress
                                                variant="determinate"
                                                value={score * 100}
                                                sx={{
                                                    height: 10,
                                                    borderRadius: 5,
                                                    backgroundColor: 'rgba(255,255,255,0.1)',
                                                    '& .MuiLinearProgress-bar': { backgroundColor: getScoreColor(score) }
                                                }}
                                            />
                                        </Box>
                                        <Typography variant="h6" sx={{ color: getScoreColor(score), fontWeight: 'bold' }}>
                                            {(score * 100).toFixed(0)}%
                                        </Typography>
                                    </Box>

                                    <Typography variant="caption" color="gray">
                                        {score < 0.8 ? "Intervention Needed: High Rebalancing Priority" : "Service Level Optimal"}
                                    </Typography>
                                </CardContent>
                            </Card>
                        </Grid>
                    ))
                )}
            </Grid>
        </Box>
    );
};

export default EquityHeatmap;
