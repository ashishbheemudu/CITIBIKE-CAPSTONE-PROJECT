import React, { useState } from 'react';
import { styled, createTheme, ThemeProvider } from '@mui/material/styles';
import CssBaseline from '@mui/material/CssBaseline';
import Box from '@mui/material/Box';
import MuiAppBar from '@mui/material/AppBar';
import Toolbar from '@mui/material/Toolbar';
import List from '@mui/material/List';
import Typography from '@mui/material/Typography';
import IconButton from '@mui/material/IconButton';
import Badge from '@mui/material/Badge';
import Container from '@mui/material/Container';
import MenuIcon from '@mui/icons-material/Menu';
import NotificationsOutlinedIcon from '@mui/icons-material/NotificationsOutlined';
import ListItemButton from '@mui/material/ListItemButton';
import ListItemIcon from '@mui/material/ListItemIcon';
import ListItemText from '@mui/material/ListItemText';
import DashboardOutlinedIcon from '@mui/icons-material/DashboardOutlined';
import MapOutlinedIcon from '@mui/icons-material/MapOutlined';
import TimelineOutlinedIcon from '@mui/icons-material/TimelineOutlined';
import PlaceOutlinedIcon from '@mui/icons-material/PlaceOutlined';
import ShowChartOutlinedIcon from '@mui/icons-material/ShowChartOutlined';
import AutoAwesomeIcon from '@mui/icons-material/AutoAwesome';
import ChevronLeftIcon from '@mui/icons-material/ChevronLeft';
import PublicIcon from '@mui/icons-material/Public';
import PsychologyIcon from '@mui/icons-material/Psychology';
import LocalShippingIcon from '@mui/icons-material/LocalShipping';
import AnalyticsIcon from '@mui/icons-material/Analytics';
import { useNavigate, useLocation } from 'react-router-dom';
// eslint-disable-next-line no-unused-vars
import { motion } from 'framer-motion';

// --- Professional Enterprise Theme ---
const professionalTheme = createTheme({
    palette: {
        mode: 'dark',
        primary: {
            main: '#6366f1', // Indigo 500 (Neon Purple/Blue mix)
        },
        secondary: {
            main: '#10b981', // Emerald
        },
        background: {
            default: '#050507', // Ultra Dark
            paper: '#0a0b10',   // Dark Glass
        },
        text: {
            primary: '#ffffff',
            secondary: 'rgba(255, 255, 255, 0.5)',
        },
        divider: 'rgba(255, 255, 255, 0.05)',
    },
    typography: {
        fontFamily: '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif',
        h1: { fontWeight: 600, letterSpacing: '-0.025em' },
        h2: { fontWeight: 600, letterSpacing: '-0.025em' },
        h3: { fontWeight: 600, letterSpacing: '-0.025em' },
        h4: { fontWeight: 600, letterSpacing: '-0.025em' },
        h6: { fontWeight: 500, letterSpacing: '0.01em' },
        button: { textTransform: 'none', fontWeight: 500 },
    },
    shape: {
        borderRadius: 6, // Tighter, more professional radius
    },
    components: {
        MuiPaper: {
            styleOverrides: {
                root: {
                    backgroundImage: 'none',
                    backgroundColor: '#18181b',
                    border: '1px solid rgba(255, 255, 255, 0.08)',
                    boxShadow: '0 1px 3px 0 rgba(0, 0, 0, 0.1), 0 1px 2px 0 rgba(0, 0, 0, 0.06)', // Subtle shadow
                },
            },
        },
        MuiAppBar: {
            styleOverrides: {
                root: {
                    backgroundColor: 'rgba(9, 9, 11, 0.8)', // Semi-transparent background
                    backdropFilter: 'blur(12px)',
                    borderBottom: '1px solid rgba(255, 255, 255, 0.08)',
                    boxShadow: 'none',
                },
            },
        },
        MuiDrawer: {
            styleOverrides: {
                paper: {
                    backgroundColor: '#09090b',
                    borderRight: '1px solid rgba(255, 255, 255, 0.08)',
                },
            },
        },
    },
});

const drawerWidth = 260;

const MainContent = styled(Box)(({ theme, open }) => ({
    flexGrow: 1,
    height: '100vh',
    overflow: 'auto',
    paddingTop: 64,
    paddingLeft: open ? drawerWidth : 72,
    transition: theme.transitions.create('padding', {
        easing: theme.transitions.easing.sharp,
        duration: theme.transitions.duration.enteringScreen,
    }),
    backgroundColor: theme.palette.background.default,
    backgroundImage: `
        radial-gradient(at 0% 0%, rgba(59, 130, 246, 0.15) 0px, transparent 50%),
        radial-gradient(at 100% 0%, rgba(16, 185, 129, 0.1) 0px, transparent 50%),
        radial-gradient(at 100% 100%, rgba(59, 130, 246, 0.1) 0px, transparent 50%),
        radial-gradient(at 0% 100%, rgba(16, 185, 129, 0.1) 0px, transparent 50%)
    `,
    backgroundAttachment: 'fixed',
    backgroundSize: '100% 100%',
}));

const NavItem = ({ item, selected, onClick, open }) => (
    <ListItemButton
        onClick={onClick}
        sx={{
            minHeight: 44,
            justifyContent: open ? 'initial' : 'center',
            px: 2.5,
            my: 0.5,
            mx: 1.5,
            borderRadius: '12px', // Softer, more modern
            backgroundColor: selected ? 'rgba(99, 102, 241, 0.15)' : 'transparent',
            color: selected ? '#818cf8' : '#a1a1aa',
            border: selected ? '1px solid rgba(99, 102, 241, 0.3)' : '1px solid transparent',
            boxShadow: selected ? '0 0 15px rgba(99, 102, 241, 0.1)' : 'none',
            '&:hover': {
                backgroundColor: selected ? 'rgba(99, 102, 241, 0.25)' : 'rgba(255, 255, 255, 0.03)',
                color: selected ? '#ffffff' : '#ffffff',
            },
            transition: 'all 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
        }}
    >
        <ListItemIcon
            sx={{
                minWidth: 0,
                mr: open ? 2 : 'auto',
                justifyContent: 'center',
                color: 'inherit',
            }}
        >
            {item.icon}
        </ListItemIcon>
        {open && (
            <ListItemText
                primary={item.text}
                primaryTypographyProps={{
                    fontSize: '0.875rem',
                    fontWeight: selected ? 600 : 400,
                }}
            />
        )}
    </ListItemButton>
);

function Layout({ children }) {
    const [open, setOpen] = useState(true);
    const navigate = useNavigate();
    const location = useLocation();

    const toggleDrawer = () => {
        setOpen(!open);
    };

    const menuItems = [
        { text: 'Mission Control', icon: <DashboardOutlinedIcon />, path: '/' },
        { text: 'Map Explorer', icon: <MapOutlinedIcon />, path: '/map' },
        { text: 'Route Explorer', icon: <TimelineOutlinedIcon />, path: '/routes' },
        { text: 'Station Intel', icon: <PlaceOutlinedIcon />, path: '/stations' },
        { text: 'Neural Demand Lab', icon: <PsychologyIcon />, path: '/prediction' },
        { text: 'Deep Analytics', icon: <AnalyticsIcon />, path: '/advanced' },
        { text: 'Fleet Command', icon: <LocalShippingIcon />, path: '/rebalancing' },
        { text: 'Social Equity', icon: <PublicIcon />, path: '/equity' },
    ];

    return (
        <ThemeProvider theme={professionalTheme}>
            <Box sx={{ display: 'flex' }}>
                <CssBaseline />

                <MuiAppBar position="fixed" sx={{ zIndex: (theme) => theme.zIndex.drawer + 1 }}>
                    <Toolbar sx={{ minHeight: 64 }}>
                        <IconButton
                            color="inherit"
                            aria-label="toggle drawer"
                            onClick={toggleDrawer}
                            edge="start"
                            sx={{ mr: 2 }}
                        >
                            {open ? <ChevronLeftIcon /> : <MenuIcon />}
                        </IconButton>

                        <Typography variant="h6" noWrap component="div" sx={{ flexGrow: 1, display: 'flex', alignItems: 'center', gap: 1 }}>
                            <Box component="span" sx={{ fontWeight: 700, letterSpacing: '-0.5px' }}>CITI BIKE</Box>
                            <Box component="span" sx={{ color: 'text.secondary', fontWeight: 400, fontSize: '0.9em' }}>ANALYTICS</Box>
                        </Typography>

                        <IconButton color="inherit">
                            <Badge badgeContent={4} color="primary" variant="dot">
                                <NotificationsOutlinedIcon />
                            </Badge>
                        </IconButton>
                    </Toolbar>
                </MuiAppBar>

                <Box
                    component="nav"
                    sx={{
                        width: open ? drawerWidth : 72,
                        flexShrink: 0,
                        position: 'fixed',
                        height: '100%',
                        borderRight: '1px solid rgba(255, 255, 255, 0.05)',
                        backgroundColor: 'rgba(5, 5, 7, 0.8)', // Glass effect base
                        backdropFilter: 'blur(20px)', // Strong blur
                        transition: 'width 0.2s ease-in-out',
                        zIndex: 1200,
                        pt: 8, // Toolbar height
                    }}
                >
                    {/* Logo Area */}
                    {open && (
                        <Box sx={{ px: 3, mb: 4, mt: 2 }}>
                            <div className="flex items-center gap-3 mb-1">
                                <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                                    <ShowChartOutlinedIcon sx={{ color: 'white', fontSize: 18 }} />
                                </div>
                                <div>
                                    <Typography variant="caption" sx={{ color: 'text.secondary', letterSpacing: '0.1em', fontSize: '0.65rem', display: 'block', mb: -0.5 }}>NYC</Typography>
                                    <Typography variant="h6" sx={{ fontWeight: 700, color: 'white', lineHeight: 1.2 }}>Citi Bike AI</Typography>
                                </div>
                            </div>
                        </Box>
                    )}
                    <List sx={{ px: 2 }}>
                        {menuItems.map((item) => (
                            <NavItem
                                key={item.text}
                                item={item}
                                selected={location.pathname === item.path}
                                onClick={() => navigate(item.path)}
                                open={open}
                            />
                        ))}
                    </List>
                </Box>

                <MainContent open={open}>
                    <Box sx={{ mt: 4, mb: 4, width: '100%', px: 0 }}>
                        <motion.div
                            initial={{ opacity: 0, y: 10 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ duration: 0.4 }}
                            style={{ width: '100%' }}
                        >
                            {children}
                        </motion.div>
                    </Box>
                </MainContent>
            </Box>
        </ThemeProvider>
    );
}

export default Layout;
