import axios from 'axios';
import { validateSystemOverview, validateStationDetails } from './utils/validators';

// API Configuration - Uses environment variable with fallback to production
const API_BASE_URL = import.meta.env.VITE_API_URL || 'https://18.218.154.66.nip.io/api';

export const fetchSystemOverview = async (filters = {}) => {
    try {
        const params = new URLSearchParams();
        if (filters.startDate) params.append('start_date', filters.startDate);
        if (filters.endDate) params.append('end_date', filters.endDate);
        if (filters.daysOfWeek && filters.daysOfWeek.length > 0) params.append('days_of_week', filters.daysOfWeek.join(','));

        const response = await axios.get(`${API_BASE_URL}/system-overview?${params.toString()}`);

        // Validate response structure
        try {
            validateSystemOverview(response.data);
        } catch (validationError) {
            console.error("Response validation failed:", validationError);
            // Continue anyway but log the issue
        }

        return response.data;
    } catch (error) {
        console.error("Error fetching system overview:", error);
        if (error.response) {
            // Server responded with error status
            throw new Error(`Server error: ${error.response.status} - ${error.response.data?.detail || 'Unknown error'}`);
        } else if (error.request) {
            // Request made but no response
            throw new Error('No response from server. Please check if the backend is running.');
        } else {
            // Something else went wrong
            throw new Error(`Request failed: ${error.message}`);
        }
    }
};

export const fetchMapData = async () => {
    const response = await axios.get(`${API_BASE_URL}/map-data`);
    return response.data;
};

export const fetchRoutes = async (topN = 50) => {
    const response = await axios.get(`${API_BASE_URL}/routes?top_n=${topN}`);
    return response.data;
};

export const fetchStations = async () => {
    const response = await axios.get(`${API_BASE_URL}/stations`);
    return response.data;
};

export const fetchStationDetails = async (stationName) => {
    try {
        const response = await axios.get(`${API_BASE_URL}/station/${encodeURIComponent(stationName)}`);

        // Validate response
        try {
            validateStationDetails(response.data);
        } catch (validationError) {
            console.error("Station details validation failed:", validationError);
        }

        return response.data;
    } catch (error) {
        console.error("Error fetching station details:", error);
        if (error.response?.status === 404) {
            throw new Error(`Station "${stationName}" not found`);
        }
        throw new Error(`Failed to fetch station details: ${error.message}`);
    }
};

export const fetchPrediction = async (stationName, startDate, endDate) => {
    const response = await axios.post(`${API_BASE_URL}/predict`, {
        station_name: stationName,
        start_date: startDate,
        end_date: endDate
    });
    return response.data;
};
