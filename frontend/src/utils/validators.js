// API Response Validators
// Provides runtime type checking for API responses to catch bugs early

export const validateSystemOverview = (data) => {
    if (!data || typeof data !== 'object') {
        throw new Error('Invalid response: expected object');
    }

    // Check required top-level fields
    const required = ['kpis', 'time_series', 'trend', 'anomalies', 'heatmap'];
    for (const field of required) {
        if (!(field in data)) {
            console.warn(`Missing field in system overview response: ${field}`);
        }
    }

    // Validate KPIs structure
    if (data.kpis && typeof data.kpis === 'object') {
        const kpiFields = ['total_trips', 'weekend_share', 'peak_hour', 'weekend_effect'];
        for (const field of kpiFields) {
            if (!(field in data.kpis)) {
                console.warn(`Missing KPI field: ${field}`);
            }
        }
    }

    // Validate arrays
    if (data.time_series && !Array.isArray(data.time_series)) {
        throw new Error('time_series should be an array');
    }

    return true;
};

export const validateStationDetails = (data) => {
    if (!data || typeof data !== 'object') {
        throw new Error('Invalid response: expected object');
    }

    if (!data.kpis || typeof data.kpis !== 'object') {
        throw new Error('Missing or invalid kpis object');
    }

    if (!Array.isArray(data.hourly_profile)) {
        throw new Error('hourly_profile should be an array');
    }

    if (!Array.isArray(data.daily_profile)) {
        throw new Error('daily_profile should be an array');
    }

    return true;
};

export const validatePredictions = (data) => {
    if (!data || typeof data !== 'object') {
        throw new Error('Invalid response: expected object');
    }

    if (!data.predictions || !Array.isArray(data.predictions)) {
        throw new Error('predictions should be an array');
    }

    // Validate first prediction structure if exists
    if (data.predictions.length > 0) {
        const first = data.predictions[0];
        if (!first.date || !('predicted' in first)) {
            throw new Error('Invalid prediction structure: missing date or predicted');
        }
    }

    return true;
};

export const safeParseJSON = (response) => {
    try {
        return typeof response === 'string' ? JSON.parse(response) : response;
    } catch (e) {
        console.error('JSON parse error:', e);
        throw new Error('Failed to parse response as JSON');
    }
};
