import requests
import time
import logging

logger = logging.getLogger(__name__)

# Constants
GBFS_STATION_INFO_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_information.json"
GBFS_STATION_STATUS_URL = "https://gbfs.citibikenyc.com/gbfs/en/station_status.json"
CACHE_TTL = 60  # seconds

class GBFSService:
    def __init__(self):
        self._cache = None
        self._last_updated = 0

    def _fetch_data(self, url):
        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch GBFS data from {url}: {e}")
            return None

    def get_live_stations(self):
        # Check cache
        if self._cache and (time.time() - self._last_updated < CACHE_TTL):
            return self._cache

        # Fetch feeds
        info_data = self._fetch_data(GBFS_STATION_INFO_URL)
        status_data = self._fetch_data(GBFS_STATION_STATUS_URL)

        if not info_data or not status_data:
            logger.warning("One or more GBFS feeds failed to load.")
            # Return cached stale data if available, else empty list
            return self._cache if self._cache else []

        # Process and Merge
        try:
            stations = {s['station_id']: s for s in info_data['data']['stations']}
            statuses = {s['station_id']: s for s in status_data['data']['stations']}

            merged_data = []
            for sid, info in stations.items():
                status = statuses.get(sid)
                if status:
                    # Calculate Fill Rate
                    capacity = info.get('capacity', 0)
                    bikes = status.get('num_bikes_available', 0)
                    ebikes = status.get('num_ebikes_available', 0)
                    docks = status.get('num_docks_available', 0)
                    
                    fill_rate = 0
                    if capacity > 0:
                        fill_rate = (bikes + ebikes) / capacity

                    merged_data.append({
                        "station_id": sid,
                        "name": info.get('name', 'Unknown Station'),
                        "lat": info.get('lat'),
                        "lon": info.get('lon'),
                        "capacity": capacity,
                        "bikes_available": bikes,
                        "ebikes_available": ebikes,
                        "docks_available": docks,
                        "is_renting": status.get('is_renting', 0) == 1,
                        "is_returning": status.get('is_returning', 0) == 1,
                        "last_reported": status.get('last_reported'),
                        "fill_rate": fill_rate
                    })
            
            self._cache = merged_data
            self._last_updated = time.time()
            logger.info(f"Updated GBFS cache with {len(merged_data)} stations.")
            return merged_data

        except Exception as e:
            logger.error(f"Error merging GBFS data: {e}")
            return self._cache if self._cache else []

# Global instance
gbfs_service = GBFSService()
