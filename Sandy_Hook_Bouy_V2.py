#!/usr/bin/env python3
"""
Sandy Hook Live Ocean Summary Monitor
Fetches data from NDBC 44065 (wave data) and CO-OPS 8531680 (tide data)
"""

import requests
import json
import time
from datetime import datetime, timedelta, timezone
import os
import signal
import sys
from typing import Optional, Dict, List, Tuple, Any
import numpy as np

# Station IDs
NDBC_STATION = "44065"  # Sandy Hook wave buoy
COOPS_STATION = "8531680"  # Sandy Hook tide gauge

# Output directory
OUTPUT_DIR = "./sandy_hook_output"


class GracefulExit:
    """Handle graceful shutdown on KeyboardInterrupt"""
    def __init__(self):
        self.shutdown = False
        signal.signal(signal.SIGINT, self.exit_gracefully)
        signal.signal(signal.SIGTERM, self.exit_gracefully)
    
    def exit_gracefully(self, signum, frame):
        print("\n\nKeyboardInterrupt received — exiting.")
        self.shutdown = True
        sys.exit(0)


def ensure_output_dir():
    """Create output directory if it doesn't exist"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def fetch_ndbc_realtime(station_id: str) -> Optional[Dict]:
    """Fetch real-time data from NDBC station"""
    try:
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # Parse header
        header = lines[0].split()
        # Parse latest data line
        data_line = lines[-1].split()
        
        if len(data_line) < len(header):
            return None
        
        data = dict(zip(header, data_line))
        return data
    except Exception as e:
        print(f"Error fetching NDBC data: {e}", file=sys.stderr)
        return None


def fetch_ndbc_historical(station_id: str, hours: int = 6) -> Optional[List[Dict]]:
    """Fetch historical data from NDBC for specified hours"""
    try:
        # NDBC provides data in various formats, try the standard realtime format
        url = f"https://www.ndbc.noaa.gov/data/realtime2/{station_id}.txt"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        
        lines = response.text.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # NDBC format: first line is header, second line is units (skip), then data
        header_line = lines[0]
        # Skip units line if present
        data_start = 1
        if len(lines) > 1 and any(unit in lines[1].lower() for unit in ['deg', 'm', 's', 'hpa']):
            data_start = 2
        
        header = header_line.split()
        data_points = []
        
        # Parse all data lines
        for line in lines[data_start:]:
            if not line.strip() or line.strip().startswith('#'):
                continue
            parts = line.split()
            # NDBC data lines have date/time first, then measurements
            # Match parts to header, handling variable spacing
            if len(parts) >= len(header):
                # Create dict, handling cases where we have more parts than headers
                data_dict = {}
                for i, key in enumerate(header):
                    if i < len(parts):
                        data_dict[key] = parts[i]
                    else:
                        data_dict[key] = 'MM'  # Missing
                data_points.append(data_dict)
        
        # NDBC realtime data typically has ~45 recent measurements
        # Filter to approximate hours (assuming ~1 measurement per hour, but often more frequent)
        # For 6 hours, we might get 6-12 data points depending on station
        return data_points[-hours*2:] if len(data_points) > hours*2 else data_points
    except Exception as e:
        print(f"Error fetching NDBC historical data: {e}", file=sys.stderr)
        return None


def fetch_coops_tide_data(station_id: str, hours: int = 6) -> Optional[List[Dict]]:
    """Fetch tide data from CO-OPS API"""
    try:
        end_time = datetime.now(timezone.utc)
        start_time = end_time - timedelta(hours=hours)
        
        url = "https://api.tidesandcurrents.noaa.gov/api/prod/datagetter"
        params = {
            "product": "water_level",
            "application": "NOS.COOPS.TAC.WL",
            "begin_date": start_time.strftime("%Y%m%d %H:%M"),
            "end_date": end_time.strftime("%Y%m%d %H:%M"),
            "datum": "MLLW",
            "station": station_id,
            "time_zone": "gmt",
            "units": "metric",
            "format": "json"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if 'data' in data:
            return data['data']
        return None
    except Exception as e:
        print(f"Error fetching CO-OPS tide data: {e}", file=sys.stderr)
        return None


def parse_float(value: str) -> Optional[float]:
    """Safely parse float, handling NDBC 'MM' (missing) values"""
    if not value or value in ['MM', '']:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def calculate_wave_stats(data_points: List[Dict], hours: int) -> Tuple[Optional[float], Optional[float]]:
    """Calculate average and max wave height from data points"""
    if not data_points:
        return None, None
    
    wave_heights = []
    for point in data_points:
        # NDBC uses 'WVHT' for wave height
        wvht = parse_float(point.get('WVHT', point.get('wvht', '')))
        if wvht is not None:
            wave_heights.append(wvht)
    
    if not wave_heights:
        return None, None
    
    avg = np.mean(wave_heights)
    max_val = np.max(wave_heights)
    return avg, max_val


def calculate_dominant_period(data_points: List[Dict]) -> Optional[float]:
    """Calculate dominant wave period"""
    if not data_points:
        return None
    
    periods = []
    for point in data_points:
        # NDBC uses 'DPD' for dominant period
        dpd = parse_float(point.get('DPD', point.get('dpd', '')))
        if dpd is not None:
            periods.append(dpd)
    
    if not periods:
        return None
    
    return np.mean(periods)


def calculate_mean_direction(data_points: List[Dict]) -> Optional[float]:
    """Calculate mean wave direction"""
    if not data_points:
        return None
    
    directions = []
    for point in data_points:
        # NDBC uses 'MWD' for mean wave direction
        mwd = parse_float(point.get('MWD', point.get('mwd', '')))
        if mwd is not None:
            directions.append(mwd)
    
    if not directions:
        return None
    
    return np.mean(directions)


def calculate_pressure_change(data_points: List[Dict], hours: int = 6) -> Optional[float]:
    """Calculate pressure change over specified hours"""
    if not data_points or len(data_points) < 2:
        return None
    
    # Get first and last valid pressure readings
    pressures = []
    for point in data_points:
        pres = parse_float(point.get('PRES', point.get('pres', '')))
        if pres is not None:
            pressures.append(pres)
    
    if len(pressures) < 2:
        return None
    
    return pressures[-1] - pressures[0]


def calculate_tide_range(tide_data: List[Dict]) -> Optional[float]:
    """Calculate tide range from CO-OPS data"""
    if not tide_data:
        return None
    
    water_levels = []
    for point in tide_data:
        try:
            level = float(point.get('v', 0))
            water_levels.append(level)
        except (ValueError, TypeError):
            continue
    
    if not water_levels:
        return None
    
    return max(water_levels) - min(water_levels)


def detect_anomaly(data_points: List[Dict], threshold: float = 0.20) -> bool:
    """Detect if there's a rapid change (>threshold) in recent data"""
    if not data_points or len(data_points) < 2:
        return False
    
    # Get recent wave heights
    wave_heights = []
    for point in data_points[-6:]:  # Last 6 data points
        wvht = parse_float(point.get('WVHT', point.get('wvht', '')))
        if wvht is not None:
            wave_heights.append(wvht)
    
    if len(wave_heights) < 2:
        return False
    
    # Calculate percentage change
    recent_avg = np.mean(wave_heights[-3:]) if len(wave_heights) >= 3 else wave_heights[-1]
    previous_avg = np.mean(wave_heights[:-3]) if len(wave_heights) >= 6 else wave_heights[0]
    
    if previous_avg == 0:
        return False
    
    change = abs((recent_avg - previous_avg) / previous_avg)
    return change > threshold


def format_value(value: Optional[float], unit: str = "", decimals: int = 2) -> str:
    """Format a value, handling None/nan"""
    if value is None or np.isnan(value):
        return "nan"
    return f"{value:.{decimals}f} {unit}".strip()


def print_summary(ndbc_data_1h: List[Dict], ndbc_data_6h: List[Dict], 
                  tide_data: List[Dict], retrieval_time: datetime):
    """Print the summary report"""
    print("\n" + "=" * 60)
    print(f"Sandy Hook Live Ocean Summary  (NDBC {NDBC_STATION}  |  CO-OPS {COOPS_STATION})")
    print(f"\nRetrieval UTC: {retrieval_time.strftime('%Y-%m-%d %H:%M:%SZ')}")
    print("=" * 60)
    
    # Calculate statistics
    avg_wvht_1h, max_wvht_1h = calculate_wave_stats(ndbc_data_1h, 1)
    avg_wvht_6h, max_wvht_6h = calculate_wave_stats(ndbc_data_6h, 6)
    dom_period = calculate_dominant_period(ndbc_data_6h)
    mean_dir = calculate_mean_direction(ndbc_data_6h)
    pressure_change = calculate_pressure_change(ndbc_data_6h, 6)
    tide_range = calculate_tide_range(tide_data)
    anomaly = detect_anomaly(ndbc_data_6h, 0.20)
    
    # Print formatted output
    print(f"\nAvg WVHT 1h:   {format_value(avg_wvht_1h, 'm', 2)}   | Max WVHT 1h: {format_value(max_wvht_1h, 'm', 2)}")
    print(f"Avg WVHT 6h:   {format_value(avg_wvht_6h, 'm', 2)}   | Max WVHT 6h: {format_value(max_wvht_6h, 'm', 2)}")
    print(f"Dom Period:    {format_value(dom_period, 's', 1)}   | Mean Dir: {format_value(mean_dir, '° (true)', 0)}")
    print(f"Wind–Wave r:   {format_value(None)}")  # Placeholder for wind-wave relationship
    print(f"\nΔPressure 6h:  {format_value(pressure_change, 'hPa', 2)}")
    print(f"Tide range 6h: {format_value(tide_range, 'm', 2)}")
    print(f"\nAnomaly >20%?  {'YES' if anomaly else 'NO'}")
    
    print("\n----")
    if anomaly:
        print("Interpretation: Rapid change (>20%) detected within last hour—monitor for short-term hazards.", end="")
    else:
        print("Interpretation: No significant rapid changes detected.", end="")
    
    if tide_range is not None:
        print(f" 6-hr tide range ~{tide_range:.2f} m at Sandy Hook.")
    else:
        print(" Tide data unavailable.")
    
    print()


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-serializable objects to JSON-compatible types"""
    # Handle numpy types
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        if np.isnan(obj):
            return None
        return float(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    # Handle nested structures
    elif isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    elif obj is None:
        return None
    else:
        # For any other type, try to convert to string as fallback
        try:
            json.dumps(obj)  # Test if it's already serializable
            return obj
        except (TypeError, ValueError):
            return str(obj)


def save_output(ndbc_data_1h: List[Dict], ndbc_data_6h: List[Dict], 
                tide_data: List[Dict], retrieval_time: datetime):
    """Save output to file"""
    ensure_output_dir()
    timestamp = retrieval_time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(OUTPUT_DIR, f"sandy_hook_summary_{timestamp}.json")
    
    output = {
        "retrieval_time": retrieval_time.isoformat(),
        "ndbc_station": NDBC_STATION,
        "coops_station": COOPS_STATION,
        "ndbc_data_1h": ndbc_data_1h,
        "ndbc_data_6h": ndbc_data_6h,
        "tide_data": tide_data,
        "statistics": {
            "avg_wvht_1h": calculate_wave_stats(ndbc_data_1h, 1)[0],
            "max_wvht_1h": calculate_wave_stats(ndbc_data_1h, 1)[1],
            "avg_wvht_6h": calculate_wave_stats(ndbc_data_6h, 6)[0],
            "max_wvht_6h": calculate_wave_stats(ndbc_data_6h, 6)[1],
            "dom_period": calculate_dominant_period(ndbc_data_6h),
            "mean_dir": calculate_mean_direction(ndbc_data_6h),
            "pressure_change_6h": calculate_pressure_change(ndbc_data_6h, 6),
            "tide_range_6h": calculate_tide_range(tide_data),
            "anomaly_detected": detect_anomaly(ndbc_data_6h, 0.20)
        }
    }
    
    try:
        # Convert numpy types to JSON-serializable types
        output_serializable = convert_to_json_serializable(output)
        with open(filename, 'w') as f:
            json.dump(output_serializable, f, indent=2)
    except Exception as e:
        print(f"Error saving output: {e}", file=sys.stderr)


def main():
    """Main monitoring loop"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Sandy Hook Ocean Monitoring')
    parser.add_argument('--once', action='store_true', 
                       help='Run once and exit (for testing)')
    parser.add_argument('--interval', type=int, default=3600,
                       help='Update interval in seconds (default: 3600 = 1 hour)')
    # Use parse_known_args to ignore Jupyter/Colab kernel arguments
    args, unknown = parser.parse_known_args()
    
    exit_handler = GracefulExit()
    
    print("Starting Sandy Hook monitor. Output dir: " + OUTPUT_DIR)
    
    try:
        while not exit_handler.shutdown:
            retrieval_time = datetime.now(timezone.utc)
            
            # Fetch data
            ndbc_data_6h = fetch_ndbc_historical(NDBC_STATION, 6)
            ndbc_data_1h = ndbc_data_6h[-1:] if ndbc_data_6h else []
            tide_data = fetch_coops_tide_data(COOPS_STATION, 6)
            
            # Print summary
            print_summary(ndbc_data_1h or [], ndbc_data_6h or [], tide_data or [], retrieval_time)
            
            # Save output
            save_output(ndbc_data_1h or [], ndbc_data_6h or [], tide_data or [], retrieval_time)
            
            # Exit if --once flag is set
            if args.once:
                break
            
            # Wait before next update
            if not exit_handler.shutdown:
                time.sleep(args.interval)
                
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt received — exiting.")
    except Exception as e:
        print(f"\nError in main loop: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

