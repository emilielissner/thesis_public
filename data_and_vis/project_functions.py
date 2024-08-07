# --------------------
# Imports
# --------------------
import numpy as np
from pyproj import CRS, Transformer
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from datetime import datetime, timedelta
import os
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.lines as mlines
import h5py
import logging
from PIL import Image
from tqdm import tqdm
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error

# --------------------
# Data Processing Functions
# --------------------

def generate_file_paths(start_date, end_date, data_dir):
    """
    Function for generating a list of file paths for the radar data between two dates for all full-range scans.
    """
    # Generate a list of dates between start_date and end_date
    start = datetime.strptime(start_date, '%Y%m%d')
    end = datetime.strptime(end_date, '%Y%m%d')
    date_list = [start + timedelta(days=x) for x in range((end-start).days + 1)]

    file_paths = []
    for date in date_list:
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        for hour in range(24):
            for minute in range(0, 60, 10):  # Assuming every 10 minutes
                time = f'{hour:02d}{minute:02d}'
                file_path = f"{data_dir}/{year}/{year}/{month}/{day}/dk.com.{year}{month}{day}{time}.500_max.h5"
                if os.path.exists(file_path):  # Check if the file actually exists
                    file_paths.append(file_path)
    return file_paths

def extract_timestamp_from_path(file_path):
    """
    Extracts a timestamp from a radar file path and formats it as "YYYY-MM-DD HH:MM:00"
    """
    # Split the file path to isolate the date and time part
    parts = file_path.split('/')
    date_part = parts[-2]  
    year_month_day_time = parts[-1].split('.')[2] 
    
    # Extract year, month, day, and time components
    year = year_month_day_time[:4]
    month = year_month_day_time[4:6]
    day = year_month_day_time[6:8]
    hour = year_month_day_time[8:10]
    minute = year_month_day_time[10:]
    
    # Combine into the desired format
    timestamp = f'{year}-{month}-{day} {hour}:{minute}:00'
    return timestamp

# --------------------
# Functions for handling image data
# --------------------

def normalize_image_values(image):
    """
    Normalizes the values in a NumPy array (image) to the range [0, 1], excluding NaN values.
    """
    min_value = np.nanmin(image)
    max_value = np.nanmax(image)
    
    # Ensure max_value is greater than min_value to avoid division by zero
    if max_value > min_value:
        normalized_image = (image - min_value) / (max_value - min_value)
    else:
        normalized_image = np.zeros_like(image)
    return normalized_image

def crop_image(image, row, col, crop_size = 63):
    """
    Function for cropping an image around a gauge's location
    """
    half_crop_size = crop_size // 2
    
    # Calculate the start and end indices for the crop
    row_start = max(0, row - half_crop_size)
    col_start = max(0, col - half_crop_size)
    
    # Adjust end indices considering the crop size, handle edge cases
    row_end = min(image.shape[0], row_start + crop_size)
    col_end = min(image.shape[1], col_start + crop_size)
    
    # Adjust start indices if the crop goes beyond the image's boundaries
    row_start = max(0, row_end - crop_size)
    col_start = max(0, col_end - crop_size)

    # Crop the image
    cropped_image = image[row_start:row_end, col_start:col_end]
    
    # If the cropped image is not of the desired size due to being near an edge, pad it
    if cropped_image.shape[0] < crop_size or cropped_image.shape[1] < crop_size:
        padding_row = crop_size - cropped_image.shape[0]
        padding_col = crop_size - cropped_image.shape[1]
        cropped_image = np.pad(cropped_image, ((0, padding_row), (0, padding_col)), 'constant', constant_values=0)
    
    return row_start, row_start + crop_size, col_start, col_start + crop_size, cropped_image

# --------------------
# Functions for working with spatial data
# --------------------
def haversine(lon1, lat1, lon2, lat2):
    """
    Function for computing the distance between two points on the Earth's surface using the Haversine formula
    (used for computing distances between gauges and radar stations)
    """
    # Radius of the Earth in km
    R = 6371.0
    # Convert degrees to radians
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])
    # Differences in coordinates
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    # Haversine formula
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distance = R * c
    return distance

def transform_coordinates_from_EUREF89_to_WGS84(easting, northing):
    """
    Function for transforming coordinates from northings/eastings 
    (used for connecting SVK data (northings/eastings) to lat/lon data)
    """
    transformer = Transformer.from_crs('epsg:25832', 'epsg:4326', always_xy=True)
    lon, lat = transformer.transform(easting, northing)
    return lon, lat

def transform_raster_grid_coordinates(x_coords, y_coords, orig_crs, dest_crs):
    """
    Transforms a raster grid's coordinates from an original to a destination coordinate reference system (CRS).
    (used for construction a grid of coordinates for the radar data)
    """
    xx, yy = np.meshgrid(x_coords, y_coords)
    transformer = Transformer.from_crs(orig_crs, dest_crs, always_xy = True)
    new_coords = transformer.transform(xx, yy)
    x_new, y_new = new_coords[0], new_coords[1]
    return(x_new, y_new)

def calculate_midpoints(coords):
    """
    Function for calculating the midpoints of a grid of coordinates
    (used for determining the pixel a gague is located in)
    """
    # Average with the adjacent (right and bottom) coordinates to find midpoints
    shifted_right = np.roll(coords, shift=-1, axis=1)
    shifted_down = np.roll(coords, shift=-1, axis=0)
    
    # Calculate the average of the current and shifted arrays
    midpoints = (coords + shifted_right + shifted_down + np.roll(shifted_down, shift=-1, axis=1)) / 4
    
    # Exclude the last row and column which don't form complete squares
    return midpoints[:-1, :-1]

# --------------------
# Functions for generating features
# --------------------

def toy(date_input):
    days_in_year = 365.25  # Average to account for leap years
    date = pd.to_datetime(date_input)
    day_of_year = date.day_of_year
    sin_time_of_year = np.sin(2 * np.pi * day_of_year / days_in_year)
    cos_time_of_year = np.cos(2 * np.pi * day_of_year / days_in_year)
    return sin_time_of_year, cos_time_of_year

def tod(time_input):
    time = pd.to_datetime(time_input)
    seconds_since_midnight = (time - time.normalize()).seconds
    seconds_in_a_day = 24 * 60 * 60
    fraction_of_day = seconds_since_midnight / seconds_in_a_day
    sin_time_of_day = np.sin(2 * np.pi * fraction_of_day)
    cos_time_of_day = np.cos(2 * np.pi * fraction_of_day)
    return sin_time_of_day, cos_time_of_day
# --------------------
# Function for splitting into test and train
# --------------------

def split_test_train_using_Kmeans(df, test_size_per_cluster=1, n_clusters = 10, random_state=123):
    # Create a deep copy of the DataFrame to ensure we're not modifying the original unintentionally
    df_copy = df.copy()
    
    # Apply K-Means Clustering based on Longitude and Latitude
    X = df_copy[['Longitude', 'Latitude']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=1)
    # Use .loc to ensure the operation is done directly on the DataFrame
    df_copy.loc[:, 'Cluster'] = kmeans.fit_predict(X)
    
    np.random.seed(random_state)
    
    test_station_ids = []
    for cluster in df_copy['Cluster'].unique():
        cluster_station_ids = df_copy[df_copy['Cluster'] == cluster]['stationId'].values
        test_station_ids.extend(np.random.choice(cluster_station_ids, size=test_size_per_cluster, replace=False))
    
    # Convert to list for test and train station IDs
    test_station_ids_list = list(test_station_ids)
    train_station_ids_list = [stationId for stationId in df_copy['stationId'] if stationId not in test_station_ids_list]
    
    return train_station_ids_list, test_station_ids_list

def split_train_val_test_using_Kmeans(df, n_clusters=10, random_state=3):
    # Create a deep copy of the DataFrame to ensure we're not modifying the original unintentionally
    df_copy = df.copy()
    
    # Apply K-Means Clustering based on Longitude and Latitude
    X = df_copy[['Longitude', 'Latitude']]
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    df_copy['Cluster'] = kmeans.fit_predict(X)
    
    np.random.seed(random_state)
    
    train_station_ids = []
    val_station_ids = []
    test_station_ids = []
    
    # Determine the number of stations for validation and test sets (10% each)
    for cluster in df_copy['Cluster'].unique():
        cluster_station_ids = df_copy[df_copy['Cluster'] == cluster]['stationId'].values
        total_stations = len(cluster_station_ids)
        
        val_test_size = max(1, int(0.1 * total_stations)) # Ensure at least one station per cluster for val and test
        
        # Randomly select validation and test stations
        val_test_station_ids = np.random.choice(cluster_station_ids, size=2 * val_test_size, replace=False)
        val_station_ids.extend(val_test_station_ids[:val_test_size])
        test_station_ids.extend(val_test_station_ids[val_test_size:])
        
        # The remaining stations in the cluster are used for training
        train_station_ids.extend([stationId for stationId in cluster_station_ids if stationId not in val_test_station_ids])
    
    return train_station_ids, val_station_ids, test_station_ids

# --------------------
# Radar functions from Jonas
# --------------------

def load_radar_image(file_path):
    """
    Loads radar image data from an HDF5 file, returning the data as a float array and a success flag.
    """
    try:
        with h5py.File(file_path, 'r') as f:
            raw_data = f['dataset1/data1/data'][()]
            raw_data_array = raw_data[()]
        raw_data_array = raw_data_array.astype(float)
        return raw_data_array, True
    except OSError as e:
        #logging.error(f"Failed to open file {file_path}: {e}")
        return None, False
    
NO_RAIN_DBZ = -50  # Adjust this value to the radar's specific "no rain detected" threshold

def raw_radar_data_to_dbz(raw_data_array, no_rain_dbz=NO_RAIN_DBZ):
    """
    Converts raw radar data to dBZ values, treating raw values representing 'no rain detected' appropriately.
    """
    # Values representing 'no rain detected' will be set to no_rain_dbz
    raw_data_array = np.array(raw_data_array, ndmin=1)
    raw_data_array[raw_data_array == 255] = np.nan  
    gain = 0.5
    offset = -32
    dbz_data = offset + gain * raw_data_array  # Convert to dBZ
    dbz_data[raw_data_array == 0] = no_rain_dbz  # Set 'no rain detected' values
    return dbz_data

def dbz_to_Z(dbz_data, no_rain_dbz=NO_RAIN_DBZ):
    """
    Converts dBZ to linear reflectivity factor Z, handling 'no rain detected' values.
    """
    dbz_data = np.array(dbz_data)
    z_data = np.power(10, dbz_data / 10)  # Convert dBZ to Z
    z_data[dbz_data <= no_rain_dbz] = 0  # Set 'no rain detected' dBZ values to 0 in Z
    return z_data

def dbz_to_R_marshall_palmer(dbz_data, a = 200, b = 1.6):
    """
    Computes rainfall intensity from dBZ using the Marshall-Palmer relation.
    """
    # Computes intensity from dBZ using the Marshall-Palmer relation
    z_data = dbz_to_Z(dbz_data)
    rain_data = np.power(z_data/a, 1/b)
    return(rain_data) 

# --------------------
# Functions for sampling data
# --------------------
def generate_train_samples(st_ID, df, station_df, data_dir, dist_coast, dist_radar, crop_size, base_path, total_samples=5000, group_fractions={'A': 1/3, 'B': 1/3, 'C': 1/3}):
    """
    Generates samples for model training based on three groups: A, B, and C, allowing for adjustable fractions of samples per group.
    """
    samples_per_group = {group: int(total_samples * fraction) for group, fraction in group_fractions.items()}
    group_counters = {'A': 0, 'B': 0, 'C': 0}
    train_samples = []
    start_date_dt, end_date_dt = pd.to_datetime('20230101'), pd.to_datetime('20231231')
    filtered_dates = df.index[(df.index >= start_date_dt) & (df.index <= end_date_dt)]
    N_samples = 0
    
    with tqdm(total=total_samples, desc="Generating samples") as pbar:
        while any(count < quota for count, quota in zip(group_counters.values(), samples_per_group.values())):
            station = np.random.choice(st_ID)
            timestamp = np.random.choice(filtered_dates)
            timestamp = pd.to_datetime(timestamp)
            rain_gauge_value = df.loc[timestamp, str(station)]

            year, month, day, hour, minute = timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute
            file_path = f"{data_dir}/{year}/{year}/{month:02d}/{day:02d}/dk.com.{year}{month:02d}{day:02d}{hour:02d}{minute:02d}.500_max.h5"
            
            # Simulate the opening and processing of radar image
            radar_image, success = load_radar_image(file_path) 
            if not success:
                continue
            radar_norm = radar_image/255

            # Cropping the radar image and spatial features
            filtered_df = station_df[station_df['stationId'] == station]
            r_idx, c_idx = filtered_df['radar_pixel_row'].iloc[0], filtered_df['radar_pixel_col'].iloc[0]
            _, _, _, _, cropped_radar_image_raw = crop_image(radar_image, r_idx, c_idx, crop_size)
            #_, _, _, _, cropped_radar_image_dbz = crop_image(dbz_data, r_idx, c_idx, crop_size)
            _, _, _, _, cropped_coast_dist = crop_image(dist_coast, r_idx, c_idx, crop_size)
            _, _, _, _, cropped_radar_dist = crop_image(dist_radar, r_idx, c_idx, crop_size)
            
            dbz_data = raw_radar_data_to_dbz(radar_image[r_idx, c_idx])
            rain_data = dbz_to_R_marshall_palmer(dbz_data)
            
            #print(dbz_data, rain_data)

            if np.isnan(dbz_data):
                continue

            if rain_gauge_value > 0 and group_counters['A'] < samples_per_group['A']:
                group = 'A'
            elif rain_gauge_value == 0 and np.all(cropped_radar_image_raw == 0) and group_counters['B'] < samples_per_group['B']:
                group = 'B'
            elif rain_gauge_value == 0 and not np.all(cropped_radar_image_raw == 0) and group_counters['C'] < samples_per_group['C']:
                group = 'C'
            else:
                continue  # Skip sample if none of the conditions are met or group quota is filled
            
            # Save sample image
            image_path = f"{base_path}/data/data_for_model/images/train/{N_samples}.tiff"
            Image.fromarray(cropped_radar_image_raw.astype(np.uint8)).save(image_path)
            
            # Save details in a dataframe
            N_samples += 1
            sample_dict = {
                'N': N_samples, 'timestamp': timestamp, 'stationId': station, 'samplingGroup': group,
                'image_path': image_path, 'rain_gauge_value': rain_gauge_value,
                'radar_value_dbz': dbz_data[0] , 'dist_radar': dist_radar[r_idx, c_idx],
                'dist_coast': dist_coast[r_idx, c_idx], 'MP': rain_data[0], 'toy': sinusoidal_toy(timestamp),
                'tod': sinusoidal_tod(timestamp),  'radar_pixel_norm' : radar_norm[r_idx, c_idx]
            }
            train_samples.append(sample_dict)
            group_counters[group] += 1
            pbar.update(1) 
        
    return pd.DataFrame(train_samples)

def generate_test_samples(stations, df, station_df, data_dir, dist_coast, dist_radar, crop_size, base_path, start_date='20231001', end_date='20231031'):
    """
    Generates time series test samples for a list of stations within a specified date range.
    """
    test_samples = []
    start_date_dt, end_date_dt = pd.to_datetime(start_date), pd.to_datetime(end_date)
    filtered_dates = df.index[(df.index >= start_date_dt) & (df.index <= end_date_dt)]
    N_samples = 0
    
    for station in tqdm(stations):
        for timestamp in filtered_dates:
            rain_gauge_value = df.loc[timestamp, str(station)]
            year, month, day, hour, minute = timestamp.year, timestamp.month, timestamp.day, timestamp.hour, timestamp.minute
            file_path = f"{data_dir}/{year}/{year}/{month:02d}/{day:02d}/dk.com.{year}{month:02d}{day:02d}{hour:02d}{minute:02d}.500_max.h5"
            
            # Attempt to open the radar image file
            radar_image, success = load_radar_image(file_path)
            if not success:
                continue
            radar_norm = radar_image/255
           
            # Cropping the radar image and spatial features
            filtered_df = station_df[station_df['stationId'] == station]
            r_idx, c_idx = filtered_df['radar_pixel_row'].iloc[0], filtered_df['radar_pixel_col'].iloc[0]
            _, _, _, _, cropped_radar_image_raw = crop_image(radar_image, r_idx, c_idx, crop_size)
            #_, _, _, _, cropped_radar_image_dbz = crop_image(dbz_data, r_idx, c_idx, crop_size)
            _, _, _, _, cropped_coast_dist = crop_image(dist_coast, r_idx, c_idx, crop_size)
            _, _, _, _, cropped_radar_dist = crop_image(dist_radar, r_idx, c_idx, crop_size)

            # Convert radar data to dBZ and then to rainfall intensity
            dbz_data = raw_radar_data_to_dbz(radar_image[r_idx, c_idx])
            rain_data = dbz_to_R_marshall_palmer(dbz_data) 


            N_samples += 1
            #if N_samples > total_samples:
            #    break  # Stop if the total_samples limit is reached
            
            image_path = f"{base_path}/data/data_for_model/images/test/{N_samples}.tiff"
            Image.fromarray(cropped_radar_image_raw.astype(np.uint8)).save(image_path)
            
            sample_dict = {
                'N': N_samples, 'timestamp': timestamp, 'stationId': station, 'samplingGroup': None,
                'image_path': image_path, 'rain_gauge_value': rain_gauge_value,
                'radar_value_dbz': dbz_data[0] , 'dist_radar': dist_radar[r_idx, c_idx],
                'dist_coast': dist_coast[r_idx, c_idx], 'MP': rain_data[0] , 'toy': sinusoidal_toy(timestamp),
                'tod': sinusoidal_tod(timestamp), 'radar_pixel_norm' : radar_norm[r_idx, c_idx]
            } 
            test_samples.append(sample_dict)
        #if N_samples > total_samples:
        #    break  # Break the outer loop if total_samples limit is reached

    return pd.DataFrame(test_samples)

# --------------------
# Functions for evaluating NN
# --------------------

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score
from scipy.stats import pearsonr

def evaluation_metrics(y_true, y_pred):

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    me = np.mean(y_pred - y_true)
  
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "ME": me,
    }

def generate_test_dataframe(model, test_dataloader):
    """
    Evaluate the model on the provided DataLoader and return a DataFrame with predictions, targets, MP_values, and station IDs.
    """ 
    model.eval()  # Set the model to evaluation mode
    
    # Initialize lists for storing results
    predictions = []
    targets = [] 
    radar_pixel_values = []  
    MP_values = []  
    groups = []  
    station_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating Model'):
            #images = batch['image']
            features = batch['features'][:, 1:]
            MP = batch['MP']
            station_id = batch['stationId']
            batch_targets = batch['target']

            # Process and extend lists with batch data
            batch_radar_pixel_values = features[:, 0].cpu().numpy()  
            radar_pixel_values.extend(batch_radar_pixel_values)
            station_ids.extend(station_id)
            batch_MP_values = MP.cpu().numpy()
            MP_values.extend(batch_MP_values.flatten().tolist())
            targets.extend(batch_targets.cpu().numpy())
    
            # Make predictions
            outputs = model(features)
            batch_predictions = outputs.cpu().numpy().flatten()  
            predictions.extend(batch_predictions)

    # Create DataFrame from the accumulated lists
    df_test = pd.DataFrame({
        'Predictions': predictions,
        'Targets': targets,
        'MP_values': MP_values,
        'Station_IDs': station_ids
    })

    df_test['MP_values'] = df_test['MP_values'].fillna(0)
    
    return df_test


def generate_test_dataframe_MP2(test_dataloader, A, B):
    """
    Evaluate the model on the provided DataLoader and return a DataFrame with predictions, targets, MP_values, and station IDs.
    """ 
    
    # Initialize lists for storing results
    predictions = []
    targets = [] 
    radar_pixel_values = []  
    MP_values = []  
    station_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating Model'):
            features = batch['features']
            MP = batch['MP']
            station_id = batch['stationId']
            batch_targets = batch['target']

            # Process and extend lists with batch data
            batch_radar_pixel_values = features[:, 0].cpu().numpy()  
            radar_pixel_values.extend(batch_radar_pixel_values)
            station_ids.extend(station_id)
            batch_MP_values = MP.cpu().numpy()
            MP_values.extend(batch_MP_values.flatten().tolist())
            targets.extend(batch_targets.cpu().numpy())
    
            # Make predictions
            outputs = dbz_to_R_marshall_palmer(features[:, 0].cpu().numpy(), A, B)
            batch_predictions = outputs.flatten()  
            predictions.extend(batch_predictions)

    # Create DataFrame from the accumulated lists
    df_test = pd.DataFrame({
        'Predictions': predictions,
        'Targets': targets,
        'MP_values': MP_values,
        'Station_IDs': station_ids,
        'Radar_dBZ_values' : radar_pixel_values
    })

    df_test['MP_values'] = df_test['MP_values'].fillna(0)
    
    return df_test

def generate_test_dataframe_MP(model, test_dataloader):
    """
    Evaluate the model on the provided DataLoader and return a DataFrame with A, B estimates,
    computed MP values using the Marshall-Palmer relation, targets, and station IDs.
    """
    model.eval()  # Set the model to evaluation mode
    
    # Initialize lists for storing results
    A_estimates = []  
    B_estimates = [] 
    targets = [] 
    radar_pixel_values = []  
    MP_values = []  
    Predicted_R = []  
    station_ids = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc='Evaluating Model'):
            features = batch['features']
            station_id = batch['stationId']
            batch_targets = batch['target']
            MP = batch['MP'].cpu().numpy()
            images = batch['image']

            Z = features[:, 0].cpu().numpy()
            radar_pixel_values.extend(Z.tolist())  # Ensure conversion to list if not already
            station_ids.extend(station_id.tolist())
            targets.extend(batch_targets.cpu().numpy().tolist())
            MP_values.extend(MP.flatten().tolist())

            A_estimates_batch, B_estimates_batch= model(features[:, 0:], images).unbind(1)
            A_estimates_batch = A_estimates_batch * 200  
            A_estimates.extend(A_estimates_batch.tolist())
            B_estimates.extend(B_estimates_batch.tolist())

            # Compute R using Marshall-Palmer relation with predicted A and B
            # Ensure broadcasting is handled correctly for batch operations
            R_pred = dbz_to_R_marshall_palmer(features[:, 0].cpu().numpy(), np.array(A_estimates_batch), np.array(B_estimates_batch))
            Predicted_R.extend(R_pred.flatten().tolist())

    # Create DataFrame from the accumulated lists
    df_test = pd.DataFrame({
        'A_estimates': A_estimates,
        'B_estimates': B_estimates,
        'Predicted_R': Predicted_R,
        'Targets': targets,
        'Radar_Z_values': radar_pixel_values,
        'Station_IDs': station_ids,
        'MP' : MP_values
    })

    return df_test

def compute_metrics(df):
    """
    Compute and print RMSE and MAE for MP values and model predictions against targets.
    """
       # Ensure station IDs are strings and trimmed
    df['Station_IDs'] = df['Station_IDs'].astype(str).str.strip()
    station_groups = df.groupby('Station_IDs')
    
    for station_id, group in station_groups:
        targets = group['Targets'].to_numpy()
        MP_values = group['MP'].to_numpy()
        predictions = group['Predicted_R'].to_numpy()
        
        # Compute RMSE
        rmse_MP = np.sqrt(mean_squared_error(targets, MP_values))
        rmse_Predictions = np.sqrt(mean_squared_error(targets, predictions))
        
        # Compute MAE
        mae_MP = mean_absolute_error(targets, MP_values)
        mae_Predictions = mean_absolute_error(targets, predictions)
        
        # Print metrics for each stationId
        print(f"Station ID: {station_id}")
        print(f"  RMSE MP: {rmse_MP}")
        print(f"  RMSE model: {rmse_Predictions}")
        print(f"  MAE MP: {mae_MP}")
        print(f"  MAE model: {mae_Predictions}")
        print("-" * 40)
    
    # Compute and print total metrics
    total_targets = df['Targets'].to_numpy()
    total_MP_values = df['MP'].to_numpy()
    total_predictions = df['Predicted_R'].to_numpy()
    
    total_rmse_MP = np.sqrt(mean_squared_error(total_targets, total_MP_values))
    total_rmse_Predictions = np.sqrt(mean_squared_error(total_targets, total_predictions))
    total_mae_MP = mean_absolute_error(total_targets, total_MP_values)
    total_mae_Predictions = mean_absolute_error(total_targets, total_predictions)
    
    print("Total Metrics:")
    print(f"  Total RMSE MP: {total_rmse_MP}")
    print(f"  Total RMSE model: {total_rmse_Predictions}")
    print(f"  Total MAE MP: {total_mae_MP}")
    print(f"  Total MAE model: {total_mae_Predictions}")

def plot_predictions_and_MP_for_station(df, stationID):
    """
    Plot predictions, Marshall Palmer (MP) estimates, and true values for a given stationID.

    Parameters:
    - df: DataFrame containing 'Predictions', 'MP_values', 'Targets', and 'Station_IDs'.
    - stationID: The specific station ID for which to plot the data.
    """
    # Filter the DataFrame for the specified stationID
    df_station = df[df['Station_IDs'] == stationID]

    # Extract data for plotting
    predictions = df_station['Predicted_R'].to_numpy()
    MP_values = df_station['MP'].to_numpy()
    targets = df_station['Targets'].to_numpy()

    idx = np.linspace(500, 700, 201, dtype=int)
    # Start plotting
    plt.figure(figsize=(12, 10))

    # Plot Predictions vs. True Values
    plt.subplot(2, 1, 1)
    plt.plot(predictions[idx]/60, label='Predictions')
    plt.plot(targets[idx], label='True values', linestyle='--')
    plt.legend()
    plt.ylim(-0.05, 0.30)
    plt.title(f'Station ID: {stationID} Predictions and True Values')

    # Plot Marshall Palmer Estimates vs. True Values
    plt.subplot(2, 1, 2)
    plt.plot(MP_values[idx], label='Marshall Palmer')
    plt.plot(targets[idx], label='True values', linestyle='--')
    plt.ylim(-0.05, 0.30)
    plt.legend()
    plt.title(f'Station ID: {stationID} Marshall Palmer Estimates and True Values')

    plt.tight_layout()
    plt.show()

# --------------------
# Functions for plotting spatial data
# --------------------

def plot_radar_pixels_and_gauges(radar_lons, radar_lats, station_df, save_path=None):
    
    """
    Plots the radar grid and the gauge locations. The pixel containing a gauge is marked by a red square.
    """
    plt.figure(figsize=(4, 4))

    # Define latitude and longitude boundaries for Denmark
    llcrnrlat = 55.6      # lower latitude
    urcrnrlat = 55.8      # upper latitude
    llcrnrlon = 12.3      # lower longitude
    urcrnrlon = 12.7      # upper longitude

    # Create a Basemap instance
    m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_ts=0, resolution='h')

    # Draw map details
    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='gainsboro', lake_color='white')

    # Draw parallels and meridians
    parallels = np.arange(54., 59., 1.)
    m.drawparallels(parallels, labels=[True, False, False, False], dashes=[2, 2], linewidth=0.5)
    meridians = np.arange(3., 17., 1.)
    m.drawmeridians(meridians, labels=[False, False, False, True], dashes=[2, 2], linewidth=0.5)

    # Convert lons and lats to x and y for plotting
    x, y = m(radar_lons, radar_lats)

    # Plot Longitude and Latitude Lines
    for col in range(x.shape[1]):
        plt.plot(x[:, col], y[:, col], color='gray', linestyle='-', linewidth=0.5, zorder=1)
    for row in range(y.shape[0]):
        plt.plot(x[row, :], y[row, :], color='gray', linestyle='-', linewidth=0.5, zorder=1)

    for idx, row in station_df.iterrows():
        pixel_row = row['radar_pixel_row']
        pixel_col = row['radar_pixel_col']
        
        # Check if pixel_row and pixel_col are not None (or np.nan in a DataFrame)
        if not np.isnan(pixel_row) and not np.isnan(pixel_col):
            pixel_row = int(pixel_row)  # Convert to integer if necessary
            pixel_col = int(pixel_col)  # Convert to integer if necessary
            
            # Calculate the coordinates of the pixel's corners
            pixel_corners_lon = [radar_lons[pixel_row, pixel_col], radar_lons[pixel_row, pixel_col+1],
                                radar_lons[pixel_row+1, pixel_col+1], radar_lons[pixel_row+1, pixel_col],
                                radar_lons[pixel_row, pixel_col]]
            pixel_corners_lat = [radar_lats[pixel_row, pixel_col], radar_lats[pixel_row, pixel_col+1],
                                radar_lats[pixel_row+1, pixel_col+1], radar_lats[pixel_row+1, pixel_col],
                                radar_lats[pixel_row, pixel_col]]
            
            # Convert pixel corner coordinates to map coordinates
            x_pixel_corners, y_pixel_corners = m(pixel_corners_lon, pixel_corners_lat)
            
            # Plot the pixel as a polygon
            polygon = plt.Polygon(list(zip(x_pixel_corners, y_pixel_corners)), edgecolor='red', facecolor='none', linewidth=2, zorder=5)
            plt.gca().add_patch(polygon)

        # Plot station points
        x, y = m(row['Longitude'], row['Latitude'])
        plt.scatter(x, y, marker='o', color='dodgerblue', zorder=10, s=4)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()

def plot_test_train_split(station_df, radar_df, test_station_ids,  llcrnrlat=None, urcrnrlat=None, llcrnrlon=None, urcrnrlon=None, save_path=None):
    """
    Plots the test, train and excluded gauges along with radar stations
    """
    plt.figure(figsize=(4, 6))

    if llcrnrlat is None:
        llcrnrlat = 54.2
    if urcrnrlat is None:
        urcrnrlat = 58.2
    if llcrnrlon is None:
        llcrnrlon = 7.0
    if urcrnrlon is None:
        urcrnrlon = 15.7

    m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_ts=0, resolution='h')

    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='gainsboro', lake_color='white')

    for idx, row in station_df[station_df['Within_25km_of_Radar'] == False].iterrows():
        x, y = m(row['Longitude'], row['Latitude'])
        color = 'darkorange' if row['stationId'] in test_station_ids else 'dodgerblue'
        size = 5 if row['stationId'] in test_station_ids else 10
        zorder = 10 if row['stationId'] in test_station_ids else 5
        m.scatter(x, y, marker='.', color=color, zorder=zorder, s=size, alpha=0.75)

    for idx, row in station_df[station_df['Within_25km_of_Radar'] == True].iterrows():
        x, y = m(row['Longitude'], row['Latitude'])
        m.scatter(x, y, marker='.', color='black', zorder=10, s=2, alpha=0.75)

    for idx, row in radar_df.iterrows():
        x, y = m(row['Longitude'], row['Latitude'])
        m.scatter(x, y, marker="^", color='black', zorder=5, s=10)

    train_station_marker = mlines.Line2D([], [], color='dodgerblue', marker='o', linestyle='None',
                                         markersize=10, label='Train Gauge')
    radar_marker = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                 markersize=7, label='Radar')
    test_station_marker = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None',
                                        markersize=5, label='Test Gauge')
    excluded_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=2, label='Excluded Gauge')
    
    plt.legend(handles=[train_station_marker, test_station_marker, excluded_marker, radar_marker], 
               loc='upper right', fontsize=6, markerscale=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)

    plt.show()

def plot_test_train_val_split(station_df, radar_df, test_station_ids, val_station_ids,  llcrnrlat=None, urcrnrlat=None, llcrnrlon=None, urcrnrlon=None, save_path=None):
    """
    Plots the test, train, validation, and excluded gauges along with radar stations.
    """
    plt.figure(figsize=(4, 6))

    if llcrnrlat is None:
        llcrnrlat = 54.2
    if urcrnrlat is None:
        urcrnrlat = 58.2
    if llcrnrlon is None:
        llcrnrlon = 7.0
    if urcrnrlon is None:
        urcrnrlon = 15.7

    m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_ts=0, resolution='h')

    m.drawmapboundary(fill_color='white')
    m.fillcontinents(color='gainsboro', lake_color='white')

    # Plot stations that are not within 25km of a radar
    for idx, row in station_df[station_df['Within_25km_of_Radar'] == False].iterrows():
        x, y = m(row['Longitude'], row['Latitude'])
        if row['stationId'] in test_station_ids:
            color, size, zorder = 'darkorange', 5, 10
        elif row['stationId'] in val_station_ids:
            color, size, zorder = 'purple', 5, 10  # Use purple for validation stations
        else:
            color, size, zorder = 'dodgerblue', 10, 5
        m.scatter(x, y, marker='.', color=color, zorder=zorder, s=size, alpha=0.75)

    # Plot stations within 25km of a radar
    for idx, row in station_df[station_df['Within_25km_of_Radar'] == True].iterrows():
        x, y = m(row['Longitude'], row['Latitude'])
        m.scatter(x, y, marker='.', color='black', zorder=10, s=2, alpha=0.75)

    # Plot radar stations
    for idx, row in radar_df.iterrows():
        x, y = m(row['Longitude'], row['Latitude'])
        m.scatter(x, y, marker="^", color='black', zorder=5, s=10)

    # Legend
    train_station_marker = mlines.Line2D([], [], color='dodgerblue', marker='o', linestyle='None',
                                         markersize=10, label='Train Gauge')
    val_station_marker = mlines.Line2D([], [], color='purple', marker='o', linestyle='None',
                                        markersize=5, label='Validation Gauge')  # Add validation marker
    test_station_marker = mlines.Line2D([], [], color='darkorange', marker='o', linestyle='None',
                                        markersize=5, label='Test Gauge')
    excluded_marker = mlines.Line2D([], [], color='black', marker='o', linestyle='None',
                                    markersize=2, label='Excluded Gauge')
    radar_marker = mlines.Line2D([], [], color='black', marker='^', linestyle='None',
                                 markersize=7, label='Radar')

    plt.legend(handles=[train_station_marker, val_station_marker, test_station_marker, excluded_marker, radar_marker], 
               loc='upper right', fontsize=6, markerscale=0.7)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)

    plt.show()
    
def plot_spatial_feature(input_array, title, cmap = 'terrain', save_path=None):
    """ 
    plots pixel data on map
    """
    # Get radar coordinate grid with longitudes and latitudes
    x_radar_coords = np.arange(-421364.8 - 500, 569635.2 + 500, 500) 
    y_radar_coords = np.arange(468631 + 500, -394369 - 500, -500)
    dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56") # raw data CRS projection
    plotting_crs = 'epsg:4326' # the CRS projection you want to plot the data in
    radar_lons, radar_lats = transform_raster_grid_coordinates(x_radar_coords, y_radar_coords, dmi_stere_crs, plotting_crs) 

    plt.figure(figsize=(4, 4))

   # Define latitude and longitude boundaries for Denmark
    llcrnrlat = 54      # lower latitude
    urcrnrlat = 59      # upper latitude
    llcrnrlon = 6       # lower longitude
    urcrnrlon = 17      # upper longitude

    # Create a Basemap instance
    m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat, llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_ts=0, resolution='h')

    # Draw parallels and meridians
    #parallels = np.arange(54., 59., 1.)  
    #m.drawparallels(parallels, labels=[True, False, False, False], dashes=[2,2], linewidth=0.5)  
    #meridians = np.arange(6., 18., 2.)  
    #m.drawmeridians(meridians, labels=[False, False, False, True], dashes=[2,2], linewidth=0.5)

    # Plot the minimum distance data
    x, y = m(radar_lons, radar_lats) 
    colormesh = m.pcolormesh(x, y, input_array, cmap=cmap)

    # Add color bar
    cbar = plt.colorbar(colormesh, shrink=0.665, aspect=12)
    cbar.set_label(title, fontsize=8)
    
    # Add coastline
    m.drawcoastlines(color = "white")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)

    plt.show()

def rain_map(input_array, title, cmap, norm, save_path=None, vmin=None, vmax=None):
    # Get radar coordinate grid with longitudes and latitudes
    x_radar_coords = np.arange(-421364.8 - 500, 569635.2 + 500, 500)
    y_radar_coords = np.arange(468631 + 500, -394369 - 500, -500)
    dmi_stere_crs = CRS("+proj=stere +ellps=WGS84 +lat_0=56 +lon_0=10.5666 +lat_ts=56")  # raw data CRS projection
    plotting_crs = 'epsg:4326'  # the CRS projection you want to plot the data in
    radar_lons, radar_lats = transform_raster_grid_coordinates(x_radar_coords, y_radar_coords, dmi_stere_crs, plotting_crs)

    plt.figure(figsize=(10, 10))

    # Define latitude and longitude boundaries for Denmark
    llcrnrlat = 54.5  # lower latitude
    urcrnrlat = 58  # upper latitude
    llcrnrlon = 8   # lower longitude
    urcrnrlon = 13  # upper longitude

    # Create a Basemap instance
    m = Basemap(projection='merc', llcrnrlat=llcrnrlat, urcrnrlat=urcrnrlat,
                llcrnrlon=llcrnrlon, urcrnrlon=urcrnrlon, lat_ts=0, resolution='h')

    # Plot the minimum distance data
    x, y = m(radar_lons, radar_lats)
    colormesh = m.pcolormesh(x, y, input_array, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax)

    # Add color bar
    cbar = plt.colorbar(colormesh, shrink=0.665, aspect=12)
    cbar.set_label(title, fontsize=8)

    # Add coastline
    m.drawcoastlines(color="black")

    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)

    plt.show()

def plot_adjusted_feature(radar_lons, radar_lats, input_array, title, cmap='terrain', save_path=None):
    from mpl_toolkits.basemap import Basemap
    import numpy as np
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    m = Basemap(projection='merc', llcrnrlat=54, urcrnrlat=59, llcrnrlon=6, urcrnrlon=17, lat_ts=0, resolution='h', ax=ax)
    x, y = m(radar_lons, radar_lats)
    colormesh = m.pcolormesh(x, y, input_array, cmap=cmap)
    m.drawcoastlines(color = "white")

    # Place colorbar in a more controlled position
    #cax = fig.add_axes([0.87, 0.10, 0.03, 0.7])  # Adjust as necessary
    #cbar = fig.colorbar(colormesh, cax=cax)
    #cbar.set_label(title)

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.01)
    
    plt.tight_layout()
    plt.show()


def draw_circle(map, center_lon, center_lat, radius_km, color = 'black', lw = 1):
    num_points = 100
    lat0, lon0 = center_lat, center_lon
    lat_points = [lat0 + (radius_km / 111.12) * np.cos(np.radians(angle)) for angle in np.linspace(0, 360, num_points)]
    lon_points = [lon0 + (radius_km / 111.12) / np.cos(np.radians(lat0)) * np.sin(np.radians(angle)) for angle in np.linspace(0, 360, num_points)]
    x, y = map(lon_points, lat_points)
    map.plot(x, y, marker=None, color=color, linestyle='-', linewidth=lw, zorder = 15)


from matplotlib.colors import BoundaryNorm
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap

def read_rgb_file(file_path):
    with open(file_path, 'r') as file:
        # Read all lines in the file
        lines = file.readlines()

    # Skip lines that do not contain color information
    color_lines = [line for line in lines if line.strip() and not line.startswith('#') and not line.startswith('ncolors')]
    
    # Extract the RGB values
    rgb_values = [tuple(map(int, line.strip().split()[:3])) for line in color_lines]
    
    # Normalize the RGB values to the range [0, 1]
    rgb_normalized = [(r/255.0, g/255.0, b/255.0) for r, g, b in rgb_values]
    
    # Create a colormap from the RGB values
    colormap = LinearSegmentedColormap.from_list("custom_radar", rgb_normalized)
    
    return colormap



# --------------------
# Model perfomance documentations plots for report
# --------------------

def plot_hyetograph(df_test, station_id, start_timestamp, end_timestamp, A_est, B_est, save_path=None, tick_interval=6):
    # Filter the DataFrame based on the stationId and timestamp range
    df_filtered = df_test[(df_test['stationId'] == station_id) & 
                          (df_test['timestamp'] >= start_timestamp) & 
                          (df_test['timestamp'] <= end_timestamp)]
    df_filtered = df_filtered.reset_index(drop=True)

    # Convert Z to dBZ
    dbz = 10 * np.log10(df_filtered['Z'])
    
    # Predict rain rates
    R_pred = np.array(dbz_to_R_marshall_palmer(dbz, A_est, B_est))
    R_true = np.array(df_filtered['R'] * 60)
    MP = np.array(dbz_to_R_marshall_palmer(dbz, 200, 1.6))

    # Convert timestamp to datetime
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
    t = df_filtered['timestamp']
    idx = np.arange(len(t))

    # Plotting
    fig, ax = plt.subplots(figsize=(14, 4.5))

    # Plot R_true as a bar plot
    ax.bar(idx, R_true, label='Gauge', color='dodgerblue', alpha=0.5)  # Bar plot for R_true

    # Plot MP and R_pred as line plots
    ax.plot(idx, MP, '-', lw=2, label='Marshall-Palmer A=200, B=1.6', color='orangered')
    ax.plot(idx, R_pred, '-', lw=2, label=f'Marshall-Palmer A={A_est:.4f}, B={B_est:.2f}', color='green')

    ax.set_ylabel('Rain rate (mm/h)')
    ax.legend()

    # Set the x-ticks and labels
    date_labels = t.map(lambda x: x.strftime('%Y-%m-%d %H:%M'))  # Apply strftime to each element
    ax.set_xticks(idx[::tick_interval])
    ax.set_xticklabels(date_labels[::tick_interval], rotation=45, ha='right')

    plt.tight_layout()

    # Save the plot
    if save_path:
        save_path = f'{save_path}_hyteograph_{station_id}.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
        
    plt.show()
    
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_hyetograph_m2(df_test, station_id, start_timestamp, end_timestamp, A_est, B_est, save_path=None, tick_interval=6):
    # Filter the DataFrame based on the stationId and timestamp range
    df_filtered = df_test[(df_test['stationId'] == station_id) & 
                          (df_test['timestamp'] >= start_timestamp) & 
                          (df_test['timestamp'] <= end_timestamp)]
    df_filtered = df_filtered.reset_index(drop=True)

    # Convert timestamp to datetime
    df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])
    t = df_filtered['timestamp']
    
    R_true = df_filtered['R_true']
    MP_opt = df_filtered['MP_opt']
    MP = df_filtered['MP']
    R_pred_R = df_filtered['R_pred_R']
    R_pred_AB = df_filtered['R_pred_AB']
    A = df_filtered['A']
    B = df_filtered['B']
    dBZ = df_filtered['dBZ']
    
    idx = np.arange(len(t))

    # Plotting
    fig, (ax1, ax2, ax3) = plt.subplots(nrows=3, figsize=(10, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1]})

    # Plot hyetograph on the first subplot
    ax1.bar(idx, R_true, label='Gauge', color='dodgerblue', alpha=0.5)  # Bar plot for R_true
    ax1.plot(idx, MP, '-', lw=1, label='Marshall-Palmer A=200, B=1.6', color='orangered')
    #ax1.plot(idx, MP_opt, '-', lw=1, label=f'M1: Marshall-Palmer A={A_est:.2f}, B={B_est:.2f}', color='green')
    ax1.plot(idx, R_pred_R, '-', lw=3, label='M2: 2 Stage PINN-R', color='blue', alpha=0.8)
    ax1.plot(idx, R_pred_AB, '--', lw=2, label='M2: 2 Stage PINN-AB', color='purple', alpha=1)
    #ax1.set_ylim(-0.1, 45)
    ax1.set_ylabel('Rain rate (mm/h)')
    ax1.legend(loc='upper right', fontsize='small')

    # Plot dBZ on the middle subplot
    ax2.plot(idx, dBZ, '-', lw=2, label='dBZ', color='k')
    ax2.set_ylabel('dBZ', color='k')
    ax2.tick_params(axis='y', labelcolor='k')
    ax2.set_ylim(-50, 60)
    
    # Plot A and B on the third subplot
    ax3.plot(idx, A, '-', lw=2, label='A', color='k')
    ax3.set_ylabel('A', color='k')
    ax3.tick_params(axis='y', labelcolor='k')
    ax3.set_ylim(0, 300)
    
    ax3b = ax3.twinx()
    ax3b.plot(idx, B, '-', lw=2, label='B', color='dimgray')
    ax3b.set_ylabel('B', color='dimgray')
    ax3b.tick_params(axis='y', labelcolor='dimgray')
    ax3b.set_ylim(0, 6) 
    
    ax3.set_xlabel('Timestamp')

    # Set the x-ticks and labels
    date_labels = t.map(lambda x: x.strftime('%Y-%m-%d %H:%M'))  # Apply strftime to each element
    ax3.set_xticks(idx[::tick_interval])
    ax3.set_xticklabels(date_labels[::tick_interval], rotation=25, ha='right', size = 8)

    plt.tight_layout()

    # Save the plot
    if save_path:
        save_path = f'{save_path}_hyetograph_{station_id}.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
        
    plt.show()

def plot_goruped_rain_intensity_error(df_test, save_path = None):
    # Define rain intensity bins
    bins = [0, 0.5, 2, 6, 10, 18, 30, np.inf]
    labels = ['No rain\n(<0.5 mm/h)', 'Weak rain\n(0.5-2 mm/h)', 'Moderate rain\n(2-6 mm/h)', 
              'Heavy rain\n(6-10 mm/h)', 'Very heavy rain\n(10-18 mm/h)', 'Shower\n(18-30 mm/h)', 'Cloudburst\n(>30 mm/h)']

    # Bin the R_true values
    df_test['rain_bin'] = pd.cut(df_test['R_true'], bins=bins, labels=labels, right=False)

    # Calculate the total absolute error for each bin
    bin_ME_standard = []
    bin_ME_opt = []

    for label in labels:
        bin_data = df_test[df_test['rain_bin'] == label]
        bin_ME_standard.append((abs(bin_data['R_true']/6 - bin_data['MP']/6)).sum())
        bin_ME_opt.append((abs(bin_data['R_true']/6 - bin_data['R_pred']/6)).sum())

    # Plotting
    plt.figure(figsize=(14, 4))

    positions = np.arange(len(labels))  # Position of bars on x-axis
    width = 0.35  

    plt.bar(positions - width/2, bin_ME_standard, width, label='Standard Parameters', color='orangered', alpha=0.5)
    plt.bar(positions + width/2, bin_ME_opt, width, label='Optimized Parameters', color='green', alpha=0.7)

    plt.xlabel('Rain intensity (mm/h)')
    plt.ylabel('Total absolute error in mm')
    plt.xticks(positions, labels)  # Assign rain intensity labels as x-tick labels
    plt.legend()

    plt.tight_layout()  # Adjust layout to not cut off labels
    
    # Save the plot
    if save_path:
        save_path = f'{save_path}_error_vs_rain_intensity.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
        
    plt.show()
    
    
def plot_goruped_rain_intensity_error_m2(df_test, TYPE, save_path = None):
    # Define rain intensity bins
    bins = [0.2, 0.5, 2, 6, 10, 18, 30, np.inf]
    labels = ['No rain\n(0.2-0.5 mm/h)', 'Weak rain\n(0.5-2 mm/h)', 'Moderate rain\n(2-6 mm/h)', 
              'Heavy rain\n(6-10 mm/h)', 'Very heavy rain\n(10-18 mm/h)', 'Shower\n(18-30 mm/h)', 'Cloudburst\n(>30 mm/h)']

    # Bin the R_true values
    df_test['rain_bin'] = pd.cut(df_test['R_true'], bins=bins, labels=labels, right=False)

    # Calculate the total absolute error for each bin
    bin_ME_standard = []
    bin_ME_opt = []
    bin_M2_R = []
    bin_M2_AB = []

    for label in labels:
        bin_data = df_test[df_test['rain_bin'] == label]
        bin_ME_standard.append((abs(bin_data['R_true']/6 - bin_data['MP']/6)).sum())
        bin_ME_opt.append((abs(bin_data['R_true']/6 - bin_data['MP_opt']/6)).sum())
        bin_M2_R.append((abs(bin_data['R_true']/6 - bin_data['R_pred_R']/6)).sum())
        bin_M2_AB.append((abs(bin_data['R_true']/6 - bin_data['R_pred_AB']/6)).sum())

    # Plotting
    plt.figure(figsize=(14, 6))

    positions = np.arange(len(labels))  # Position of bars on x-axis
    width = 0.2  # Width of each bar

    plt.bar(positions - 1.5*width, bin_ME_standard, width, label='Standard Parameters', color='orangered', alpha=0.5)
    plt.bar(positions - 0.5*width, bin_ME_opt, width, label='Optimized Parameters', color='green', alpha=0.7)
    plt.bar(positions + 0.5*width, bin_M2_R, width, label=f'{TYPE} (R)', color='blue', alpha=0.5)
    plt.bar(positions + 1.5*width, bin_M2_AB, width, label=f'{TYPE} (AB)', color='purple', alpha=0.7)

    plt.xlabel('Rain intensity (mm/h)')
    plt.ylabel('Total absolute error in mm')
    plt.xticks(positions, labels)  # Assign rain intensity labels as x-tick labels
    plt.legend()

    plt.tight_layout()  # Adjust layout to not cut off labels
    
    # Save the plot
    if save_path:
        save_path = f'{save_path}_error_vs_rain_intensity.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
        
    plt.show()
    
    
def plot_rmse_by_month(df_test, save_path = None):
    months = df_test['month'].unique()
    RMSE_standard = []
    RMSE_opt = []

    for month in months:
        df_filtered = df_test[df_test['month'] == month]
        RMSE_standard.append(np.sqrt(mean_squared_error(df_filtered['R_true'] / 6, df_filtered['MP'] / 6)))
        RMSE_opt.append(np.sqrt(mean_squared_error(df_filtered['R_true'] / 6, df_filtered['R_pred'] / 6)))

    # Assuming test_stations is sorted or in the desired order for plotting
    positions = np.arange(len(months))  # Position of bars on x-axis
    width = 0.35  # Width of each bar

    plt.figure(figsize=(14, 4))
    # Plot RMSE for standard model
    plt.bar(positions - width/2, RMSE_standard, width, label='Standard Parameters', color='orangered', alpha=0.5)
    # Plot RMSE for optimized model
    plt.bar(positions + width/2, RMSE_opt, width, label='Optimized Parameters', color='green', alpha=0.7)

    plt.xlabel('Month')
    plt.ylabel('RMSE (mm)')
    plt.xticks(positions, months)  # Assign month as x-tick labels
    plt.legend()

    if save_path:        
        save_path = f'{save_path}_error_vs_month.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)

    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()
    
def rmse(predictions, targets):
    return np.sqrt(((targets - predictions) ** 2).mean())

def mae(predictions, targets):
    return np.abs(targets - predictions).mean()

def mse(predictions, targets):
    return ((targets - predictions) ** 2).mean()

def me(predictions, targets):
    return (targets - predictions).mean()

def calculate_metrics(df):
    results = []
    for station_id in df['stationId'].unique():
        df_station = df[df['stationId'] == station_id].copy()
        df_station['timestamp'] = pd.to_datetime(df_station['timestamp'])
        df_station.set_index('timestamp', inplace=True)

        # Convert R to mm for each 10-minute observation
        df_station['R_true_mm'] = df_station['R'] * 60 / 6  # 60 minutes/hour / 6 for 10-minute intervals
        df_station['R_pred_mm'] = df_station['R_pred'] / 6

        metrics = {
            'StationId': station_id,
            'RMSE': rmse(df_station['R_pred_mm'], df_station['R_true_mm']),
            'MAE': mae(df_station['R_pred_mm'], df_station['R_true_mm']),
            'MSE': mse(df_station['R_pred_mm'], df_station['R_true_mm']),
            'ME': me(df_station['R_pred_mm'], df_station['R_true_mm'])
        }

        # Aggregation intervals
        intervals = ['30T', '1H', '1D']
        for interval in intervals:
            aggregated = df_station[['R_pred_mm', 'R_true_mm']].resample(interval).sum()
            metrics[f'RMSE_{interval}'] = rmse(aggregated['R_pred_mm'], aggregated['R_true_mm'])
            metrics[f'MAE_{interval}'] = mae(aggregated['R_pred_mm'], aggregated['R_true_mm'])
            metrics[f'MSE_{interval}'] = mse(aggregated['R_pred_mm'], aggregated['R_true_mm'])
            metrics[f'ME_{interval}'] = me(aggregated['R_pred_mm'], aggregated['R_true_mm'])
        
        results.append(metrics)
    
    return pd.DataFrame(results)

def calculate_metrics2(df, pred_column):
    results = []
    for station_id in df['stationId'].unique():
        df_station = df[df['stationId'] == station_id].copy()
        df_station['timestamp'] = pd.to_datetime(df_station['timestamp'])
        df_station.set_index('timestamp', inplace=True)

        # Convert R to mm for each 10-minute observation
        df_station['R_true_mm'] = df_station['R'] * 60 / 6  # 60 minutes/hour / 6 for 10-minute intervals
        df_station['R_pred_mm'] = df_station[pred_column] / 6

        metrics = {
            'StationId': station_id,
            'RMSE': rmse(df_station['R_pred_mm'], df_station['R_true_mm']),
            'MAE': mae(df_station['R_pred_mm'], df_station['R_true_mm']),
            'MSE': mse(df_station['R_pred_mm'], df_station['R_true_mm']),
            'ME': me(df_station['R_pred_mm'], df_station['R_true_mm'])
        }

        # Aggregation intervals
        intervals = ['30T', '1H', '1D']
        for interval in intervals:
            aggregated = df_station[['R_pred_mm', 'R_true_mm']].resample(interval).sum()
            metrics[f'RMSE_{interval}'] = rmse(aggregated['R_pred_mm'], aggregated['R_true_mm'])
            metrics[f'MAE_{interval}'] = mae(aggregated['R_pred_mm'], aggregated['R_true_mm'])
            metrics[f'MSE_{interval}'] = mse(aggregated['R_pred_mm'], aggregated['R_true_mm'])
            metrics[f'ME_{interval}'] = me(aggregated['R_pred_mm'], aggregated['R_true_mm'])
        
        results.append(metrics)
    
    return pd.DataFrame(results)

def calculate_metrics_MP(df):
    results = []
    for station_id in df['stationId'].unique():
        df_station = df[df['stationId'] == station_id].copy()
        df_station['timestamp'] = pd.to_datetime(df_station['timestamp'])
        df_station.set_index('timestamp', inplace=True)

        # Convert R to mm for each 10-minute observation
        df_station['R_true_mm'] = df_station['R'] * 60 / 6  # 60 minutes/hour / 6 for 10-minute intervals
        df_station['MP_pred_mm'] = df_station['MP'] / 6

        metrics = {
            'StationId': station_id,
            'RMSE': rmse(df_station['MP_pred_mm'], df_station['R_true_mm']),
            'MAE': mae(df_station['MP_pred_mm'], df_station['R_true_mm']),
            'MSE': mse(df_station['MP_pred_mm'], df_station['R_true_mm']),
            'ME': me(df_station['MP_pred_mm'], df_station['R_true_mm'])
        }

        # Aggregation intervals
        intervals = ['30T', '1H', '1D']
        for interval in intervals:
            aggregated = df_station[['MP_pred_mm', 'R_true_mm']].resample(interval).sum()
            metrics[f'RMSE_{interval}'] = rmse(aggregated['MP_pred_mm'], aggregated['R_true_mm'])
            metrics[f'MAE_{interval}'] = mae(aggregated['MP_pred_mm'], aggregated['R_true_mm'])
            metrics[f'MSE_{interval}'] = mse(aggregated['MP_pred_mm'], aggregated['R_true_mm'])
            metrics[f'ME_{interval}'] = me(aggregated['MP_pred_mm'], aggregated['R_true_mm'])
        
        results.append(metrics)
    
    return pd.DataFrame(results)

def plot_error_metrics(metrics_df, metrics_df_MP, save_path = None):
    error_metrics = ['RMSE', 'MAE', 'MSE', 'ME']
    aggregation_levels = ['30T', '1H', '1D']
    aggregation_levels_str = ['30 min', '1 hour', '1 day']
    
    num_rows = len(aggregation_levels) + 1  # +1 for the non-aggregated metrics
    num_cols = len(error_metrics)
    
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(25, 15))
    
    for col, metric in enumerate(error_metrics):
        # Top row: non-aggregated metrics
        ax = axs[0, col]
        positions = np.arange(len(metrics_df['StationId']))
        width = 0.4  # Width of the bars
        ax.bar(positions - width/2, metrics_df[metric], width, label='Optimized Parameters', color='green', alpha=0.7)
        ax.bar(positions + width/2, metrics_df_MP[metric], width, label='Standard Parameters', color='orangered', alpha=0.5)
        ax.set_title(f'{metric}', size = 18)
        ax.set_xticks([])
        ax.set_xticklabels([])
        if col == 0:
            ax.set_ylabel('5 min', fontsize=18)

        # Subsequent rows: aggregated metrics
        for row, interval in enumerate(aggregation_levels, start=1):
            ax = axs[row, col]
            interval_col = f'{metric}_{interval}'
            ax.bar(positions - width/2, metrics_df[interval_col], width, label='Optimized Parameters', color='green', alpha=0.7)
            ax.bar(positions + width/2, metrics_df_MP[interval_col], width, label='Standard Parameters', color='orangered', alpha=0.5)
            
            if row == 0:
                ax.set_title(f'{metric}', fontsize=14)
            if col == 0:
                ax.set_ylabel(f'{aggregation_levels_str[row-1]}', fontsize=18)
            if row == num_rows - 1:
                ax.set_xticks(positions)
                ax.set_xticklabels(metrics_df['StationId'], rotation='vertical', fontsize=16)
            else:
                ax.set_xticks([])
                ax.set_xticklabels([])

    # Create a single legend for the entire figure
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=18, prop={'size': 18})
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)  # Adjust the top to make space for the legend
    
    if save_path:        
        save_path = f'{save_path}_error_aggregated.png'
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0.01, dpi=300)
        
    plt.show()
    
def Z_to_R_marshall_palmer(Z, A=200, B=1.6):
    Z = np.asarray(Z)
    R = (Z / A)**(1/B)
    return R
