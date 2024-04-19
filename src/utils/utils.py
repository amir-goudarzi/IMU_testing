import pandas as pd
from scipy.interpolate import interp1d
import numpy as np
import os
import logging


def load_data(src_dir: os.PathLike, video_id: str, is_csv=True):
    '''
    Load the WAV file and its label.
    Args :
        • filename: str. The path of a WAV file.
    Returns A tuple of two Pandas DataFrame objects:
        • Accelerometer: A DataFrame with the following columns:
            • seconds: The time in seconds.
            • AcclX: The acceleration along the x-axis.
            • AcclY: The acceleration along the y-axis.
            • AcclZ: The acceleration along the z-axis.
        • Gyroscope: A DataFrame with the following columns:
            • seconds: The time in seconds.
            • GyroX: The angular velocity along the x-axis.
            • GyroY: The angular velocity along the y-axis.
            • GyroZ: The angular velocity along the z-axis.
    '''
    df_accl = None
    df_gyro = None

    if is_csv:
        df_accl = pd.read_csv(src_dir, video_id + '-accl.csv').dropna().reset_index(drop=True)
        df_gyro = pd.read_csv(src_dir, video_id + '-gyro.csv').dropna().reset_index(drop=True)
    else:
        df_accl = pd.read_pickle(src_dir, video_id + '-accl.pkl').dropna().reset_index(drop=True)
        df_gyro = pd.read_pickle(src_dir, video_id + '-gyro.pkl').dropna().reset_index(drop=True)

    return df_accl, df_gyro


def save_data(
        data: pd.DataFrame,
        dst_dir: os.PathLike,
        video_id: str,
        is_accl=True) -> None:
    '''
    Save the interpolated data.
    Args:
        • data: A DataFrame with the following columns:
            • seconds: The time in seconds.
            • {sensor}X: x-axis.
            • {sensor}Y: y-axis.
            • {sensor}Z: z-axis.
        • filename: str. The path of a WAV file.
    '''

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
        logging.debug(f'Directory created: {dst_dir=}')
    
    if is_accl:
        data.to_pickle(os.path.join(dst_dir, video_id +  '-accl.pkl'), index=False)
    else:
        data.to_pickle(os.path.join(dst_dir, video_id + '-gyro.pkl'), index=False)


def interpolate_data(
        df: pd.DataFrame,
        is_accl=True) -> pd.DataFrame:
    '''
    Interpolate the data to the desired sampling rate.
    Args:
        • df: A DataFrame with the following columns:
            • seconds: The time in seconds.
            • {sensor}X: x-axis.
            • {sensor}Y: y-axis.
            • {sensor}Z: z-axis.
        • sample_rate: int. The sampling rate of the WAV file.
        • downsampling_rate: int. The desired sampling rate.
        • is_accl: bool. The sensor to interpolate. Default is `True`.
    '''
    sample_rate = 200 if is_accl else 400
    

    start_time = df['seconds'].iloc[0]
    end_time = df['seconds'].iloc[-1]

    target_time_s = np.linspace(start_time, end_time,
                        num=int(round(1+sample_rate*(end_time - start_time))),
                        endpoint=True)

    # TODO: Interpolate the data. Choose which is the correct way to interpolate.
    b, a = interp1d()


def datetime_to_milliseconds(timestamp: str):
    '''
    Convert a timestamp to milliseconds.
    Args:
        • timestamp: str. The timestamp in the format HH:MM:SS.SSS.
    Returns: int. The timestamp in milliseconds.
    '''
    time = timestamp.split(':')
    seconds, micro = time[2].split('.')
    hours = int(time[0])
    minutes = int(time[1])
    return (hours * 3600 + minutes * 60 + int(seconds)) * 1000 + int(micro)


def center_timestamp(start: str, stop: str):
    '''
    Center the timestamps.
    Args:
        • start: str. The start timestamp.
        • stop: str. The stop timestamp.
    Returns: A tuple of two strings. The centered timestamps.
    '''
    start_ms = datetime_to_milliseconds(start)
    stop_ms = datetime_to_milliseconds(stop)
    center = (start_ms + stop_ms) // 2
    return center
