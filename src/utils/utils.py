import pandas as pd

def load_data(filename):
    '''
    Load the WAV file and its label.
    Args :
        • filename: str. The path of a WAV file.
    Returns A tuple of two Pandas DataFrame objects:
        • signals: A DataFrame with the following columns:
            • seconds: The time in seconds.
            • AcclX: The acceleration along the x-axis.
            • AcclY: The acceleration along the y-axis.
            • AcclZ: The acceleration along the z-axis.
            • GyroX: The angular velocity along the x-axis.
            • GyroY: The angular velocity along the y-axis.
            • GyroZ: The angular velocity along the z-axis.
        • sampling_rate: The sampling rate of the WAV file.
    '''
    # TODO: Load the WAV file and its label
    df_accl = pd.read_csv(filename + '-accl.csv').dropna().reset_index(drop=True)
    df_gyro = pd.read_csv(filename + '-gyro.csv').dropna().reset_index(drop=True)

    return df_accl, df_gyro