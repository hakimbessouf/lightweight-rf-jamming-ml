            import h5py
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, skew
import scipy.signal as signal

file_path = r"D:\dataset\w3.mat"

def load_iq_chunk(file_path, condition, start_idx, end_idx):
    with h5py.File(file_path, 'r') as mat_file:
        iq_data = mat_file[condition]
        i_samples = iq_data[0, start_idx:end_idx]
        q_samples = iq_data[1, start_idx:end_idx]
        return i_samples, q_samples

def calculate_features(i_samples, q_samples, relative_jamming_power, distance_tx_rx):
    i_samples = i_samples.astype(np.float32)
    q_samples = q_samples.astype(np.float32)
    
    amplitude = np.sqrt(i_samples**2 + q_samples**2)
    phase = np.arctan2(q_samples, i_samples)
    power = amplitude**2
    phase_diff = np.diff(phase)
    instantaneous_frequency = np.concatenate(([0], phase_diff))

    kurt_i = kurtosis(i_samples)
    skew_i = skew(i_samples)
    kurt_q = kurtosis(q_samples)
    skew_q = skew(q_samples)

    length = len(i_samples)
    features = {
        'Amplitude': amplitude[:length],
        'Phase': phase[:length],
        'Power': power[:length],
        'Instantaneous_Frequency': instantaneous_frequency[:length],
        'Kurtosis_I': [kurt_i] * length,
        'Skewness_I': [skew_i] * length,
        'Kurtosis_Q': [kurt_q] * length,
        'Skewness_Q': [skew_q] * length,
        'Relative_Jamming_Power': [relative_jamming_power] * length,
        'Distance_Tx_Rx': [distance_tx_rx] * length
    }

    # Vérification des longueurs
    for key, value in features.items():
        print(f"Clé '{key}' : Longueur = {len(value)}")
    
    return features

def generate_iq_features_csv(file_path, output_csv_prefix, chunk_size=100000):
    parameters = {
        'Gaussian': {'Relative_Jamming_Power': 0.1, 'Distance_Tx_Rx': 10},
        'Nojamming': {'Relative_Jamming_Power': 0.0, 'Distance_Tx_Rx': 10},
        'Sine': {'Relative_Jamming_Power': 0.1, 'Distance_Tx_Rx': 10},
    }

    conditions = ['Gaussian', 'Nojamming', 'Sine']

    for condition in conditions:
        with h5py.File(file_path, 'r') as mat_file:
            iq_data = mat_file[condition]
            total_samples = iq_data.shape[1]

        for start_idx in range(0, total_samples, chunk_size):
            end_idx = min(start_idx + chunk_size, total_samples)
            
            i_samples, q_samples = load_iq_chunk(file_path, condition, start_idx, end_idx)

            params = parameters[condition]
            relative_jamming_power = params['Relative_Jamming_Power']
            distance_tx_rx = params['Distance_Tx_Rx']

            features = calculate_features(i_samples, q_samples, relative_jamming_power, distance_tx_rx)

            df = pd.DataFrame(features)

            output_filename = f"{output_csv_prefix}_{condition}_part{start_idx // chunk_size + 1}.csv"
            df.to_csv(output_filename, index=False)
            print(f"Fichier sauvegardé : {output_filename}, lignes {start_idx} à {end_idx}")

output_csv_prefix = 'iq_features_data'
generate_iq_features_csv(file_path, output_csv_prefix, chunk_size=100000)
