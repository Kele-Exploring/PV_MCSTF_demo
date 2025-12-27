"""
The MCSTF method is a fundamental MTF technique designed for three-channel time series imaging.
Functionality achieved:
Conversion of power and irradiance time-series data into image data using the MCSTF method.
Firstly, MTF images are generated for both datasets.
Subsequently, the difference and mean values between the two datasets are calculated.
Cross-state transition probabilities are then computed for the mean and difference values,
yielding a fusion image of the two datasets.
Finally, the three images (MTF_R, MTF_G, MTF_B) are stitched into a single RGB image,
yielding the final MCSTF image.
"""
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, KBinsDiscretizer
import matplotlib.pyplot as plt
from matplotlib import rcParams
import time

def load_data(file_path):
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    power_data = df.iloc[:, 1].values  # Power
    radiation_data = df.iloc[:, 2].values  # Irradiance
    labels = df.iloc[:, 4].values  # Label

    scaler = MinMaxScaler()
    power_data = scaler.fit_transform(power_data.reshape(-1, 1)).flatten()
    radiation_data = scaler.fit_transform(radiation_data.reshape(-1, 1)).flatten()

    # Grouping the data into sets of 72
    data_groups = []
    for i in range(0, len(power_data), 72):
        data_groups.append((power_data[i:i + 72], radiation_data[i:i + 72], labels[i]))

    return data_groups

def calculate_mtf(data):

    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    quantized_data = discretizer.fit_transform(data.reshape(-1, 1)).astype(int).flatten()

    num_states = 10
    transition_matrix = np.zeros((num_states, num_states))

    for (i, j) in zip(quantized_data[:-1], quantized_data[1:]):
        transition_matrix[i, j] += 1

    row_sums = transition_matrix.sum(axis=1, keepdims=True)

    row_sums = np.where(row_sums == 0, 1, row_sums)

    transition_matrix /= row_sums

    n = len(data)
    mtf_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf_matrix[i, j] = transition_matrix[quantized_data[i], quantized_data[j]]

    return mtf_matrix

def calculate_mtf_fusion(data_1, data_2):

    discretizer = KBinsDiscretizer(n_bins=10, encode='ordinal', strategy='uniform')
    quantized_data_1 = discretizer.fit_transform(data_1.reshape(-1, 1)).astype(int).flatten()
    quantized_data_2 = discretizer.fit_transform(data_2.reshape(-1, 1)).astype(int).flatten()

    num_states_fusion = 10
    transition_matrix_fusion = np.zeros((num_states_fusion, num_states_fusion))

    for (i, j) in zip(quantized_data_1[:-1], quantized_data_2[1:]):
        transition_matrix_fusion[i, j] += 1

        for i in range(num_states_fusion):
            for j in range(num_states_fusion):

                count_i_in_Q1 = (quantized_data_1 == i).sum()
                count_j_in_Q2 = (quantized_data_2 == j).sum()

                total_count = (count_i_in_Q1 + count_j_in_Q2) / 2

                if total_count > 0:
                    transition_matrix_fusion[i, j] /= total_count

    n = len(data_1)
    mtf_matrix_fusion = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            mtf_matrix_fusion[i, j] = transition_matrix_fusion[quantized_data_1[i], quantized_data_2[j]]

    return mtf_matrix_fusion

def create_mtf_image(power_data, radiation_data):

    MTF_R = calculate_mtf(power_data)
    MTF_G = calculate_mtf(radiation_data)

    MTF_B = calculate_mtf_fusion((radiation_data - power_data), (power_data + radiation_data) / 2)

    MTF_R = (MTF_R - MTF_R.min()) / (MTF_R.max() - MTF_R.min())
    MTF_G = (MTF_G - MTF_G.min()) / (MTF_G.max() - MTF_G.min())
    MTF_B = (MTF_B - MTF_B.min()) / (MTF_B.max() - MTF_B.min())

    mtf_image = np.stack([MTF_R, MTF_G, MTF_B], axis=-1)

    return MTF_R, MTF_G, MTF_B, mtf_image

if __name__ == "__main__":

    file_path = 'dataset/dataset.csv'
    # file_path = 'dataset/dataset_test.csv'
    df = load_data(file_path)

    data_groups = preprocess_data(df)

# Generate MCSTF images for all data
#     output_dir = 'dataset/MCSTF_RGB'
#     # output_dir = 'dataset/MCSTF_RGB_test'
#     os.makedirs(output_dir, exist_ok=True)
#
#     start_time = time.time()
#
#     for group_num, (power_data, radiation_data, label) in enumerate(data_groups[:8796]):
#         MTF_R, MTF_G, MTF_B, mtf_image = create_mtf_image(power_data, radiation_data)
#
#         group_num_str = f'{group_num + 1:03d}'
#
#         file_name = f'MCSTF_image_group_{group_num_str}_label_{label}.png'
#
#         plt.imsave(os.path.join(output_dir, file_name), mtf_image)
#
#         print(f'Group {group_num_str} processed and saved.')
#
#     end_time = time.time()
#     time_taken = end_time - start_time
#     print(f'All MCSTF RGB images saved to directory: {output_dir}')
#     print(f'Takes {time_taken:.2f} seconds')

# Select a data set (for demonstration purposes)
    power_data, radiation_data, label = data_groups[6425]

    GAFR, GAFG, GAFB, dsgaf_image = create_mtf_image(power_data, radiation_data)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 4, 1)
    plt.imshow(np.stack([GAFR, np.zeros_like(GAFR), np.zeros_like(GAFR)], axis=-1))
    plt.title("Red Channel (MCSTF_R)")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlabel('（a）', fontsize=14)
    # plt.axis("off")

    plt.subplot(1, 4, 2)
    plt.imshow(np.stack([np.zeros_like(GAFG), GAFG, np.zeros_like(GAFG)], axis=-1))
    plt.title("Green Channel (MCSTF_G)")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlabel('（b）', fontsize=14)
    # plt.axis("off")

    plt.subplot(1, 4, 3)
    plt.imshow(np.stack([np.zeros_like(GAFB), np.zeros_like(GAFB), GAFB], axis=-1))
    plt.title("Blue Channel (MCSTF_B)")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlabel('（c）', fontsize=14)

    plt.subplot(1, 4, 4)
    plt.imshow(dsgaf_image)
    plt.title("MCSTF Image")
    plt.gca().set_xticks([])
    plt.gca().set_yticks([])
    plt.xlabel('（d）', fontsize=14)

    plt.tight_layout()
    plt.show()
