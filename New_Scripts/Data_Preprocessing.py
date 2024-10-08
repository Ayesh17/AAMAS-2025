import os
import numpy as np
import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import MinMaxScaler

# Define all column headers based on the provided list
column_headers = [
    'TIME', 'BLUE_ID', 'BLUE_LAT', 'BLUE_LON', 'BLUE_HEADING', 'BLUE_SPEED',
    'RED_ID', 'RED_LAT', 'RED_LON', 'RED_HEADING', 'RED_SPEED',
    'DISTANCE', 'BEH_PHASE', 'COLREGS', 'AVOIDANCE', 'BEH_LABEL',
    'abs_b_r', 'abs_r_b', 'rel_b_r', 'rel_r_b', 'abs_dheading',
    'r_accel', 'dvx', 'dvy', 'xr', 'yr', 'cpa_time', 'cpa_dist',
    'S_SIDE', 'S_BEAM', 'S_REG_b_r', 'S_REG_r_b', 'S_DREL_BEAR',
    'S_DELTA_DIST', 'S_ACCEL', 'S_DABS_HEADING', 'S_CPA_TIME',
    'S_CPA_DELTA_DIST', 'S_CPA_TIME_DIST'
]

# List of features to exclude based on the previous analysis
features_to_exclude = [
    'TIME', 'BLUE_ID', 'RED_ID', 'DISTANCE', 'xr', 'yr', 'COLREGS', 'abs_b_r', 'abs_r_b'
]

# List of features to normalize and retain
features_to_normalize = [
    'BLUE_SPEED', 'RED_SPEED', 'BLUE_HEADING', 'RED_HEADING', 'abs_dheading',
    'r_accel', 'dvx', 'dvy', 'cpa_dist', 'S_DELTA_DIST', 'S_DABS_HEADING',
    'S_CPA_TIME', 'S_CPA_DELTA_DIST', 'S_CPA_TIME_DIST'
]

# Retain only columns that are not excluded
filtered_columns = [col for col in column_headers if col not in features_to_exclude]

# Create new names for the normalized features
normalized_feature_names = [f"{col}_normalized" for col in features_to_normalize]

# Replace original feature names with the new normalized ones
final_columns = [
    col if col not in features_to_normalize else f"{col}_normalized" for col in filtered_columns
]

# Ensure the output directory exists
output_folder = "Data2"
os.makedirs(output_folder, exist_ok=True)

# Path to the root folder containing behavior subfolders (update this path as needed)
root_folder = "HMM_train_data_noise_preprocessed"

# Set the maximum sequence length for padding/truncation
max_sequence_length = 200

# Function to preprocess, normalize selected features, and assign updated column headers
def preprocess_and_normalize(file_path):
    try:
        # Read the CSV file without headers and assign the custom column headers
        df = pd.read_csv(file_path, header=None)  # Read without headers, treat all rows as data
        df.columns = column_headers  # Assign custom headers
        print(f"Read CSV file: {file_path} with shape: {df.shape}")

        # Remove unwanted columns
        df = df[filtered_columns]

        # Check if the DataFrame is empty
        if df.empty:
            print(f"Warning: {file_path} is empty and will be skipped.")
            return None

        # Normalize the selected features and update column names
        scaler = MinMaxScaler()
        df[features_to_normalize] = scaler.fit_transform(df[features_to_normalize])

        # Assign the new column names for normalized features
        df.columns = final_columns

        # Rename the `BEH_LABEL` column to `Label` and move it to the last position
        df = df.rename(columns={'BEH_LABEL': 'Label'})
        label_column = df.pop('Label')  # Remove the 'Label' column
        df['Label'] = label_column  # Add it back as the last column

        # Adjust the DataFrame to ensure it has exactly 200 rows
        if len(df) < max_sequence_length:
            # Pad with zeros if the sequence is shorter than 200 rows
            padding_df = pd.DataFrame(0, index=np.arange(max_sequence_length - len(df)), columns=df.columns)
            df = pd.concat([df, padding_df], ignore_index=True)
            print(f"Padded sequence in {file_path} to {max_sequence_length} rows.")
        elif len(df) > max_sequence_length:
            # Truncate if the sequence is longer than 200 rows
            df = df.iloc[:max_sequence_length]
            print(f"Truncated sequence in {file_path} to {max_sequence_length} rows.")

        # Convert the DataFrame to a NumPy array
        return df
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

# Read and preprocess all CSV files, and collect them into sequences with labels
data_sequences = []
labels = []
behavior_classes = ["Benign", "block", "ram", "cross", "headon", "herd", "overtake"]

# Mapping for behavior names to ensure correct file naming
behavior_mapping = {0: "benign", 1: "block", 2: "ram", 3: "cross", 4: "headon", 5: "herd", 6: "overtake"}

# Define label mapping dictionary for label transformation
label_mapping = {1: 0, 8: 1, 6: 2, 5: 3, 7: 5, 3: 4, 4: 6}

# New labels
# Benign -> 0, block -> 1, ram -> 2, cross -> 3, headon -> 4, herd -> 5, overtake -> 6

# Iterate over each behavior and corresponding files
for behavior in behavior_classes:
    folder_path = os.path.join(root_folder, behavior, "scenario")  # Include "scenario" subfolder in path
    if not os.path.exists(folder_path):
        print(f"Warning: The folder '{folder_path}' does not exist. Skipping this behavior.")
        continue

    print(f"Processing behavior: {behavior}")
    behavior_index = behavior_classes.index(behavior)
    sequence_index = 0  # Track sequence numbers for file naming

    for file in os.listdir(folder_path):
        if file.endswith("hmm_formatted.csv"):
            file_path = os.path.join(folder_path, file)
            print(f"Reading file: {file_path}")

            # Process the file and normalize it
            sequence_df = preprocess_and_normalize(file_path)
            if sequence_df is not None:
                # Apply the label mapping transformation
                sequence_df['Label'] = sequence_df['Label'].map(label_mapping).fillna(sequence_df['Label']).astype(int)

                data_sequences.append(sequence_df.values)
                labels.append(sequence_df['Label'].iloc[0])  # Use the label from the 'Label' column

                # Save each sequence as a CSV file with the specified naming convention and 'Label' column included
                file_name = f"{output_folder}/{behavior_mapping[behavior_index]}_{sequence_index}.csv"
                sequence_df.to_csv(file_name, index=False)
                print(f"Saved sequence {sequence_index} to {file_name}")
                sequence_index += 1

print("Preprocessing completed. All sequences have been adjusted to the required length.")
