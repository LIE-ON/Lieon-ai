import os
import pandas as pd

# Define constants
CHUNK_DURATION = 10  # Duration for each segment in seconds


def create_output_dir(input_dir):
    # Create output directory for the split label files
    output_dir = f"{input_dir}_edit"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir


def split_labels(label_file_path, output_dir):
    # Read the label file
    labels = pd.read_csv(label_file_path)
    file_name = os.path.basename(label_file_path)
    file_stem, ext = os.path.splitext(file_name)

    # Get the maximum time to determine how many chunks we need
    max_time = labels['End'].max()
    num_chunks = int(max_time // CHUNK_DURATION) + 1

    # Process each chunk
    for i in range(num_chunks):
        start_time = i * CHUNK_DURATION
        end_time = (i + 1) * CHUNK_DURATION

        # Filter rows within the current chunk
        chunk_data = labels[(labels['End'] > start_time) & (labels['Start'] < end_time)].copy()

        # Adjust start and end times to be relative to the chunk
        chunk_data['Start'] = chunk_data['Start'].clip(lower=start_time) - start_time
        chunk_data['End'] = chunk_data['End'].clip(upper=end_time) - start_time

        # Save the chunked data
        chunk_name = f"{file_stem}_{i + 1}{ext}"
        chunk_data.to_csv(os.path.join(output_dir, chunk_name), index=False)


def process_directory(input_dir):
    # Create output directory for processed label files
    output_dir = create_output_dir(input_dir)

    # List label files in the directory
    label_files = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Process each label file
    for label_file in label_files:
        split_labels(label_file, output_dir)


if __name__ == "__main__":
    input_directory = "your_label_dir"
    process_directory(input_directory)
