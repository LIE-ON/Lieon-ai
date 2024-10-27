
# Audio and Label Segmentation Toolkit

This repository provides a set of Python scripts for handling audio segmentation, label segmentation, and surplus label detection. These tools are designed to facilitate the segmentation of audio files into 10-second chunks and ensure corresponding label files are aligned correctly. A simple mechanism is also provided to detect and remove surplus labels generated due to edge cases.

## Contents

- [`data_segmentation.py`](#data_segmentationpy): Segments audio files into 10-second chunks.
- [`label_segmentation.py`](#label_segmentationpy): Segments labels based on corresponding audio lengths.
- [`surplus_detector.py`](#surplus_detectorpy): Detects and prints surplus labels that may have been created due to edge cases.

---

### `data_segmentation.py`

This script processes all audio files in the specified directory and segments them into 10-second chunks. The segmented audio files are then stored in an automatically created `Audio_edit` folder in the parent directory.

#### Usage:

```bash
python data_segmentation.py --input_dir <path_to_audio_directory>
```

**Output**:  
- Segmented audio files saved in the `Audio_edit` folder in the parent directory.

---

### `label_segmentation.py`

This script segments label files to match the corresponding segmented audio files (which are divided into 10-second intervals). It takes the audio files and their corresponding labels as input, processes the labels based on the duration of the audio, and outputs the segmented labels into the `Label_edit` folder in the parent directory.

#### Usage:

```bash
python label_segmentation.py --input_dir <path_to_label_directory> --audio_dir <path_to_audio_directory>
```

**Output**:  
- Segmented labels saved in the `Label_edit` folder in the parent directory.

---

### `surplus_detector.py`

This script helps detect any surplus label files that may have been generated due to an edge case where more labels are created than necessary. This occurs when labels outnumber the segmented audio files. The script prints the names of these surplus (empty) label files so they can be manually removed.

#### Usage:

```bash
python surplus_detector.py --label_dir <path_to_label_directory>
```

**Output**:  
- Prints a list of surplus label files that are safe to remove manually.

---

### Author

github ID: LimDoHyeon
