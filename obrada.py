import os
import mne
import numpy as np

#pristup dokumentima
path = os.path.join(os.getcwd(), 'files')

def loading_files():
    eeg_data = []
    annotations = []

# Walk through the directory and load .edf files and their corresponding .event files
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith('.edf'):
                edf_path = os.path.join(root, file)
                event_path = edf_path + '.event'

                try:
                    # Load EEG data using MNE; use Latin-1 encoding for EDF annotations
                    raw = mne.io.read_raw_edf(edf_path, preload=True, encoding="latin1")
                    eeg_data.append(raw)

                    # Load annotations if available
                    if os.path.exists(event_path):
                        try:
                            with open(event_path, 'r', encoding='latin1', errors='replace') as f:
                                events = f.readlines()
                        except Exception:
                            with open(event_path, 'rb') as f:
                                events = [f.read()]
                        annotations.append(events)
                except Exception as e:
                    print(f"Error loading {edf_path}: {e}")

    return eeg_data, annotations

# Call the function to load files
eeg_data, annotations = loading_files()




