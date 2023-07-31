import os
import numpy as np
import matplotlib.pyplot as plt
import librosa, librosa.display
import math
import json

# dataset link = https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification

DATASET_PATH = "D:\COLLEGE\PROJECTS\Audio DL Basics\datasets\Data\genres_original"
JSON_PATH = "D:\COLLEGE\PROJECTS\Audio DL Basics\datasets\dataset.json"
SAMPLE_RATE = 22050
DURATION = 30 # in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * DURATION

def save_mfcc(dataset_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_segments = 5):
    
    # dictionary to store data
    data = {
        'mapping': [], 
        'mfcc': [],
        'labels': []
    }

    num_samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    expected_num_mfcc_vectors_per_segment = math.ceil(num_samples_per_segment / hop_length)

    # looping through all genres
    for i, (dirpath, dirnames, filenames) in enumerate(os.walk(dataset_path)):
        # print(dirpath)
        # print(dirnames)
        # print(filenames)
        # break
        # ensuring that we're not at the root level
        # flag = False
        if dirpath is not dataset_path:
            
            # save semantic label
            semantic_label = dirpath.split('\\')[-1]
            # if semantic_label != 'jazz':
            #     continue
            # print(semantic_label)
            # break
            data['mapping'].append(semantic_label)
            print(f"Processing {semantic_label}")
            # process files for spefic genre
            for f in filenames:
                # load audio file
                filepath = os.path.join(dirpath, f)
                # print(filepath)
                signal, sr = librosa.load(filepath, sr = SAMPLE_RATE)

                # process segments, extracting mfcc and storing data
                for s in range(num_segments):
                    start_sample = num_samples_per_segment * s
                    finish_sample = start_sample + num_samples_per_segment

                    mfcc = librosa.feature.mfcc(y = signal[start_sample:finish_sample], 
                                                sr = sr, n_mfcc = n_mfcc,
                                                n_fft = n_fft, 
                                                hop_length = hop_length)
                    mfcc = mfcc.T

                    # store mfcc for segment if it has the expected length
                    if len(mfcc) == expected_num_mfcc_vectors_per_segment:
                        # flag = True
                        data['mfcc'].append(mfcc.tolist())
                        data['labels'].append(i-1)
                        # print(f"{filepath}, segment: {s}")
    #     if flag:
    #         break
    # print(data)
    with open(json_path, 'w') as fp:
        json.dump(data, fp, indent = 4)


if __name__ == "__main__":
    save_mfcc(DATASET_PATH, JSON_PATH, num_segments = 10)
                    


