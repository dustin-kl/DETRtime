import numpy as np

print('Loading Data')
file = f'../../segmentation/ICML_dots_min_segmentation_minseq_4000_margin_2_amp_thresh_150/tensors/train.npz'
output_file = f'../../segmentation/ICML_dots_min_segmentation_minseq_4000_margin_2_amp_thresh_150/tensors/train_short.npz'

full_data = np.load(file)
print(len(full_data['EEG']))
short_EEG = full_data['EEG'][:1000]
short_labels = full_data['labels'][:1000]

np.savez(output_file, EEG=short_EEG, labels=short_labels)