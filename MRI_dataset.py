import torch
import h5py
import os
import torch
from torch.utils.data import Dataset

class MRIdataset_all_dims_hdf5(Dataset):
    """
        Initializes the dataset from a flattened and shuffled HDF5 file.

        Args:
        - hdf5_path (str): Path to the HDF5 file.
        - transform (callable): Transform to apply to each slice.
    """
    def __init__(self, hdf5_path='MRI_dataset_all_dims.h5', transform=None):
        
        self.transform = transform
        self.hdf5_path = hdf5_path

        if not os.path.exists(hdf5_path):
            raise FileNotFoundError(f"HDF5 file not found at path: {hdf5_path}")

        # Open the file to read the dataset dimensions
        with h5py.File(hdf5_path, 'r') as hf:
            if "data" not in hf or "roi" not in hf:
                raise ValueError(f"Datasets 'data' or 'roi' not found in the HDF5 file.")
            self.num_slices = hf["data"].shape[0]  # Total number of slices
            print(f"Dataset loaded. Total slices: {self.num_slices}")

    def __len__(self):
        return self.num_slices

    def __getitem__(self, idx):
        """
        Retrieves a slice and its corresponding ROI.

        :param idx: The index of the slice to retrieve.
        :return data_slice, roi_slice: The data slice and the corresponding ROI slice.
        """

        with h5py.File(self.hdf5_path, 'r') as hf:
            # Load the slice directly from the HDF5 file
            data_slice = hf["data"][idx]
            roi_slice = hf["roi"][idx]

        # Apply transformations if any
        if self.transform:
            data_slice = self.transform(data_slice)
            roi_slice = self.transform(roi_slice)

        # Convert to PyTorch tensors
        return data_slice.type(torch.float32), roi_slice.type(torch.float32)


