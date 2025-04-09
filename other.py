import numpy as np
import nibabel as nib
import os

def get_patient_file_paths(num_patient):
    """
    Get the file paths for the MRI and ROI files of a patient.

    :param num_patient: The patient number.
    :return: The MRI file path and the ROI file path. None, None: if the files do not exist.
    """
    file_paths = [
        f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-T2sel_FLAIR.nii.gz',
        f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-tse3dvfl_FLAIR.nii.gz'
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-corhipp4mm_FLAIR.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-traAcpc4mm_FLAIR.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-traacpcVNS_FLAIR.nii.gz',
        f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-32ch10_FLAIR.nii.gz'
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-cor3mmOPT_FLAIR.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-tra4mm_FLAIR.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-corAcpc4mm_FLAIR.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-cor4mm_FLAIR.nii.gz'
    ]
    roi_paths = [
        f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-T2sel_FLAIR_roi.nii.gz',
        f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-tse3dvfl_FLAIR_roi.nii.gz'
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-corhipp4mm_FLAIR_roi.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-traAcpc4mm_FLAIR_roi.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-traacpcVNS_FLAIR_roi.nii.gz',
        f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-32ch10_FLAIR_roi.nii.gz'
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-cor3mmOPT_FLAIR_roi.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-tra4mm_FLAIR_roi.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-corAcpc4mm_FLAIR_roi.nii.gz',
        # f'sub-00{num_patient:03d}/anat/sub-00{num_patient:03d}_acq-cor4mm_FLAIR_roi.nii.gz'
    ]
    for file_path, roi_path in zip(file_paths, roi_paths):
        if os.path.exists(file_path):
            if os.path.exists(roi_path):
                return file_path, roi_path
            else:
                return file_path, None
    return None, None

def pad_slice_to_target_shape(slice_data, target_shape=(256, 256)):
    """Pads a slice to the target shape with zeros."""

    pad_height = max(0, target_shape[0] - slice_data.shape[0])
    pad_width = max(0, target_shape[1] - slice_data.shape[1])

    pad_top = pad_height // 2
    pad_bottom = pad_height - pad_top
    pad_left = pad_width // 2
    pad_right = pad_width - pad_left

    padded_slice = np.pad(slice_data, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=0)
    return padded_slice

def load_nii(file_path):
    """
    Load a NIfTI file and return the data as a numpy array.
    """
    return nib.load(file_path).get_fdata()