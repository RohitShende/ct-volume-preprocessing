import os
import pydicom
import joblib
import dicom_numpy
from fastai.medical.imaging import get_dicom_files
from preprocess_volumes import CleanCTScans
from joblib import Parallel, delayed


def get_ct_scan_as_list_of_pydicoms(folder, destination_folder):
    list_of_dicom_files = get_dicom_files(folder)
    scan = [pydicom.dcmread(f) for f in list_of_dicom_files]
    scan = dicom_numpy.sort_by_slice_position(scan)
    file_path = f"{destination_folder}/{scan[0].PatientID}.pkl"
    joblib.dump(scan, file_path)
    return f'Saved file - {file_path}'


def save_each_scan_as_pickle(source_folder, destination_folder):
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)

    folders = [os.path.join(source_folder, folder) for folder in os.listdir(source_folder) if
               not folder.startswith('.')]

    results = Parallel(n_jobs=max(10,len(folders)))(delayed(get_ct_scan_as_list_of_pydicoms)(im_file, destination_folder)
                                            for im_file in folders)
    print(*results, sep='\n')
    print(len(folders))


source = "../data/Tampered_Scans/Experiment-2-Open/"
destination = './pickled_scans/test'
# save_each_scan_as_pickle(source, destination)
source = "../data/Tampered_Scans/Experiment-1-Blind/"
destination = './pickled_scans/train'
# save_each_scan_as_pickle(source, destination)

cleaner = CleanCTScans("run", "", "", "logs", "./pickled_scans/train", "../cleaned_scans")
