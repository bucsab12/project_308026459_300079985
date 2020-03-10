import subprocess
from glob import glob
import os


def n4itk_norm(input_path, n_dims=3, n_iters='[20,20,10,5]'):
    """
    This function executes the N4BiasFieldCorrection algorithm and writes n4itk normalized image to parent_dir under orig_filename_n.nii.gz
    :param input_path: input_path to the nifti file to be normalized
    :param n_dims: Number of dimensions to be used by the N4BiasFieldCorrection algorithm
    :param n_iters: Number of iterations that will be used by the N4BiasFieldCorrection algorithm
    :return: None
    """
    output_fn = input_path[:-7] + '_n.nii.gz'
    print(input_path)
    subprocess.call('N4BiasFieldCorrection -i ' + '"' + input_path + '"' + ' -d ' + str(n_dims) + ' -c ' + n_iters + ' -o ' + '"' + output_fn + '"', shell=True)


def modality_normalization(path):
    """
    This function is used to run N4 bias field correction on the selected modalities of brains in the dataset. Needs to be run in LINUX environment.
    Script command: python3 /mnt/c/Users/bucsa/PycharmProjects/deep_brain/n4_norm.py
    :param path: Path to the folder containing all the data according to the Linux environment
    :return: None
    """
    # /home/bucsa/build/ANTS-build
    # Set environment variables in the Linux environment
    os.system('export ANTSPATH=/opt/ANTs/bin/')
    os.system('export PATH=${ANTSPATH}:$PATH')
    # Change DIR to the ANTS DIR
    os.chdir('/home/bucsa/build/ANTS-build')
    CWD = os.getcwd()
    os.chdir(path)
    t1 = glob(os.path.join(path, '**/*t1.nii.gz'))
    t1ce = glob(os.path.join(path, '**/*t1ce.nii.gz'))
    t2 = glob(os.path.join(path, '**/*t2.nii.gz'))
    flair = glob(os.path.join(path, '**/*flair.nii.gz'))
    for modality in [t1, t1ce, t2, flair]:
        for file_path in modality:
            n4itk_norm(file_path, n_dims=3, n_iters='[50,50,30,20]')  # normalize files
    os.chdir(CWD)


PATH = "TRAIN"
modality_normalization(PATH)
