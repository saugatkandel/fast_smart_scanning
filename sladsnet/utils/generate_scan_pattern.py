import numpy as np
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
from skimage.transform import rescale

from sladsnet.input_params import ERDInputParams, GeneralInputParams, SampleParams
from sladsnet.code.measurement_interface import TransmissionSimulationMeasurementInterface, ExternalMeasurementInterface
from sladsnet.code.erd import SladsSklearnModel
from sladsnet.code.results import Result
from sladsnet.code.base import ExperimentalSample, Sample
from sladsnet.code.sampling import run_sampling
import argparse


def generate_scan_pattern(numx: int, numy: int,
                          ratio_initial_scan: float = 0.1,
                          num_scan_points: int=None,
                          save: bool = False):
    if num_scan_points is not None:
        print("Numer of scan points is provided. This overrides ratio_initial_scan.")
        ratio_initial_scan = None
    sample_params = SampleParams(image_shape=(numy, numx),
                                 initial_scan_points_num=num_scan_points,
                                 initial_scan_ratio=ratio_initial_scan,
                                 stop_ratio=0.3,
                                 random_seed=11)
    num_scan_points = np.shape(sample_params.initial_idxs)[0]
    print("Initial ratio is", num_scan_points / sample_params.image_size)
    if save:
        np.savetxt(f'initial_points_{numx}_{numy}_points_{num_scan_points}.csv',
                   sample_params.initial_idxs, delimiter=',', fmt='%10d')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate initial scanning structure')
    parser.add_argument('-x','--numx', type=int, help='number of x points', required=True)
    parser.add_argument('-y', '--numy', type=int, help='number of y points', required=True)
    parser.add_argument('-s', '--nums', type=int, help='number of initial scan points', required=True)
    args = parser.parse_args()

    generate_scan_pattern(args.numx, args.numy, num_scan_points=args.nums)
    



