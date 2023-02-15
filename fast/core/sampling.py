# ----------------------------------------------------------------------- #
# Copyright (c) 2023, UChicago Argonne, LLC. All rights reserved.         #
#                                                                         #
# Software Name:    Fast Autonomous Scanning Toolkit (FAST)               #
# By: Argonne National Laboratory                                         #
# OPEN SOURCE LICENSE                                                     #
#                                                                         #
# Redistribution and use in source and binary forms, with or without      #
# modification, are permitted provided that the following conditions      #
# are met:                                                                #
#                                                                         #
# 1. Redistributions of source code must retain the above copyright       #
#    notice, this list of conditions and the following disclaimer.        #
#                                                                         #
# 2. Redistributions in binary form must reproduce the above copyright    #
#    notice, this list of conditions and the following disclaimer in      #
#    the documentation and/or other materials provided with the           #
#    distribution.                                                        #
#                                                                         #
# 3. Neither the name of the copyright holder nor the names of its        #
#    contributors may be used to endorse or promote products derived from #
#    this software without specific prior written permission.             #
#                                                                         #
# *********************************************************************** #
#                                                                         #
# DISCLAIMER                                                              #
#                                                                         #
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS     #
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT       #
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS       #
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE          #
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,    #
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,    #
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS   #
# OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED      #
# AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,  #
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF   #
# THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY            #
# OF SUCH DAMAGE.                                                         #
# *********************************************************************** #
import numpy as np
from tqdm import tqdm

from .base import Sample
from .results import Result


def run_sampling(
    sample: Sample,
    max_iterations: int = np.inf,
    results: Result = None,
    results_frequency_percentage: int = 1,
    results_frequency_step: int = 0,
    stop_percentage: float = 100,
    disable_progress_bar: bool = True,
    debug: bool = False,
    indices_to_actually_measure: int = None,
    verbose: bool = True,
):

    # Indicate that the stopping condition has not yet been met
    stop_ratio = sample.params_sample.stop_ratio
    sampling_iters = 0

    # Scan initial sets
    if sample.ratio_measured == 0:
        sample.perform_initial_scan()

    if results is not None:
        results.add(sample)
        store_at = 1

    # Check stopping criteria, just in case of a bad input
    completed_run_flag = check_stopping_criteria(sample, sampling_iters, max_iterations, stop_percentage, verbose)

    # Until the stopping criteria has been met
    with tqdm(
        total=stop_ratio * 100,
        desc="% sampled",
        leave=False,
        ascii=True,
        disable=disable_progress_bar,
    ) as pbar:

        # Initialize progress bar state according to % measured
        percent_measured = round(sample.ratio_measured * 100, 2)
        pbar.n = np.clip(percent_measured, 0, stop_ratio * 100)
        pbar.refresh()

        # Until the program has completed
        while not completed_run_flag:

            # Find next measurement locations
            new_idxs = sample.find_new_measurement_idxs()
            if indices_to_actually_measure is not None:
                new_idxs = new_idxs[:indices_to_actually_measure]
            # Perform measurements, reconstructions and ERD/RD computations
            if len(new_idxs) != 0:
                sample.perform_measurements(new_idxs)

                sample.reconstruct_and_compute_erd()
                if debug:
                    print(sample.iteration, sample.ERD.sum())
                sampling_iters += 1

                if results is not None:
                    if results_frequency_step > 0:
                        if sampling_iters % results_frequency_step == 0:
                            results.add(sample)
                    elif sample.ratio_measured / results_frequency_percentage * 100 > store_at:
                        results.add(sample)
                        store_at += 1
            else:
                if verbose:
                    print("No new scan position found. Stopping scan.")
                completed_run_flag = True
                break

            percent_measured = round(sample.ratio_measured * 100, 2)

            # Check stopping criteria
            completed_run_flag = check_stopping_criteria(sample, sampling_iters, max_iterations, stop_percentage, verbose=verbose)

            # Update the progress bar
            pbar.set_postfix({"total ERD": sample.ERD.sum()})
            pbar.n = np.clip(percent_measured, 0, stop_ratio * 100)

            pbar.refresh()
        pbar.close()
    return results, sampling_iters


def check_stopping_criteria(
    sample: Sample, current_iter: int, max_iterations: int = np.inf, stop_percentage: float = 100, verbose: bool = True
):
    percent_measured = round(sample.ratio_measured * 100, 2)
    if sample.params_sample.scan_method in ["pointwise", "random"]:
        if sample.ratio_measured >= sample.params_sample.stop_ratio:
            if verbose:
                print("Reached the stopping ratio set in the sample parameters. Stopping scan.")
            return True
    if np.sum(sample.ERD) == 0:
        if verbose:
            print("No more improvements expected. Stopping scan.")
        return True
    if current_iter >= max_iterations:
        if verbose:
            print("Reached the maximum iterations for this sampling run. Stopping scan.")
        return True
    if percent_measured > stop_percentage:
        if verbose:
            print("Reached the maximum sampling percentage for this sampling run. Stopping scan.")
        return True
    return False
