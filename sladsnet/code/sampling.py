import numpy as np
from tqdm import tqdm

from .base import Sample
from .results import Result


def run_sampling(sample: Sample, max_iterations: int = np.inf,
                 results: Result = None,
                 results_frequency_percentage: int = 1,
                 disable_progress_bar: bool = True):

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
    completed_run_flag = _check_stopping_criteria(sample, sampling_iters, max_iterations)

    # Until the stopping criteria has been met
    with tqdm(total=stop_ratio * 100, desc='% sampled', leave=False, ascii=True, disable=disable_progress_bar) as pbar:

        # Initialize progress bar state according to % measured
        percent_measured = round(sample.ratio_measured * 100, 2)
        pbar.n = np.clip(percent_measured, 0, stop_ratio * 100)
        pbar.refresh()

        # Until the program has completed
        while not completed_run_flag:

            # Find next measurement locations
            new_idxs = sample.find_new_measurement_idxs()

            # Perform measurements, reconstructions and ERD/RD computations
            if len(new_idxs) != 0:
                sample.perform_measurements(new_idxs)

                sample.reconstruct_and_compute_erd()
                sampling_iters += 1
                if results is not None and (sample.ratio_measured / results_frequency_percentage * 100 > store_at):
                    results.add(sample)
                    store_at += 1
            else:
                break

            # Check stopping criteria
            completed_run_flag = _check_stopping_criteria(sample, sampling_iters, max_iterations)

            # Update the progress bar
            percent_measured = round(sample.ratio_measured * 100, 2)
            pbar.n = np.clip(percent_measured, 0, stop_ratio * 100)
            pbar.refresh()
        pbar.close()
    return results


def _check_stopping_criteria(sample: Sample, current_iter: int, max_iterations: int):
    if sample.params_sample.scan_method in ['pointwise', 'random']:
        if sample.ratio_measured >= sample.params_sample.stop_ratio:
            return True
    if np.sum(sample.ERD) == 0:
        return True
    if current_iter >= max_iterations:
        return True

    return False
