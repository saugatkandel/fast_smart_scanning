
import fast.s26_analysis.utils as utils 
from fast.utils.generate_scan_pattern import generate_scan_pattern as gcn

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib as mpl
import skimage
from tqdm.notebook import tqdm
import tifffile as tif
import joblib
import numpy as np

import tkinter as tk
from PIL import Image, ImageTk

root = tk.Tk()

array = np.ones((500,500))*150
img =  ImageTk.PhotoImage(image=Image.fromarray(array))

canvas = tk.Canvas(root,width=300,height=300)
w = tk.Scale(root, from_=0, to=200, orient=tk.HORIZONTAL)
canvas.pack()
w.pack()
canvas.create_image(20,20, anchor="nw", image=img)

root.mainloop()



def init_image():
    coffee = skimage.color.rgb2gray(skimage.data.coffee())

    coffee = skimage.transform.resize(coffee[:256,100:356], (128, 128))
    coffee = (coffee - coffee.min()) / (coffee.max() - coffee.min()) * 100

    return coffee

def run_scan():
    init_pattern = gcn(*coffee.T.shape, 0.01)


    # Specifying the trained nn model to use
    # We use the model generated for c=2.
    erd_model_to_load = Path.cwd().parent / 'training/cameraman/c_2/erd_model_relu.pkl'

    # Creating a simulated sample
    # Setting the initial batch size to 50 (inner_batch_size=50)
    # Supplying the nn model path
    sample_fast = utils.create_experiment_sample(numx=coffee.shape[1], numy=coffee.shape[0],
                                            inner_batch_size=50,
                                            initial_idxs=init_pattern,
                                            erd_model_file_path=erd_model_to_load)

    # Actual simulation run. This can be fairly time consuming
    masks_all = []
    recons_fast_all = []
    ratios_all = []
    tot_erds_all = []
    count = 0
    new_idxs = init_pattern

    n_scan_points = int(0.4 * coffee.size)
    pbar = tqdm(total=n_scan_points, desc='Scanned points')
    # Continue until we measure 40% of the points.
    while sample_fast.mask.sum() < n_scan_points:
        # Supply the measurement values.
        sample_fast.measurement_interface.finalize_external_measurement(coffee[new_idxs[:,0], new_idxs[:,1]])
        
        # Supply in measurement positions
        sample_fast.perform_measurements(new_idxs)
        
        # Use the measurement values to reconstruct the sample and calculate the ERDs
        sample_fast.reconstruct_and_compute_erd()
        
        # Compute new positions.
        new_idxs = sample_fast.find_new_measurement_idxs()[:50]
        
        ratio = sample_fast.ratio_measured
        ratios_all.append(ratio)
        tot_erds_all.append(sample_fast.ERD.sum())
        recons_fast_all.append(sample_fast.recon_image.copy())
        masks_all.append(sample_fast.mask.copy())
        pbar.update(int(sample_fast.mask.sum() - pbar.n))



def render_scan():
    for ix in range(10, len(ratios_all), 20):
        plt.figure(figsize=[6, 3])
        plt.subplot(1,2,1)
        plt.imshow(recons_fast_all[ix])
        plt.title('Reconstruction')
        plt.subplot(1,2,2)
        plt.imshow(masks_all[ix])
        plt.title('Measurement mask')
        plt.suptitle(f'Scan coverage is {ratios_all[ix] * 100: 3.2f}%')
        plt.show()
        
