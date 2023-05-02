import dash
from dash import dcc
from dash import html
import cv2

from flask import Flask, Response
import numpy as np
import time

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

def init_image():
    coffee = skimage.color.rgb2gray(skimage.data.coffee())

    coffee = skimage.transform.resize(coffee[:256,100:356], (128, 128))
    coffee = (coffee - coffee.min()) / (coffee.max() - coffee.min()) * 100

    return coffee

def run_scan(input_im):
    init_pattern = gcn(*input_im.T.shape, 0.01)


    # Specifying the trained nn model to use
    # We use the model generated for c=2.
    erd_model_to_load = Path.cwd().parent / 'training/cameraman/c_2/erd_model_relu.pkl'

    # Creating a simulated sample
    # Setting the initial batch size to 50 (inner_batch_size=50)
    # Supplying the nn model path
    sample_fast = utils.create_experiment_sample(numx=input_im.shape[1], numy=input_im.shape[0],
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

    n_scan_points = int(0.4 * input_im.size)
    while sample_fast.mask.sum() < n_scan_points:
        # Supply the measurement values.
        sample_fast.measurement_interface.finalize_external_measurement(input_im[new_idxs[:,0], new_idxs[:,1]])
        
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

        print(sample_fast.mask.shape)
        print(sample_fast.recon_image.shape)

        # Hardcoded for now
        out_im = np.zeros((128, 256))
        out_im[:, 0:128] = sample_fast.mask * 256
        out_im[:, 128:256] = sample_fast.recon_image

        yield out_im



def render_scan(recon_fast, mask, ratio):
    plt.figure(figsize=[6, 3])
    plt.subplot(1,2,1)
    plt.imshow(recon_fast)
    plt.title('Reconstruction')
    plt.subplot(1,2,2)
    plt.imshow(mask)
    plt.title('Measurement mask')
    plt.suptitle(f'Scan coverage is {ratio * 100: 3.2f}%')
    plt.show()
        


def get_image(seed=0):
    # strip slide
    size = 400
    res = np.mod((np.arange(size)[..., None] + np.arange(size)[None, ...]) + seed, [255])

    ret, jpeg = cv2.imencode('.jpg', res)

    return jpeg.tobytes()


def gen():
    im = init_image()
    scan_gen = run_scan(im)


    for frame_arr in scan_gen:
        ret, jpeg = cv2.imencode('.jpg', frame_arr)
        frame = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        time.sleep(0.5)


server = Flask(__name__)
app = dash.Dash(__name__, server=server)


@server.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


app.layout = html.Div([
    html.H1("Stream"),
    html.Img(src="/video_feed"),
    html.H1("Historical View"),
])

if __name__ == '__main__':
    app.run_server(debug=True)