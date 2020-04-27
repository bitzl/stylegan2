import argparse
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import re
import sys

import pretrained_networks

from run_generator import _parse_num_range
from tqdm import tqdm
import matplotlib.pyplot as plt

from matplotlib.animation import FuncAnimation

def generate_frames(network_pkl, zs_path, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    zs = np.load(zs_path)

    for z_idx, z in tqdm(enumerate(zs)):
        z = np.array([z])
        assert z.shape == (1, *Gs.input_shape[1:]) # [minibatch, component]
        rnd = np.random.RandomState(1000)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(dnnlib.make_run_dir_path('%06d.png' % z_idx))



def generate_movie(network_pkl, zs_path, truncation_psi):
    print('Loading networks from "%s"...' % network_pkl)
    _G, _D, Gs = pretrained_networks.load_networks(network_pkl)
    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    fig, ax = plt.subplots(1, figsize=(1, 1))
    fig.subplots_adjust(0, 0, 1, 1)
    ax.axis("off")
    

    zs = np.load(zs_path)
    def animate(z):
        z = np.array([z])
        assert z.shape == (1, *Gs.input_shape[1:]) # [minibatch, component]
        rnd = np.random.RandomState(1000)
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars}) # [height, width]
        images = Gs.run(z, None, **Gs_kwargs) # [minibatch, height, width, channel]
        ax.imshow(images[0], vmin=0, vmax=1)

    animation = FuncAnimation(
        # Your Matplotlib Figure object
        fig,
        # The function that does the updating of the Figure
        animate,
        # Frame information (here just frame number)
        zs,
        # Extra arguments to the animate function
        fargs=[],
        # Frame-time in ms; i.e. for a given frame-rate x, 1000/x
        interval=1000 / 25
    )

    # Try to set the DPI to the actual number of pixels you're plotting
    animation.save(dnnlib.make_run_dir_path("movie.mp4"), dpi=zs.shape[1])


def main():
    parser = argparse.ArgumentParser(
        description='''StyleGAN2 generator.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('frames', help='Generate frames')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--zs_path', type=str, help='File of feature vectors', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Resulting directory for run results (default: %(default)s)', default='results', metavar='DIR')

    parser_generate_images = subparsers.add_parser('movie', help='Generate movie')
    parser_generate_images.add_argument('--network', help='Network pickle filename', dest='network_pkl', required=True)
    parser_generate_images.add_argument('--zs_path', type=str, help='File of feature vectors', required=True)
    parser_generate_images.add_argument('--truncation-psi', type=float, help='Truncation psi (default: %(default)s)', default=0.5)
    parser_generate_images.add_argument('--result-dir', help='Resulting directory for run results (default: %(default)s)', default='results', metavar='DIR')

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    sc = dnnlib.SubmitConfig()
    sc.num_gpus = 1
    sc.submit_target = dnnlib.SubmitTarget.LOCAL
    sc.local.do_not_copy_source_files = True
    sc.run_dir_root = kwargs.pop('result_dir')
    sc.run_desc = subcmd

    func_name_map = {
        'frames': 'run_image_sequence.generate_frames',
        'movie': 'run_image_sequence.generate_movie',
    }
    dnnlib.submit_run(sc, func_name_map[subcmd], **kwargs)


if __name__ == "__main__":
    main()
