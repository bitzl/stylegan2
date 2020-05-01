import numpy as np
import argparse
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description='''Create sequence of z values to generate images for.

Run 'python %(prog)s <subcommand> --help' for subcommand help.''',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_generate_images = subparsers.add_parser('random-walk', help='Create a random walk')
    parser_generate_images.add_argument('--length', type=int, help='Length of the random walk', required=True)
    # parser_generate_images.add_argument('--image_size', type=int, help='Size of the images to create later', required=True)
    parser_generate_images.add_argument('--speed', type=float, help='Max variance per dimension between two steps (default: %(default)s)', default=0.01, required=False)
    parser_generate_images.add_argument('--target', type=str, help='Target array file (.npy)', required=True)

    parser_generate_images = subparsers.add_parser('line', help='Create a straight line')
    parser_generate_images.add_argument('--min_length', type=float, default=1.0, help='Minimal length of the straight line', required=False)
    parser_generate_images.add_argument('--speed', default=0.01, type=float, help='Max variance per dimension between two steps (default: %(default)s)', required=False)
    parser_generate_images.add_argument('--target', type=str, help='Target array file (.npy)', required=True)

    args = parser.parse_args()
    kwargs = vars(args)
    subcmd = kwargs.pop('command')

    if subcmd is None:
        print ('Error: missing subcommand.  Re-run with --help for usage.')
        sys.exit(1)

    # Prepare target
    target = kwargs.pop('target')
    target_path = Path(target).parent
    target_path.mkdir(parents=True, exist_ok=True)

    if subcmd == "random-walk":
        generate_random_walk(target, kwargs.pop("length"), kwargs.pop("speed"))
    elif subcmd == "line":
        generate_line(target, kwargs.pop("min_length"), kwargs.pop("speed"))


def generate_random_walk(target: str, length: int, speed: float):
    latent_size = 512
    x = np.random.normal(scale=speed, size=(length, latent_size))
    np.save(target, x.cumsum(axis=1))


def generate_line(target: str, min_length: int, speed: float):
    latent_size = 512
    z0 = np.random.normal(size=(latent_size))
    distance = 0
    guesses = 0
    while distance < min_length and guesses < 50:
        z1 = np.random.normal(size=(latent_size))
        distance = np.linalg.norm(z1 - z0)
        guesses += 1
    
    if distance < min_length:
        print(f"Could not guess a line with minimum length {min_length}, giving up")
        return

    steps = int(distance / speed)
    np.save(target, np.linspace(z0, z1, steps))
    print(f"Saved line with {steps} points to {target}")


if __name__ == "__main__":
    main()
