import argparse
import pathlib


def parse_terminal():

    # Define parser
    parser = argparse.ArgumentParser(description="Aging of Evolving Genomes in Silico")
    parser.add_argument(
        "-c",
        "--config_path",
        type=str,
        help="path to config file",
        default=None,
    )
    parser.add_argument(
        "-p",
        "--pickle_path",
        type=str,
        help="path to pickle file",
        default=None,
    )
    parser.add_argument(
        "--overwrite",
        "-o",
        action="store_true",
        help="overwrite old data with new simulation",
        default=None,
    )
    parser.add_argument(
        "--visor",
        "-v",
        action="store_true",
        help="run visor – the interactive GUI",
        default=False,
    )

    # Parse arguments
    args = parser.parse_args()

    # Decision tree
    config_path = pathlib.Path(args.config_path).absolute() if args.config_path else None
    pickle_path = pathlib.Path(args.pickle_path).absolute() if args.pickle_path else None

    return config_path, pickle_path, args.overwrite, args.visor
