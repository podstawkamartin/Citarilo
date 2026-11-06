"""Command-line entry point."""

import argparse

from citarilo import __version__


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="citarilo",
        description="Evidence-backed customer support for TrailSupply.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    parser.parse_args()
