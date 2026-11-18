"""Execute a native detector stage."""

import argparse

from holdtrack.detect.runner import worker


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("request")
    worker(parser.parse_args().request)


if __name__ == "__main__":
    main()
