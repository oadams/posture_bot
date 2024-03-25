"""Preprocess data to prepare it for model training."""


import logging
from pathlib import Path
import sys
from typing import Union


def main(input_filepath: Union[str, Path], output_filepath: Union[str, Path]) -> None:
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")
    with open(input_filepath) as f:
        s = f.read()
    with open(output_filepath, "w") as f:
        print(s, file=f, end="")


if __name__ == "__main__":
    LOG_FMT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FMT)

    main(sys.argv[1], sys.argv[2])
