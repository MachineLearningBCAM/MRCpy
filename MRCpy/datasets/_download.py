# -*- coding: utf-8 -*-
"""
Utility for downloading large dataset files on demand from GitHub releases.
"""
import os
import urllib.request
import sys

# Base URL for large dataset files hosted on GitHub releases
_BASE_URL = ("https://github.com/MachineLearningBCAM/MRCpy/"
             "releases/download/v3.0.0-data/")

# Large files that are NOT bundled with the PyPI package
_LARGE_FILES = [
    'catsvsdogs_features_resnet18_1.csv.zip',
    'catsvsdogs_features_resnet18_2.csv.zip',
    'mnist_features_resnet18_1.csv.zip',
    'mnist_features_resnet18_2.csv.zip',
    'mnist_features_resnet18_3.csv.zip',
    'mnist_features_resnet18_4.csv.zip',
    'mnist_features_resnet18_5.csv.zip',
    'yearbook_features_resnet18_1.csv.zip',
    'yearbook_features_resnet18_2.csv.zip',
    'yearbook_attributes.csv',
    'comp-vs-sci_Train.csv',
    'comp-vs-sci_Test.csv',
    'comp-vs-talk_Train.csv',
    'comp-vs-talk_Test.csv',
    'rec-vs-sci_Train.csv',
    'rec-vs-sci_Test.csv',
    'rec-vs-talk_Train.csv',
    'rec-vs-talk_Test.csv',
    'sci-vs-talk_Train.csv',
    'sci-vs-talk_Test.csv',
]


def _reporthook(block_num, block_size, total_size):
    """Progress bar for urllib downloads."""
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100, downloaded * 100 / total_size)
        mb_downloaded = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(
            f"\r  Downloading: {mb_downloaded:.1f}/{mb_total:.1f} MB "
            f"({percent:.0f}%)")
    else:
        mb_downloaded = downloaded / (1024 * 1024)
        sys.stdout.write(f"\r  Downloading: {mb_downloaded:.1f} MB")
    sys.stdout.flush()


def ensure_data_file(filename):
    """Ensure a data file exists locally, downloading it if necessary.

    Parameters
    ----------
    filename : str
        Name of the file (e.g., 'mnist_features_resnet18_1.csv.zip').

    Returns
    -------
    filepath : str
        Full path to the local file.

    Raises
    ------
    RuntimeError
        If the file cannot be downloaded.
    """
    data_dir = os.path.join(os.path.dirname(__file__), 'data')
    filepath = os.path.join(data_dir, filename)

    if os.path.exists(filepath):
        return filepath

    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)

    url = _BASE_URL + filename
    print(f"Dataset file '{filename}' not found locally. "
          f"Downloading from GitHub...")

    try:
        urllib.request.urlretrieve(url, filepath, reporthook=_reporthook)
        print()  # newline after progress bar
    except Exception as e:
        # Clean up partial download
        if os.path.exists(filepath):
            os.remove(filepath)
        raise RuntimeError(
            f"Failed to download '{filename}' from {url}. "
            f"You can download it manually from:\n"
            f"  {url}\n"
            f"and place it in:\n"
            f"  {data_dir}\n"
            f"Error: {e}"
        ) from e

    return filepath
