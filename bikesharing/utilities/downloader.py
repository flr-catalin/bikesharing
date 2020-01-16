import os
import zipfile
from datetime import date
from pathlib import Path

import wget

import utilities.bikesharing_info as bikesharing


class Downloader:
    """
    Provides utility functions for fetching a dataset from an URL and for extracting .zip archives.
    """

    def __init__(self, filename=bikesharing.FILENAME, url=bikesharing.URL, save_path=bikesharing.SAVE_PATH):
        downloaded_file = save_path.joinpath(Path(filename))
        self.downloaded_file = str(downloaded_file) + "-" + str(date.today()) + ".zip"
        self.url = url
        self.filename = filename
        self.download_to = save_path

    def fetch_dataset(self):
        if not os.path.exists(self.downloaded_file):
            os.makedirs(self.download_to, exist_ok=True)
            wget.download(self.url, str(self.downloaded_file))
            print("Downloaded " + self.downloaded_file)
            return True
        else:
            print(self.downloaded_file + " already exists")
            return False

    def extract_zip(self):
        zip_file = zipfile.ZipFile(self.downloaded_file)
        zip_file.extractall(self.download_to)
        zip_file.close()
        print("Successfully extracted " + str(self.filename))

    def run(self):
        if self.fetch_dataset():
            self.extract_zip()
