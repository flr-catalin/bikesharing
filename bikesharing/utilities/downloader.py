import os
import zipfile
from datetime import date
from pathlib import Path

import wget

from bikesharing.utilities import dataset


class Downloader:
    """
    Provides utility functions for fetching a utilities from an URL and for extracting .zip archives.
    """

    def __init__(self, filename=dataset.FILENAME, url=dataset.URL, save_path=dataset.SAVE_PATH):
        """
        Initialises the downloader.

        :param filename: the name of the file to download, without extension
        :type filename: str
        :param url: the URL of the file to download
        :type url: str
        :param save_path: the save path
        :type save_path: Path
        """

        downloaded_file = save_path.joinpath(Path(filename))
        self.downloaded_file = str(downloaded_file) + "-" + str(date.today()) + ".zip"
        self.url = url
        self.filename = filename
        self.download_to = save_path

    def fetch_dataset(self):
        """
        Downloads the file from the instance URL.

        :return:
            True if the file was successfully downloaded
            False if the file already exists
        :rtype: bool
        """

        if not os.path.exists(self.downloaded_file):
            os.makedirs(str(self.download_to), exist_ok=True)
            wget.download(self.url, str(self.downloaded_file))
            print("Downloaded " + self.downloaded_file)
            return True
        else:
            print(self.downloaded_file + " already exists")
            return False

    def extract_zip(self):
        """
        Extracts a .zip file.
        """

        zip_file = zipfile.ZipFile(self.downloaded_file)
        zip_file.extractall(self.download_to)
        zip_file.close()
        print("Successfully extracted " + str(self.filename))

    def run(self):
        """
        Fetches and extracts the utilities if it does not already exist on the disk.
        """

        if self.fetch_dataset():
            self.extract_zip()
