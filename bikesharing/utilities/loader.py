import pandas
from sklearn.model_selection import train_test_split


class Loader:
    """
    Provides utility functions handling data from .csv files.
    """

    def __init__(self, csv_path):
        """
        Initialises the loader.
        @param csv_path: the path to the .csv file
        @type csv_path: Path
        """
        self.csv_path = csv_path
        self.csv_data = pandas.read_csv(self.csv_path)
        self.csv_data.sample(replace=True, frac=1.0, random_state=1)

    def get_header(self):
        """
        Gets the header of the .csv file.
        @return: the header
        @rtype: list
        """
        return list(self.csv_data.columns.values)

    def get_split_data(self):
        """
        Gets the data from the .csv file split with the ratio:
            60% - train
            20% - test
            20% - validation
        @return: the train, test and validation
        @rtype: list
        """
        train, test = train_test_split(self.csv_data, test_size=0.2, random_state=1)
        train, validation = train_test_split(train, test_size=0.25, random_state=1)

        return train, test, validation

    def get_data(self):
        """
        Gets all the data from the .csv file.
        @return: the data
        @rtype: DataFrame
        """
        return self.csv_data
