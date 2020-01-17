import os
import pickle
import random

from sklearn.ensemble import RandomForestRegressor

from bikesharing.utilities.loader import Loader


class RandomForest:
    """
    RandomForestRegressor wrapper for the bike sharing dataset.
    """

    def __init__(self, csv_path):
        """
        Initialises the random forest wrapper.
        @param csv_path: the path to the .csv file
        @type csv_path: Path
        """
        loader = Loader(csv_path)
        random.seed(5)

        self.csv_data = {"full": loader.get_data()}
        self.target = ['cnt']
        self.features = ['season', 'mnth', 'hr', 'holiday', 'weekday', 'workingday',
                         'weathersit', 'temp', 'atemp', 'hum', 'windspeed']

        self.samples = {"full": self.csv_data["full"][self.features].values}
        self.labels = {"full": self.csv_data["full"][self.target].values.ravel()}

        self.model = RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
                                           max_features='auto', max_leaf_nodes=None,
                                           min_impurity_decrease=0.0, min_impurity_split=None,
                                           min_samples_leaf=1, min_samples_split=4,
                                           min_weight_fraction_leaf=0.0, n_estimators=200, n_jobs=None,
                                           oob_score=False, random_state=100, verbose=0, warm_start=False)

    def save_model(self, save_path):
        """
        Saves the model to a local file.
        @param save_path: The save path
        @type save_path: Path
        @return:
            True if the model was successfully saved
            False if the model is not initialised
        @rtype: bool
        """
        if self.model is not None:
            pickle.dump(self.model, open(save_path, "wb"))
            return True
        else:
            return False

    def load_model(self, load_path):
        """
        Loads the model from a local file.
        @param load_path: the load path
        @type load_path: Path
        @return:
            True if the model was successfully loaded
            False if the model file does not exist
        @rtype: bool
        """
        if os.path.exists(load_path):
            self.model = pickle.load(open(load_path, 'rb'))
            return True
        else:
            return False
