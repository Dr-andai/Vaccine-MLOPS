import logging
from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics import accuracy_score

class Evaluation(ABC):

    @abstractmethod
    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        pass


class accuracy(Evaluation):

    def calculate_scores(self, y_true: np.ndarray, y_pred: np.ndarray):
        try:
            logging.info("Calculating Accuracy")
            acc = accuracy_score(y_true, y_pred)
            logging.info("accuracy: {}".format(acc))
            return acc
        
        except Exception as e:
            logging.error ("Error in calculating accuracy: {}".format(e))
            raise e
    