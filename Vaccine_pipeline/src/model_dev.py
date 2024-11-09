import logging
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier

class Model(ABC):
    
    @abstractmethod
    def train(self, X_train, y_train):

        pass


class RandomForestClassifierModel(Model):

    def train (self, X_train, y_train, **kwargs):

        try: 
            model = RandomForestClassifier(**kwargs)
            model.fit(X_train, y_train)
            logging.info("Model training completed")
            return model
        
        except Exception as e:
            logging.error("Error in training model:{}".format(e))
            raise e