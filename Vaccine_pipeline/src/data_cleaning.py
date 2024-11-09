import logging
from abc import ABC, abstractclassmethod
from typing import Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


class DataStrategy(ABC):

    @abstractclassmethod
    def handle_data(self, data: pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:
        pass


class DataPreProcessStrategy(DataStrategy):
    def handle_data(self, data: pd.DataFrame)-> pd.DataFrame:
        try: 
            data = data.drop(

                [
                    'covid_vacc','other_vaccine','vacc_2','vacc_3','prefer','pers_7',
                    'record_id','redcap_survey_identifier','consent_timestamp','date',
                    'consent','consent_complete','survey_timestamp','nationality',
                    'race','other_race','other_comorb','other_worried','prefer_other',
                    'vaccinate_children','survey_complete','siteFinal',
                ],
                axis=1
            )

            hospital_mapping = {
                'Aga Khan University, Nairobi':'private', 
                'Avenue Hospital':'private',
                'Mediheal Hospital':'private',
                'Penda Health':'private',
                'PCEA Hospital': 'faith',
                'Coast General Hospital': 'public'
            }

            data['studysite'] = data['studysite'].replace(hospital_mapping)

            # bins = [18, 25, 35, 55]  
            # labels = [1, 2, 3]  

            # data['age_code'] = pd.cut(data['age'], bins=bins, labels=labels, right=False)
            # data = data.drop(columns=['age'])

            for column in data:
                data[column]=data[column].astype('category')
                mode = data[column].mode()[0]
                data[column].fillna(mode, inplace=True)

            le = LabelEncoder()
            
            for column in data.columns:
                if column != 'VaccineStatus':
                    data[column] = le.fit_transform(data[column])
            
            return data
        except Exception as e:
            logging.error("Error in preprocessing data: {}".format(e))
            raise e
        

class DataDivideStrategy(DataStrategy):

    def handle_data(self, data:pd.DataFrame) -> Union[pd.DataFrame, pd.Series]:

        try:
            X = data.drop(['VaccineStatus'], axis=1)
            y = data['VaccineStatus']
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            return X_train, X_test, y_train, y_test
        except Exception as e:
            logging.error("Error in dividing data: {}".format(e))
            raise e

class DataCleaning:
    def __init__(self, data: pd.DataFrame, strategy: DataStrategy):
        self.data = data
        self.strategy = strategy

    def handle_data(self) -> Union[pd.DataFrame, pd.Series]:
        try:
            return self.strategy.handle_data(self.data)
        except Exception as e:
            logging.error("Error in handling data:{}".format(e))
            raise e

    