#!/bin/python

import os

import pandas as pd
import numpy as np

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from config import Config


class DataSplitter:
    def __init__(self, df):
        self.df = df
        self.hospitals = self.df['Last_Facility'].unique()

    def _preprocess(self, df):
        imputed = KNNImputer().fit_transform(df)
        df = pd.DataFrame(imputed, index=df.index, columns=df.columns)

        # Separate outcome and scale
        oc = df['MORTALITY'].astype('int8')
        df = df.drop('MORTALITY', axis=1)

        scaled = MinMaxScaler().fit_transform(df)
        df = pd.DataFrame(scaled, index=df.index, columns=df.columns)
        df['MORTALITY'] = oc.values

        return df

    def per_hospital_processing(self):
        # Store here, return this, dance around
        datasets_by_facility = {}

        # Remove values with missingness below threshold
        # Preprocessing should take care of column rearrangement
        df_count = self.df.count() / self.df.shape[0]
        df_count = df_count[df_count >= Config.presence_threshold]
        self.df = self.df[df_count.index]

        # Do a per-facility split and impute / scale accordingly
        for hosp in self.hospitals:
            df_hosp_data = self.df.query('Last_Facility == @hosp')
            df_hosp_data = df_hosp_data.drop(['Last_Facility', 'Admit_Date'], axis=1)
            df_hosp_data = self._preprocess(df_hosp_data)

            datasets_by_facility[hosp] = df_hosp_data

        return datasets_by_facility


class StratifiedDatasetCreator:
    def __init__(self, seed):
        self.random_state = seed

        self.df_mortality = pd.read_pickle(
            os.path.join(Config.data_dir, 'MORTALITY_7.pickle'))

        splitter = DataSplitter(self.df_mortality)
        self.datasets_by_facility = splitter.per_hospital_processing()

        # Store SKF split data here
        self.training_datasets = {}
        self.testing_datasets = {}

    def retching_maw(self):
        for facility, df_facility in self.datasets_by_facility.items():
            X = df_facility.drop('MORTALITY', axis=1)
            y = df_facility[['MORTALITY']]

            # Store dataframes as iterables
            self.training_datasets[facility] = []
            self.testing_datasets[facility] = []

            # Splitter
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.3, stratify=y,
                random_state=self.random_state)

            training_data = pd.concat((X_train, y_train), axis=1)
            testing_data = pd.concat((X_test, y_test), axis=1)

            self.training_datasets[facility].append(training_data)
            self.testing_datasets[facility].append(testing_data)
