# -*- coding: utf-8 -*-

import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

class MessageEncoder(json.JSONEncoder):

    def default(self, data):
        if isinstance(data, np.ndarray):
            return {'__type__':'np.ndarray', 'data':data.tolist()}
        elif isinstance(data, LogisticRegression) or isinstance(data, SGDClassifier):
            if isinstance(data, LogisticRegression):
                model_dict = {'__type__':'LogisticRegression', 'params':data.get_params()}
            elif isinstance(data, SGDClassifier):
                model_dict = {'__type__':'SGDClassifier', 'params':data.get_params()}
            if hasattr(data, 'intercept_'):
                model_dict['intercept_'] = data.intercept_
            if hasattr(data, 'classes_'):
                model_dict['classes_'] = data.classes_
            return model_dict
        elif isinstance(data, pd.Series):
            pd_series_dict = {'__type__':'pd.Series', 'data':{**data.to_dict()}}
            return pd_series_dict
        return super().default(data)

class MessageDecoder(json.JSONDecoder):

    def __init__(self, *args, **kwargs):
        json.JSONDecoder.__init__(self, object_hook=self.object_hook, *args, **kwargs)
    def object_hook(self, json_msg):
        if '__type__' in json_msg:
            if json_msg['__type__'] == 'np.ndarray':
                return np.array(json_msg['data'])
            elif json_msg['__type__'] == 'LogisticRegression' or json_msg['__type__'] == 'SGDClassifier':
                if json_msg['__type__'] == 'LogisticRegression':
                    model = LogisticRegression()
                elif json_msg['__type__'] == 'SGDClassifier':
                    model = SGDClassifier()
                model.set_params(**json_msg['params'])
                if 'intercept_' in json_msg:
                    model.intercept_ = json_msg['intercept_']
                if 'classes_' in json_msg:
                    model.classes_ = json_msg['classes_']
                return model
            elif json_msg['__type__'] == 'pd.Series':
                return pd.Series(json_msg['data'])
            else:
                raise TypeError()
        return json_msg
