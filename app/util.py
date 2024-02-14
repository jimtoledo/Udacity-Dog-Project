import pickle
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

dog_model_metrics = {}
with open('../model.metrics', 'rb') as file:
    dog_model_metrics = pickle.load(file)

model_preds = None
with open('../model_preds.df', 'rb') as file:
    model_preds = pickle.load(file)

