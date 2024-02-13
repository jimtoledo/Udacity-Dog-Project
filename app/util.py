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

#Returns image of matplotlib figure of training/validation/testing metrics
def get_metric_graphs():
    #For each model, plot accuracy and loss for both training and validation
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20,18))
    ax1.set_title('Categorical cross-entropy loss during training')
    ax1.set_xlabel('Epoch')
    for network in dog_model_metrics:
        loss = dog_model_metrics[network]['history']['loss']
        ax1.plot(loss, label=network)
    ax1.set_xticks(range(len(dog_model_metrics[network]['history']['loss'])))
    ax1.legend()

    ax2.set_title('Accuracy during training')
    ax2.set_xlabel('Epoch')
    for network in dog_model_metrics:
        loss = dog_model_metrics[network]['history']['accuracy']
        ax2.plot(loss, label=network)
    ax2.set_xticks(range(len(dog_model_metrics[network]['history']['accuracy'])))
    ax2.legend()

    ax3.set_title('Categorical cross-entropy validation loss during training')
    ax3.set_xlabel('Epoch')
    for network in dog_model_metrics:
        loss = dog_model_metrics[network]['history']['val_loss']
        ax3.plot(loss, label=network)
    ax3.set_xticks(range(len(dog_model_metrics[network]['history']['val_loss'])))
    ax3.legend()

    ax4.set_title('Validation accuracy during training')
    ax4.set_xlabel('Epoch')
    for network in dog_model_metrics:
        loss = dog_model_metrics[network]['history']['val_accuracy']
        ax4.plot(loss, label=network)
    ax4.set_xticks(range(len(dog_model_metrics[network]['history']['val_accuracy'])))
    ax4.legend()

    ax5.set_title('Categorical cross-entropy test loss')
    for network in dog_model_metrics:
        ax5.bar(network, dog_model_metrics[network]['test_loss'])

    ax6.set_title('Test accuracy')
    for network in dog_model_metrics:
        ax6.bar(network, dog_model_metrics[network]['test_accuracy'])

    #Save as png to buffer
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    #Encode to html
    img_content = base64.b64encode(buffer.getvalue()).decode('ascii')
    img = f'data:image/png;base64,{img_content}'
    return img

