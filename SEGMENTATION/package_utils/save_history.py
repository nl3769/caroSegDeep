import matplotlib.pyplot as plt
import os

def save_loss(loss_train: list, loss_val: list, pathToSave: str, ModelName: str):
    ''' Saves the evolution of loss during training. '''
    plt.plot(loss_train)
    plt.plot(loss_val)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(pathToSave, ModelName + '_loss.png'))
    plt.close()

def save_IOU(IOU_train: list, IOU_val: list, pathToSave: str, ModelName: str):
    ''' Saves the evolution of IOU during training. '''
    plt.plot(IOU_train)
    plt.plot(IOU_val)
    plt.title('model IOU')
    plt.ylabel('IOU')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(os.path.join(pathToSave, ModelName + '_IOU.png'))
    plt.close()

def save_DICE(DICE_train: list, DICE_val: list, pathToSave: str, ModelName: str):
    ''' Saves the evolution of DICE during training. '''
    plt.plot(DICE_train)
    plt.plot(DICE_val)
    plt.title('model DICE')
    plt.ylabel('DICE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(os.path.join(pathToSave, ModelName + '_DICE.png'))
    plt.close()