import matplotlib.pyplot as plt
import os
def SaveLoss(history, pathToSave, ModelName):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(os.path.join(pathToSave, ModelName + '_loss.png'))
    plt.close()

def SaveIOU(history, pathToSave, ModelName):
    plt.plot(history.history['iou'])
    plt.plot(history.history['val_iou'])
    plt.title('model IOU')
    plt.ylabel('IOU')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(os.path.join(pathToSave, ModelName + '_IOU.png'))
    plt.close()

def SaveDICE(history, pathToSave, ModelName):
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('model DICE')
    plt.ylabel('DICE')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='lower right')
    plt.savefig(os.path.join(pathToSave, ModelName + '_DICE.png'))
    plt.close()