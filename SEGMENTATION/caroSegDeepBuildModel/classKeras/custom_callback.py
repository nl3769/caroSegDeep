import tensorflow.keras
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import os

LR_SCHEDULE = [
    # (epoch to start, learning rate) tuples
    (1, 0.005),
    (15, 0.001),
    (30, 0.0005),
    (40, 0.0001),
    (50, 0.00005)
]


# ----------------------------------------------------------------
def lr_schedule(epoch, lr):
    """Helper function to retrieve the scheduled learning rate based on epoch."""
    if epoch < LR_SCHEDULE[0][0] or epoch > LR_SCHEDULE[-1][0]:
        return lr
    for i in range(len(LR_SCHEDULE)):
        if epoch == LR_SCHEDULE[i][0]:
            return LR_SCHEDULE[i][1]
    return lr


# ----------------------------------------------------------------
class CustomPlotPrediction(tensorflow.keras.callbacks.Callback):

    def __init__(self, images, p):
        super().__init__()
        'Initialization'
        self.images = images
        self.pred = []
        self.p = p

    #
    # def on_epoch_end(self, epoch, logs=None):
    #     # print(f"Finished epoch {epoch}, loss is {logs['loss']}, iou is {logs['iou']}, dice is {logs['dice_coef']}"


    def on_epoch_begin(self, epoch, logs=None):
        if self.images.shape[0] == 1:
            print("Prediction of ", 1, " image.")
        else:
            print("Prediction of ", self.images.shape[0], " images.")

        self.pred.append(self.model.predict(self.images, batch_size=1, verbose=1))

    def on_train_end(self, logs=None):
        pred = np.asarray(self.pred)

        for k in range(pred.shape[0]):
            for i in range(pred.shape[1]):
                plt.imsave(os.path.join(self.p.PATH_TO_SAVE_PREDICTION_DURING_TRAINING, "prediction_img_" + str(i) + "_epoch_" + str(k) + ".png"), pred[k, i, :, :, 0], cmap='gray')
                # plt.imsave("caroSegDeepBuildModel/predictionDuringTraining/prediction_img_" + str(i) + "_epoch_" + str(k) + ".png", pred[k, i, :, :, 0], cmap='gray')


# ----------------------------------------------------------------
class CustomLearningRateScheduler(tensorflow.keras.callbacks.Callback):
    """Learning rate scheduler which sets the learning rate according to schedule.

  Arguments:
      schedule: a function that takes an epoch index
          (integer, indexed from 0) and current learning rate
          as inputs and returns a new learning rate as output (float).
  """

    def __init__(self, schedule):
        super().__init__()
        self.schedule = schedule
        self.learningRate = [] #to do (save the learning rate at each epoch and plot it)

    def on_epoch_begin(self, epoch, logs=None):

        if not hasattr(self.model.optimizer, "lr"):
            raise ValueError('Optimizer must have a "lr" attribute.')

        # Get the current learning rate from model's optimizer.
        lr = float(K.get_value(self.model.optimizer.learning_rate))
        # Call schedule function to get the scheduled learning rate.
        scheduled_lr = self.schedule(epoch, lr)
        # Set the value back to the optimizer before this epoch starts
        K.set_value(self.model.optimizer.lr, scheduled_lr)
        print("\nEpoch %05d: Learning rate is %6.4f." % (epoch, scheduled_lr))
