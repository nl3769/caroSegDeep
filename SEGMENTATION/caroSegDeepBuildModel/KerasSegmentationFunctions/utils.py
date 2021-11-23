import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mtpli
import tensorflow as tf
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.eager import context

# Runtime data augmentation
def get_augmented(X_train,
                  Y_train,
                  X_val=None,
                  Y_val=None,
                  batch_size=32,
                  seed=0,
                  data_gen_args = dict(rotation_range=5,
                                       shear_range=5,
                                       horizontal_flip=True,
                                       vertical_flip=False,
                                       zoom_range = 0.2,
                                       fill_mode='constant')
                  ):


    # Train data, provide the same seed and keyword arguments to the fit and flow methods
    X_datagen = ImageDataGenerator(**data_gen_args)
    Y_datagen = ImageDataGenerator(**data_gen_args)

    X_datagen.fit(X_train, augment=True, seed=seed)
    Y_datagen.fit(Y_train, augment=True, seed=seed)

    X_train_augmented = X_datagen.flow(X_train, batch_size=batch_size, shuffle=True, seed=seed)
    Y_train_augmented = Y_datagen.flow(Y_train, batch_size=batch_size, shuffle=True, seed=seed)
    
    train_generator = zip(X_train_augmented, Y_train_augmented)

    if not (X_val is None) and not (Y_val is None):
        # Validation data, no data augmentation, but we create a generator anyway
        X_datagen_val = ImageDataGenerator(**data_gen_args)
        Y_datagen_val = ImageDataGenerator(**data_gen_args)
        X_datagen_val.fit(X_val, augment=True, seed=seed)
        Y_datagen_val.fit(Y_val, augment=True, seed=seed)
        X_val_augmented = X_datagen_val.flow(X_val, batch_size=batch_size, shuffle=True, seed=seed)
        Y_val_augmented = Y_datagen_val.flow(Y_val, batch_size=batch_size, shuffle=True, seed=seed)

        # combine generators into one which yields image and masks
        val_generator = zip(X_val_augmented, Y_val_augmented)
        
        return train_generator, val_generator
    else:
        return train_generator


def plot_segm_history(history, metrics=['iou', 'val_iou'], losses=['loss', 'val_loss']):
    # summarize history for iou
    plt.figure(figsize=(12,6))
    for metric in metrics:
        plt.plot(history.history[metric], linewidth=3)
    plt.suptitle('metrics over epochs', fontsize=20)
    plt.ylabel('metric', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0.3, 1, step=0.02), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(metrics, loc='center right', fontsize=15)
    plt.show()
    # summarize history for loss
    plt.figure(figsize=(12,6))    
    for loss in losses:
        plt.plot(history.history[loss], linewidth=3)
    plt.suptitle('loss over epochs', fontsize=20)
    plt.ylabel('loss', fontsize=20)
    plt.xlabel('epoch', fontsize=20)
    #plt.yticks(np.arange(0, 0.2, step=0.005), fontsize=35)
    #plt.xticks(fontsize=35)
    plt.legend(losses, loc='center right', fontsize=15)
    plt.show()


def mask_to_red(mask):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    c1 = mask.reshape(img_size,img_size)
    c2 = np.zeros((img_size,img_size))
    c3 = np.zeros((img_size,img_size))
    c4 = mask.reshape(img_size,img_size)

    return np.stack((c1, c2, c3, c4), axis=-1).astype(np.int)


def mask_to_rgba(mask, color='red'):
    '''
    Converts binary segmentation mask from white to red color.
    Also adds alpha channel to make black background transparent.
    '''
    img_size = mask.shape[0]
    zeros = np.zeros((img_size,img_size))
    ones = mask.reshape(img_size,img_size)
    if color == 'red':
        return np.stack((ones, zeros, zeros, ones), axis=-1)
    elif color == 'green':
        return np.stack((zeros, ones, zeros, ones), axis=-1)
    elif color == 'blue':
        return np.stack((zeros, zeros, ones, ones), axis=-1)
    elif color == 'yellow':
        return np.stack((ones, ones, zeros, ones), axis=-1)
    elif color == 'magenta':
        return np.stack((ones, zeros, ones, ones), axis=-1)
    elif color == 'cyan':
        return np.stack((zeros, ones, ones, ones), axis=-1)


def image_to_rgba(image, alpha=255):
    img_size = image.shape[0]
    img = image.reshape(img_size,img_size)
    img_a = np.ones((img_size,img_size))*alpha
    return np.stack((img, img, img, img_a), axis=-1)


def save_image(image, filename):
    im = Image.fromarray(image)
    im.save(filename)


def save_imgs(org_imgs, 
              mask_imgs, 
              pred_imgs=None, 
              figsize=4,
              alpha=0.5,
              text_of_GT="ground truth",
              output_prefix="./test/my_images_"
             ):
    nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]
    
    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if  not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3
    
    for m in range(0, nm_img_to_plot):
        image_filename = output_prefix + str(m).zfill(4) + "_image.png"
        mask_filename = output_prefix  + str(m).zfill(4) + "_mask.png"
        overlay_filename =  output_prefix + str(m).zfill(4) + "_overlay.png"
        montage_filename =  output_prefix + str(m).zfill(4) + "_montage.png"
        
        if (m%5):
            print(f'creating png : {m}/{nm_img_to_plot}', end = '\r')
        
        mtpli.imsave(image_filename, org_imgs[m], cmap='gray' ) 
        mtpli.imsave(mask_filename, mask_imgs[m], cmap='gray' ) 
        
        # prepare figure
        fig, axes = plt.subplots(1, cols, figsize=(cols*figsize, figsize))
        axes[0].set_title("original", fontsize=15) 
        axes[1].set_title(text_of_GT, fontsize=15)
        axes[0].imshow(org_imgs[m], cmap=get_cmap(org_imgs))
        axes[0].set_axis_off()
        axes[1].imshow(mask_imgs[m], cmap=get_cmap(mask_imgs))
        axes[1].set_axis_off()     
        
        if pred_imgs is None:   # only one mask provided, no comparison
            axes[2].set_title("overlay", fontsize=15) 
            axes[2].imshow(org_imgs[m], cmap=get_cmap(org_imgs))
            axes[2].imshow(mask_to_red(zero_pad_mask(mask_imgs[m], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[2].set_axis_off()
                        
        else:
            #m_0 = mask_to_rgba(mask_imgs[m],'green')
            #m_1 = mask_to_rgba(pred_imgs[m],'red')
            #mtpli.imsave(comp_filename , m_0[:,:,0:3] + m_1[:,:,0:3] )
            m1 = mask_imgs[m]
            m2 = pred_imgs[m]
            
            lab_rgb = np.zeros((m1.shape[0], m1.shape[1],3), dtype=np.uint8)
            red  = np.array([255,0,0],dtype=np.uint8)
            green = np.array([0,255,0],dtype=np.uint8)
            violet = np.array([255,0,255],dtype=np.uint8)

            m1_bool = m1 > 0.5
            m2_bool = m2 > 0.5
            m1_and_m2 = np.logical_and(m1_bool , m2_bool)

            lab_rgb[m1_bool] = red
            lab_rgb[m2_bool] = violet
            lab_rgb[m1_and_m2] = green
            
            axes[2].set_title("prediction", fontsize=15) 
            axes[3].set_title("overlay", fontsize=15) 
            axes[2].imshow(pred_imgs[m], cmap=get_cmap(pred_imgs))
            axes[2].set_axis_off()
            axes[3].imshow(org_imgs[m], cmap=get_cmap(org_imgs))
            axes[3].imshow(mask_to_red(zero_pad_mask(pred_imgs[m], desired_size=org_imgs_size)), cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[3].set_axis_off()

            prediction_filename = output_prefix  + str(m).zfill(4) + "_pred.png"
            mtpli.imsave(prediction_filename, pred_imgs[m]*255, cmap='gray' ) 
            
            comp_filename =  output_prefix + str(m).zfill(4) + "_comp.png"
            mtpli.imsave(comp_filename , lab_rgb )

        plt.savefig( montage_filename, dpi=150)
        plt.close()
    print(f'wrote : {nm_img_to_plot}                     ')        
    
    
    
def save_imgs2(org_imgs, 
              mask_imgs, 
              pred_imgs=None, 
              figsize=4,
              alpha=0.5,
              text_of_GT="ground truth",
              output_prefix="./test/my_images_"
             ):
    nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]
    
    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if  not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3
    
    for m in range(0, nm_img_to_plot):
        image_filename = output_prefix + str(m).zfill(4) + "_image.png"
        mask_filename = output_prefix  + str(m).zfill(4) + "_mask.png"
        overlay_filename =  output_prefix + str(m).zfill(4) + "_overlay.png"
        montage_filename =  output_prefix + str(m).zfill(4) + "_montage.png"
        
        if (m%5):
            print(f'creating png : {m}/{nm_img_to_plot}', end = '\r')
        
        mtpli.imsave(image_filename, org_imgs[m], cmap='gray' ) 
        mtpli.imsave(mask_filename, mask_imgs[m], cmap='gray' ) 
        
        # prepare figure
        fig, axes = plt.subplots(1, cols, figsize=(cols*figsize, figsize))
        axes[0].set_title("original", fontsize=15) 
        axes[0].imshow(org_imgs[m], cmap=get_cmap(org_imgs))
        axes[0].set_axis_off()
         
        if pred_imgs is None:   # only one mask provided, no comparison
            axes[1].set_title(text_of_GT, fontsize=15)
            axes[1].imshow(mask_imgs[m], cmap=get_cmap(mask_imgs))
            axes[1].set_axis_off()     
            axes[2].set_title("overlay", fontsize=15) 
            axes[2].imshow(org_imgs[m], cmap=get_cmap(org_imgs))
            axes[2].imshow(mask_to_red(zero_pad_mask(mask_imgs[m], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[2].set_axis_off()
                        
        else:
            #m_0 = mask_to_rgba(mask_imgs[m],'green')
            #m_1 = mask_to_rgba(pred_imgs[m],'red')
            #mtpli.imsave(comp_filename , m_0[:,:,0:3] + m_1[:,:,0:3] )
            m1 = mask_imgs[m]
            m2 = pred_imgs[m]
            
            lab_rgb = np.zeros((m1.shape[0], m1.shape[1],3), dtype=np.uint8)
            red  = np.array([255,0,0],dtype=np.uint8)
            green = np.array([0,255,0],dtype=np.uint8)
            violet = np.array([255,0,255],dtype=np.uint8)

            m1_bool = m1 > 0.5
            m2_bool = m2 > 0.5
            m1_and_m2 = np.logical_and(m1_bool , m2_bool)

            lab_rgb[m1_bool] = red
            lab_rgb[m2_bool] = violet
            lab_rgb[m1_and_m2] = green
            
            axes[1].set_title("compare", fontsize=15)
            axes[1].imshow(lab_rgb)
            axes[1].set_axis_off()     
            
            axes[2].set_title("prediction", fontsize=15) 
            axes[3].set_title("overlay", fontsize=15) 
            axes[2].imshow(pred_imgs[m], cmap=get_cmap(pred_imgs))
            axes[2].set_axis_off()
            axes[3].imshow(org_imgs[m], cmap=get_cmap(org_imgs))
            axes[3].imshow(mask_to_red(zero_pad_mask(pred_imgs[m], desired_size=org_imgs_size)), cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[3].set_axis_off()

            prediction_filename = output_prefix  + str(m).zfill(4) + "_pred.png"
            mtpli.imsave(prediction_filename, pred_imgs[m]*255, cmap='gray' ) 
            
            comp_filename =  output_prefix + str(m).zfill(4) + "_comp.png"
            mtpli.imsave(comp_filename , lab_rgb )

        plt.savefig( montage_filename, dpi=150)
        plt.close()
    print(f'wrote : {nm_img_to_plot}                     ')       
    
            
def plot_imgs(org_imgs, 
              mask_imgs, 
              pred_imgs=None, 
              nm_img_to_plot=10, 
              figsize=4,
              alpha=0.5,
              text_of_GT="ground truth",
              output_pdf_name=None
             ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if  not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols*figsize, nm_img_to_plot*figsize))
    axes[0, 0].set_title("original", fontsize=15) 
    axes[0, 1].set_title(text_of_GT, fontsize=15)
    if not (pred_imgs is None):
        axes[0, 2].set_title("prediction", fontsize=15) 
        axes[0, 3].set_title("overlay", fontsize=15) 
    else:
        axes[0, 2].set_title("overlay", fontsize=15) 
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
        axes[m, 1].set_axis_off()        
        if not (pred_imgs is None):
            axes[m, 2].imshow(pred_imgs[im_id], cmap=get_cmap(pred_imgs))
            axes[m, 2].set_axis_off()
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1

    if not( output_pdf_name is None): 
        plt.savefig(output_pdf_name, dpi=150)
    plt.show()

def plotOrgGtPred(org,
                  gt,
                  pred,
                  NbImgToPlot = 1,
                  figSize = 4,
                  OutputPDF = None):

    fig, axes = plt.subplots(NbImgToPlot, 3, figsize=(3 * figSize, NbImgToPlot * figSize))

    axes[0, 0].set_title("original", fontsize=15)
    axes[0, 1].set_title("GT", fontsize=15)
    axes[0, 2].set_title("Prediction", fontsize=15)

    for m in range(NbImgToPlot):
        axes[m, 0].imshow(org[m], cmap=get_cmap(org))
        axes[m, 1].imshow(gt[m], cmap=get_cmap(gt))
        axes[m, 2].imshow(pred[m], cmap=get_cmap(pred))

    if not( OutputPDF is None):
        plt.savefig(OutputPDF, dpi=150)

    plt.show()



def plot_imgs2(org_imgs, 
              mask_imgs, 
              pred_imgs=None, 
              nm_img_to_plot=10, 
              figsize=4,
              alpha=0.5,
              text_of_GT="ground truth",
              output_pdf_name=None
             ):
    '''
    Image plotting for semantic segmentation data.
    Last column is always an overlay of ground truth or prediction
    depending on what was provided as arguments.
    '''
    if nm_img_to_plot > org_imgs.shape[0]:
        nm_img_to_plot = org_imgs.shape[0]
    im_id = 0
    org_imgs_size = org_imgs.shape[1]

    org_imgs = reshape_arr(org_imgs)
    mask_imgs = reshape_arr(mask_imgs)
    if  not (pred_imgs is None):
        cols = 4
        pred_imgs = reshape_arr(pred_imgs)
    else:
        cols = 3

    fig, axes = plt.subplots(nm_img_to_plot, cols, figsize=(cols*figsize, nm_img_to_plot*figsize))
    axes[0, 0].set_title("original", fontsize=15) 
    
    if not (pred_imgs is None):
        axes[0, 1].set_title("GT", fontsize=15)
        axes[0, 2].set_title("prediction", fontsize=15) 
        axes[0, 3].set_title("overlay", fontsize=15) 
    else:
        axes[0, 1].set_title(text_of_GT, fontsize=15)
        axes[0, 2].set_title("overlay", fontsize=15) 
    for m in range(0, nm_img_to_plot):
        axes[m, 0].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
        axes[m, 0].set_axis_off()
        if not (pred_imgs is None):
            m1 = mask_imgs[im_id]
            m2 = pred_imgs[im_id]
            
            lab_rgb = np.zeros((m1.shape[0], m1.shape[1],3), dtype=np.uint8)
            red  = np.array([255,0,0],dtype=np.uint8)
            green = np.array([0,255,0],dtype=np.uint8)
            violet = np.array([255,0,255],dtype=np.uint8)

            m1_bool = m1 > 0.5 
            m2_bool = m2 > 0.5
            m1_and_m2 = np.logical_and(m1_bool , m2_bool)
            lab_rgb[m1_bool] = red        # GT  -> True Negative
            lab_rgb[m2_bool] = violet     # Pred ->False Positive
            lab_rgb[m1_and_m2] = green    # Union commun

            # axes[m, 1].imshow(lab_rgb)
            axes[m, 1].imshow(m1, cmap=get_cmap(pred_imgs))
            axes[m, 1].set_axis_off()

            axes[m, 2].imshow( pred_imgs[im_id], cmap=get_cmap(pred_imgs) )
            axes[m, 2].set_axis_off()
            
            axes[m, 3].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            # axes[m, 3].imshow(mask_to_red(zero_pad_mask(pred_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(pred_imgs), alpha=alpha)
            axes[m, 3].set_axis_off()
        else:
            axes[m, 1].imshow(mask_imgs[im_id], cmap=get_cmap(mask_imgs))
            axes[m, 1].set_axis_off()   
            
            axes[m, 2].imshow(org_imgs[im_id], cmap=get_cmap(org_imgs))
            axes[m, 2].imshow(mask_to_red(zero_pad_mask(mask_imgs[im_id], desired_size=org_imgs_size)), cmap=get_cmap(mask_imgs), alpha=alpha)
            axes[m, 2].set_axis_off()
        im_id += 1

    if not( output_pdf_name is None): 
        plt.savefig(output_pdf_name, dpi=150)
    plt.show()
    

def reshape_arr(arr):
    if arr.ndim == 3:
        return arr
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return arr
        elif arr.shape[3] == 1:
            return arr.reshape(arr.shape[0], arr.shape[1], arr.shape[2])


def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'



class TrainValTensorBoard(TensorBoard):
    def __init__(self, log_dir='./logs', **kwargs):
        self.val_log_dir = os.path.join(log_dir, 'validation')
        training_log_dir = os.path.join(log_dir, 'training')
        os.makedirs(self.val_log_dir, exist_ok=True)
        os.makedirs(training_log_dir, exist_ok=True)
        super(TrainValTensorBoard, self).__init__(training_log_dir, **kwargs)

    def set_model(self, model):
        if context.executing_eagerly():
            self.val_writer = tf.contrib.summary.create_file_writer(self.val_log_dir)
        else:
            self.val_writer = tf.summary.FileWriter(self.val_log_dir)
        super(TrainValTensorBoard, self).set_model(model)

    def _write_custom_summaries(self, step, logs=None):
        logs = logs or {}
        val_logs = {k.replace('val_', ''): v for k, v in logs.items() if 'val_' in k}
        if context.executing_eagerly():
            with self.val_writer.as_default(), tf.contrib.summary.always_record_summaries():
                for name, value in val_logs.items():
                    tf.contrib.summary.scalar(name, value.item(), step=step)
        else:
            for name, value in val_logs.items():
                summary = tf.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = value.item()
                summary_value.tag = name
                self.val_writer.add_summary(summary, step)
        self.val_writer.flush()

        logs = {k: v for k, v in logs.items() if not 'val_' in k}
        super(TrainValTensorBoard, self)._write_custom_summaries(step, logs)

    def on_train_end(self, logs=None):
        super(TrainValTensorBoard, self).on_train_end(logs)
        self.val_writer.close()
