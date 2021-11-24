import matplotlib.pyplot as plt

def plot_org_gt_pred(org,
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


def get_cmap(arr):
    if arr.ndim == 3:
        return 'gray'
    elif arr.ndim == 4:
        if arr.shape[3] == 3:
            return 'jet'
        elif arr.shape[3] == 1:
            return 'gray'
