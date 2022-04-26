def pixelAccuracy(pred_img, label_img):
    pixel_correct = np.sum(pred_img == label_img)
    pixel_labeled = pred_img.shape[0] * pred_img.shape[1]
    pixel_accuracy = pixel_correct / pixel_labeled
    return pixel_accuracy, pixel_correct, pixel_labeled
    
def MeanPixelAccuracy(pred, label):
    """
    Function to compute the mean pixel accuracy for semantic segmentation between mini-batch tensors
    :param pred: Tensor of predictions
    :param label: Tensor of ground-truth
    :return: Mean pixel accuracy for all the mini-bath
    """
    # Convert tensors to numpy arrays
    imPred = np.asarray(torch.squeeze(pred))
    imLab = np.asarray(torch.squeeze(label))

    # Create empty numpy arrays
    pixel_accuracy = np.empty(imLab.shape[0])
    pixel_correct = np.empty(imLab.shape[0])
    pixel_labeled = np.empty(imLab.shape[0])

    # Compute pixel accuracy for each pair of images in the batch
    for i in range(imLab.shape[0]):
        pixel_accuracy[i], pixel_correct[i], pixel_labeled[i] = pixelAccuracy(imPred[i], imLab[i])

    # Compute the final accuracy for the batch
    acc = 100.0 * np.sum(pixel_correct) / (np.spacing(1) + np.sum(pixel_labeled))

    return acc
