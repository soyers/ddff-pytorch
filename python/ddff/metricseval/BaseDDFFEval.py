import numpy as np
import torch
import skimage.filters as skf
import cv2

class BaseDDFFEval:
    def __init__(self, trainer):
        self.trainer = trainer

    def evaluate(self, dataloader, accthrs = [1.25, 1.25**2, 1.25**3]):
        avgmetrics = np.zeros((1, 7+len(accthrs)), dtype=float)
        count = 0
        for i, data in enumerate(dataloader):
            inputs, output = data["input"], data["output"]
            if torch.cuda.is_available():
                    inputs = inputs.cuda()
            output_approx = self.trainer.evaluate(inputs)
            metrics = self.__calmetrics(output_approx.permute(0,2,3,1).squeeze().data.cpu().numpy(), output.permute(0,2,3,1).squeeze().numpy(), 1.0, accthrs, bumpinessclip=0.05, ignore_zero=True)
            count += 1
            avgmetrics += metrics
        return avgmetrics/count

    # Metrics calculation provided by Caner Hazirbas
    def __calmetrics(self, pred, target, mse_factor, accthrs, bumpinessclip=0.05, ignore_zero=True):
        metrics = np.zeros((1, 7+len(accthrs)), dtype=float)

        if target.sum() == 0:
            return metrics

        pred_ = np.copy(pred)
        if ignore_zero:
            pred_[target==0.0] = 0.0
            numPixels = (target>0.0).sum() # number of valid pixels
        else:
            numPixels = target.size

        #euclidean norm
        metrics[0,0] = np.square(pred_-target).sum() / numPixels * mse_factor

        # RMS
        metrics[0,1] = np.sqrt(metrics[0,0])

        # log RMS
        logrms = (np.ma.log(pred_)-np.ma.log(target))
        metrics[0,2] = np.sqrt(np.square(logrms).sum() / numPixels)

        # absolute relative
        metrics[0,3] = np.ma.divide(np.abs(pred_-target), target).sum() / numPixels

        #square relative
        metrics[0,4] = np.ma.divide(np.square(pred_-target), target).sum() / numPixels

        # accuracies
        acc = np.ma.maximum(np.ma.divide(pred_,target), np.ma.divide(target, pred_))
        for i, thr in enumerate(accthrs):
            metrics[0, 5+i] = (acc < thr).sum() / numPixels * 100.

        # badpix
        metrics[0, 8]= (np.abs(pred_-target) > 0.07).sum() / numPixels * 100.

        # bumpiness -- Frobenius norm of the Hessian matrix
        diff = np.asarray(pred-target, dtype='float64') # PRED or PRED_
        chn = diff.shape[2] if len(diff.shape) > 2 else 1
        bumpiness = np.zeros_like(pred_).astype('float')
        for c in range(0,chn):
            if chn > 1:
                diff_ = diff[:, :, c]
            else:
                diff_ = diff
            dx = skf.scharr_v(diff_)
            dy = skf.scharr_h(diff_)
            dxx = skf.scharr_v(dx)
            dxy = skf.scharr_h(dx)
            dyy = skf.scharr_h(dy)
            dyx = skf.scharr_v(dy)
            hessiannorm = np.sqrt(np.square(dxx) + np.square(dxy) + np.square(dyy) + np.square(dyx))
            bumpiness += np.clip(hessiannorm, 0, bumpinessclip)
        bumpiness = bumpiness[target>0].sum() if ignore_zero else bumpiness.sum()
        metrics[0, 9] = bumpiness / chn / numPixels * 100.

        return metrics
