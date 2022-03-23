from torch import nn

import skimage.measure
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import sklearn.metrics as metrics

import numpy as np
import torch
from skorch.callbacks import ProgressBar, Checkpoint, LRScheduler
from skorch.dataset import CVSplit
import os
from skorch.callbacks import EarlyStopping
from skorch import NeuralNet, NeuralNetBinaryClassifier, NeuralNetClassifier
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

def seed_torch(seed):
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def init_weights(m):
    if type(m) == nn.Linear:
        # torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.kaiming_uniform_(m.weight)
        m.bias.data.fill_(0.01)


def norm_ah(X):
    num = X.shape[1]
    out = np.zeros(X.shape)
    for i in range(num):
        dis = X[:, i].max() - X[:, i].min()
        out[:, i] = (X[:, i] - X[:, i].min())/dis

    # ipdb.set_trace()

    return out

def nanmean(v, *args, inplace=False, **kwargs):
    if not inplace:
        v = v.clone()
    is_nan = torch.isnan(v)
    v[is_nan] = 0
    return v.sum(*args, **kwargs) / (~is_nan).float().sum(*args, **kwargs)


def norm_1d(data):
    return data/data.sum()


def R2(x, y):
    from scipy import stats
    return stats.pearsonr(x, y)[0] ** 2


def noncalibrate(X_train, Y_train, X_test, Y_test, net):

    net.fit(X_train, Y_train)
    net.load_params(f_params='params.pt')
    probs = net.predict_proba(X_test)[:, 1]
    fop_origin, mpv_origin = calibration_curve(
        Y_test, probs, n_bins=10, normalize=True)
    # import ipdb; ipdb.set_trace()

    return fop_origin, mpv_origin


def calibrate(X_train, Y_train, X_test, Y_test, net):
    # import ipdb; ipdb.set_trace()

    net_cali = CalibratedClassifierCV(net, cv=5)
    net_cali.fit(X_train, Y_train)
    
    # net_cali.load_params(f_params='params.pt')
    probs = net_cali.predict_proba(X_test.cpu())[:, 1]
    # import ipdb; ipdb.set_trace()
    fop, mpv = calibration_curve(
        Y_test, probs, n_bins=10, normalize=True)
    # import ipdb; ipdb.set_trace()
    return fop, mpv, net_cali
    # return net_cali


def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr,
                                                    threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point


def Find_Optimal_Cutoff(TPR, FPR, threshold):

    # import ipdb; ipdb.set_trace()
    y = TPR - FPR
    Youden_index = np.argmax(y)  # Only the first occurrence is returned.
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point


def CB_loss(labels, logits, samples_per_cls,
            no_of_classes=2,
            loss_type='softmax',
            beta=0.9999,
            gamma=2.0):
    """Compute the Class Balanced Loss between `logits` and the ground truth `labels`.
    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits)
    where Loss is one of the standard losses used for Neural Networks.
    Args:
      labels: A int tensor of size [batch].
      logits: A float tensor of size [batch, no_of_classes].
      samples_per_cls: A python list of size [no_of_classes].
      no_of_classes: total number of classes. int
      loss_type: string. One of "sigmoid", "focal", "softmax".
      beta: float. Hyperparameter for Class balanced loss.
      gamma: float. Hyperparameter for Focal loss.
    Returns:
      cb_loss: A float tensor representing class balanced loss
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes

    labels_one_hot = F.one_hot(labels, no_of_classes).float()

    weights = torch.tensor(weights).float()
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0],1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1,no_of_classes)

    if loss_type == "focal":
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(input = logits,target = labels_one_hot, weights = weights)
    elif loss_type == "softmax":
        pred = logits.softmax(dim = 1)
        cb_loss = F.binary_cross_entropy(input = pred, target = labels_one_hot, weight = weights)
    return cb_loss


def my_BCEloss(y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    if np.isnan(y_pred.cpu().detach().numpy().sum()):

        # import ipdb;ipdb.set_trace()
        pass
    # smooth prediction
    x = np.arange(y.shape[1])  # .reshape(-1, 1)
    y_hat_pred = np.zeros(y_pred.shape)
    y_pri_pred = y_pred.clone().cpu().detach().numpy()

    # import ipdb;ipdb.set_trace()

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] == 1)[0]
        idx_neg = torch.where(y[:, i] == 0)[0]
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            CE_pos = -1*nanmean(torch.log(y_pred[idx_pos, i]+eps))
            loss += CE_pos*weight[i]/2
        if len(idx_neg) != 0:
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            CE_neg = -1*nanmean(y_pred[idx_neg, i] *
                                torch.log(1 - y_pred[idx_neg, i] + eps))
            loss += CE_neg*weight[i]/2

    if np.isnan(loss.cpu().detach().numpy()):
        
        pass
        # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    return loss


def my_weight_BCEloss(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    if np.isnan(y_pred.cpu().detach().numpy().sum()):

        # import ipdb;ipdb.set_trace()
        pass
    # smooth prediction
    x = np.arange(y.shape[1])  # .reshape(-1, 1)
    y_hat_pred = np.zeros(y_pred.shape)
    y_pri_pred = y_pred.clone().cpu().detach().numpy()

    # import ipdb;ipdb.set_trace()

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] == 1)[0]
        idx_neg = torch.where(y[:, i] == 0)[0]
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            CE_pos = -1*nanmean(torch.log(y_pred[idx_pos, i]+eps))
            loss += CE_pos*weight[i]/2
        if len(idx_neg) != 0:
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            CE_neg = -1*nanmean(y_pred[idx_neg, i] *
                                torch.log(1 - y_pred[idx_neg, i] + eps))
            loss += CE_neg*weight[i]/2

    if np.isnan(loss.cpu().detach().numpy()):
        
        pass
        # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    return loss


def my_L1_loss(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    if np.isnan(y_pred.cpu().detach().numpy().sum()):

        # import ipdb;ipdb.set_trace()
        pass
    # smooth prediction
    x = np.arange(y.shape[1])  # .reshape(-1, 1)
    y_hat_pred = np.zeros(y_pred.shape)
    y_pri_pred = y_pred.clone().cpu().detach().numpy()

    # import ipdb;ipdb.set_trace()

    for i in range(y.shape[0]):

        '''
        try:
            # reg = lr.fit(x, y_pri_pred[i])
            coef = np.polyfit(x, y_pri_pred[i], 2)
            reg = np.poly1d(coef)
            
            # import ipdb;ipdb.set_trace()
            y_hat_pred[i] = reg(x) - y_pri_pred[i]

            for j in range(y.shape[1]):
                y_pred[i, j] += y_hat_pred[i, j]
        except:
            # import ipdb;ipdb.set_trace()
            pass
        '''
        # y_pred[i] = y_hat_pred[i]
        # import ipdb;ipdb.set_trace()   
         

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] == 1)[0]
        idx_neg = torch.where(y[:, i] == 0)[0]
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            CE_pos = nanmean((y_pred[idx_pos, i]+eps)**2)
            loss += CE_pos*weight[i]/2
        if len(idx_neg) != 0:
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            CE_neg = nanmean(y_pred[idx_neg, i] *
                                (1 - y_pred[idx_neg, i] - eps)**2)
            loss += CE_neg*weight[i]/2

    if np.isnan(loss.cpu().detach().numpy()):
        
        pass
        # import ipdb;ipdb.set_trace()
    # import ipdb;ipdb.set_trace()
    return loss

def my_weight_rmse_CB(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    # import ipdb;ipdb.set_trace()
    resi = y_pred - y
    resi = resi**2

    # a = l1_ratio*alpha
    # b = (1-l1_ratio)*alpha

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= 0.5)[0]
        idx_neg = torch.where(y[:, i] < 0.5)[0]
        beta = 0.9999
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            E_pos = (1-beta^len(idx_pos))/(1-beta) 
            MSE_pos = torch.sum(resi[idx_pos, i])/E_pos
            loss += MSE_pos*weight[i]/2
        if len(idx_neg) != 0:
            E_neg = (1-beta^len(idx_neg))/(1-beta) 
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.sum(resi[idx_neg, i])/E_neg
            loss += MSE_neg*weight[i]/2
    # loss += a*np.sum(np.abs(weight))

    return loss


def my_weight_rmse(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    # import ipdb;ipdb.set_trace()
    resi = y_pred - y
    resi = resi**2

    # a = l1_ratio*alpha
    # b = (1-l1_ratio)*alpha

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= 0.5)[0]
        idx_neg = torch.where(y[:, i] < 0.5)[0]

        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            MSE_pos = torch.mean(resi[idx_pos, i])
            loss += MSE_pos*weight[i]/2
        if len(idx_neg) != 0:
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.mean(resi[idx_neg, i])
            loss += MSE_neg*weight[i]/2
    # loss += a*np.sum(np.abs(weight))

    return loss

def my_weight_rmse_single(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    # import ipdb;ipdb.set_trace()
    resi = y_pred - y
    resi = resi**2

    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= 0.5)[0]
        idx_neg = torch.where(y[:, i] < 0.5)[0]

        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            MSE_pos = torch.mean(resi[idx_pos, i])
            loss += MSE_pos*weight[i]/2
        if len(idx_neg) != 0:
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.mean(resi[idx_neg, i])
            loss += MSE_neg*weight[i]/2

    return loss

def my_weight_rmse2(weight, y_pred, y, pre_weight='False'):

    loss = 0
    eps = 1e-7
    # weight = np.ones(y.shape[1])/y.shape[1]

    if pre_weight:
        weight = np.logspace(2, 0, num=y.shape[1])
        # conv_point = weight[-10]
        weight[-1:] = 0/y.shape[1]
        weight = norm_1d(weight)

    # weights = torch.from_numpy(weight).type(torch.FloatTensor).cuda()
    resi = y_pred - y
    resi = resi**2

    # import ipdb;ipdb.set_trace()
    P = torch.cos(0.9*np.pi*(y - 0.5))
    
    resi = resi*P
    '''
    for i in range(y.shape[1]):
        loss += torch.mean(resi[:, i])*weight[i]

    '''
    for i in range(y.shape[1]):
        idx_pos = torch.where(y[:, i] >= 0.5)[0]
        idx_neg = torch.where(y[:, i] < 0.5)[0]
        beta = 0.9999
        # import ipdb;ipdb.set_trace()
        if len(idx_pos) != 0:
            E_pos = (1-beta**len(idx_pos))/(1-beta) 
            MSE_pos = torch.sum(resi[idx_pos, i])/E_pos
            loss += MSE_pos*weight[i]/2
        if len(idx_neg) != 0:
            E_neg = (1-beta**len(idx_neg))/(1-beta) 
            # CE_neg = -1*torch.mean(torch.log(1 - y_pred[idx_neg, i] - eps))
            MSE_neg = torch.sum(resi[idx_neg, i])/E_neg
            loss += MSE_neg*weight[i]/2
    
    return loss





class PhysinformedNet(NeuralNet):
    def __init__(self, *args, pre_weight='False',
                 num_output, weight, alpha, l1_ratio,
                 # P,
                 loss,
                 **kwargs):
        super().__init__(*args, **kwargs)
        # self.ReLU = nn.ReLU()
        # self.ReLU6 = nn.ReLU6()
        # self.criterion = criterion
        # self.X = 0
        # self.loss = my_BCEloss
        # self.loss = my_weight_BCEloss
        # self.loss = my_L1_loss
        # self.P = P
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.loss = loss
        self.pre_weight = pre_weight
        self.weight = weight
        self.num_output = num_output

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # loss_ori = torch.zeros(1).cuda()

        if np.isnan(y_pred.cpu().detach().numpy().sum()):

            pass

            # import ipdb;ipdb.set_trace()
        '''
        print('BCE value:', self.loss(y_pred.type(torch.FloatTensor).cuda(),
                                      y_true.type(torch.FloatTensor).cuda()))
        '''        
       
             
        loss_ori = self.loss(y_pred=y_pred.type(torch.FloatTensor).cuda(),
                             y=y_true.type(torch.FloatTensor).cuda(),
                             pre_weight=self.pre_weight,
                             # alpha=self.alpha,
                             # l1_ratio=self.l1_ratio,
                             # P=self.P,
                             weight=self.weight)
        # import ipdb; ipdb.set_trace()
        
        l1_lambda = self.alpha*self.l1_ratio
        l1_reg = torch.tensor(0.)
        l1_reg = l1_reg.to(self.device)
        for param in self.module.parameters():
            l1_reg += torch.sum(torch.abs(param))
        loss1 = l1_lambda * l1_reg
        loss_ori += loss1
        
        l2_lambda = self.alpha*(1-self.l1_ratio) 
        l2_reg = torch.tensor(0.)
        l2_reg = l2_reg.to(self.device)
        for param in self.module.parameters():
            l2_reg += torch.norm(param).sum()
        loss_ori += l2_lambda * l2_reg
        
        # loss_ori += loss_ori_t

        # loss_ori = loss_ori
        # import ipdb; ipdb.set_trace()
        return loss_ori

class PhysinformedNet_single(NeuralNet):
    def __init__(self, *args, pre_weight='False',
                 num_output, weights, **kwargs):
        super().__init__(*args, **kwargs)
        self.ReLU = nn.ReLU()
        self.ReLU6 = nn.ReLU6()
        # self.criterion = criterion
        self.X = 0
        # self.loss = my_BCEloss
        # self.loss = my_weight_BCEloss
        # self.loss = my_L1_loss
        # self.P = P
        self.loss = my_weight_rmse
        self.pre_weight = pre_weight
        self.weight = weights

    def get_loss(self, y_pred, y_true, X, training=False):
        # print('y_pred:', y_pred.shape)
        # print('y_true:', y_true.shape)
        # print('X:', X.shape)
        # loss_ori = torch.zeros(1).cuda()

        if np.isnan(y_pred.cpu().detach().numpy().sum()):

            pass

            # import ipdb;ipdb.set_trace()
        '''
        print('BCE value:', self.loss(y_pred.type(torch.FloatTensor).cuda(),
                                      y_true.type(torch.FloatTensor).cuda()))
        '''        
        # import ipdb; ipdb.set_trace()
             
        loss_ori = self.loss(y_pred=y_pred.type(torch.FloatTensor).cuda(),
                             y=y_true.type(torch.FloatTensor).cuda(),
                             pre_weight=self.pre_weight,
                             # P = self.P,
                             weight=self.weight)
        # loss_ori += loss_ori_t

        # loss_ori = loss_ori

        return loss_ori

def proba_comba2(theta, P, y):
    out = np.zeros(P.shape[0])
    for i in np.arange(P.shape[0]):
        # import ipdb;ipdb.set_trace()
        idx = np.where(P[i] != 0)[0]
        out[i] = np.dot(P[i], theta)/len(idx)
        # out[i] = np.dot(P[i], np.exp(theta))/len(idx)
    # import ipdb;ipdb.set_trace()
    return y - out


def proba_comba(theta, P, y):
    out = np.zeros(P.shape[0])
    for i in np.arange(P.shape[0]):
        # import ipdb;ipdb.set_trace()
        idx = np.where(P[i] != 0)[0]
        # out[i] = np.dot(P[i], theta)/len(idx)
        out[i] = np.dot(P[i], norm21(theta))/len(idx)
    # import ipdb;ipdb.set_trace()
    return y - out

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def norm21(x):
    x -= x.min()
    return x / x.sum()


def grayify_colormap(cmap, mode='hsp'):
    """
    Return a grayscale version a the colormap.

    The grayscale conversion of the colormap is bases on perceived luminance of
    the colors. For the conversion either the `~skimage.color.rgb2gray` or a
    generic method called ``hsp`` [1]_ can be used. The code is loosely based
    on [2]_.


    Parameters
    ----------
    cmap : str or `~matplotlib.colors.Colormap`
        Colormap name or instance.
    mode : {'skimage, 'hsp'}
        Grayscale conversion method. Either ``skimage`` or ``hsp``.

    References
    ----------

    .. [1] Darel Rex Finley, "HSP Color Model - Alternative to HSV (HSB) and HSL"
       http://alienryderflex.com/hsp.html

    .. [2] Jake VanderPlas, "How Bad Is Your Colormap?"
       https://jakevdp.github.io/blog/2014/10/16/how-bad-is-your-colormap/
    """
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))

    if mode == 'skimage':
        from skimage.color import rgb2gray
        luminance = rgb2gray(np.array([colors]))
        colors[:, :3] = luminance[0][:, np.newaxis]
    elif mode == 'hsp':
            RGB_weight = [0.299, 0.587, 0.114]
            luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
            colors[:, :3] = luminance[:, np.newaxis]
    else:
        raise ValueError('Not a valid grayscale conversion mode.')

    return cmap.from_list(cmap.name + "_grayscale", colors, cmap.N)


def illustrate_colormap(cmap, **kwargs):
    """
    Illustrate color distribution and perceived luminance of a colormap.

    Parameters
    ----------
    cmap : str or `~matplotlib.colors.Colormap`
        Colormap name or instance.
    kwargs : dicts
        Keyword arguments passed to `grayify_colormap`.
    """
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap(cmap)
    cmap_gray = grayify_colormap(cmap, **kwargs)
    figure = plt.figure(figsize=(6, 4))
    v = np.linspace(0, 1, 4 * cmap.N)

    # Show colormap
    show_cmap = figure.add_axes([0.1, 0.8, 0.8, 0.1])
    im = np.outer(np.ones(50), v)
    show_cmap.imshow(im, cmap=cmap, origin='lower')
    show_cmap.set_xticklabels([])
    show_cmap.set_yticklabels([])
    show_cmap.set_yticks([])
    show_cmap.set_title('RGB & Gray Luminance of colormap {0}'.format(cmap.name))

    # Show colormap gray
    show_cmap_gray = figure.add_axes([0.1, 0.72, 0.8, 0.09])
    show_cmap_gray.imshow(im, cmap=cmap_gray, origin='lower')
    show_cmap_gray.set_xticklabels([])
    show_cmap_gray.set_yticklabels([])
    show_cmap_gray.set_yticks([])

    # Plot RGB profiles
    plot_rgb = figure.add_axes([0.1, 0.1, 0.8, 0.6])
    plot_rgb.plot(v, [cmap(_)[0] for _ in v], color='r')
    plot_rgb.plot(v, [cmap(_)[1] for _ in v], color='g')
    plot_rgb.plot(v, [cmap(_)[2] for _ in v], color='b')
    plot_rgb.plot(v, [cmap_gray(_)[0] for _ in v], color='k', linestyle='--')
    plot_rgb.set_ylabel('Luminance')
    plot_rgb.set_ylim(-0.005, 1.005)

    
