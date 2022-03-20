from cProfile import label
import numpy as np
import h5py
from tqdm import tqdm
import skimage.measure
from ipdb import set_trace as st
import datetime as dt
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.cm import register_cmap, cmap_d
from matplotlib import rc
import matplotlib.colors as colors

from Model.Modules import CNN_batch, CNN_batch_multi, \
    ResidualAttentionModel_andong_pre
from ML_Modules import seed_torch, init_weights, \
    PhysinformedNet, norm21
from ML_Modules import calibrate, ROC, noncalibrate, \
    norm_1d, proba_comba, AdjustWithConstraint_AH
from ML_Modules import illustrate_colormap, \
    grayify_colormap, my_weight_rmse2

import torch
from skorch.callbacks import ProgressBar, Checkpoint
from skorch.callbacks import EarlyStopping, \
    WarmRestartLR, LRScheduler
from skorch.dataset import CVSplit
import sklearn
from sklearn.metrics import confusion_matrix
from qpsolvers import solve_qp, solve_ls

font = {'family': 'serif',
        # 'weight': 'bold',
        'size': 22}

rc('font', **font)

def create_cdict(r, g, b):
    """
    Create the color tuples in the correct format.
    """
    i = np.linspace(0, 1, r.size)
    cdict = {name: list(zip(i, el / 255.0, el / 255.0))
             for el, name in [(r, 'red'), (g, 'green'), (b, 'blue')]}
    return cdict


def _cmap_from_rgb(r, g, b, name):
    cdict = create_cdict(r, g, b)
    return colors.LinearSegmentedColormap(name, cdict)


def metric(y_pred, y_real, thres=0):

    if thres !=0:
        y_pred = (y_pred >= thres).astype(bool)
        
    MCC_sta = sklearn.metrics.matthews_corrcoef(y_real,
                                                y_pred)
    TN, FP, FN, TP = confusion_matrix(
            y_real, y_pred).ravel()
    TSS = TPR = TP/(TP+FN) + TN/(TN+FP)-1

    return MCC_sta, TSS


def dt2date(date0, time_res):

    year_image = int(date0[0]/1e4)
    month_image = int((date0[0] - year_image*1e4)/1e2)
    dom_image = int(date0[0] - year_image*1e4 - month_image*1e2)
    UT_image = int(date0[1])*time_res
    date = dt.datetime(int(year_image), int(month_image),
                       int(dom_image), int(UT_image))
    return date

def SoHO_read(channels, win_size=4):

    filename_Y = 'Data/solar_data_dst_single_Y_regressor.h5'
    with h5py.File(filename_Y,'r') as f:
            
        Y_test = np.array(f['Y_test'])
        Y_train = np.array(f['Y_train'])    
        Y_valid = np.array(f['Y_valid'])
        f.close()

    filename_head = 'Data/solar_data_min_single_'
    filename_end = '.h5'
                
    for idx in range(len(channels)):
            
        filename = filename_head+channels[idx]+filename_end
        
        with h5py.File(filename,'r') as f:
            
            if idx == 0:

                X_train = np.zeros([f['X_train'].shape[0],
                                    len(channels),
                                    int(256/win_size),
                                    int(256/win_size)])
                X_test = np.zeros([f['X_test'].shape[0],
                                    len(channels),
                                    int(256/win_size),
                                    int(256/win_size)])
                X_valid = np.zeros([f['X_valid'].shape[0],
                                    len(channels),
                                    int(256/win_size),
                                    int(256/win_size)])
            if win_size>1:
                for i in tqdm(range(len(Y_train))):
                    X_train[i,idx,:,:] = skimage.measure.block_reduce(
                        f['X_train'][i][0], 
                        (win_size, win_size), 
                        np.max)
                for i in tqdm(range(len(Y_test))):
                    X_test[i,idx,:,:] = skimage.measure.block_reduce(
                        f['X_test'][i][0], 
                        (win_size, win_size), 
                        np.max)
                for i in tqdm(range(len(Y_valid))):
                    X_valid[i,idx,:,:] = skimage.measure.block_reduce(
                        f['X_valid'][i][0], 
                        (win_size, win_size), 
                        np.max)
            else:
                # st()
                X_train[:,idx,:,:] = f['X_train'][:, 0]
                X_test[:,idx,:,:] = f['X_test'][:, 0]
                X_valid[:,idx,:,:] = f['X_valid'][:, 0]

            f.close()    

    X_all = np.vstack((X_valid, X_train, X_test))    
    Y_all = np.hstack([Y_valid, Y_train, Y_test]) 

    return X_all, Y_all


def Read_solar_image(ind, omni_data, 
                     omni_date, all_date,
                     time_res, delay_range, delay):

    # import ipdb;ipdb.set_trace()

    year_image = int(all_date[ind, 0]/1e4)
    month_image = int((all_date[ind, 0] - year_image*1e4)/1e2)
    dom_image = int(all_date[ind, 0] - year_image*1e4 - month_image*1e2)
    UT_image = int(all_date[ind, 1])*time_res
    date = dt.datetime(int(year_image), int(month_image),
                       int(dom_image), int(UT_image))
    X_omni = np.zeros(delay_range)

    date_t = date+dt.timedelta(hours=delay)

    index_omni = np.where((omni_date.year == date_t.year)
                          & (omni_date.month == date_t.month)
                          & (omni_date.day == date_t.day)
                          & (omni_date.hour == date_t.hour)
                          )[0]
    if len(index_omni) != 0:
        X_omni = omni_data.values[index_omni[0]:\
            index_omni[0]+delay_range]

    return X_omni[::time_res]


def norm_cdf(cdf, thres_cdf):
    cdf_05 = np.zeros(cdf.shape)
    cdf_05[np.where(cdf == thres_cdf)[0]] = 0.5
    # import ipdb;ipdb.set_trace()
    idx_pos = np.where(cdf >= thres_cdf)[0]
    idx_neg = np.where(cdf < thres_cdf)[0]

    cdf_05[idx_pos] = (cdf[idx_pos] - thres_cdf)/(cdf.max() - thres_cdf)*0.5+0.5
    cdf_05[idx_neg] = 0.5 - (cdf[idx_neg] - thres_cdf)/(cdf.min() - thres_cdf)*0.5

    return cdf_05

def norm_05(X, mean):
    X_hat = np.zeros(X.shape)
    for i in np.arange(X.shape[0]):
        if X[i] >= mean:
            X_hat[i] = (X[i] - mean)/(X.max() - mean)*0.5+0.5
        else:
            X_hat[i] = (X[i] - mean)/(mean - X.min())*0.5+0.5
    return X_hat

def cdf_AH(X, Dst_peak):
    X = X.astype(np.int)
    cdf = np.zeros(X.max() - X.min()+1)
    ccdf = np.zeros(X.max() - X.min()+1)
    CDF = np.zeros(X.shape[0])
    CCDF = np.zeros(X.shape[0])
    x = np.arange(X.min(), X.max()+1)

    for i in x:
        idx = np.where(X <= i)[0]
        cdf[i-X.min()] = len(idx)/len(X)
        ccdf[i-X.min()] = 1 - len(idx)/len(X)

    for i, dst_clu in enumerate(X):
        try:
            idx = np.where(x == dst_clu)[0]
            CDF[i] = cdf[idx]
            CCDF[i] = ccdf[idx]
        except:
            st()

        CDF[i] = cdf[idx]
        CCDF[i] = ccdf[idx]

    # import ipdb;ipdb.set_trace()
    return CCDF, CDF, ccdf[np.where(x == Dst_peak)[0]]


def residual(params, x, data, eps_data):
    amp = params['amp']
    phaseshift = params['phase']
    freq = params['frequency']
    decay = params['decay']

    model = amp * np.sin(x*freq + phaseshift) * np.exp(-x*x*decay)

    return (data-model) / eps_data


def Prob_train(X_train, Y_train, X_t, Y_bin, 
               delay, num_channel, 
               dst_peak, callname, 
               train=True):

    weight_len = 12
    for_num = delay//2
    weight_clu = []
    weight_scale = []
    l1_ratio = 0.05
    # gap = 24/weight_len
    weights = np.ones(weight_len)/weight_len
    weights_0 = weights
    batch = 64
    # module = CNN_batch_multi(0.6, 
    #                          X_train.shape[1], 
    #                          int(X_train.shape[2]/2), 
    #                          weight_len)
    my_callbacks = [Checkpoint(f_params=callname),
                LRScheduler(WarmRestartLR),
                EarlyStopping(patience=5)]
    Resi_opt = 1000
    module = ResidualAttentionModel_andong_pre(num_channel)
    MCC_opt = 0
    TSS_opt = 0
    weights_opt = np.ones(weight_len)
    TP_best, FP_best, TN_best, FN_best, TSS_best, MCC_best \
        = 0, 0, 0, 0, 0, 0
    
    X_train = torch.from_numpy(X_train).float()
    X_t = torch.from_numpy(X_t).float()

    model = PhysinformedNet(
        module=module,
        max_epochs=100,
        lr=1e-3,
        train_split=CVSplit(5, stratified=False),  
        criterion=torch.nn.MSELoss,
        # train_split=None,
        batch_size=batch,
        optimizer=torch.optim.Adam,
        callbacks=my_callbacks,
        loss=my_weight_rmse2,
        alpha=1e-4,
        l1_ratio=l1_ratio,
        device='cuda',
        weight=weights_0,
        num_output=weight_len,
        # weight=weights,
        verbose=1,
        optimizer__weight_decay=0,
        iterator_train__shuffle=True,
    )
    
    Y_cdf, _, dst_thres = cdf_AH(
        Y_train.reshape(-1, 1), dst_peak)
    
    if train:
        Y_cdf = norm_cdf(Y_cdf, dst_thres).reshape(Y_train.shape)
        model.fit(X_train, Y_cdf)
        model.load_params(f_params=callname)
    else:
        model.initialize()
        model.load_params(f_params=callname)

    Y_train_prob = model.predict_proba(X_t)
    _, _, _, thres_train, _ = ROC(Y_bin[:, -1],
                                Y_train_prob[:, -1])
    Y_pred = (Y_train_prob >= thres_train).astype(bool)
    MCC, TSS = metric(Y_pred[:, -1], Y_bin[:, -1])
    # print('Simple CNN MCC/TSS: {}/{}'.format(MCC, TSS))

    return Y_train_prob


def Ensemble_data(Y_pred, Y_reg, Y, 
            train_idx_clu, weight_len, Dst_peak, 
            weights):

    j = 0
    for i in np.arange(len(train_idx_clu)-1):
        # i is used to seperate the storm events

        event_len = train_idx_clu[i+1] - train_idx_clu[i]
        if event_len < 30:
            continue

        j += 1
        Y_train_clu_pred = np.zeros([weight_len,
                                        event_len-2*weight_len])
        Y_train = np.zeros(event_len-2*weight_len)
        Y_train_prob = np.zeros(event_len-2*weight_len)
        Y_train_reg = np.zeros(event_len-2*weight_len)

                
        # import ipdb;ipdb.set_trace()
        for m in range(Y_train.shape[0]):

            for n in range(weight_len):

                Y_train_clu_pred[n, m] = Y_pred[train_idx_clu[i]+weight_len+m,
                                                n]

        # Y_train_reg = Y_reg[train_idx_clu[i]+weight_len:train_idx_clu[i+1], 0]
        Y_train_reg = Y_reg[train_idx_clu[i]+weight_len:train_idx_clu[i+1]-weight_len, 0]
        
        Y_train = Y[train_idx_clu[i]+weight_len:train_idx_clu[i+1]-weight_len, 0]
        
        # import ipdb;ipdb.set_trace()

        _, _, _, optimal_th, _ = ROC(Y_train,
                                        np.dot(Y_train_clu_pred.T, weights))

        if j == 1:
            thres_clu = optimal_th
            Y_all_reg = Y_train_reg
            Y_all = Y_train
            Y_pred_all = Y_train_clu_pred.T
        else:
            # import ipdb;ipdb.set_trace()
            thres_clu = np.hstack((thres_clu, optimal_th))
            Y_all_reg = np.hstack((Y_all_reg, Y_train_reg))
            Y_all = np.hstack((Y_all, Y_train))
            Y_pred_all = np.vstack([Y_pred_all, Y_train_clu_pred.T])

    # st()
    Y_ori = Y_all_reg
    Y_all_reg, _, dst_thres = cdf_AH(Y_all_reg, Dst_peak)
    Y_all_reg = norm_cdf(Y_all_reg, dst_thres)

    idx_pos = np.where(Y_all_reg >= 0.5)[0]
    idx_neg = np.where(Y_all_reg < 0.5)[0]
        
    P = np.eye(Y_pred_all.shape[0])

    # make the covariance flexible
    for n in range(Y_all_reg.shape[0]):
        P[n, n] = np.cos(np.pi*(Y_all_reg[n] - 0.5))

    return Y_pred_all, Y_all_reg, Y_all, thres_clu, P


def LS(Y_pred_all, Y_all_reg, P,
       weight_len, lr, iter, weights):

    solver='quadprog'
    A = Y_pred_all
    l = Y_all_reg

    h = np.ones(2*(weight_len-1))*0.1
    a = np.expand_dims(np.ones(weight_len), axis=0)
    W = -1*np.eye(Y_pred_all.shape[0])
    b = 1
    lb = np.zeros(weight_len)+0.0
    ub = np.ones(weight_len)
    alpha = 1
    l1 = 0.05
    l2 = alpha*(1-l1)
    
    ATPA = 2*np.dot(np.dot(A.T, P), A)+2*l2*np.eye(12)
    ATPL = -2*np.dot(np.dot(A.T, P), l)+l1*np.ones(12)
    
    x_sol = solve_qp(P=ATPA, q=ATPL,
                        lb=lb,
                        solver=solver,
                        verbose=True)
    # print("LS solution: x = {}".format(x_sol))

    weights = weights + (x_sol-weights)*lr
    V = np.dot(weights, Y_pred_all.T) - Y_all_reg
    VTV = np.dot(np.dot(V.T, P), V)
    # print('VTV in {}th is {}'.format(iter, VTV))

    return weights


def prob_pred(y_pred, y_reg, 
              date, thres, figname):

    fig, ax = plt.subplots(figsize=(16, 8))
    lm1 = ax.plot(date, y_pred, 'r.-', label='Prob')
    lm2 = ax.plot(date, np.tile(thres, y_reg.shape[0]),
            'kx-', label='threshold')
    ax1 = ax.twinx()
    lm3 = ax1.plot(date, y_reg, 'b.-', label='Dst')

    lm = lm1 + lm2 + lm3
    labs = [l.get_label() for l in lm]
    ax.legend(lm, labs, loc=1)

    ax.set_xlabel('Date')
    ax.set_ylabel('Probability')
    ax1.set_ylabel('Dst')

    plt.savefig(figname, dpi=300)
    # plt.close()

    return None


def SoHO_plot(X, Y_reg, date,
              var_idx, 
              figname,
              n_max=6):

    Vari_text = ['MDI', 'EIT', 'LASCO',
             'MDI$\Delta$', 'EIT$\Delta$', 'LASCO$\Delta$',
             'MDI', 'EIT', 'LASCO',
             'MDI$\Delta$', 'EIT$\Delta$', 'LASCO$\Delta$']

    cmap_list = ['ds9grey', 'EIT195', 'ds9bb',
                'ds9grey', 'EIT195', 'ds9bb',
                'ds9grey', 'EIT195', 'ds9bb',
                'ds9grey', 'EIT195', 'ds9bb']

    ds9he = {'red': lambda v : np.interp(v, [0, 0.015, 0.25, 0.5, 1],
                                            [0, 0.5, 0.5, 0.75, 1]),
            'green': lambda v : np.interp(v, [0, 0.065, 0.125, 0.25, 0.5, 1],
                                            [0, 0, 0.5, 0.75, 0.81, 1]),
            'blue': lambda v : np.interp(v, [0, 0.015, 0.03, 0.065, 0.25, 1],
                                            [0, 0.125, 0.375, 0.625, 0.25, 1])}

    ds9rainbow = {'red': lambda v : np.interp(v, [0, 0.2, 0.6, 0.8, 1], [1, 0, 0, 1, 1]),
              'green': lambda v : np.interp(v, [0, 0.2, 0.4, 0.8, 1], [0, 0, 1, 1, 0]),
              'blue': lambda v : np.interp(v, [0, 0.4, 0.6, 1], [1, 1, 0, 0])}

    data = np.array(pd.read_csv('Data/eit_dark_green.csv'))
    EIT195 = _cmap_from_rgb(data[: ,0],
                            data[: ,1]*3,
                            data[: ,2]*3,
                            'EIT195')
    # Set aliases, where colormap exists in matplotlib
    cmap_d['ds9bb'] = cmap_d['afmhot']
    cmap_d['ds9grey'] = cmap_d['gray']

    # Register all other colormaps
    # register_cmap('ds9b', data=ds9b)
    # register_cmap('ds9cool', data=ds9cool)
    # register_cmap('ds9a', data=ds9a)
    # register_cmap('ds9i8', data=ds9i8)
    # register_cmap('ds9aips0', data=ds9aips0)
    # register_cmap(cmap=ds9rainbow)
    # register_cmap('EIT195', data=EIT195)
    # register_cmap('ds9heat', data=ds9heat)

    # st()
    n_max = n_max
    fig_image, axs_image = plt.subplots(len(var_idx), n_max,
                                        figsize=(32, 16))

    idx = np.argmin(Y_reg[:, 0])

    # ub = np.min([X.shape[0], idx+12])
    # lb = np.max([0, idx-36])
    ub, lb = X.shape[0], 0
    # import ipdb;ipdb.set_trace()


    for n in range(n_max):
        for m, var in enumerate(var_idx):
            # import ipdb;ipdb.set_trace()
            # st()
            if m == 1:
                axs_image[m, n].contourf(X[lb+(ub-lb)//n_max*n, m],
                                        cmap=EIT195)
            else:
                axs_image[m, n].contourf(X[lb+(ub-lb)//n_max*n, m],
                                        cmap=cmap_list[m])

            plt.setp(axs_image[m, n].get_xticklabels(), \
                visible=False)
            plt.setp(axs_image[m, n].get_yticklabels(), \
                visible=False)
            axs_image[m, n].tick_params(axis=u'both', \
                which=u'both', length=0)
            if n == 0:
                axs_image[m, n].set_ylabel(Vari_text[var])

            # axs_image[m, n].get_xaxis().set_visible(False)
            # axs_image[m, n].get_yaxis().set_visible(False)

        axs_image[m, n].set_xlabel(
            date[lb+(ub-lb)//n_max*n])

    plt.savefig(figname, dpi=300)
    # plt.close()

    return None