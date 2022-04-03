import pandas as pd
import numpy as np
import seaborn as sns
from rfpimp import *
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.datasets import load_boston
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

from mrmr import mrmr_classif
from sklearn.feature_selection import f_regression

from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
from sklearn.base import clone
import warnings

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from xgboost import XGBRegressor
import matplotlib.gridspec as gridspec

import xgboost
import shap
import xgboost as xgb

def boston_data_processing():
    boston_data = load_boston()
    boston = pd.DataFrame(boston_data.data, columns=boston_data.feature_names)
    boston['MEDV'] = boston_data.target
    
    #split data
    msk = np.random.rand(len(boston)) < 0.8
    X_train = boston[msk].reset_index().drop(columns=['index'])
    X_val = boston[~msk].reset_index().drop(columns=['index'])
    
    #normalize data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_train = pd.DataFrame(X_train, columns=boston.columns)
    X_val = pd.DataFrame(X_val, columns=boston.columns)
    
    x_train = X_train.iloc[:, 0:-1]
    y_train = X_train.iloc[:, -1]
    x_val = X_val.iloc[:, 0:-1]
    y_val = X_val.iloc[:, -1]
    return x_train, y_train, x_val, y_val


def spearman(x_train, y_train, graph=False):
    feature_corr = {}
    for column in list(x_train.columns):
        spearson_corr, _ = spearmanr(x_train[column], y_train)
        feature_corr[column] = np.abs(spearson_corr)
    feature_corr_ = {k: v for k, v in sorted(feature_corr.items(), key=lambda item: item[1], reverse=True)}
    
    if graph==True:
        fig, ax = plt.subplots(figsize=(10,4)) 
        ax.bar(feature_corr_.keys(), feature_corr_.values(), color='#76b5c5')
        ax.set_xlabel(f"Features (total {len(feature_corr_.keys())})", fontsize=12, fontname="Times")
        ax.set_ylabel("Feature Importance Score", fontsize=12, fontname="Times")
        ax.set_title("Spearman Feature Importance Ranking (abslute value)",  fontsize=18, fontname="Times")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.show()
    return feature_corr_

def pca(X_train, graph=False):
    n_components = min(len(X_train), len(list(X_train.columns))-1)
    pca = PCA(n_components=n_components)
    X_new = pca.fit_transform(np.array(X_train.iloc[:, 0:-1]))
    if graph == True:
        fig, axes = plt.subplots(1,2,figsize=(10,4))
        plt.style.use('ggplot')
        axes[0].scatter(X_new[:,0], X_new[:,1], c=['#8d542a'])
        axes[0].set_xlabel('PC1')
        axes[0].set_ylabel('PC2')
        axes[0].set_title('PCA Plot')

        PC_values = np.arange(pca.n_components_) + 1
        axes[1].plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
        axes[1].set_title('Scree Plot')
        axes[1].set_xlabel('Principal Component')
        axes[1].set_ylabel('Variance Explained')
        plt.show()

    explained_variance = pca.explained_variance_ratio_
    pca_component = abs(pca.components_)
    
    pc1 = pca_component[0]
    pc1_feaimp_rank_idx = pc1.argsort()[::-1][:len(pc1)]
    columns = list(X_train.columns[:-1])
    features = [x for _, x in sorted(zip(pc1_feaimp_rank_idx, columns))]
    imp_table = pd.DataFrame([list(sorted(pc1, reverse=True))], columns=list(features))
    
    return imp_table, explained_variance

def mRMR(x_train, y_train, k):
    X = x_train
    y = pd.Series(y_train)
    selected_features = mrmr_classif(X, y, K = k)
    return selected_features

def mkdf(columns, importances):
    I = pd.DataFrame(data={'Feature':columns, 'Importance':importances})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I

def fit_linear_regression_model(x_train, y_train, x_val, y_val):
    reg = LinearRegression().fit(x_train, y_train)
    reg_pred = reg.predict(x_val)

    print('MAE: ', mean_absolute_error(reg_pred, y_val))

    I = mkdf(x_train.columns,np.abs(reg.coef_))
    viz = plot_importances(I, title="Feature importance via Linear Regression Coefficient Rank(sklearn)")
    return reg

def eval_metrics(model, x_train, y_train, x_val, y_val, scoring, graph=False):
#     scoring = ['r2', 'neg_mean_absolute_percentage_error', 'neg_mean_squared_error']
#     perm = PermutationImportance(model, random_state=0).fit(x_val, y_val)
#     chart = eli5.show_weights(perm, feature_names = x_val.columns.tolist())
    features = []
    records = []
    imp_mean = []
    model.fit(x_train, y_train)
    r = permutation_importance(model, x_val, y_val, 
                               n_repeats=30, 
                               random_state=0,
                               scoring=scoring)
    for i in r.importances_mean.argsort()[::-1]:
#         if r.importances_mean[i] - 2 * r.importances_std[i] > 0:
        features.append(x_val.columns[i])
        record = f"{r.importances_mean[i]:.3f} +/- {r.importances_std[i]:.3f}"
        records.append(record)
        imp_mean.append(round(r.importances_mean[i], 3))
    
    chart = pd.DataFrame({'features':features, 'imp_mean +- std':records})
    if graph == True:
        I = mkdf(chart.features, imp_mean)
        viz = plot_importances(I, title="Feature Importance Rank via permutation")
        return chart
    return chart

def dropcol_regression(model, x_train, y_train, x_val, y_val):
    reg = model
    reg_ = clone(reg)
    reg_.fit(x_train, y_train)
    reg_predict = reg_.predict(x_val)
    baseline = metrics.mean_absolute_error(y_val, reg_predict)

    imp = []
    for col in x_train.columns:
        X = x_train.drop(col, axis=1)
        X_val = x_val.drop(col, axis=1)
        reg_new = clone(reg_)
        reg_new.fit(X, y_train)
        reg_predict_new = reg_new.predict(X_val)
        o = metrics.mean_absolute_error(y_val, reg_predict_new)
        imp.append(np.abs(baseline - o))
    imp = np.array(imp)
    I = pd.DataFrame(
            data={'Feature':x_train.columns,
                  'Importance':imp})
    I = I.set_index('Feature')
    I = I.sort_values('Importance', ascending=False)
    return I

def linear_regression_best_feaimp(x_train, y_train, x_val, y_val, k):
    reg = LinearRegression().fit(x_train, y_train)
    reg_yhat = reg.predict(x_val)
    
    rank_spearman = list(spearman(x_train, y_train).keys())
    rank_pca = pca(x_train)[0].columns.tolist()
    mrmr = mRMR(x_train, y_train, 13)
    permutation = list(eval_metrics(reg, x_train, y_train, x_val, y_val, 'neg_mean_absolute_percentage_error', graph=False)['features'])
    drop_col = list(dropcol_regression(reg, x_train, y_train, x_val, y_val).index)
    
    feature_lists = [rank_spearman, rank_pca, mrmr, permutation, drop_col]
    types = ['rank_spearman', 'rank_pca', 'mrmr', 'permutation', 'drop_col']
    
    mae_baseline = metrics.mean_absolute_error(y_val, reg_yhat)
    
    mae_total_list = {}
    for i, feature_selected in enumerate(feature_lists): #for every selection method
        X_train = x_train
        X_val = x_val
        mae_list = []
        for j in range(k):
            if len(feature_selected) >= 2:
                feature_selected.pop()
                X_train = X_train[feature_selected]
                X_val = X_val[feature_selected]

                reg_new = LinearRegression().fit(X_train, y_train)
                y_hat = reg_new.predict(X_val)
                mae_new = metrics.mean_absolute_error(y_val, y_hat)
                mae_list.append(mae_new)
            else: continue
        key = types[i]
        mae_total_list[key] = mae_list
    return mae_total_list

def rf_regression_best_feaimp(x_train, y_train, x_val, y_val, k):
    rf = RandomForestRegressor(n_estimators=100,
                                min_samples_leaf=1,
                                n_jobs=-1,
                                oob_score=True)

    rf.fit(x_train, y_train)
    y_hat = rf.predict(x_val)
    base_mae = metrics.mean_absolute_error(y_val,y_hat)
    
    
    rank_spearman = list(spearman(x_train, y_train).keys())
    rank_pca = pca(x_train)[0].columns.tolist()
    mrmr = mRMR(x_train, y_train, 13)
    permutation = list(eval_metrics(rf, x_train, y_train, x_val, y_val, 'neg_mean_absolute_percentage_error', graph=False)['features'])
    drop_col = list(dropcol_regression(rf, x_train, y_train, x_val, y_val).index)
    
    feature_lists = [rank_spearman, rank_pca, mrmr, permutation, drop_col]
    types = ['rank_spearman', 'rank_pca', 'mrmr', 'permutation', 'drop_col']

    
    mae_total_list = {}
    for i, feature_selected in enumerate(feature_lists): #for every selection method
        X_train = x_train
        X_val = x_val
        mae_list = []
        for j in range(k):
            if len(feature_selected) >= 2:
                feature_selected.pop()
                X_train = X_train[feature_selected]
                X_val = X_val[feature_selected]

                rf_new = RandomForestRegressor(n_estimators=100,
                                min_samples_leaf=1,
                                n_jobs=-1,
                                oob_score=True).fit(X_train, y_train)
                y_hat = rf_new.predict(X_val)
                mae_new = metrics.mean_absolute_error(y_val, y_hat)
                mae_list.append(mae_new)
            else: continue
        key = types[i]
        mae_total_list[key] = mae_list
    return mae_total_list

def xgboost_best_feaimp(x_train, y_train, x_val, y_val, k):
    model = XGBRegressor()
    model.fit(x_train, y_train)
    cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
    scores = np.abs(cross_val_score(model, x_train, y_train, scoring='neg_mean_absolute_error', cv=cv, n_jobs=-1))
    base_mae = scores.mean()
    
    rank_spearman = list(spearman(x_train, y_train).keys())
    rank_pca = pca(x_train)[0].columns.tolist()
    mrmr = mRMR(x_train, y_train, 13)
    permutation = list(eval_metrics(model, x_train, y_train, x_val, y_val, 'neg_mean_absolute_percentage_error', graph=False)['features'])
    drop_col = list(dropcol_regression(model, x_train, y_train, x_val, y_val).index)
    
    feature_lists = [rank_spearman, rank_pca, mrmr, permutation, drop_col]
    types = ['rank_spearman', 'rank_pca', 'mrmr', 'permutation', 'drop_col']
    
    mae_total_list = {}
    for i, feature_selected in enumerate(feature_lists): #for every selection method
        X_train = x_train
        X_val = x_val
        mae_list = []
        for j in range(k):
            if len(feature_selected) >= 2:
                feature_selected.pop()
                X_train = X_train[feature_selected]
                X_val = X_val[feature_selected]

                model = XGBRegressor()
                model.fit(X_train, y_train)
                mae_new = metrics.mean_absolute_error(y_val, model.predict(X_val))
                mae_list.append(mae_new)
            else: continue
        key = types[i]
        mae_total_list[key] = mae_list
    return mae_total_list

def plot_comparison(mae, model_name):
    methods = list(mae.keys())
    mae_list = list(mae.values())
    fig,ax = plt.subplots(figsize = (10,8))
    

    ax.plot(mae_list[0][::-1],label = 'Spearman Rank Coefficient',color = '#003f5c',marker = 'o')
    ax.plot(mae_list[1][::-1],label = 'PCA',color = '#58508d',marker = '+')
    ax.plot(mae_list[2][::-1],label = 'mRMR',color = '#bc5090',marker = '>')
    ax.plot(mae_list[3][::-1],label = 'Permutation',color = '#ff6361',marker = 'x')
    ax.plot(mae_list[4][::-1],label = 'Drop Column',color = '#ffa600',marker = '+')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title(model_name)
    ax.set_xlabel('number of features')
    ax.set_ylabel('Metric: MAE score')
    plt.legend()
    plt.show()

def auto_select(model, x_train, y_train, x_val, y_val, k):
    
    rank_spearman = list(spearman(x_train, y_train).keys())[:k]
    rank_pca = pca(x_train)[0].columns.tolist()[:k]
    mrmr = mRMR(x_train, y_train, 13)[:k]
    permutation = list(eval_metrics(model, x_train, y_train, x_val, y_val, 'neg_mean_absolute_percentage_error', graph=False)['features'])[:k]
    drop_col = list(dropcol_regression(model, x_train, y_train, x_val, y_val).index)[:k]
    
    feature_lists = [rank_spearman, rank_pca, mrmr, permutation, drop_col]
    methods = ['rank_spearman', 'rank_pca', 'mrmr', 'permutation', 'drop_col']
    
    model.fit(x_train, y_train)
    y_hat = model.predict(x_val)
    baseline = metrics.mean_absolute_error(y_val, y_hat)

    scores = {}
    for i, feature in enumerate(feature_lists):
        X_train = x_train[feature] 
        X_val = x_val[feature]

        model_new = model.fit(X_train, y_train)
        y_hat_new = model_new.predict(X_val)
        mae_new = round(metrics.mean_absolute_error(y_val, y_hat_new), 3)
        method = methods[i]
        scores[method] = mae_new
        
    best_method = min(scores, key=scores.get)
    features = feature_lists[methods.index(best_method)]

    fig,ax = plt.subplots(figsize = (8,6))
    methods = list(scores.keys())
    mae_scores = list(scores.values())

    ax.bar(methods,mae_scores,color = '#003f5c')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Auto-select Feature Selection Method')
    ax.set_xlabel('Type of Feature Selection Method')
    ax.set_ylabel('Metric: MAE score')
    plt.show() 
    return scores, best_method, model, features

def auto_select_graph(scores):
    fig,ax = plt.subplots(figsize = (8,6))
    methods = list(scores.keys())
    mae_scores = list(scores.values())

    ax.bar(methods,mae_scores,color = '#003f5c')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_title('Auto-select Feature Selection Method')
    ax.set_xlabel('Type of Feature Selection Method')
    ax.set_ylabel('Metric: MAE score')
    plt.show()


def get_feature_importances(x_train, y_train, x_val, y_val, shuffle=None):
    if shuffle == True:
        y_train = y_train.copy().sample(frac=1.0)
    
    # Fit the model
    dtrain = xgb.DMatrix(x_train, label=y_train)
    param = {'max_depth': 6, 'learning_rate': 0.03}
    num_round = 100
    bst = xgb.train(param, dtrain, num_round)
    
    # Get feature importances
    imp_df = pd.DataFrame()
    imp_df["feature"] = list(x_train.columns)
    imp_df["importance_gain"] = list(bst.get_score(importance_type='total_gain').values())
    imp_df["importance_split"] = list(bst.get_score(importance_type='weight').values())
    dtest = xgb.DMatrix(x_val)
    imp_df['MAE_loss'] = metrics.mean_absolute_error(y_val, bst.predict(dtest))
    
    return imp_df

def null_imp_df(x_train, y_train, x_val, y_val):
    null_imp_df = pd.DataFrame()
    nb_runs = 80
    import time
    start = time.time()
    dsp = ''
    for i in range(nb_runs):
        # Get current run importances
        imp_df = get_feature_importances(x_train, y_train, x_val, y_val, shuffle=True)
        imp_df['run'] = i + 1 
        # Concat the latest importances with the old ones
        null_imp_df = pd.concat([null_imp_df, imp_df], axis=0)
        # Erase previous message
        for l in range(len(dsp)):
            print('\b', end='', flush=True)
        # Display current run and time used
        spent = (time.time() - start) / 60
        dsp = 'Done with %4d of %4d (Spent %5.1f min)' % (i + 1, nb_runs, spent)
    return null_imp_df


def display_distributions(actual_imp_df_, null_imp_df_, feature_):
    plt.figure(figsize=(13, 6))
    gs = gridspec.GridSpec(1, 2)
    # Plot Split importances
    ax = plt.subplot(gs[0, 0])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_split'].values, color = "#ffa600", label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_split'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Split Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (split) Distribution for %s ' % feature_.upper())
    # Plot Gain importances
    ax = plt.subplot(gs[0, 1])
    a = ax.hist(null_imp_df_.loc[null_imp_df_['feature'] == feature_, 'importance_gain'].values, color = "skyblue", label='Null importances')
    ax.vlines(x=actual_imp_df_.loc[actual_imp_df_['feature'] == feature_, 'importance_gain'].mean(), 
               ymin=0, ymax=np.max(a[0]), color='r',linewidth=10, label='Real Target')
    ax.legend()
    ax.set_title('Gain Importance of %s' % feature_.upper(), fontweight='bold')
    plt.xlabel('Null Importance (gain) Distribution for %s ' % feature_.upper())