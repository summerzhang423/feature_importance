# Feature Importance

This report is dedicated to provide a detailed explaination on feature importance, different technics used to measure feature importance and an auto selection function that could pick the best feature selection method for a given ML model.

Strategies applied directly to data:
- Option 1: Spearman's Rank Correlation Coefficient
- Option 2: Principle Component Analysis (PCA)
- Option 3: Minimal-redundancy-maximal-relevance (mRMR)

Strategies applied to models:
- Option 1: Permutation importance
- Option 2: drop column importance


Given above 5 different feature importance strategies, different strategies will rank their feature imporance differently (We used MAE as our metric for model evaluation). I have chosen 3 models for comparison purpose. 1) Regression, 2) Random Forest and 3) XGBoostRegressor. 

<table border="0">
<tr valign="top" border="0">
<td border="0"><img src="image/Regression_comp.png" width="100%"></a></td>
<td border="0"><img src="image/RF_comp.png" width="100%"></a></td>	
<td border="0"><img src="image/XGBoostRegressor_comp.png" width="100%"></a></td>	
</tr>
</table>

From the comparison above we can see:
- The more features there are, the better the metric presents (specifica to our dataset)
- While some features seems to be more appealing to one of the models, it might not be the case for other models. i.e. Permutation strategy seems to work well with regression model, but it did not perform as well with XGBoostRegressor model
- Depending on how many features we want to select, our choice of strategy might differ as well. i.e. for XGBoostRegressor, if only 4 features are wanted, **drop column** is a better strategy in comparison. However, if we want top 10 features instead, **permutation** will be a better strategy than drop column (due to smaller MAE score)

# Auto Feature Importance Selection Algorithm

The question the becomes: "How can we know which feature selection method shall be used given a model"? The answer to that is to use an automated feature importance selection algorithm. Let the algorithm run iteratively through different methods and return the method that can generate the lowest MAE score. 
```
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
 ```

Given model, x_train, x_valid, y_train, y_valid and ideal number of feature, this `auto_select` function is able to return a table with the most suitable feature selection method, ranked features and a graph indicates each method's corresponding MAE score

```
scores, best_method, model, features = auto_select(reg, x_train, y_train, x_val, y_val, k)
print(f'The best feature selection method is {best_method} and top {k} features selected by this method is: {features}')
```

<img src="image/reg_selection.png" width="500">
<img src="image/result.png" width="600">



