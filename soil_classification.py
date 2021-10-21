import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# from kennard_stone import train_test_split
from sklearn.model_selection import train_test_split
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# TODO: impliment conditioned Latin Hypercube Sampling instead of kennard stone /regular train test split???

# TODO: plot all the figures as well

# TODO: for the higherarchical model: build model based on the the 'lower' class (for training data can use real labels, for test data must use predicted labels to avoid biasing results)

# hierarchical: lower levels constrain predictions of higher levels

"""
### FOR EACH ONE STILL DO THE FEATURE SELECTION USING ALL DATA 

for great group: 
1. create subsets by order, and then take the first 10???? columns????? 
2. then subset test data by predicted results 
3. set case weights for each order 
4. then do prediction for great groups by order (same method with second highest, etc. as before) 
5. then we concat results (recombine results) and do the whole confusion matrix thing




for subgroup: 
1. before prediction for terrain features, etc. need to create weights by order (line 489)
2. build models for each great group 
3. make sure order for test and train is the same??????????? 


overall: then create maps 
"""

"""

"""


# TODO: add debug and verbose features
# TODO: make function more efficient
# emulate findCorr function in R
def find_corr(df, cutoff=0.9):
    # find correlations
    corr = df.corr()
    # find mean of correlations
    mean_corr = abs(corr.mean(axis=0)).sort_values(ascending=False)
    # find correlated pairs above the cutoff
    corr = corr.where(abs(corr.values) > cutoff).stack().index.values
    # remove self-correlation
    corr = [unique for unique in corr if unique[0] != unique[1]]
    # starting with the highest mean correlation, go through pairs and append to drop list if found, then remove the item from list of lists
    to_drop = []
    for highest in mean_corr.index:
        pre_len = len(corr)
        corr = [tup for tup in corr if highest not in tup]
        if pre_len != len(corr):
            to_drop.append(highest)
    return to_drop
    # final_df = df.drop(columns=to_drop)
    # return final_df, final_df.corr()


def get_weights(df, level='Order', balance=False):
    # balance case weights at the order level
    w = (1 / df[level].value_counts()) / len(df)
    if balance:
        w = w ** (1 / 4)
    w = pd.DataFrame(w / sum(w)).reset_index()
    weights = pd.DataFrame(df[level])
    weights = np.array(weights.merge(w, left_on=level, right_on='index', how='left')[f'{level}_y'])
    return weights


def band_eng(train, weights, feature):
    if feature != 'compile':
        feature_dict = {'terrain': [1, 20], 'band_ratios': [20, 44], 'sar': [44, 52], 'optical': [52, 72]}
        feature_df = train.iloc[:, [0] + list(range(feature_dict[feature][0], feature_dict[feature][1]))]
    else:
        feature_df = train.copy()

    # remove ndwi
    if feature == 'band_ratios':
        ndwi = [x for x in feature_df.columns if 'ndwi' in x]
        feature_df.drop(columns=ndwi, inplace=True)

    if feature == 'optical':
        drop_cols = find_corr(feature_df.iloc[:, 1:])
        feature_df = feature_df.drop(columns=drop_cols)

    # TODO: chcek to make sure the translation from ranger to random forest works; confirm with preston
    forest = RandomForestClassifier(n_estimators=500)
    forest.fit(X=feature_df.iloc[:, 1:], y=feature_df.iloc[:, 0], sample_weight=weights)
    features = list(
        pd.Series(forest.feature_importances_, index=feature_df.columns[1:]).sort_values(ascending=False).index)
    val = []
    for x in range(1, len(features)):
        temp = feature_df[features[0:x + 1]]
        forest = RandomForestClassifier(n_estimators=500, oob_score=True, verbose=True)
        forest.fit(X=temp, y=feature_df.iloc[:, 0], sample_weight=weights)
        # print(x+1, forest.oob_score_)
        # algorithm calculates oob score, so we need to convert to error
        val.append(1 - forest.oob_score_)

    # as per preston, we don't need to use the feature dict, just subset where it is a minimum
    # TODO: potentially create a condition that stops stops min if the delta between the two steps is less than a certain value
    return features[:val.index(min(val)) + 1]


def model_build(train, test, level, weights, hierarchical=False, **kwargs):
    level_dict = {'Soil.Subgr': 4, 'grt.grp': 5, 'Order': 6, 'class': 3, 'Soil.Serie': 7}
    train_sub = train.iloc[:, [level_dict[level]] + list(range(10, 81))]
    test_sub = test.iloc[:, [level_dict[level]] + list(range(10, 81))]

    band_var = []
    for bands in ['terrain', 'band_ratios', 'sar', 'optical']:
        print(bands)
        temp = band_eng(train_sub, weights, bands)
        band_var.extend(temp)

    # now do the same process for the final model with all the bands
    final_bands = band_eng(train_sub[[train_sub.columns[0]] + band_var], weights, 'compile')

    # subset data
    final_x = train_sub[final_bands]
    final_y = train_sub[level]
    test_x = test_sub[final_bands]
    test_y = test_sub[level]

    # then do a final model
    # modification for hierarchical
    if hierarchical:
        predict = pd.DataFrame()
        most_likely = pd.DataFrame()
        for higher_level in train[kwargs['grp_type']].unique():
            # because the indices don't change, I can use the original train and test sets to subset the train/test sub by prior level
            level_x_train = final_x[train[kwargs['grp_type']] == higher_level]
            level_y_train = final_y[train[kwargs['grp_type']] == higher_level]
            level_x_test = test_x[kwargs['predict_df'] == higher_level]

            level_weight = get_weights(train[train[kwargs['grp_type']] == higher_level], level, balance=True)
            forest = RandomForestClassifier(n_estimators=500, oob_score=True)
            forest.fit(X=level_x_train, y=level_y_train, sample_weight=level_weight)

            predict_temp = pd.DataFrame(forest.predict_proba(X=level_x_test), columns=forest.classes_,
                                        index=level_x_test.index)
            if level_y_train.nunique() > 1:
                most_likely_temp = np.argsort(-predict_temp.values, axis=1)[:, :2]
                column_name = ['Most Likely', 'Second Most Likely']
            # for the case where all values are the same
            # TODO: print warning when this is the case
            else:
                most_likely_temp = np.argsort(-predict_temp.values, axis=1)
                column_name = ['Most Likely']

            # TODO: deal with futurewarning
            most_likely_temp = pd.DataFrame(predict.columns[most_likely_temp], columns=column_name,
                                            index=predict_temp.index)

            predict = pd.concat([predict, predict_temp])
            most_likely = pd.concat([most_likely, most_likely_temp])
            # TODO: make sure index values are preserved

    else:

        forest = RandomForestClassifier(n_estimators=500, oob_score=True)
        forest.fit(X=final_x, y=final_y, sample_weight=weights)
        # TODO: have a summary table -- r prints out a type/number of trees, sample size, etc. see if this is possible

        # TODO: this just needs to be an output as well
        features = pd.Series(forest.feature_importances_, index=final_x.columns)

        # predict data
        predict = pd.DataFrame(forest.predict_proba(X=test_x), columns=forest.classes_, index=test_y.index)
        most_likely = np.argsort(-predict.values, axis=1)[:, :2]
        most_likely = pd.DataFrame(predict.columns[most_likely], columns=['Most Likely', 'Second Most Likely'],
                                   index=predict.index)

        # TODO: r-like confusion matrix with sensitivyt/specificicty/ pso neg values etc (will need to look it up)
        # TODO: if hierarch, sort index to order it was in before? (we can match on the dataset anyway.. but still)
    return predict, most_likely


if __name__ == '__main__':
    # get rid of unnamed column
    eia = pd.read_csv('EIA_soil_predictors_27Jan2021_PS.csv').iloc[:, 1:]

    # TODO: figure out the kennard stone split... for now just do regular train/test
    train, test = train_test_split(eia, train_size=0.75, random_state=0)  # , metric='mahal')

    # get training weights
    weights = get_weights(train)
