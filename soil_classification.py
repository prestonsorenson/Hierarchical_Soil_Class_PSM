import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
#from kennard_stone import train_test_split
from sklearn.model_selection import train_test_split
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#TODO: impliment conditioned Latin Hypercube Sampling instead of kennard stone /regular train test split???

# TODO: plot all the figures as well

# hierarchical: lower levels constrain predictions of higher levels

"""
ISSUES/FOR PRESTON: 
A) CODE AVAILABILITY ISSUES
1. kennard stone train-test split doesn't really exist/needs to be investigated further before use 
4. there may not be an equivalent to the 'probability forest' option  -- predict proba 

B) QUESTIONS 
3. need an output of the plots so they can be emulated (since I currently cannot run the script 
4. What are we building at the very end? I lose the thread 
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
    final_df = df.drop(columns=to_drop)
    return final_df, final_df.corr()


def get_weights(train):
    # balance case weights at the order level
    w = (1/train['Order'].value_counts())/len(train)
    w = pd.DataFrame(w/sum(w)).reset_index()
    weights = pd.DataFrame(train['Order'])
    weights = np.array(weights.merge(w, left_on='Order', right_on='index', how='left')['Order_y'])
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

    # TODO: determine what it is that is happening with these correlations
    corr_df, corr_matrix = find_corr(feature_df.iloc[:, 1:])

    # TODO: chcek to make sure the translation from ranger to random forest works; confirm with preston
    forest = RandomForestClassifier(n_estimators=500)
    forest.fit(X=feature_df.iloc[:, 1:], y=feature_df.iloc[:, 0], sample_weight=weights)
    features = list(pd.Series(forest.feature_importances_, index=feature_df.columns[1:]).sort_values(ascending=False).index)
    val = []
    for x in range(1, len(features)):
        temp = feature_df[features[0:x+1]]
        forest = RandomForestClassifier(n_estimators=500, oob_score=True, verbose=True)
        forest.fit(X=temp, y=feature_df.iloc[:, 0], sample_weight=weights)
        #print(x+1, forest.oob_score_)
        # algorithm calculates oob score, so we need to convert to error
        val.append(1 - forest.oob_score_)

    # as per preston, we don't need to use the feature dict, just subset where it is a minimum
    # TODO: potentially create a condition that stops stops min if the delta between the two steps is less than a certain value
    return features[:val.index(min(val))+1]


def model_build(train, test, level, weights):
    level_dict = {'subgroup': 4, 'greatgroup': 5, 'order': 6, 'class': 3, 'series': 7}
    train_sub = train.iloc[:, [level_dict[level]] + list(range(10, 81))]
    test_sub = test.iloc[:, [level_dict[level]] + list(range(10, 81))]

    band_var = []
    for bands in ['terrain', 'band_ratios', 'sar', 'optical']:
        print(bands)
        temp = band_eng(train_sub, weights, bands)
        band_var.extend(temp)

    # now do the same process for the final model with all the bands
    final_bands = band_eng(train_sub[[train_sub.columns[0]] + band_var], weights, 'compile')

    # then do a final model
    final_x = train_sub[final_bands]
    final_y = train_sub.iloc[level_dict[level]]
    # TODO: determine if there would be an equivalent to probability forest (predict_proba?)
    forest = RandomForestClassifier(n_estimators=500, importance='impurity', oob_error=True, split_rule='extratrees')
    forest.fit(X=final_x, y=final_y, sample_weight=weights)
    # what are being done with these???
    features = sorted(forest.feature_importances_).index()

    test_x = test_sub[final_bands]
    test_y = test_sub[level_dict[level]]
    predict = forest.predict(X=test_x)
    # then there's some sort of operation with .... column... names????
    return predict


if __name__ == '__main__':
    # get rid of unnamed column
    eia = pd.read_csv('EIA_soil_predictors_27Jan2021_PS.csv').iloc[:, 1:]

    # TODO: figure out the kennard stone split... for now just do regular train/test
    train, test = train_test_split(eia, train_size=0.75)  # , metric='mahal')

    # get training weights
    weights = get_weights(train)
