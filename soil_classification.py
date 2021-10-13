import numpy as np
import pandas as pd
from skranger.ensemble import RangerForestClassifier
#from kennard_stone import train_test_split
from sklearn.model_selection import train_test_split
import warnings
from pandas.core.common import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

#TODO: use random forest; no need to use rangerforest
#TODO: impliment Conditioned Latin Hypercube Sampling in Python instead of kennard stone /regular train test split
# TODO: findCorr hardcode: cutoff val = keep only one of the variables in correlation (the one with the lowest mean absolute correlation)


# hierarchical: lower levels constrain predictions of higher levels

"""
ISSUES/FOR PRESTON: 
A) CODE AVAILABILITY ISSUES
1. kennard stone train-test split doesn't really exist/needs to be investigated further before use 
2. no equivalent to the type of correlation that is sdone in r 
3. python ranger function has an oob error but doesn't seem to use it anywehre? there is no documentation for it; will have to dig through the code 
4. there may not be an equivalent to the 'probability forest' option  -- predict proba 

B) QUESTIONS
1. confirm: which ndwi bands are to be removed, all that contain 'ndwi'? 
2. we need to subset to the specific bands that are useful... why even do the test here in python??? 
3. need an output of the plots so they can be emulated (since I currently cannot run the script 
4. What are we building at the very end? I lose the thread 
"""



# TODO: plot all the figures as well


def band_eng(train, weights, feature):
    if feature != 'compile':
        feature_dict = {'terrain': [1, 20], 'band_ratios': [20, 44], 'sar': [44, 52], 'optical': [52, 72]}
        feature_df = train.iloc[:, [0] + list(range(feature_dict[feature][0], feature_dict[feature][1]))]
    else:
        feature_df = train.copy()
    if feature == 'band_ratios':
        # remove ndwi
        # TODO: need to test this to ensure the proper bands are being removed; confirm with preston which ones are to be removed
        ndwi = [x for x in feature_df.columns if 'ndwi' in x]
        feature_df.drop(columns=ndwi, inplace=True)
    corr_matrix = feature_df.iloc[:, 1:].corr()
    # TODO: FIND SOMETHING TO MATCH FINDCORRELATION FUNCTION IN R ... CURRENTLY NO EQUIVALENT
    # IS THIS EVEN USED???
    # RESEARCH: https://stackoverflow.com/questions/44994791/equivalent-r-findcorrelationcorr-cutoff-0-75-in-python-pandas/44995470
    # https://www.projectpro.io/recipes/drop-out-highly-correlated-features-in-python

    # use sk ranger function; we are translating from the r presents, so will use presets found below to populate python function
    # https://www.rdocumentation.org/packages/ranger/versions/0.13.1/topics/ranger
    ranger = RangerForestClassifier(n_estimators=500, importance='impurity')
    ranger_pred = ranger.fit(X=feature_df.iloc[:, 1:], y=feature_df.iloc[:, 0], sample_weight=weights)
    features = list(pd.Series(ranger_pred.feature_importances_, index=feature_df.columns[1:]).sort_values(ascending=False).index)
    val = []
    for x in range(len(features)):
        temp = feature_df[features[0:x+1]]
        ranger = RangerForestClassifier(n_estimators=500, importance='impurity', oob_error=True, verbose=True)
        ranger_pred = ranger.fit(X=temp, y=feature_df.iloc[:, 0], sample_weight=weights)
        # TODO: it doesn't seem as if the oob_error feature actually leads anywhere
        val.append(ranger.score(X=temp, y=feature_df.iloc[:, 0], sample_weight=weights))

    # TODO: subset the proper bands that were decided by preston???
    # with OOB error, it would be min val, because this is just accuracy, will be doing max val
    return features[:val.index(max(val))]

def get_weights(train):
    # balance case weights at the order level
    w = (1/train['Order'].value_counts())/len(train)
    w = pd.DataFrame(w/sum(w)).reset_index()
    weights = pd.DataFrame(train['Order'])
    weights = np.array(weights.merge(w, left_on='Order', right_on='index', how='left')['Order_y'])
    return weights

def model_build(train, test, level, weights):
    level_dict = {'subgroup': 4, 'greatgroup': 5, 'order': 6, 'class': 3, 'series': 7}
    train_sub = train.iloc[:, [level_dict[level]] + list(range(10, 81))]
    test_sub = test.iloc[:, [level_dict[level]] + list(range(10, 81))]

    band_var = []
    for bands in ['terrain', 'band_ratios', 'sar', 'optical']:
        temp = band_eng(train_sub, weights, bands)
        band_var.extend(temp)

    # now do the same process for the final model with all the bands
    final_bands = band_eng(train_sub[[train_sub.columns[0]] + band_var], weights, 'compile')

    # then do a final model
    final_x = train_sub[final_bands]
    final_y = train_sub.iloc[level_dict[level]]
    # TODO: determine if there would be an equivalent to probability forest (predict_proba?)
    ranger = RangerForestClassifier(n_estimators=500, importance='impurity', oob_error=True, split_rule='extratrees')
    ranger_pred = ranger.fit(X=final_x, y=final_y, sample_weight=weights)
    # what are being done with these???
    features = sorted(ranger_pred.feature_importances_).index()

    test_x = test_sub[final_bands]
    test_y = test_sub[level_dict[level]]
    predict = ranger.predict(X=test_x)
    # then there's some sort of operation with .... column... names????
    return predict

if __name__ == '__main__':
    # get rid of unnamed column
    eia = pd.read_csv('EIA_soil_predictors_27Jan2021_PS.csv').iloc[:,1:]

    # TODO: figure out the kennard stone split... for now just do regular train/test
    train, test = train_test_split(eia, train_size=0.75)  # , metric='mahal')

    # get training weights
    weights = get_weights(train)
