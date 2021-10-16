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

#TODO: for the higherarchical model: build model based on the the 'lower' class (for training data can use real labels, for test data must use predicted labels to avoid biasing results)

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
Questions: 
1. ter_cor=findCorrelation(cor(terrain[,-1]), cutoff=0.9)
    ter_cor
    cor(terrain[,-1]) --- what exactly is the point of this??? this isn't used and you just output it. is the idea that you
    will actually REMOVE the columns with too high correlation?? 
    
2. optical_cor=optical_cor+1 -- what are you doing here??? -- why are you adding 1 to the columns? 
3. pred_order=colnames(pred_order)[max.col(pred_order, ties.method="first")] -- what is happening here; you're selecting your final prediction ...??? 
    what's the logic at the end with the prediction order and the second column
4. eia_train_gg_luv$grt.grp=as.character(eia_train_gg_luv$grt.grp)
    eia_train_gg_org$grt.grp=as.character(eia_train_gg_org$grt.grp) -- what does the 'as character' do? 
5. what are you doing with model importance in separating soil subgroups for test  (lines 362~~) 
6. why are you subsetting in the first 10 columns wwhen you've already done that ??? (348) 
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


def get_weights(train, level='Order'):
    # balance case weights at the order level
    w = (1/train[level].value_counts())/len(train)
    w = pd.DataFrame(w/sum(w)).reset_index()
    weights = pd.DataFrame(train[level])
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


def model_build(train, test, level, weights, higherarcical=False, **kwargs):
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

    # TODO: in higher groups, this is where we start building higherarchical models
    # then do a final model
    if higherarcical:
        predict = pd.DataFrame()
        for level in train[kwargs['grp_type']].unique():
            # because the indices don't change, I can use the original train and test sets to subset the train/test sub by prior level
            level_train = train_sub[train[kwargs['grp_type']] == level]
            level_test = test_sub[kwargs['predict_df'][kwargs['grp_type']] == level]

            final_x = level_train[final_bands]
            final_y = level_train.iloc[level_dict[level]]
            test_x = level_test[final_bands]
            test_y = level_test[level_dict[level]]

            level_weight = get_weights(final_x, level)
            forest = RandomForestClassifier(n_estimators=500, oob_score=True)
            forest.fit(X=final_x, y=final_y, sample_weight=level_weight)
            # TODO: find somehwere/method to store feature importances
            predict_temp = forest.predict(X=test_x)

            # TODO: do the whole second highest thing
            predict = pd.concat([predict, predict_temp])
            # TODO: make sure index values are preserved

    else:
        final_x = train_sub[final_bands]
        final_y = train_sub.iloc[level_dict[level]]
        # TODO: determine if there would be an equivalent to probability forest (predict_proba?)
        forest = RandomForestClassifier(n_estimators=500, oob_score=True)
        forest.fit(X=final_x, y=final_y, sample_weight=weights)
        # TODO: have a summary table -- r prints out a type/number of trees, sample size, etc. see if this is possible

        # TODO: this just needs to be an output as well
        features = pd.Series(forest.feature_importances_, index=final_x.columns)

        test_x = test_sub[final_bands]
        test_y = test_sub[level_dict[level]]
        # predict data
        # TODO: this is a softmax regression: % that it is each group
        predict = forest.predict(X=test_x)
        # then select the NAMES of the SECOND HIGHEST PREDICTION for each data point (second most likely)
        # then subset the PERCENTAGES of the MOST likely and SECOND most likely
        # and then see..... if they... .tie??????

        # TODO: r-like confusion matrix with sensitivyt/specificicty/ pso neg values etc (will need to look it up)
    return predict




if __name__ == '__main__':
    # get rid of unnamed column
    eia = pd.read_csv('EIA_soil_predictors_27Jan2021_PS.csv').iloc[:, 1:]

    # TODO: figure out the kennard stone split... for now just do regular train/test
    train, test = train_test_split(eia, train_size=0.75)  # , metric='mahal')

    # get training weights
    weights = get_weights(train)
