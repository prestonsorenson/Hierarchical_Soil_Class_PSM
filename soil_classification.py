import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pandas.core.common import SettingWithCopyWarning
from sklearn.ensemble import RandomForestClassifier
import warnings

logging.basicConfig(level=logging.WARNING)
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# TODO: plot all the figures/outputs
# TODO: add debug and verbose features
# TODO: make function more efficient if possible
# TODO: potentially make classes


# emulate findCorr function in R
def find_corr(df, cutoff=0.9):
    """
    Find most highly correlated values in dataframe. Emulates the findCorr Function in R:
    The absolute values of pair-wise correlations are considered. If two variables have a high correlation,
    the function looks at the mean absolute correlation of each variable and removes the variable with the largest
    mean absolute correlation.
    (https://www.rdocumentation.org/packages/DescTools/versions/0.99.40/topics/FindCorr)

    :param df: (dataframe) input data for which to measure correlation
    :param cutoff: (float) correlation cutoff, pairwise correlation of two values >= cutoff will be flagged for
    removal as per methodology above
    :return: (list) list of column names to remove from df
    """
    # find correlations
    corr = df.corr()
    # find mean of correlations
    mean_corr = abs(corr.mean(axis=0)).sort_values(ascending=False)
    # find correlated pairs above the cutoff
    corr = corr.where(abs(corr.values) >= cutoff).stack().index.values
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


def get_weights(df, level='Order', balance=False):
    """
    Create weights for input into randomforrest classier.

    :param df: (dataframe) dataset that contains the column from which to calculate weights
    :param level: (str) column name of data from which to calculate weights. Default = 'Order'
    :param balance: (bool) balance weights. Default = False.
    :return: (array) numpy array of weights
    """
    # balance case weights at the order level
    w = (1 / df[level].value_counts()) / len(df)
    if balance:
        w = w ** (1 / 4)
    w = pd.DataFrame(w / sum(w)).reset_index()
    weights = pd.DataFrame(df[level])
    weights = np.array(weights.merge(w, left_on=level, right_on='index', how='left')[f'{level}_y'])
    return weights


def band_eng(df, weights, feature):
    """
    for each feature, use random forest classificaition to determine which bands to use
    :param df: (datafrme) n x d dimension dataset in which feature bands are located
    :param weights: (array) an n-length array of weights for each sample
    :param feature: (str) {terrain, band_ratios, sar, optical, compile} indicate which feature to determine optimal
    bands on. If 'compile', no subset is made and all features are considered.
    :return: (list) list of features (column names) on which to subset df where the oob error is minimized
    """
    # subset dataset on feature input
    if feature != 'compile':
        feature_dict = {'terrain': [1, 20], 'band_ratios': [20, 44], 'sar': [44, 52], 'optical': [52, 72]}
        feature_df = df.iloc[:, [0] + list(range(feature_dict[feature][0], feature_dict[feature][1]))]
    else:
        # no subset if 'compile'
        feature_df = df.copy()

    # remove ndwi
    if feature == 'band_ratios':
        ndwi = [x for x in feature_df.columns if 'ndwi' in x]
        feature_df.drop(columns=ndwi, inplace=True)

    # remove highly correlated features for optical
    if feature == 'optical':
        drop_cols = find_corr(feature_df.iloc[:, 1:])
        feature_df = feature_df.drop(columns=drop_cols)

    # measure oob error using an increasing number of features, sorted by importance
    forest = RandomForestClassifier(n_estimators=500)
    forest.fit(X=feature_df.iloc[:, 1:], y=feature_df.iloc[:, 0], sample_weight=weights)
    features = list(
        pd.Series(forest.feature_importances_, index=feature_df.columns[1:]).sort_values(ascending=False).index)
    val = []
    for x in range(1, len(features)):
        temp = feature_df[features[0:x + 1]]
        forest = RandomForestClassifier(n_estimators=500, oob_score=True)
        forest.fit(X=temp, y=feature_df.iloc[:, 0], sample_weight=weights)
        # algorithm calculates oob score, so we need to convert to error
        val.append(1 - forest.oob_score_)
    # plot figures
    plt.figure()
    plt.plot(val)
    plt.title(f'{feature.title()} OOB Error vs Num. Features, by Importance')
    plt.show()
    # subset to feature combination where oob error is at minimum
    # TODO: potentially create a condition that stops stops min if the delta between the two steps is less than a certain value
    return features[:val.index(min(val)) + 1]


def model_build(train, test, level, weights, hierarchical=False, **kwargs):
    """
    Build soil classification model for given soil grouping level
    :param train: (dataframe) n x d dimension training data
    :param test: (dataframe) m x d dimension testing data
    :param level: (str) {Soil.Subgr, grt.grp, Order, class, Soil.Serie} Indicate which soil level to train model for
    :param weights: (array) n-dimension array of weights to apply to each sample
    :param hierarchical: (bool) whether to apply hierarchical training logic for higher groupings. If True, additional
    arguments will need to be passed via kwargs. Default ==False
    :param kwargs: Optional parameters for hierarchical==True.
        grp_type: (str) {grt.grp, Order, class, Soil.Serie} the lower level soil grouping with which to subset models
        for current training data
        predict_df: (dataframe|list) an m x 1 dimension datataset with the names of most likely predicted soil
        classification for grp_type
    :return:
       - m x len(unique classifiers) - dimension dataframe with % liklihood of possible classifications for given level
       - m x 2 - dimension dataframe indicating the most and second most likely soil classification for each sample
    """
    # subset data based on level
    level_dict = {'Soil.Subgr': 4, 'grt.grp': 5, 'Order': 6, 'class': 3, 'Soil.Serie': 7}
    train_sub = train.iloc[:, [level_dict[level]] + list(range(10, 81))]
    test_sub = test.iloc[:, [level_dict[level]] + list(range(10, 81))]

    # determine which feature bands are the most useful
    band_var = []
    for bands in ['terrain', 'band_ratios', 'sar', 'optical']:
        print(f'{level} selecting {bands} bands')
        temp = band_eng(train_sub, weights, bands)
        band_var.extend(temp)

    # now do the same process for the final model with all the bands
    print('selecting final bands')
    final_bands = band_eng(train_sub[[train_sub.columns[0]] + band_var], weights, 'compile')

    # subset data using the final bands
    final_x = train_sub[final_bands]
    final_y = train_sub[level]
    test_x = test_sub[final_bands]
    test_y = test_sub[level]

    # if model hierarchical, split model training by lower-level classifications
    if hierarchical:
        # initialize empty dataframes to append to
        predict = pd.DataFrame()
        most_likely = pd.DataFrame()
        for higher_level in train[kwargs['grp_type']].unique():
            print(f'Predicting {level} for {higher_level} values')
            # because the indices don't change, use original train/est set idx to subset
            level_x_train = final_x[train[kwargs['grp_type']] == higher_level]
            level_y_train = final_y[train[kwargs['grp_type']] == higher_level]
            # subset test data based on prediction of lower level
            level_x_test = test_x[kwargs['predict_df'] == higher_level]

            # if no data found in specific lower level classification, move onto next part of loop
            if len(level_x_test) == 0:
                continue

            # if subset only provides one type option, we don't need to do prediction
            # TODO: print warning when this is the case
            if level_y_train.nunique() == 1:
                predict_temp = pd.DataFrame(np.full(len(level_x_test), 1), columns=[level_y_train.unique()],
                                            index=level_x_test.index)
                most_likely_temp = pd.DataFrame([level_y_train.unique()]*len(level_x_test),
                                                columns=[f'{level.title()} Most Likely'], index=level_x_test.index)

            # for all other cases, do prediction
            else:

                level_weight = get_weights(train[train[kwargs['grp_type']] == higher_level], level, balance=True)
                forest = RandomForestClassifier(n_estimators=500, oob_score=True)
                forest.fit(X=level_x_train, y=level_y_train, sample_weight=level_weight)

                predict_temp = pd.DataFrame(forest.predict_proba(X=level_x_test), columns=forest.classes_,
                                            index=level_x_test.index)
                most_likely_temp = np.argsort(-predict_temp.values, axis=1)[:, :2]

                # TODO: deal with futurewarning
                most_likely_temp = pd.DataFrame(predict_temp.columns[most_likely_temp],
                                                columns=[f'{level.title()} Most Likely',
                                                         f'{level.title()} Second Most Likely'],
                                                index=predict_temp.index)

            predict = pd.concat([predict, predict_temp])
            most_likely = pd.concat([most_likely, most_likely_temp])

    # general model, without subsetting by lower level features
    else:

        forest = RandomForestClassifier(n_estimators=500)
        forest.fit(X=final_x, y=final_y, sample_weight=weights)
        # TODO: model summary table?

        # TODO: should this be an outout as well?
        features = pd.Series(forest.feature_importances_, index=final_x.columns)

        # predict data
        predict = pd.DataFrame(forest.predict_proba(X=test_x), columns=forest.classes_, index=test_y.index)
        most_likely = np.argsort(-predict.values, axis=1)[:, :2]
        most_likely = pd.DataFrame(predict.columns[most_likely],
                                   columns=[f'{level.title()} Most Likely',
                                            f'{level.title()} Second Most Likely'],
                                   index=predict.index)

        # TODO: r-like confusion matrix with sensitivity/specificity
    return predict, most_likely


def final_compile(df, *args):
    """
    Compile predictions with relevant columns
    :param df: (dataframe) n x d -dimension dataframe of training dataset features we want to combine with predictions
    :param args: (dataframes) a variable number of n - length dataframes of predictions to combine with training
    dataset features. Please note: all column names should be unique and all index numbers in df should be present.
    :return: Compiled dataframe
    """
    for temp in args:
        df = pd.concat([df, temp], axis=1)
    return df.sort_index()


if __name__ == '__main__':
    import kennard_stone_split as ks

    # get rid of unnamed column
    eia = pd.read_csv('EIA_soil_predictors_27Jan2021_PS.csv').iloc[:, 1:]

    # spit data using custom kennard-stone split algorithm
    train, test = ks.ks_split(eia)

    # get training weights
    weights = get_weights(train)

    # train order
    # start with order
    order_predict, order_likely = model_build(train, test, 'Order', weights)

    # great group
    gg_predict, gg_likely = model_build(train, test, 'grt.grp', weights, hierarchical=True, grp_type='Order',
                                        predict_df=order_likely['Most Likely'])

    # subgroup
    sb_predict, sb_likely = model_build(train, test, 'Soil.Subgr', weights, hierarchical=True, grp_type='grt.grp',
                                        predict_df=gg_likely['Most Likely'])

    # compile datasets
    final_predictions = final_compile(test[['field_1', 'Site.ID', 'Northing', 'Easting']],
                                      order_likely, gg_likely, sb_likely)

    order_predict = final_compile(test[['field_1', 'Site.ID', 'Northing', 'Easting']], order_predict)

    gg_predict = final_compile(test[['field_1', 'Site.ID', 'Northing', 'Easting']], gg_predict)

    sb_predict = final_compile(test[['field_1', 'Site.ID', 'Northing', 'Easting']], sb_predict)

