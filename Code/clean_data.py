import numpy as np


def clean_data(input_data,y):

    # Seperate in 4 different batches according to PRI_jet_num
    jet_feature_num = 22

    tx0, y0 = filter(input_data,y,jet_feature_num,0)
    tx1, y1 = filter(input_data,y,jet_feature_num,1)
    tx2, y2 = filter(input_data,y,jet_feature_num,2)
    tx3, y3 = filter(input_data,y,jet_feature_num,3)


    # Delete features which have same value for all data (also deletes missing features)

    tx0 = delete_redundant_features(tx0)
    tx1 = delete_redundant_features(tx1)
    tx2 = delete_redundant_features(tx2)
    tx3 = delete_redundant_features(tx3)


    # Normalizing data

    tx0 = standardize(tx0)
    tx1 = standardize(tx1)
    tx2 = standardize(tx2)
    tx3 = standardize(tx3)


    # Deletes the outliers of the standardized data above the threshold in absolute value

    threshold = 10; 

    tx0, y0 = delete_outliers(tx0,y0,threshold)
    tx1, y1 = delete_outliers(tx1,y1,threshold)
    tx2, y2 = delete_outliers(tx2,y2,threshold)
    tx3, y3 = delete_outliers(tx3,y3,threshold)

    return tx0, tx1, tx2, tx3, y0, y1, y2, y3


def delete_redundant_features(data):
    """ Deletes the features with the same value for the whole data """

    return data[:,np.any(data != data[0,:],0)]


def filter(data, y, feature, value):
    """ Filters the data with the same value for a feature """

    idxs = np.ravel(np.where(data[:,feature]==value))

    data_i = data[idxs,:]
    y_i = y[idxs]

    return data_i, y_i


def standardize(x):
    """ Standardize the data set """

    mean_x = np.mean(x,0)
    x = x - mean_x
    std_x = np.std(x,0)
    x = x / std_x

    return x


def delete_outliers(data, y, threshold):
    """ Deletes standardized data points which have at least one value above the threshold """
    idxs = np.all(abs(data)<threshold,1)

    data_clean = data[idxs,:]
    y_clean = y[idxs]

    return data_clean, y_clean