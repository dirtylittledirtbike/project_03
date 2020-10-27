import numpy as np
import pandas as pd

def get_word_covariance(word_vec, classifier, n=100, top=True):
        """
        Description:
        get covariance matrix and dataframe for words with
        highest coefficients for each class in our trained model
        
        Paramters:
        word_vec: fitted word vector,
        classifier: trained classifier (multinomial nb)
        n: number of top coefficients
        top: default True returns dataframe and covariance matrix for top coefficients
        otherwise returns lowest covariance matrix/dataframe for lowest coefficients
        
        Returns: covariance matrix of top coefficients for each word
        dataframe of cofficients for each word.
        """
        
        
        if top:
            neg_class_prob_sorted = classifier.feature_log_prob_[0, :].argsort()
            pos_class_prob_sorted = classifier.feature_log_prob_[1, :].argsort()

            negative_class_feats = np.take(word_vec.get_feature_names(), neg_class_prob_sorted[-n:])
            positive_class_feats = np.take(word_vec.get_feature_names(), pos_class_prob_sorted[-n:])

            neg_coefs = np.take(classifier.feature_log_prob_[0, :],neg_class_prob_sorted[-n:])
            pos_coefs = np.take(classifier.feature_log_prob_[1, :],pos_class_prob_sorted[-n:])

            pos_dict = dict(zip(positive_class_feats.tolist(), pos_coefs.tolist()))
            neg_dict = dict(zip(negative_class_feats.tolist(), neg_coefs.tolist()))
            
            df_original = pd.DataFrame([neg_dict, pos_dict]).T
            df_original.columns = ['non_toxic_coefs', 'toxic_coefs']
            
            df = pd.DataFrame([neg_dict, pos_dict]).fillna(np.log(.0000001))
            df_temp = df.T
            df_temp.columns = ['non_toxic_coefs', 'toxic_coefs']
            df_scaled = df_temp - df_temp.mean()

            return df_original, df_scaled.T.cov()
        else:
            neg_class_prob_sorted = classifier.feature_log_prob_[0, :].argsort()
            pos_class_prob_sorted = classifier.feature_log_prob_[1, :].argsort()

            negative_class_feats = np.take(word_vec.get_feature_names(), neg_class_prob_sorted[:n])
            positive_class_feats = np.take(word_vec.get_feature_names(), pos_class_prob_sorted[:n])

            neg_coefs = np.take(classifier.feature_log_prob_[0, :],neg_class_prob_sorted[:n])
            pos_coefs = np.take(classifier.feature_log_prob_[1, :],pos_class_prob_sorted[:n])

            pos_dict = dict(zip(positive_class_feats.tolist(), pos_coefs.tolist()))
            neg_dict = dict(zip(negative_class_feats.tolist(), neg_coefs.tolist()))
            
            df_original = pd.DataFrame([neg_dict, pos_dict]).T
            df_original.columns = ['non_toxic_coefs', 'toxic_coefs']

            return df_original


def get_class_features(vector, classifier, n=10, top=True, indices=False):
    
    """
    Description:
    get words for the highest coefficients for each class on model
    
    Paramters:
    vector: word vector,
    classifier: fitted classifier (multinomial nb)
    n: number of top coefficients
    top: default True returns words with highest coefficients
    otherwise returns lowest words with lowest coefficients
    indices: default False, returns indices of coefficients
    
    Returns: array of words or indices with the highest coefficients for each class
    in trained model.
    """
    
    if top:
        neg_class_prob_sorted = classifier.feature_log_prob_[0, :].argsort()
        pos_class_prob_sorted = classifier.feature_log_prob_[1, :].argsort()
        
        neg_idxs =  neg_class_prob_sorted[-n:]
        pos_idxs = pos_class_prob_sorted[-n:]
        
        negative_class_feats = np.take(vector.get_feature_names(), neg_idxs)
        positive_class_feats = np.take(vector.get_feature_names(), pos_idxs)
        
        # return idices for highest coefficients of each class
        if indices:
            return neg_idxs, pos_idxs
        else:
            return negative_class_feats, positive_class_feats

    else:
        neg_class_prob_sorted = classifier.feature_log_prob_[0, :].argsort()
        pos_class_prob_sorted = classifier.feature_log_prob_[1, :].argsort()
        
        neg_idxs =  neg_class_prob_sorted[:n]
        pos_idxs = pos_class_prob_sorted[:n]
        
        negative_class_feats = np.take(vector.get_feature_names(), neg_idxs)
        positive_class_feats = np.take(vector.get_feature_names(), pos_idxs)
        
        if indices:
            return neg_idxs, pos_idx
        else:
            return negative_class_feats, positive_class_feats
