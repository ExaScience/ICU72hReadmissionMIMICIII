# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 22:53:59 2018

@authors: a.pakbin, T.J. Ashby
"""
from sklearn.model_selection import StratifiedKFold
from auxiliary import grid_search,ICD9_categorizer, save_fold_data, convert_numbers_to_names, min_max_mean_auc_for_labels, train_test_one_hot_encoder, possible_values_finder,train_test_normalizer, train_test_imputer, feature_importance_saver, feature_importance_updator, save_roc_curve, data_reader, vectors_to_csv, create_subfolder_if_not_existing, feature_rankings_among_all_labels_saver
import numpy as np
import pandas as pd
from fmeasure import roc, maximize_roc
from xgboost.sklearn import XGBClassifier  
import random as rnd
from sklearn.metrics import roc_auc_score
import pickle
import gc
import sys
import logging as lg



#
# NB: the original code base contains code that will trigger
# "pandas.core.common.SettingWithCopyError: A value is trying to be set on a
# copy of a slice from a DataFrame" errors if the code is run with 
# pd.set_option('mode.chained_assignment', 'raise'). Hence I'm not using it.
#

def main(file_name,
         data_address,
         writing_address):

    lg.basicConfig(stream=sys.stderr, level=lg.DEBUG)
    mpl_logger = lg.getLogger('matplotlib')
    mpl_logger.setLevel(lg.WARNING)
    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.max_columns', 20)

    data_address = str(data_address)
    writing_address = str(writing_address)

    #the address where MIMIC III tables are in .csv.gz format. The tables are: D_ICD_PROCEDURES.csv.gz, D_ITEMS.csv.gz and D_LABITEMS.csv.gz
    #conversion_tables_address='../data'
    conversion_tables_address = data_address

    #outcome labels can contain: '24hrs' ,'48hrs','72hrs', '24hrs~72hrs','7days','30days', 'Bounceback'
    outcome_labels=['24hrs' ,'48hrs','72hrs', '24hrs~72hrs','7days','30days', 'Bounceback']
    normalize_data=False
    save_folds_data=True
    values_for_grid_search=[np.linspace(start=1, stop=6, num=6),[50,100,200,1000,1500],[0.1]]
    num_of_folds=5
    #################################

    categorical_column_names=['ADMISSION_TYPE', 'INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY','FIRST_CAREUNIT', 'GENDER']

    # Read the CSV file
    # - The form of the CSV file is:
    #   - 
    data=data_reader(data_address, file_name)

    # Returns a dictionary where each column name is a key, and the result is the
    # set of values that can appear (with NaN etc removed)
    possible_values=possible_values_finder(data, categorical_column_names)

    # Fill in the target data column
    data['IsReadmitted_24hrs~72hrs']=[1 if x>0 else 0 for x in (data['IsReadmitted_72hrs']-data['IsReadmitted_24hrs'])]

    # List of non-feature column names
    non_attribute_column_names=['HADM_ID', 'ICUSTAY_ID', 'INTIME', 'OUTTIME', 'SUBJECT_ID', 'IsReadmitted_24hrs','IsReadmitted_Bounceback','IsReadmitted_24hrs~72hrs' ,'IsReadmitted_48hrs','IsReadmitted_72hrs','IsReadmitted_7days','IsReadmitted_30days', 'Time_To_readmission', 'hospital_expire_flag']

    if 'Subset' in data.columns:
        #
        # NB: If doing subsetting, you should NOT add the test fold from subset A to
        # the real test data from subset B, otherwise you'll get better results than
        # you should (as the model is trained on subset A and so will do well on the
        # slice of subset A included in the test set).
        #
        testOnSubsetA = False
    else:
        #
        # However, if there is no subsetting (everything is subset A), then you need
        # to use the test data from subset A, otherwise there is no test data. Hence
        # the flag.
        #
        lg.info("No subsetting in input data")
        data.loc[:, 'Subset'] = 'A'
        testOnSubsetA = True

    non_attribute_column_names.append('Subset')


    #TODO: for excludig insurance, language, religion, marital status and ethnicity from the data, uncomment the following line
    #non_attribute_column_names += ['INSURANCE', 'LANGUAGE', 'RELIGION', 'MARITAL_STATUS', 'ETHNICITY']

    # 
    # The function ICD9_categorizer() coarsens the ICD codes to a higher level
    # by dropping the last code digit - but, it looks like there may be some
    # issues with the original code as it treats the ICD codes as numbers rather
    # than strings and so doesn't take into account the semantically meaningful
    # leading and trailing zeros.
    #
    data=ICD9_categorizer(data)

    model_type='XGB'

    PREDICTIONS=list()
    current_folder=writing_address


    #
    # Loop over target labels to predict
    #
    for idx, label_column_name in enumerate(['IsReadmitted_'+outcome_label for outcome_label in outcome_labels]):

        #
        # Original code (replaced because we need to handle subsets for the
        # experiments):
        #   icu_stays=data['ICUSTAY_ID'].values
        #   y=data[label_column_name].values
        #   X=data.drop(non_attribute_column_names, axis=1)
        #

        #
        # Subsetting
        #

        # Labels to predict (sklearn format)
        y=data.loc[data['Subset'] == "A", label_column_name].values
        y_testB = data.loc[data['Subset'] == "B", label_column_name].values

        # Input features 
        X = data.loc[data['Subset'] == "A", :].drop(non_attribute_column_names, axis=1)
        X_testB = data.loc[data['Subset'] == "B", :].drop(non_attribute_column_names, axis=1)

        # Output folder
        current_subfolder=current_folder+'/'+outcome_labels[idx]
        create_subfolder_if_not_existing(current_subfolder)

        auc_list=list()

        ICUstayID=list()
        Prediction=list()

        accumulative_feature_importance=None

        print ('\n',model_type, ' '*5,'LABEL: ', outcome_labels[idx])

        skf=StratifiedKFold(n_splits=num_of_folds, shuffle=True, random_state=rnd.randint(1,1e6))

        #
        # Loop over folds
        # - Each fold is a train/test split, with the test being used for the final score
        #
        fold_number=0
        for train_index, test_index in skf.split(X, y):

            fold_number+=1
            print ('\n  fold',fold_number)

            #
            # Original code (replaced because we need to handle subsets for the
            # experiments):
            #   X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            #   y_train, y_test = y[train_index], y[test_index]
            #   icustay_id_train, icustay_id_test=icu_stays[train_index],icu_stays[test_index]
            #

            X_train = X.iloc[train_index]
            y_train = y[train_index]

            if testOnSubsetA == True:
                X_test = pd.concat([X_testB, X.iloc[test_index]])
                y_test = np.concatenate((y_testB, y[test_index]))
            else:
                X_test = X_testB
                y_test = y_testB

            lg.debug("len X_test: {}, len y_test: {}".format(len(X_test), len(y_test)))

            #
            # Original code (replaced because we need to handle subsets for the
            # experiments):
            #   icustay_id_train, icustay_id_test=icu_stays[train_index],icu_stays[test_index]
            #

            icustay_id_train = (data.loc[data['Subset'] == "A", 'ICUSTAY_ID'].values)[train_index]


            testB = data.loc[data['Subset'] == "B", 'ICUSTAY_ID'].values

            if testOnSubsetA == True:
                testA = (data.loc[data['Subset'] == "A", 'ICUSTAY_ID'].values)[test_index]
                icustay_id_test = np.concatenate((testB, testA))
            else:
                icustay_id_test = testB

            lg.debug("len icustay_id_test: {}".format(len(icustay_id_test)))


            # Fill in missing values in train and test sets
            [X_TRAIN_IMPUTED, X_TEST_IMPUTED]=train_test_imputer(X_train, X_test, categorical_column_names)

            if normalize_data:
                [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=train_test_normalizer(X_TRAIN_IMPUTED, X_TEST_IMPUTED, categorical_column_names)  
            else:
                [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=[X_TRAIN_IMPUTED, X_TEST_IMPUTED]

            # Do one-hot encoding for categorical variables
            [X_TRAIN_NORMALIZED, X_TEST_NORMALIZED]=train_test_one_hot_encoder(X_TRAIN_NORMALIZED, X_TEST_NORMALIZED, categorical_column_names, possible_values)

            if save_folds_data:

                # Save the train and test inputs for this fold
                save_fold_data(current_subfolder, fold_number, icustay_id_train, X_TRAIN_NORMALIZED, y_train, icustay_id_test, X_TEST_NORMALIZED, y_test, convert_names=True, conversion_tables_address=conversion_tables_address)

            [max_depths, n_estimators, learning_rates]=values_for_grid_search

            #
            # Grid search to find best hyperparams
            #  - Hyper params picked per fold (?)
            #  - Hyper params picked using nested k-fold with 2 folds (?)
            #
            best_settings=grid_search(X=X_TRAIN_NORMALIZED, y=y_train, num_of_folds=2, verbose=True, return_auc_values=False, first_dim=max_depths, second_dim=n_estimators, third_dim=learning_rates)

            print ('{:<4s}{:<16s}: max_depth: {:<1s}, n_estimators: {:<2s}, learning_rate: {:<2s}'.format('','best hyperparameters', str(best_settings[0]), str(best_settings[1]), str(best_settings[2])))


            model=XGBClassifier(max_depth=int(best_settings[0]), n_estimators=int(best_settings[1]), learning_rate=best_settings[2])

            #
            # Do the actual training (with the best hyperparams)
            #
            model.fit(X_TRAIN_NORMALIZED, y_train)

            feature_importance=model.feature_importances_
            accumulative_feature_importance=feature_importance_updator(accumulative_feature_importance, feature_importance)

            # Dump the feature importances to file
            pd.DataFrame(data={'FEATURE_NAME': convert_numbers_to_names(X_TRAIN_NORMALIZED.columns, conversion_tables_address), 'IMPORTANCE': feature_importance}).sort_values(by='IMPORTANCE', ascending=False).reset_index(drop=True).to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_ranked_feature_importances.csv')

            #
            # Make the predictions on the test set
            #
            predictions=model.predict_proba(X_TEST_NORMALIZED)[:,1]

            # Append results to an array (?)
            # These variables seem to be only assigned to, never used
            ICUstayID=np.append(ICUstayID,icustay_id_test)
            Prediction=np.append(Prediction,predictions)

            # Write stuff out...
            lg.debug("Vector lengths: 1 icustay_id_test: {}, 2 predictions: {}, 3 y_test: {}".format(len(icustay_id_test), len(predictions), len(y_test)))
            vectors_to_csv(current_subfolder, file_name='fold_'+str(fold_number), vector_one=icustay_id_test, label_one='ICUSTAY_ID', vector_two=predictions, label_two='PREDICTION', vector_three=y_test, label_three='LABEL')

            auc=roc_auc_score(y_true=y_test, y_score=predictions)
            auc_list.append(auc)
            ROC=roc(predicted=predictions, labels=y_test)
            ROC.to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_roc.csv')

            maximum=maximize_roc(ROC, maximization_criteria='fscore')
            maximum.to_csv(current_subfolder+'/'+'fold_'+str(fold_number)+'_optimum_point.csv')

            TPR, FPR = ROC['recall'].values, 1-ROC['specificity'] 

            # Minor change here to allow different figure formats
            figtype = 'png'
            save_roc_curve(current_subfolder+'/'+'fold_'+str(fold_number)+'_roc_curve.'+figtype, TPR, FPR, auc)        
            pickle.dump(model, open(current_subfolder+'/'+'fold_'+str(fold_number)+'.model','wb'))
            print (' '+'-'*30)


        feature_importance_saver(address=current_subfolder, col_names=convert_numbers_to_names(X_TRAIN_NORMALIZED.columns, conversion_tables_address), accumulative_feature_importance=accumulative_feature_importance, num_of_folds=num_of_folds)

        # Minor change here to avoid complications with python generator functions
        vectors_to_csv(current_subfolder, file_name='folds_AUC', vector_one=auc_list, label_one='AUC', vector_two=list(range(1,num_of_folds+1)), label_two='FOLD_NUMBER')
        gc.collect()

    current_folder=writing_address
    min_max_mean_auc_for_labels(current_folder, outcome_labels)
    feature_rankings_among_all_labels_saver(current_folder,outcome_labels, conversion_tables_address)



if __name__=='__main__':

    file_name = sys.argv[1]
    data_address = sys.argv[2]
    writing_address = sys.argv[3]

    main(file_name, data_address, writing_address)
