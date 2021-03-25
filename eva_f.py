import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, plot_confusion_matrix, plot_roc_curve

def evaluate (model, X_train, X_test, y_train, y_test, use_decision_function='yes'):
    '''
    Evaluate a classifier model based on both training and testing predictions.
    In terms of Accuracy, F1-score and Roc-Auc score.
    
    Corresponding training and testing results will be save to the list for later model comparison.
    
    Also shows confusion matrix and roc-curve for the testing
    -*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*-*
    
    Inputs:
    - use_decision_function allows you to toggle whether you use decision_function or
    predict_proba in order to get the output needed for roc_auc_score
    - If use_decision_function == 'skip', then it ignores calculating the roc_auc_score
    
    Outputs:
    - train_acc, test_acc, train_f1, test_f1, train_roc_auc, test_roc_auc
    '''
    
    
    # accuracy
    train_acc = []
    test_acc = []
    
    # f1-score
    train_f1 = []
    test_f1 = []
    
    # roc-auc score
    train_roc_auc = []
    test_roc_auc = []
    
    # Grab predictions
    y_train_preds = model.predict(X_train)
    y_test_preds = model.predict(X_test)
    
    # out-put need to roc-auc score
    if use_decision_function == 'skip': # skips calculating the roc_auc_score
        train_score = False
        test_score = False
    
    elif use_decision_function == 'yes': # not all classifiers have decision_function
        train_score = model.decision_function(X_train)
        test_score = model.decision_function(X_test)
    
    elif use_decision_function == 'no':
        train_score = model.predict_proba(X_train)[:, 1] # proba for the 1 class
        test_score = model.predict_proba(X_test)[:, 1]
    
    else:
        raise Exception ("The value for use_decision_function should be 'skip', 'yes' or 'no'.")

    
    # Print scores and append the values to corresponding lists
    # Train    
    print("Training Scores")
    print("-*-*-*-*-*-*-*-*")
    
    print(f"Accuracy: {accuracy_score(y_train, y_train_preds)}")
    train_acc.append(accuracy_score(y_train, y_train_preds))
    
    print(f"F1 Score: {f1_score(y_train, y_train_preds)}")
    train_f1.append(f1_score(y_train, y_train_preds))
    
    if type(train_score) == np.ndarray:
        print(f"ROC-AUC: {roc_auc_score(y_train, train_score)}")
    train_roc_auc.append(roc_auc_score(y_train, train_score))
    
    print('\n')

    
    # Test    
    print("Testing Scores")
    print("-*-*-*-*-*-*-*-*")
    
    print(f"Accuracy: {accuracy_score(y_test, y_test_preds)}")
    test_acc.append(accuracy_score(y_test, y_test_preds))
    
    print(f"F1 Score: {f1_score(y_test, y_test_preds)}")
    test_f1.append(f1_score(y_test, y_test_preds))
    
    if type(test_score) == np.ndarray:
        print(f"ROC-AUC: {roc_auc_score(y_test, test_score)}")
    test_roc_auc.append(roc_auc_score(y_test, test_score))

    
    # plot test confusion matrix    
    plot_confusion_matrix(model, X_test, y_test, cmap=plt.cm.Blues)
    
    # plot test roc-curve
    plot_roc_curve(model, X_test, y_test)
    
    plt.show()
    
    
    return train_acc, test_acc, train_f1, test_f1, train_roc_auc, test_roc_auc