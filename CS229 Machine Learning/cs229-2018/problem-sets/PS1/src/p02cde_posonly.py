import numpy as np
import util

from p01b_logreg import LogisticRegression

# Character to replace with sub-problem letter in plot_path/pred_path
WILDCARD = 'X'


def main(train_path, valid_path, test_path, pred_path):
    """Problem 2: Logistic regression for incomplete, positive-only labels.

    Run under the following conditions:
        1. on y-labels,
        2. on l-labels,
        3. on l-labels with correction factor alpha.

    Args:
        train_path: Path to CSV file containing training set.
        valid_path: Path to CSV file containing validation set.
        test_path: Path to CSV file containing test set.
        pred_path: Path to save predictions.
    """
    pred_path_c = pred_path.replace(WILDCARD, 'c')
    pred_path_d = pred_path.replace(WILDCARD, 'd')
    pred_path_e = pred_path.replace(WILDCARD, 'e')

    x_train, y_train = util.load_dataset(train_path, add_intercept=True) # add intercept term
    _, t_train = util.load_dataset(train_path, label_col='t')
    
    x_valid, y_valid = util.load_dataset(valid_path, add_intercept=True)
    _, t_valid = util.load_dataset(valid_path, label_col='t')
    
     x_test, y_test = util.load_dataset(test_path, add_intercept=True)
    _, t_test = util.load_dataset(test_path, label_col='t')
    
    # *** START CODE HERE ***
    # Part (c): Train and test on true labels
    # Make sure to save outputs to pred_path_c
    model1 = LogisticRegression()
    model1.fit(x_train, t_train)
    print("Part (c):")
    print("Theta is: ", model1.theta)
    print("The accuracy on test set is: ", np.mean(t_test == model1.predict(x_test)))
    util.plot(x_test, t_test, model1.theta)
    
    # Part (d): Train on y-labels and test on true labels
    # Make sure to save outputs to pred_path_d
    model2 = LogisticRegression()
    model2.fit(x_train, y_train)
    print("Part (d):")
    print("Theta is: ", model2.theta)
    print("The accuracy on test set is: ", np.mean(t_test == model2.predict(x_test)))
    util.plot(x_test, t_test, model2.theta)
    
    # Part (e): Apply correction factor using validation set and test on true labels
    # Plot and use np.savetxt to save outputs to pred_path_e
    
    
    
    # *** END CODER HERE
