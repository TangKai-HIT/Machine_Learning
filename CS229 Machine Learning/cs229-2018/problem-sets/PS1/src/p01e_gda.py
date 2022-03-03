import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(e): Gaussian discriminant analysis (GDA)

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    # Load dataset
    x_train, y_train = util.load_dataset(train_path, add_intercept=False)
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=False)
    # *** START CODE HERE ***
    classifier = GDA()
    classifier.fit(x_train, y_train)
    
    output = classifier.predict(x_valid)
    np.savetxt(pred_path, output, fmt="%d")
    
    # output accuracy
    print("1.For training set:")
    print("Theta is: ", classifier.theta)
    print("The accuracy on training set is: ", np.mean(classifier.predict(x_train) == y_train))
    
    print("2.For test set:")
    print("The accuracy on validation set is: ", np.mean(output == y_valid))
    # *** END CODE HERE ***


class GDA(LinearModel):
    """Gaussian Discriminant Analysis.

    Example usage:
        > clf = GDA()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Fit a GDA model to training set given by x and y.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).

        Returns:
            theta: GDA model parameters.
        """
        # *** START CODE HERE ***
        m, n = x.shape
        phi = 1/m * y.sum()
        mu_1 = (x.T @ y)/(y.sum())
        mu_0 = (x.T @ (y!=1))/(m-y.sum())
        
        sigma = np.zeros((n,n))
        for i in range(m):
            bias = (x[i,:] - mu_1) if y[i]==1 else (x[i,:] - mu_0)
            bias = bias.reshape((n,1))
            sigma += bias @ bias.T
        sigma/=m
        sigma_inv = np.linalg.inv(sigma)
        
        theta = sigma_inv @ (mu_1-mu_0)
        theta0 = 0.5*(mu_0 @ sigma_inv @ mu_0 - mu_1 @ sigma_inv @ mu_1 - 2*np.log((1-phi)/phi))
        
        self.theta = np.insert(theta, 0, theta0)
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        m = x.shape[0]
        x_res = x.reshape((m,-1))
        hypo = 1/(1 + np.exp(-(np.insert(x_res,0,0,axis=1) @ self.theta)))
        return hypo>0.5
        # *** END CODE HERE
