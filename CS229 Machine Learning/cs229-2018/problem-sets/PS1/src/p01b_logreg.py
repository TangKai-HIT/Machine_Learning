import numpy as np
import util

from linear_model import LinearModel


def main(train_path, eval_path, pred_path):
    """Problem 1(b): Logistic regression with Newton's Method.

    Args:
        train_path: Path to CSV file containing dataset for training.
        eval_path: Path to CSV file containing dataset for evaluation.
        pred_path: Path to save predictions.
    """
    x_train, y_train = util.load_dataset(train_path, add_intercept=True) # add intercept term
    x_valid, y_valid = util.load_dataset(eval_path, add_intercept=True)
    # *** START CODE HERE ***
    LRM = LogisticRegression()
    LRM.fit(x_train, y_train)
    
    output = LRM.predict(x_valid)
    np.savetxt(pred_path, output, fmt="%d")
    
    # plot results
    print("1.For training set:")
    util.plot(x_train, y_train, LRM.theta, save_path="./output/Problem_1(b)_train1.jpg")
    print("Theta is: ", LRM.theta)
    print("The accuracy on training set is: ", np.mean(LRM.predict(x_train) == y_train))
    
    print("2.For test set:")
    util.plot(x_valid, y_valid, LRM.theta, save_path="./output/Problem_1(b)_valid1.jpg")
    print("The accuracy on validation set is: ", np.mean(output == y_valid))
    # *** END CODE HERE ***


class LogisticRegression(LinearModel):
    """Logistic regression with Newton's Method as the solver.

    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_eval)
    """

    def fit(self, x, y):
        """Run Newton's Method to minimize J(theta) for logistic regression.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        # *** START CODE HERE ***
        # hypothesis
        def hypo(x):
            z = np.matmul(x, self.theta)
            h = 1/(1 + np.exp(-z))
            return h
            
        # gradient
        def grad(x, y):
            m = x.shape[0]
            g = -1/m * np.matmul(x.T, (y - hypo(x)))
            return g
        
        # Hessian
        def hessian(x):
            m, n = x.shape
            H = np.zeros((n,n))
            for k in range(n):
                for l in range(n):
                    x_k = x[:,k]; x_l = x[:,l]
                    H[k,l] = 1/m * np.dot(x_k*x_l, hypo(x)*(1-hypo(x)))
            return H
        
        # Initialize theta
        if self.theta is None:
            self.theta = np.zeros(x.shape[1])
        # Newton Method
        count = 0
        while True:
            H = hessian(x)
            g = grad(x, y)
            theta_new = self.theta - np.linalg.inv(H) @ g
            
            count+=1
            if np.linalg.norm(theta_new - self.theta)<self.eps or count>self.max_iter:
                self.theta = theta_new
                break
            else:
                self.theta = theta_new
            
        # *** END CODE HERE ***

    def predict(self, x):
        """Make a prediction given new inputs x.

        Args:
            x: Inputs of shape (m, n).

        Returns:
            Outputs of shape (m,).
        """
        # *** START CODE HERE ***
        z = np.matmul(x, self.theta)
        h = 1/(1 + np.exp(-z))
        return h>0.5
        # *** END CODE HERE ***
