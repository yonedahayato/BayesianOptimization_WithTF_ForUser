# -*- coding: utf-8 -*-
from BO.bayesian_optimization import BayesianOptimization

import sys
from collections import OrderedDict
import numpy as np
import tensorflow as tf

class BayesianOpt():
    def __init__(self, function, hyper_parameter_range, hyper_parameter_explore
                 X_train, X_test, y_train, y_test):
        self.function = function
        self.hyper_parameter_range = hyper_parameter_range
        self.hyper_parameter_explore = hyper_parameter_explore

        self.X_train, self.X_test, self.y_train, self.y_test = X_train, X_test, y_train, y_test

    def session_manager(self):
        def session_manager_function(**hyper_parameter_dict):
            X, label, train_step, accuracy = self.function(**hyper_parameter_dict)
            init = tf.initialize_all_variables()
            with tf.Session() as sess:
                sess.run(init)
                print("Training...")
                for i in range(20000):
                    batch_index = np.random.choice(self.X_train.shape[0], 50, replace=False)
                    batch_x = self.X_train[batch_index, :]
                    batch_y = self.y_train[batch_index, 0]

                    train_step.run({X: batch_x, label: batch_y})
                    if i % 2000==0:
                        train_accuracy = accuracy.eval({X: batch_x, label: batch_y})
                accuracy = accuracy.eval({X: self.X_test, label: self.y_test[:,0]})
                print("accuracy %6.3f" % accuracy)
                return accuracy

        return session_manager_function

    def BO_function(self, k_num, acq, verbose=True):
        gp_params = {"alpha": 1e-5}

        MPL_Session = self.session_manager()
        BO = BayesianOptimization(MPL_Session,
                                  self.hyper_parameter_range,
                                  verbose=verbose, kernel_num = k_num)

        BO.explore(self.hyper_parameter_explore)

        BO.maximize(n_iter=200, acq=acq, **gp_params)

        print("-"*53)
        print("Final Results")
        print("kernel: {}".format(str(BO.kernel)))
        print("acquisition function: {}".format(BO.acquisition))

        print("score: {}".format(BO.res["max"]["max_val"]))
        print("best_parameter: ")
        print(BO.res["max"]["max_params"])
        print("-"*53)

    def main(self):
        self.BO_function(0, "ucb")
