# Copyright 2022-present, Lorenzo Bonicelli, Pietro Buzzega, Matteo Boschini, Angelo Porrello, Simone Calderara.
# All rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

best_args = {
    'perm-mnist': {
    'sgd': {-1: {'lr': 0.2, 'batch_size': 128, 'n_epochs': 1}},
    'ewc_on': {-1: {'lr': 0.1,
                    'e_lambda': 0.7,
                    'gamma': 1.0,
                    'batch_size': 128,
                    'n_epochs': 1}},
    'si': {-1: {'lr': 0.1,
                'c': 0.5,
                'xi': 1.0,
                'batch_size': 128,
                'n_epochs': 1}},
    'er': {200: {'lr': 0.2,
                 'minibatch_size': 128,
                 'batch_size': 128,
                 'n_epochs': 1},
           500: {'lr': 0.2,
                 'minibatch_size': 128,
                 'batch_size': 128,
                 'n_epochs': 1},
           5120: {'lr': 0.2,
                  'minibatch_size': 128,
                  'batch_size': 128,
                  'n_epochs': 1}},
    'gem': {200: {'lr': 0.1,
                  'gamma': 0.5,
                  'batch_size': 128,
                  'n_epochs': 1},
            500: {'lr': 0.1, 'gamma': 0.5, 'batch_size': 128,
                  'n_epochs': 1},
            5120: {'lr': 0.1, 'gamma': 0.5, 'batch_size': 128,
                   'n_epochs': 1}},
    'agem': {200: {'lr': 0.1,
                   'minibatch_size': 128,
                   'batch_size': 128,
                   'n_epochs': 1},
             500: {'lr': 0.1,
                   'minibatch_size': 128,
                   'batch_size': 128,
                   'n_epochs': 1},
             5120: {'lr': 0.1,
                    'minibatch_size': 128,
                    'batch_size': 128,
                    'n_epochs': 1}},
    'hal': {200: {'lr': 0.1,
                  'minibatch_size': 128,
                  'batch_size': 128,
                  'hal_lambda': 0.1,
                  'beta': 0.5,
                  'gamma': 0.1,
                  'n_epochs': 1},
            500: {'lr': 0.1,
                  'minibatch_size': 128,
                  'batch_size': 128,
                  'hal_lambda': 0.1,
                  'beta': 0.3,
                  'gamma': 0.1,
                  'n_epochs': 1},
            5120: {'lr': 0.1,
                   'minibatch_size': 128,
                   'batch_size': 128,
                   'hal_lambda': 0.1,
                   'beta': 0.5,
                   'gamma': 0.1,
                   'n_epochs': 1}},
    'gss': {200: {'lr': 0.2,
                  'minibatch_size': 10,
                  'gss_minibatch_size': 128,
                  'batch_size': 128,
                  'batch_num': 1,
                  'n_epochs': 1},
            500: {'lr': 0.1,
                  'minibatch_size': 128,
                  'gss_minibatch_size': 10,
                  'batch_size': 128,
                  'batch_num': 1,
                  'n_epochs': 1},
            5120: {'lr': 0.03,
                   'minibatch_size': 128,
                   'gss_minibatch_size': 10,
                   'batch_size': 128,
                   'batch_num': 1,
                   'n_epochs': 1}},
    'agem_r': {200: {'lr': 0.1,
                     'minibatch_size': 128,
                     'batch_size': 128,
                     'n_epochs': 1},
               500: {'lr': 0.1,
                     'minibatch_size': 128,
                     'batch_size': 128,
                     'n_epochs': 1},
               5120: {'lr': 0.1,
                      'minibatch_size': 128,
                      'batch_size': 128,
                      'n_epochs': 1}},
    'fdr': {200: {'lr': 0.1,
                  'minibatch_size': 128,
                  'alpha': 1.0,
                  'batch_size': 128,
                  'n_epochs': 1},
            500: {'lr': 0.1,
                  'minibatch_size': 128,
                  'alpha': 0.3,
                  'batch_size': 128,
                  'n_epochs': 1},
            5120: {'lr': 0.1,
                   'minibatch_size': 128,
                   'alpha': 1,
                   'batch_size': 128,
                   'n_epochs': 1}},
    'der': {200: {'lr': 0.2,
                  'minibatch_size': 128,
                  'alpha': 1.0,
                  'batch_size': 128,
                  'n_epochs': 1},
            500: {'lr': 0.2,
                  'minibatch_size': 128,
                  'alpha': 1.0,
                  'batch_size': 128,
                  'n_epochs': 1},
            5120: {'lr': 0.2,
                   'minibatch_size': 128,
                   'alpha': 0.5,
                   'batch_size': 128,
                   'n_epochs': 1}},
    'derpp': {200: {'lr': 0.1,
                    'minibatch_size': 128,
                    'alpha': 1.0,
                    'beta': 1.0,
                    'batch_size': 128,
                    'n_epochs': 1},
              500: {'lr': 0.2,
                    'minibatch_size': 128,
                    'alpha': 1.0,
                    'beta': 0.5,
                    'batch_size': 128,
                    'n_epochs': 1},
              5120: {'lr': 0.2,
                     'minibatch_size': 128,
                     'alpha': 0.5,
                     'beta': 1.0,
                     'batch_size': 128,
                     'n_epochs': 1}}},
    'rot-mnist': {
        'sgd': {-1: {'lr': 0.2, 'batch_size': 128, 'n_epochs': 1}},
        'ewc_on': {-1: {'lr': 0.1,
                        'e_lambda': 0.7,
                        'gamma': 1.0,
                        'batch_size': 128,
                        'n_epochs': 1}},
        'si': {-1: {'lr': 0.1,
                    'c': 1.0,
                    'xi': 1.0,
                    'batch_size': 128,
                    'n_epochs': 1}},
        'er': {200: {'lr': 0.2,
                     'minibatch_size': 128,
                     'batch_size': 128,
                     'n_epochs': 1},
               500: {'lr': 0.2,
                     'minibatch_size': 128,
                     'batch_size': 128,
                     'n_epochs': 1},
               5120: {'lr': 0.2,
                      'minibatch_size': 128,
                      'batch_size': 128,
                      'n_epochs': 1}},
        'gem': {200: {'lr': 0.01,
                      'gamma': 0.5,
                      'batch_size': 128,
                      'n_epochs': 1},
                500: {'lr': 0.01, 'gamma': 0.5, 'batch_size': 128,
                      'n_epochs': 1},
                5120: {'lr': 0.01, 'gamma': 0.5, 'batch_size': 128,
                       'n_epochs': 1}},
        'agem': {200: {'lr': 0.1,
                       'minibatch_size': 128,
                       'batch_size': 128,
                       'n_epochs': 1},
                 500: {'lr': 0.3,
                       'minibatch_size': 128,
                       'batch_size': 128,
                       'n_epochs': 1},
                 5120: {'lr': 0.3,
                        'minibatch_size': 128,
                        'batch_size': 128,
                        'n_epochs': 1}},
        'hal': {200: {'lr': 0.1,
                      'minibatch_size': 128,
                      'batch_size': 128,
                      'hal_lambda': 0.2,
                      'beta': 0.5,
                      'gamma': 0.1,
                      'n_epochs': 1},
                500: {'lr': 0.1,
                      'minibatch_size': 128,
                      'batch_size': 128,
                      'hal_lambda': 0.1,
                      'beta': 0.5,
                      'gamma': 0.1,
                      'n_epochs': 1},
                5120: {'lr': 0.1,
                       'minibatch_size': 128,
                       'batch_size': 128,
                       'hal_lambda': 0.1,
                       'beta': 0.3,
                       'gamma': 0.1,
                       'n_epochs': 1}},
        'gss': {200: {'lr': 0.2,
                      'minibatch_size': 10,
                      'gss_minibatch_size': 128,
                      'batch_size': 128,
                      'batch_num': 1,
                      'n_epochs': 1},
                500: {'lr': 0.2,
                      'minibatch_size': 128,
                      'gss_minibatch_size': 128,
                      'batch_size': 128,
                      'batch_num': 1,
                      'n_epochs': 1},
                5120: {'lr': 0.2,
                       'minibatch_size': 128,
                       'gss_minibatch_size': 128,
                       'batch_size': 128,
                       'batch_num': 1,
                       'n_epochs': 1}},
        'agem_r': {200: {'lr': 0.1,
                         'minibatch_size': 128,
                         'batch_size': 128,
                         'n_epochs': 1},
                   500: {'lr': 0.3,
                         'minibatch_size': 128,
                         'batch_size': 128,
                         'n_epochs': 1},
                   5120: {'lr': 0.3,
                          'minibatch_size': 128,
                          'batch_size': 128,
                          'n_epochs': 1}},
        'fdr': {200: {'lr': 0.2,
                      'minibatch_size': 128,
                      'alpha': 1.0,
                      'batch_size': 128,
                      'n_epochs': 1},
                500: {'lr': 0.2,
                      'minibatch_size': 128,
                      'alpha': 0.3,
                      'batch_size': 128,
                      'n_epochs': 1},
                5120: {'lr': 0.2,
                       'minibatch_size': 128,
                       'alpha': 1,
                       'batch_size': 128,
                       'n_epochs': 1}},
        'der': {200: {'lr': 0.2,
                      'minibatch_size': 128,
                      'alpha': 1.0,
                      'batch_size': 128,
                      'n_epochs': 1},
                500: {'lr': 0.2,
                      'minibatch_size': 128,
                      'alpha': 0.5,
                      'batch_size': 128,
                      'n_epochs': 1},
                5120: {'lr': 0.2,
                       'minibatch_size': 128,
                       'alpha': 0.5,
                       'batch_size': 128,
                       'n_epochs': 1}},
        'derpp': {200: {'lr': 0.1,
                        'minibatch_size': 128,
                        'alpha': 1.0,
                        'beta': 0.5,
                        'batch_size': 128,
                        'n_epochs': 1},
                  500: {'lr': 0.2,
                        'minibatch_size': 128,
                        'alpha': 0.5,
                        'beta': 1.0,
                        'batch_size': 128,
                        'n_epochs': 1},
                  5120: {'lr': 0.2,
                         'minibatch_size': 128,
                         'alpha': 0.5,
                         'beta': 0.5,
                         'batch_size': 128,
                         'n_epochs': 1}}},
    'seq-mnist': {
        'sgd': {-1: {'lr': 0.03, 'batch_size': 10, 'n_epochs': 1}},
        'ewc_on': {-1: {'lr': 0.03,
                        'e_lambda': 90,
                        'gamma': 1.0,
                        'batch_size': 10,
                        'n_epochs': 1}},
        'si': {-1: {'lr': 0.1,
                    'c': 1.0,
                    'xi': 0.9,
                    'batch_size': 10,
                    'n_epochs': 1}},
        'lwf': {-1: {'lr': 0.03,
                     'alpha': 1,
                     'softmax_temp': 2.0,
                     'batch_size': 10,
                     'n_epochs': 1,
                     'optim_wd': 0.0005}},
        'pnn': {-1: {'lr': 0.1, 'batch_size': 10, 'n_epochs': 1}},
        'er': {200: {'lr': 0.01,
                     'minibatch_size': 10,
                     'batch_size': 10,
                     'n_epochs': 1},
               500: {'lr': 0.1,
                     'minibatch_size': 10,
                     'batch_size': 10,
                     'n_epochs': 1},
               5120: {'lr': 0.1,
                      'minibatch_size': 10,
                      'batch_size': 10,
                      'n_epochs': 1}},
        'mer': {200: {'lr': 0.1,
                      'minibatch_size': 128,
                      'beta': 1,
                      'gamma': 1,
                      'batch_num': 1,
                      'batch_size': 1,
                      'n_epochs': 1},
                500: {'lr': 0.1,
                      'minibatch_size': 128,
                      'beta': 1,
                      'gamma': 1,
                      'batch_num': 1,
                      'batch_size': 1,
                      'n_epochs': 1},
                5120: {'lr': 0.03,
                       'minibatch_size': 128,
                       'beta': 1,
                       'gamma': 1,
                       'batch_num': 1,
                       'batch_size': 1,
                       'n_epochs': 1}},
        'gem': {200: {'lr': 0.01,
                      'gamma': 1.0,
                      'batch_size': 10,
                      'n_epochs': 1},
                500: {'lr': 0.03, 'gamma': 0.5, 'batch_size': 10,
                      'n_epochs': 1},
                5120: {'lr': 0.1, 'gamma': 1.0, 'batch_size': 10,
                       'n_epochs': 1}},
        'agem': {200: {'lr': 0.1,
                       'minibatch_size': 128,
                       'batch_size': 10,
                       'n_epochs': 1},
                 500: {'lr': 0.1,
                       'minibatch_size': 128,
                       'batch_size': 10,
                       'n_epochs': 1},
                 5120: {'lr': 0.1,
                        'minibatch_size': 128,
                        'batch_size': 10,
                        'n_epochs': 1}},
        'hal': {200: {'lr': 0.1,
                      'minibatch_size': 128,
                      'batch_size': 128,
                      'hal_lambda': 0.1,
                      'beta': 0.7,
                      'gamma': 0.5,
                      'n_epochs': 1},
                500: {'lr': 0.1,
                      'minibatch_size': 128,
                      'batch_size': 128,
                      'hal_lambda': 0.1,
                      'beta': 0.2,
                      'gamma': 0.5,
                      'n_epochs': 1},
                5120: {'lr': 0.1,
                       'minibatch_size': 128,
                       'batch_size': 128,
                       'hal_lambda': 0.1,
                       'beta': 0.7,
                       'gamma': 0.5,
                       'n_epochs': 1}},
        'gss': {200: {'lr': 0.1,
                      'minibatch_size': 10,
                      'gss_minibatch_size': 10,
                      'batch_size': 128,
                      'batch_num': 1,
                      'n_epochs': 1},
                500: {'lr': 0.1,
                      'minibatch_size': 10,
                      'gss_minibatch_size': 10,
                      'batch_size': 128,
                      'batch_num': 1,
                      'n_epochs': 1},
                5120: {'lr': 0.1,
                       'minibatch_size': 128,
                       'gss_minibatch_size': 10,
                       'batch_size': 128,
                       'batch_num': 1,
                       'n_epochs': 1}},
        'agem_r': {200: {'lr': 0.1,
                         'minibatch_size': 128,
                         'batch_size': 10,
                         'n_epochs': 1},
                   500: {'lr': 0.1,
                         'minibatch_size': 128,
                         'batch_size': 10,
                         'n_epochs': 1},
                   5120: {'lr': 0.1,
                          'minibatch_size': 128,
                          'batch_size': 10,
                          'n_epochs': 1}},
        'icarl': {200: {'lr': 0.1,
                        'minibatch_size': 10,
                        'optim_wd': 0,
                        'batch_size': 10,
                        'n_epochs': 1},
                  500: {'lr': 0.1,
                        'minibatch_size': 10,
                        'optim_wd': 0,
                        'batch_size': 10,
                        'n_epochs': 1},
                  5120: {'lr': 0.1,
                         'minibatch_size': 10,
                         'optim_wd': 0,
                         'batch_size': 10,
                         'n_epochs': 1}},
        'fdr': {200: {'lr': 0.03,
                      'minibatch_size': 128,
                      'alpha': 0.5,
                      'batch_size': 128,
                      'n_epochs': 1},
                500: {'lr': 0.1,
                      'minibatch_size': 128,
                      'alpha': 0.2,
                      'batch_size': 128,
                      'n_epochs': 1},
                5120: {'lr': 0.1,
                       'minibatch_size': 128,
                       'alpha': 0.2,
                       'batch_size': 128,
                       'n_epochs': 1}},
        'der': {200: {'lr': 0.03,
                      'minibatch_size': 10,
                      'alpha': 0.2,
                      'batch_size': 10,
                      'n_epochs': 1},
                500: {'lr': 0.03,
                      'minibatch_size': 128,
                      'alpha': 1.0,
                      'batch_size': 10,
                      'n_epochs': 1},
                5120: {'lr': 0.1,
                       'minibatch_size': 128,
                       'alpha': 0.5,
                       'batch_size': 10,
                       'n_epochs': 1}},
        'derpp': {200: {'lr': 0.03,
                        'minibatch_size': 128,
                        'alpha': 0.2,
                        'beta': 1.0,
                        'batch_size': 10,
                        'n_epochs': 1},
                  500: {'lr': 0.03,
                        'minibatch_size': 10,
                        'alpha': 1.0,
                        'beta': 0.5,
                        'batch_size': 10,
                        'n_epochs': 1},
                  5120: {'lr': 0.1,
                         'minibatch_size': 64,
                         'alpha': 0.2,
                         'beta': 0.5,
                         'batch_size': 10,
                         'n_epochs': 1}}},
    'seq-cifar10': {'sgd': {-1: {'lr': 0.1,
                                 'batch_size': 32,
                                 'n_epochs': 50}},
                    'ewc_on': {-1: {'lr': 0.03,
                                    'e_lambda': 10,
                                    'gamma': 1.0,
                                    'batch_size': 32,
                                    'n_epochs': 50}},
                    'si': {-1: {'lr': 0.03,
                                'c': 0.5,
                                'xi': 1.0,
                                'batch_size': 32,
                                'n_epochs': 50}},
                    'lwf': {-1: {'lr': 0.01,
                                 'alpha': 3.0,
                                 'softmax_temp': 2.0,
                                 'batch_size': 32,
                                 'n_epochs': 50,
                                 'optim_wd': 0.0005}},
                    'pnn': {-1: {'lr': 0.03, 'batch_size': 32,
                                 'n_epochs': 50}},
                    'er': {200: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 50},
                           500: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 50},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50}},
                    'gem': {200: {'lr': 0.03,
                                  'gamma': 0.5,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            500: {'lr': 0.03, 'gamma': 0.5,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            5120: {'lr': 0.03, 'gamma': 0.5,
                                   'batch_size': 32,
                                   'n_epochs': 50}},
                    'agem': {200: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'batch_size': 32,
                                   'n_epochs': 50},
                             500: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'batch_size': 32,
                                   'n_epochs': 50},
                             5120: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'batch_size': 32,
                                    'n_epochs': 50}},
                    'hal': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50,
                                  'hal_lambda': 0.2,
                                  'beta': 0.5,
                                  'gamma': 0.1,
                                  'steps_on_anchors': 100,
                                  'finetuning_epochs': 1},
                            500: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50,
                                  'hal_lambda': 0.1,
                                  'beta': 0.3,
                                  'gamma': 0.1,
                                  'steps_on_anchors': 100,
                                  'finetuning_epochs': 1},
                            5120: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'batch_size': 32,
                                   'n_epochs': 50,
                                   'hal_lambda': 0.1,
                                   'beta': 0.3,
                                   'gamma': 0.1,
                                   'steps_on_anchors': 100,
                                   'finetuning_epochs': 1}},
                    'gss': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'gss_minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50,
                                  'batch_num': 1},
                            500: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'gss_minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 50,
                                  'batch_num': 1},
                            5120: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'gss_minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 1,
                                  'batch_num': 1}},
                    'agem_r': {200: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50},
                               500: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 50},
                               5120: {'lr': 0.03,
                                      'minibatch_size': 32,
                                      'batch_size': 32,
                                      'n_epochs': 50}},
                    'icarl': {200: {'lr': 0.1,
                                    'minibatch_size': 0,
                                    'softmax_temp': 2.0,
                                    'optim_wd': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              500: {'lr': 0.1,
                                    'minibatch_size': 0,
                                    'softmax_temp': 2.0,
                                    'optim_wd': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 0,
                                     'softmax_temp': 2.0,
                                     'optim_wd': 0.00001,
                                     'batch_size': 32,
                                     'n_epochs': 50}},
                    'fdr': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.3,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            500: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 1,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            5120: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'alpha': 0.3,
                                   'batch_size': 32,
                                   'n_epochs': 50}},
                    'der': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.3,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            500: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.3,
                                  'batch_size': 32,
                                  'n_epochs': 50},
                            5120: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'alpha': 0.3,
                                   'batch_size': 32,
                                   'n_epochs': 50}},
                    'derpp': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.1,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'n_epochs': 50},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.1,
                                     'beta': 1.0,
                                     'batch_size': 32,
                                     'n_epochs': 50}}},
    'seq-tinyimg': {'sgd': {-1: {'lr': 0.03,
                                 'batch_size': 32,
                                 'n_epochs': 100}},
                    'ewc_on': {-1: {'lr': 0.03,
                                    'e_lambda': 25,
                                    'gamma': 1.0,
                                    'batch_size': 32,
                                    'n_epochs': 100}},
                    'si': {-1: {'lr': 0.03,
                                'c': 0.5,
                                'xi': 1.0,
                                'batch_size': 32,
                                'n_epochs': 100}},
                    'lwf': {-1: {'lr': 0.01,
                                 'alpha': 1.0,
                                 'softmax_temp': 2.0,
                                 'batch_size': 32,
                                 'n_epochs': 100,
                                 'optim_wd': 0.0005}},
                    'pnn': {-1: {'lr': 0.03, 'batch_size': 32,
                                 'n_epochs': 100}},
                    'er': {200: {'lr': 0.1,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 100},
                           500: {'lr': 0.03,
                                 'minibatch_size': 32,
                                 'batch_size': 32,
                                 'n_epochs': 100},
                           5120: {'lr': 0.1,
                                  'minibatch_size': 32,
                                  'batch_size': 32,
                                  'n_epochs': 100}},
                    'agem': {200: {'lr': 0.01,
                                   'minibatch_size': 32,
                                   'batch_size': 32,
                                   'n_epochs': 100},
                             500: {'lr': 0.01,
                                   'minibatch_size': 32,
                                   'batch_size': 32,
                                   'n_epochs': 100},
                             5120: {'lr': 0.01,
                                    'minibatch_size': 32,
                                    'batch_size': 32,
                                    'n_epochs': 100}},
                    'agem_r': {200: {'lr': 0.01,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 100},
                               500: {'lr': 0.01,
                                     'minibatch_size': 32,
                                     'batch_size': 32,
                                     'n_epochs': 100},
                               5120: {'lr': 0.01,
                                      'minibatch_size': 32,
                                      'batch_size': 32,
                                      'n_epochs': 100}},
                    'icarl': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'softmax_temp': 2.0,
                                    'optim_wd': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'softmax_temp': 2.0,
                                    'optim_wd': 0.00001,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'softmax_temp': 2.0,
                                     'optim_wd': 0.00001,
                                     'batch_size': 32,
                                     'n_epochs': 100}},
                    'fdr': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.3,
                                  'batch_size': 32,
                                  'n_epochs': 100},
                            500: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 1,
                                  'batch_size': 32,
                                  'n_epochs': 100},
                            5120: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'alpha': 0.3,
                                   'batch_size': 32,
                                   'n_epochs': 100}},
                    'der': {200: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'softmax_temp': 2.0,
                                  'alpha': 0.1,
                                  'batch_size': 32,
                                  'n_epochs': 100},
                            500: {'lr': 0.03,
                                  'minibatch_size': 32,
                                  'alpha': 0.1,
                                  'batch_size': 32,
                                  'n_epochs': 100},
                            5120: {'lr': 0.03,
                                   'minibatch_size': 32,
                                   'alpha': 0.1,
                                   'batch_size': 32,
                                   'n_epochs': 100}},
                    'derpp': {200: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.1,
                                    'beta': 1.0,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              500: {'lr': 0.03,
                                    'minibatch_size': 32,
                                    'alpha': 0.2,
                                    'beta': 0.5,
                                    'batch_size': 32,
                                    'n_epochs': 100},
                              5120: {'lr': 0.03,
                                     'minibatch_size': 32,
                                     'alpha': 0.1,
                                     'beta': 0.5,
                                     'batch_size': 32,
                                     'n_epochs': 100}}},
    'mnist-360': {
        'sgd': {-1: {'lr': 0.1, 'batch_size': 4}},
        'er': {200: {'lr': 0.2,
                     'batch_size': 1,
                     'minibatch_size': 16},
               500: {'lr': 0.2, 'batch_size': 1,
                     'minibatch_size': 16},
               1000: {'lr': 0.2,
                      'batch_size': 4,
                      'minibatch_size': 16}},
        'mer': {200: {'lr': 0.2,
                      'minibatch_size': 128,
                      'beta': 1,
                      'gamma': 1,
                      'batch_num': 3},
                500: {'lr': 0.1,
                      'minibatch_size': 128,
                      'beta': 1,
                      'gamma': 1,
                      'batch_num': 3},
                1000: {'lr': 0.2,
                       'minibatch_size': 128,
                       'beta': 1,
                       'gamma': 1,
                       'batch_num': 3}},
        'agem_r': {200: {'lr': 0.1,
                         'batch_size': 16,
                         'minibatch_size': 128},
                   500: {'lr': 0.1,
                         'batch_size': 16,
                         'minibatch_size': 128},
                   1000: {'lr': 0.1,
                          'batch_size': 4,
                          'minibatch_size': 128}},
        'der': {200: {'lr': 0.1,
                      'batch_size': 16,
                      'minibatch_size': 64,
                      'alpha': 0.5},
                500: {'lr': 0.2,
                      'batch_size': 16,
                      'minibatch_size': 16,
                      'alpha': 0.5},
                1000: {'lr': 0.1,
                       'batch_size': 8,
                       'minibatch_size': 16,
                       'alpha': 0.5}},
        'derpp': {200: {'lr': 0.2,
                        'batch_size': 16,
                        'minibatch_size': 16,
                        'alpha': 0.5,
                        'beta': 1.0},
                  500: {'lr': 0.2,
                        'batch_size': 16,
                        'minibatch_size': 16,
                        'alpha': 0.5,
                        'beta': 1.0},
                  1000: {'lr': 0.2,
                         'batch_size': 16,
                         'minibatch_size': 128,
                         'alpha': 0.2,
                         'beta': 1.0}}},
    'seq-cifar100': {
        'sgd': {-1: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0}},
        'gdumb': {
            500:  {'lr': 0.1, 'maxlr': 0.05, 'minlr': 5e-4, 'cutmix_alpha': 1, 'fitting_epochs': 250, 'optim_mom': 0.9, 'optim_wd': 1e-6},
            2000: {'lr': 0.1, 'maxlr': 0.05, 'minlr': 5e-4, 'cutmix_alpha': 1, 'fitting_epochs': 250, 'optim_mom': 0, 'optim_wd': 1e-6},
        },
        'lucir': {
            500: {'lr': 0.03, 'lr_finetune':0.01,  'optim_mom': 0.9, 'optim_wd': 0, 'lamda_base': 5, 'k_mr':  2, 'fitting_epochs': 20, 'mr_margin': 0.5, 'lamda_mr': 1.},
            2000: {'lr': 0.03, 'lr_finetune':0.01,  'optim_mom': 0.9, 'optim_wd': 0, 'lamda_base': 5, 'k_mr':  2, 'fitting_epochs': 20, 'mr_margin': 0.5, 'lamda_mr': 1.},
        },
        
        
        'icarl': {
            500: {'lr': 0.3, 'optim_mom': 0, 'optim_wd': 1e-05},
            2000: {'lr': 0.3, 'optim_mom': 0, 'optim_wd': 1e-05}
        },
        'bic': {
            500: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0},
            2000: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0},
        },
        'lwf': {
            -1: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 5e-4},
        },


        'er_ace': {
            500: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0},
            2000: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0}
        },
        'rpc': {
            500: {'lr': 0.1, 'optim_mom': 0, 'optim_wd': 0},
            2000: {'lr': 0.1, 'optim_mom': 0, 'optim_wd': 0}
        },
        'der': {
            500: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0, 'alpha': 0.3},
            2000: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0, 'alpha': 0.3}
        },
        'derpp': {
            500: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0, 'alpha': 0.1, 'beta': 0.5, 'svrg': 1},
            2000: {'lr': 0.03, 'optim_mom': 0, 'optim_wd': 0, 'alpha': 0.1, 'beta': 0.5, 'svrg': 1}
        },
        'xder': {
            500: {'m': 0.7, 'alpha':0.3, 'beta': 0.8, 'gamma': 0.85, 'optim_wd': 0, 'lambd': 0.05, 'eta': 0.001, 'lr': 0.03, 'simclr_temp': 5, 'optim_mom': 0, 'simclr_batch_size':64, 'simclr_num_aug': 2, 'svrg': 1},
            2000: {'m': 0.2, 'alpha':0.6, 'beta': 0.9, 'gamma': 0.85, 'optim_wd': 0, 'lambd': 0.05, 'eta': 0.01, 'lr': 0.03, 'simclr_temp': 5, 'optim_mom': 0, 'simclr_batch_size':64, 'simclr_num_aug': 2, 'svrg': 1}
        },
        'er': {
            500: {'lr': 0.1, 'optim_mom': 0, 'optim_wd': 0, 'svrg': 1},
            2000: {'lr': 0.1, 'optim_mom': 0, 'optim_wd': 0, 'svrg': 1}
        },
        'hal': {
            500: {'lr': 0.03, 'hal_lambda': 0.1, 'beta': 0.3, 'gamma': 0.1, 'svrg': 1},
            2000: {'lr': 0.03, 'hal_lambda': 0.1, 'beta': 0.3, 'gamma': 0.1, 'svrg': 1}
        }
    }

}
