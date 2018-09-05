# Author: Olivier Fercoq <olivier.fercoq@telecom-paristech.fr>

import numpy as np

import scipy.sparse as sp
import cd_solver

# imports for loading datasets
from scipy import io
from sklearn.datasets.mldata import fetch_mldata
#from sklearn.externals.joblib import Memory
#import sys
#sys.path.append("../tv_l1_solver")
#from load_poldrack import load_gain_poldrack


probs = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10]  # range(11)

for prob in probs:
    if prob == 0:
        f = ["square", "square"]
        cf = [0.5]*2
        bf = [1,-8]
        A = np.array([[1,4],[4, 3]])

        g = ["abs", "abs"]


        pb_toy = cd_solver.Problem(N=2, f=f, Af=A, bf=bf, cf=cf, g=g)

        cd_solver.coordinate_descent(pb_toy, max_iter=100, verbose=0.5, print_style='smoothed_gap', min_change_in_x=0.)

    if (prob >= 1 and prob <= 4) or prob == 6:
        dataset = 'leukemia'
        data = fetch_mldata(dataset)
        X = data.data
        X = sp.csc_matrix(X)
        y = data.target

    if prob == 1:
        # Lasso
        print("Lasso on Leukemia")
        pb_leukemia_lasso = cd_solver.Problem(N=X.shape[1],
                                              f=["square"] * X.shape[0],
                                              Af=X,
                                              bf=y,
                                              cf=[0.5] * X.shape[0],
                                              g=["abs"] * X.shape[1],
                                              cg=[0.1*np.linalg.norm(X.T.dot(y), np.inf)] * X.shape[1])

        cd_solver.coordinate_descent(pb_leukemia_lasso, max_iter=100, verbose=5, print_style='smoothed_gap')

    if prob == 2:
        # Logistic regression
        print("logistic regression on Leukemia")
        pb_leukemia_logreg = cd_solver.Problem(N=X.shape[1],
                                               f=["log1pexp"] * X.shape[0],
                                               Af=(X.T.multiply(y)).T,
                                               bf=y,
                                               cf=[1] * X.shape[0],
                                               g=["square"] * X.shape[1],
                                               cg=[0.5*0.01*np.linalg.norm(X.T.dot(y), np.inf)] * X.shape[1])

        cd_solver.coordinate_descent(pb_leukemia_logreg, max_iter=150, verbose=2., print_style='smoothed_gap')

    if prob == 3:
        # SVM
        print("dual SVM on Leukemia")
        alpha = 1000
        pb_leukemia_svm = cd_solver.Problem(N=X.shape[0],
                                            f=["square"] * X.shape[1] + ["linear"] * X.shape[0],
                                            Af=sp.vstack([X.T.multiply(y), -sp.eye(X.shape[0])], format="csc"),
                                            bf=np.zeros(X.shape[1] + X.shape[0]),
                                            cf=[0.5/alpha] * X.shape[1] + [1] * X.shape[0],
                                            g=["box_zero_one"] * X.shape[0])

        cd_solver.coordinate_descent(pb_leukemia_svm, max_iter=100, verbose=0.5, print_style='smoothed_gap')

    if prob == 4:
        # Lasso by ISTA
        print("Lasso on Leukemia by ISTA")
        pb_leukemia_lasso = cd_solver.Problem(N=X.shape[1],
                                              f=["square"] * X.shape[0],
                                              Af=X,
                                              bf=y,
                                              cf=[0.5] * X.shape[0],
                                              g=["abs"],
                                              blocks=[0, X.shape[1]],
                                              cg=[0.1*np.linalg.norm(X.T.dot(y), np.inf)])

        cd_solver.coordinate_descent(pb_leukemia_lasso, max_iter=100, verbose=0.5, print_style='smoothed_gap')

    if prob == 5:
        # basic problem with constraints
        print('basic problem with constraints')
        f = ["square", "square"]
        cf = [0.5, 0.5]
        bf = [1, -0.5]
        Af = np.eye(2)

        h = ["eq_const", "eq_const"]
        Ah = np.array([[1, 1], [1,0]])

        pb_toy_const = cd_solver.Problem(N=2, f=f, Af=Af, bf=bf, cf=cf, h=h, Ah=Ah)

        cd_solver.coordinate_descent(pb_toy_const, max_iter=100, verbose=0.001, print_style='smoothed_gap')

    if prob == 6:
        # SVM with intercept
        
        print("dual SVM with intercept on Leukemia")
        alpha = 1000
        Xred = np.linalg.cholesky(np.array((X.dot(X.T)).todense()))
        pb_leukemia_svm_intercept = cd_solver.Problem(N=X.shape[0],
                                            f=["square"] * Xred.shape[1] + ["linear"],
                                            Af=sp.vstack([Xred.T * y, -np.ones((1,X.shape[0]))], format="csc"),
                                            bf=np.zeros(Xred.shape[1]+1),
                                            cf=[0.5/alpha] * Xred.shape[1] + [1],
                                            # g=["box_zero_one"] * X.shape[0],
                                            h=["eq_const"],
                                            Ah=sp.csc_matrix(y)
                                                          )

        cd_solver.coordinate_descent(pb_leukemia_svm_intercept, max_iter=10000, verbose=0.5, print_style='smoothed_gap')

    if prob == 7:
        print("dual SVM with intercept on RCV1")

        data = io.loadmat('/data/ofercoq/datasets/Classification/rcv1_train.binary.mat')

        X = data['X'].astype(np.float)
        y = data['y'].astype(np.float).ravel()
        
        C = 1. / X.shape[0]
        alpha = 0.25 / X.shape[0]

        pb_rcv1_svm_intercept = cd_solver.Problem(N=X.shape[0],
                                            f=["square"] * X.shape[1] + ["linear"],
                                            Af=sp.vstack([X.T.multiply(y), -np.ones((1,X.shape[0]))], format="csc"),
                                            bf=np.zeros(X.shape[1] + 1),
                                            cf=[C] * X.shape[1] + [1],
                                            g=["box_zero_one"] * X.shape[0],
                                            Dg=alpha*sp.eye(X.shape[0]),
                                            h=["eq_const"],
                                            Ah=sp.csc_matrix(y)
                                                          )
        
        cd_solver.coordinate_descent(pb_rcv1_svm_intercept, max_iter=100, verbose=2., print_style='smoothed_gap', step_size_factor=alpha/1000)

    if prob == 8:
        print("TV regularized least squares on toy dataset")

        X = np.array([[1,2,3,4,5,6,7], [-7, -6, -5, -4, -3, -2, -1]])
        y = [0, 2]
        
        alpha = 1*1e-2

        mask = np.array([[[True,True], [True,True]], [[False,True], [True,True]]])
        integer_mask = np.cumsum(mask).reshape(mask.shape) * mask
        ravelling_array = np.cumsum(mask==mask).reshape(mask.shape) - 1
        correspondance = ravelling_array[mask]

        N = np.prod(mask.shape)

        X = sp.csr_matrix(X)
        Af = sp.csr_matrix((X.data, correspondance[X.indices], X.indptr), (X.shape[0], N))

        Dx = sp.diags([-np.ones(mask.shape[0]), np.ones(mask.shape[0])], offsets=[0, 1])
        Dy = sp.diags([-np.ones(mask.shape[1]), np.ones(mask.shape[1])], offsets=[0, 1])
        Dz = sp.diags([-np.ones(mask.shape[2]), np.ones(mask.shape[2])], offsets=[0, 1])

        Dx = sp.kron(Dx, sp.eye(mask.shape[1]*mask.shape[2]))
        Dy = sp.kron(sp.eye(mask.shape[0]), sp.kron(Dy, sp.eye(mask.shape[2])))
        Dz = sp.kron(sp.eye(mask.shape[0]*mask.shape[1]), Dz)
        
        threeDgradient = sp.vstack([Dx, Dy, Dz], format='csc')
        threeDgradient.eliminate_zeros()
        # reorder the matrix
        threeDgradient = sp.csc_matrix((threeDgradient.data, 3 * (threeDgradient.indices % N) + threeDgradient.indices // N, threeDgradient.indptr), (3*N, N))

        pb_toy_tvl1 = cd_solver.Problem(N=N,
                                        f=["square"] * X.shape[0],
                                        Af=Af,
                                        bf=y,
                                        cf=[0.5] * X.shape[0],
                                        h=["norm2"] * N,
                                        ch=[1.] * N,
                                        blocks_h=np.arange(0, 3*N + 1, 3),
                                        Ah=alpha*threeDgradient
                                        )

        cd_solver.coordinate_descent(pb_toy_tvl1, max_iter=300000, verbose=0.1, max_time=100., print_style='smoothed_gap')



    if prob == 9:
        print("l1+TV regularized least squares on fmri dataset")

        mem = Memory(cachedir='cache', verbose=3)
        X, y, subjects, mask, affine = mem.cache(load_gain_poldrack)(smooth=0, folder='../tv_l1_solver')

        l1_ratio = 0.5
        alpha = 1e-2

        integer_mask = np.cumsum(mask).reshape(mask.shape) * mask
        ravelling_array = np.cumsum(mask==mask).reshape(mask.shape) - 1
        correspondance = ravelling_array[mask]

        N = np.prod(mask.shape)

        X = sp.csr_matrix(X)
        Af = sp.csr_matrix((X.data, correspondance[X.indices], X.indptr), (X.shape[0], N))

        Dx = sp.diags([-np.ones(mask.shape[0]), np.ones(mask.shape[0])], offsets=[0, 1])
        Dy = sp.diags([-np.ones(mask.shape[1]), np.ones(mask.shape[1])], offsets=[0, 1])
        Dz = sp.diags([-np.ones(mask.shape[2]), np.ones(mask.shape[2])], offsets=[0, 1])

        Dx = sp.kron(Dx, sp.eye(mask.shape[1]*mask.shape[2]))
        Dy = sp.kron(sp.eye(mask.shape[0]), sp.kron(Dy, sp.eye(mask.shape[2])))
        Dz = sp.kron(sp.eye(mask.shape[0]*mask.shape[1]), Dz)
        
        threeDgradient = sp.vstack([Dx, Dy, Dz], format='csc')
        threeDgradient.eliminate_zeros()
        # reorder the matrix
        threeDgradient = sp.csc_matrix((threeDgradient.data, 3 * (threeDgradient.indices % N) + threeDgradient.indices // N, threeDgradient.indptr), (3*N, N))
        
        pb_fmri_tvl1 = cd_solver.Problem(N=N,
                                        f=["square"] * X.shape[0],
                                        Af=Af,
                                        bf=y,
                                        cf=[0.5] * X.shape[0],
                                        g=["abs"] * N,
                                        cg=[alpha*l1_ratio] * N,
                                        h=["norm2"] * N,
                                        ch=[(1-l1_ratio)] * N,
                                        blocks_h=np.arange(0, 3*N + 1, 3),
                                        Ah=alpha*threeDgradient
                                        )

        cd_solver.coordinate_descent(pb_fmri_tvl1, max_iter=100, verbose=20., max_time=100., step_size_factor=10., print_style='smoothed_gap')

    if prob == 10:
        # LP  --  min c.dot(x) : Mx <= b
        print('basic LP')
        d = 3
        n = 4
        M = np.array([[2,4,5,7], [1,1,2,2], [1,2,3,3]])
        c = -np.array([7,9,18,17])
        b = np.array([41,17,24])

        pb_basic_lp = cd_solver.Problem(N=n,
                                f=["linear"],
                                Af=c,
                                g=["ineq_const"]*n,
                                h=["ineq_const"]*d,
                                Ah=-M,
                                bh=-b
                                )
        
        cd_solver.coordinate_descent(pb_basic_lp, max_iter=1000000, verbose=1., max_time=10., print_style='smoothed_gap')
        