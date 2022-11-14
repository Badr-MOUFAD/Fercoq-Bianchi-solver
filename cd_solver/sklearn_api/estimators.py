from cd_solver import Problem, coordinate_descent


class Lasso:
    """Scikit-learn like Lasso estimator.

    Solve un normalized Lasso problem using Fercoq & Bianchi algorithm.

    Objective::

        min_w (1/2) * ||y - Xw||^2 + alpha * ||w||_1

    Parameters
    ----------
        smooth_formulation : bool, default=True
            If True, considers the datafit term a smooth
            and uses its gradient to minimize objective.
            If False, uses a saddle point formulation of objective
            and uses prox of the conjugate of datafit to minimize objective.

    Attributes
    ----------
        coef_ : array, shape (n_features,)
            regression coefficients.

        p_objs_ : array, shape (max_iter,)
            Values of the primal objective in each iteration.
    """

    def __init__(self, alpha, smooth_formulation=True,
                 max_iter=1000, verbose=0):
        self.alpha = alpha
        self.max_iter = max_iter
        self.verbose = verbose
        self.smooth_formulation = smooth_formulation

    def fit(self, X, y):
        alpha = self.alpha
        n_samples, n_features = X.shape

        # cf. cd_solver.pyx
        # Problem docstring for the choice of params
        if self.smooth_formulation:
            datafit_params = {
                'f': ["square"], 'Af': X, 'bf': y,
                'blocks_f': [0, n_samples],
                'cf': [0.5],
            }
        else:
            datafit_params = {
                'h': ["square"], 'Ah': X, 'bh': y,
                'blocks_h': [0, n_samples],
                'ch': [0.5],
            }

        # cf. file comparison.py
        # examples leukemia on Lasso
        pb = Problem(
            N=n_features,
            # datafit
            ** datafit_params,
            # penalty
            g=["abs"] * n_features, cg=[alpha] * n_features
        )
        coordinate_descent(pb, max_iter=self.max_iter,
                           verbose=self.verbose, per_pass=1)

        self.coef_ = pb.sol
        self.p_objs_ = pb.p_objs
        return self


if __name__ == '__main__':
    pass
