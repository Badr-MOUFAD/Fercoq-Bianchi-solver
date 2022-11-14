import pytest
import numpy as np

from sklearn.datasets import make_regression
from sklearn.linear_model import Lasso as sk_Lasso

from cd_solver.sklearn_api import Lasso


@pytest.mark.parametrize('smooth_formulation', [True, False])
def test_lasso_estimator(smooth_formulation):
    rho = 1e-1
    n_samples, n_features = 100, 100
    X, y = make_regression(n_samples, n_features, noise=1, random_state=123)

    alpha_max = np.linalg.norm(X.T @ y, ord=np.inf)
    alpha = rho * alpha_max

    # Fercoq & Bianchi Lasso
    fq_estimator = Lasso(alpha, smooth_formulation=smooth_formulation,
                         max_iter=10_000, verbose=0)
    fq_estimator.fit(X, y)

    # scikit learn Lasso
    sk_estimator = sk_Lasso(alpha / n_samples,
                            fit_intercept=False, tol=1e-9)
    sk_estimator.fit(X, y)

    np.testing.assert_allclose(
        sk_estimator.coef_.flatten(), fq_estimator.coef_.flatten())


if __name__ == '__main__':
    pass
