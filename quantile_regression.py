import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.formula.api as smf
from sklearn.ensemble import GradientBoostingRegressor


BETA = np.array([2.0])
BUFFER_X = 2.0
QUANTILE_LOWER = 0.1
QUANTILE_UPPER = 0.9


def get_epsilon_sd(x):

    return np.exp(0.2 * x) + 0.5


def get_X_y(n_obs=1000):

    X = np.random.normal(size=(n_obs, 1), scale=2)

    epsilon_sd = get_epsilon_sd(X[:, 0])
    epsilon = np.random.normal(size=n_obs, scale=epsilon_sd)

    y = np.dot(X, BETA) + epsilon

    return X, y


def get_y_expected(min_x, max_x):

    x_linspace = np.linspace(min_x, max_x)

    y_expected = np.dot(x_linspace.reshape((x_linspace.size, 1)), BETA)

    return x_linspace, y_expected


def get_y_quantile(x_linspace, y_expected, quantile):

    epsilon_sd = get_epsilon_sd(x_linspace)
    epsilon_quantile = norm.ppf(quantile, scale=epsilon_sd)
    y_quantile = y_expected + epsilon_quantile

    return y_quantile


def get_gbm(X, y, quantile):

    gbm = GradientBoostingRegressor(loss="quantile", alpha=quantile, subsample=0.5)

    # TODO Early stopping with quantile loss on validation set
    gbm.fit(X=X, y=y)

    return gbm


def get_quantile_regression(X, y, quantile):

    df = pd.DataFrame({"y": y, "x": X[:, 0]})
    quantile_regression = smf.quantreg("y ~ x", data=df)

    return quantile_regression.fit(q=quantile)


def get_y_quantiles_predicted(X, y, x_linspace):

    gbm_upper = get_gbm(X, y, QUANTILE_UPPER)
    gbm_lower = get_gbm(X, y, QUANTILE_LOWER)

    quantile_regression_upper = get_quantile_regression(X, y, QUANTILE_UPPER)
    quantile_regression_lower = get_quantile_regression(X, y, QUANTILE_LOWER)

    df_prediction = pd.DataFrame({"x": x_linspace.flatten()})

    return {
        "upper_reg": quantile_regression_upper.predict(exog=df_prediction),
        "lower_reg": quantile_regression_lower.predict(exog=df_prediction),
        "upper_gbm": gbm_upper.predict(x_linspace.reshape(-1, 1)),
        "lower_gbm": gbm_lower.predict(x_linspace.reshape(-1, 1)),
    }


def get_y_quantiles_true(x_linspace, y_expected):

    return {
        "lower": get_y_quantile(x_linspace, y_expected, QUANTILE_LOWER),
        "upper": get_y_quantile(x_linspace, y_expected, QUANTILE_UPPER),
    }


def save_plot(
    plot_filename,
    x_linspace,
    X,
    y,
    y_expected,
    y_quantiles,
    include_observations=True,
    include_gbm=True,
    include_reg=True,
):

    f, ax = plt.subplots(figsize=(14, 8))

    if include_observations:

        plt.plot(X[:, 0], y, "kx", alpha=0.2, label="observations")

    plt.plot(
        x_linspace,
        y_quantiles["true"]["upper"],
        color="darkcyan",
        linestyle="dashed",
        label=f"{QUANTILE_UPPER} quantile",
    )

    plt.plot(
        x_linspace,
        y_quantiles["true"]["lower"],
        color="darkcyan",
        linestyle="dashed",
        label=f"{QUANTILE_LOWER} quantile",
    )

    if include_gbm:

        plt.plot(
            x_linspace,
            y_quantiles["predicted"]["upper_gbm"],
            color="darkorange",
            linestyle="dashed",
            label=f"{QUANTILE_UPPER} quantile (GBM prediction)",
        )
        plt.plot(
            x_linspace,
            y_quantiles["predicted"]["lower_gbm"],
            color="darkorange",
            linestyle="dashed",
            label=f"{QUANTILE_LOWER} quantile (GBM prediction)",
        )

    if include_reg:

        plt.plot(
            x_linspace,
            y_quantiles["predicted"]["upper_reg"],
            color="darkorange",
            linestyle="dotted",
            label=f"{QUANTILE_UPPER} quantile (linear prediction)",
        )

        plt.plot(
            x_linspace,
            y_quantiles["predicted"]["lower_reg"],
            color="darkorange",
            linestyle="dotted",
            label=f"{QUANTILE_LOWER} quantile (linear prediction)",
        )

    plt.plot(
        x_linspace,
        y_expected,
        linestyle="dotted",
        color="midnightblue",
        label="E[ Y | X ]",
    )

    if not include_observations:

        # Add a "rug plot" instead of points for each observation
        ymin, ymax = ax.get_ylim()
        rug_plot_y = ymin + 0.02 * (ymax - ymin)
        plt.plot(X[:, 0], [rug_plot_y] * X.shape[0], "|", color="k", alpha=0.1)

        plt.ylim((ymin, ymax))

    ax.legend()

    plt.xlabel("x")
    plt.ylabel("y")
    plt.savefig(plot_filename, bbox_inches="tight")


def main():

    X, y = get_X_y()
    x_linspace, y_expected = get_y_expected(X.min() - BUFFER_X, X.max() + BUFFER_X)

    y_quantiles = {
        "true": get_y_quantiles_true(x_linspace, y_expected),
        "predicted": get_y_quantiles_predicted(X, y, x_linspace),
    }

    save_plot(
        "quantile_regression_rug_plot.png",
        x_linspace,
        X,
        y,
        y_expected,
        y_quantiles,
        include_observations=False,
    )

    save_plot(
        "quantile_regression_rug_plot_without_gbm.png",
        x_linspace,
        X,
        y,
        y_expected,
        y_quantiles,
        include_observations=False,
        include_gbm=False,
    )

    save_plot(
        "quantile_regression_scatter_plot.png",
        x_linspace,
        X,
        y,
        y_expected,
        y_quantiles,
        include_observations=True,
    )


if __name__ == "__main__":
    main()
