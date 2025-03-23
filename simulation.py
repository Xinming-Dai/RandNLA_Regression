import numpy as np
import time
from sketch_n_solve.solve.least_squares import LeastSquares
from utils import generate_cotaminated_error, coefficients, data_generation, parse_args
from tqdm import tqdm

# Recommended to use either "clarkson_woodruff" or "uniform_sparse"
# for best results out of the box

if __name__ == "__main__":
    params = parse_args()
    n = params.n
    p = params.p
    delta = params.delta
    dist = params.dist
    param = params.param
    seed = params.seed
    num_trials = params.n_trials
    sketch_fn = params.sketch_fn
    tau = params.tau
    k = params.k

    res = []
    times = []
    for m in tqdm(range(num_trials)):
        beta = coefficients(p, seed+m)
        X, Y = data_generation(beta, n, p, delta, dist, param, seed+m)
        if sketch_fn in ['proposal1', 'proposal2']:
            start_time = time.time()
            # split the data into two parts
            X1, X2 = X[:n//2], X[n//2:]
            Y1, Y2 = Y[:n//2], Y[n//2:]
            # train on the first part
            kw = {'k': k}
            lsq = LeastSquares("clarkson_woodruff", seed+m)
            beta_hat, _, _, time_elapsed = lsq(X1, Y1, **kw)
            # refit on the second part
            resid = Y2 - X2 @ beta_hat
            # tau = 1.35 is optimal for normal inliers
            if tau == 0:
                # use median absolute deviation normalized to estimate tau
                tau = 1.4826 * np.median(np.abs(resid))
            kw = {'k': k, 'resid': resid, 'tau': tau}
            lsq = LeastSquares(sketch_fn, seed+m)
            beta_hat, _, _, time_elapsed = lsq(X2, Y2, **kw)
            end_time = time.time()
            time_elapsed = end_time - start_time
            tau = np.round(tau, 2)
        else:
            kw = {'k': k}
            lsq = LeastSquares(sketch_fn, seed+m)
            beta_hat, _, _, time_elapsed = lsq(X, Y, **kw)
        # We use MSE a the metric
        mse = np.mean((beta - beta_hat)**2)
        res.append(mse)
        times.append(time_elapsed)
    
    # save to pickle
    res = np.array(res)
    times = np.array(times)
    np.save(f"results/{sketch_fn}_{n}_{k}_{delta}_{dist}_{param}_{tau}_results.npy", res)
    np.save(f"results/{sketch_fn}_{n}_{k}_{delta}_{dist}_{param}_{tau}_times.npy", times)

