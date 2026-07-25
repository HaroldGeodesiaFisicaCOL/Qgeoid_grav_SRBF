import numpy as np
import pandas as pd
from numpy.linalg import solve
from tqdm import tqdm
from datetime import datetime

np.random.seed(42)


def random_sign_matrix(n, s):
    return np.random.choice([-1, 1], size=(n, s))


def build_N(X_list, sigma2_list, sigma_mu2):
    """
    Construye N de forma incremental.
    No guarda XtX_list en memoria.
    """
    n_params = X_list[0].shape[1]
    N = np.zeros((n_params, n_params))

    for X, s2 in zip(X_list, sigma2_list):
        N += (1 / s2) * (X.T @ X)

    N += (1 / sigma_mu2) * np.eye(n_params)
    return N


def precompute_static_terms(X_list, y_list, U_obs_list):
    """
    Precalcula únicamente términos pequeños o reutilizables:
      - Xty_list: X.T @ y
      - XtUobs_list: X.T @ U_obs

    No se precomputan XtX_list ni matrices grandes.
    """
    Xty_list = [X.T @ y for X, y in zip(X_list, y_list)]
    XtUobs_list = [X.T @ U_obs for X, U_obs in zip(X_list, U_obs_list)]
    return Xty_list, XtUobs_list


def estimate_beta(X_list, Xty_list, sigma2_list, mu, sigma_mu2):
    """
    Calcula beta usando la construcción incremental de N
    y el lado derecho precomputado.
    """
    N = build_N(X_list, sigma2_list, sigma_mu2)

    rhs = np.zeros((mu.shape[0], 1))
    for Xty, s2 in zip(Xty_list, sigma2_list):
        rhs += (1 / s2) * Xty

    rhs += (1 / sigma_mu2) * mu
    beta = solve(N, rhs)

    return beta, N


def estimate_sigma_i2(X, y, beta, r_i):
    e = X @ beta - y
    return ((e.T @ e).item()) / r_i, e


def estimate_sigma_mu2(beta, mu, r_mu):
    e_mu = beta - mu
    return ((e_mu.T @ e_mu).item()) / r_mu, e_mu


def compute_lambda(sigma1_2, sigma_mu_2):
    return sigma1_2 / sigma_mu_2


def save_weights(filepath, hist_sigma):
    pd.DataFrame(hist_sigma.T).to_csv(filepath, index=False, header=False)


def iterative_estimation(
    X_list,
    y_list,
    mu,
    sigma2_list,
    sigma_mu2_init,
    max_iter=20,
    tol=1e-6,
    num_trace_samples=1,
):
    print(f"[{datetime.now()}] Iniciando VCE", flush=True)

    n_obs_list = [X.shape[0] for X in X_list]
    p = mu.shape[0]
    sigma_mu2 = sigma_mu2_init

    hist_sigma = np.zeros((len(X_list) + 1, max_iter))

    print(f"[{datetime.now()}] Generando matrices Hutchinson", flush=True)
    U_obs_list = [random_sign_matrix(X.shape[0], num_trace_samples) for X in X_list]
    U_par = random_sign_matrix(p, num_trace_samples)

    # Precomputación de términos que sí conviene guardar
    Xty_list, XtUobs_list = precompute_static_terms(X_list, y_list, U_obs_list)

    normas_iter = []

    for it in range(max_iter):
        t0 = datetime.now()

        beta, N = estimate_beta(X_list, Xty_list, sigma2_list, mu, sigma_mu2)

        r_i_list = []
        for X, XtUobs, n_obs in zip(X_list, XtUobs_list, n_obs_list):
            Alpha = solve(N, XtUobs)
            XA = X @ Alpha
            tr = (XtUobs * Alpha).sum() / num_trace_samples
            r_i_list.append(n_obs - tr)

        Alpha_mu = solve(N, U_par)
        tr_mu = (U_par * Alpha_mu).sum() / num_trace_samples
        r_mu = p - tr_mu

        new_sigma2 = []
        e_all = []

        for X, y, ri in zip(X_list, y_list, r_i_list):
            sigma2_i, e_i = estimate_sigma_i2(X, y, beta, ri)
            new_sigma2.append(sigma2_i)
            e_all.append(e_i)

        e_vec = np.concatenate(e_all, axis=0)
        new_sigma_mu2, e_mu = estimate_sigma_mu2(beta, mu, r_mu)

        norma_e = np.linalg.norm(e_vec)
        norma_e_mu = np.linalg.norm(e_mu)
        normas_iter.append([norma_e, norma_e_mu])

        hist_sigma[:-1, it] = new_sigma2
        hist_sigma[-1, it] = new_sigma_mu2

        err = max(
            abs(new_sigma2[0] - sigma2_list[0]),
            abs(new_sigma_mu2 - sigma_mu2),
        )

        elapsed = (datetime.now() - t0).total_seconds()

        print(
            f"[{datetime.now()}] Iter {it+1:02d}: "
            f"sigma_i={new_sigma2[0]:.6e}  "
            f"sigma_mu={new_sigma_mu2:.6e}  "
            f"delta={err:.2e}  tiempo={elapsed:.2f}s",
            flush=True,
        )

        sigma2_list, sigma_mu2 = new_sigma2, new_sigma_mu2

        if err < tol:
            print(f"[{datetime.now()}] Convergencia alcanzada", flush=True)
            break

    lam = compute_lambda(sigma2_list[0], sigma_mu2)

    return {
        "beta": beta,
        "N": N,
        "sigma2_list": sigma2_list,
        "sigma_mu2": sigma_mu2,
        "lambda": lam,
        "iterations": it + 1,
        "histo_sigmas": hist_sigma[:, : it + 1],
        "normas": normas_iter,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Variance Component Estimation")

    parser.add_argument("--rutas_archivo", required=True)
    parser.add_argument("--sigma2_init_list", required=True)
    parser.add_argument("--sigma_mu2_init", type=float, required=True)
    parser.add_argument("--max_iter", type=int, default=20)
    parser.add_argument("--tol", type=float, default=1e-6)
    parser.add_argument("--num_trace_samples", type=int, default=20)
    parser.add_argument("--output_beta", required=True)
    parser.add_argument("--output_weights", required=True)
    parser.add_argument("--output_residuales", required=True)
    parser.add_argument("--output_normas", required=True)
    parser.add_argument(
        "--output_covariance",
        default=None,
        help="Ruta opcional para guardar matriz de covarianza",
    )

    args = parser.parse_args()

    rutas_df = pd.read_csv(args.rutas_archivo, sep="\t", header=None)

    X_list = []
    y_list = []
    proyectos = []

    for _, row in rutas_df.iterrows():
        x_path, y_path, flag = row[0], row[1], row[2].strip().lower()

        X = np.load(x_path)
        df_obs = pd.read_csv(y_path, sep="\t")
        y = df_obs["Perturbaciones_residuales"].values.reshape(-1, 1) / 100000

        if flag == "si":
            for proj in df_obs["Proyecto"].unique():
                mask = df_obs["Proyecto"] == proj
                X_list.append(X[mask.values, :])
                y_list.append(y[mask.values, :])
                proyectos.append(proj)
        else:
            X_list.append(X)
            y_list.append(y)
            proyectos.append("Terrestres")

    sigma2_list = [float(v) for v in args.sigma2_init_list.split(",")]
    if len(sigma2_list) != len(proyectos):
        raise ValueError(f"Hay {len(proyectos)} proyectos pero {len(sigma2_list)} sigmas")

    print("\n[INFO] Asociación de proyectos con σ asignados:")
    print(f"{'Índice':<10}{'Proyecto':<30}{'σ':>15}")
    print("-" * 60)

    for i, (nombre, sigma) in enumerate(zip(proyectos, sigma2_list)):
        print(f"{i:<10}{nombre:<30}{sigma:>15.6e}")

    mu = np.zeros((X_list[0].shape[1], 1))

    resultados = iterative_estimation(
        X_list,
        y_list,
        mu,
        sigma2_list,
        args.sigma_mu2_init,
        args.max_iter,
        args.tol,
        args.num_trace_samples,
    )

    beta = resultados["beta"]
    N = resultados["N"]

    np.savetxt(args.output_beta, beta)
    save_weights(args.output_weights, resultados["histo_sigmas"])

    np.savetxt(
        args.output_residuales,
        np.vstack([(X @ beta - y) for X, y in zip(X_list, y_list)]),
    )

    np.savetxt(args.output_normas, resultados["normas"])

    if args.output_covariance:
        Cov = np.linalg.inv(N)
        np.save(args.output_covariance, Cov)
        print("Matriz de covarianza guardada")

    print("Iteraciones:", resultados["iterations"])
    print("Lambda:", resultados["lambda"])