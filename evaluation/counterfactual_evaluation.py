import re
import os
import pickle as pkl
import numpy as np
import pandas as pd
from collections import Counter
import random

from counterfactuals.find_many_counterfactuals import make_individual_plot, make_raw_plot
from evaluation.metrics.fid import compute_validity, compute_fid, compute_MAE, compute_MSE, compute_class_fid, get_importance_region_information, mean_lsp_calculation
from evaluation.metrics.summarizing_fkts import mean_cf, max_cf, min_loss, weighted

def compute_rel_L2(x_org, x_cf, x_rec, eps: float = 1e-8) -> float:
    """
    Compute a relative L2 distance between original images and counterfactuals,
    normalized by the reconstruction distance.

    rel_L2 = mean( ||x_cf - x_org||_2 / (||x_rec - x_org||_2 + eps) )

    Args:
        x_org: np.ndarray, shape (N, H, W, C) or similar
        x_cf:  np.ndarray, same shape as x_org (counterfactuals)
        x_rec: np.ndarray, same shape as x_org (reconstructions)
        eps:   small constant to avoid divide-by-zero

    Returns:
        float: scalar relative L2 distance.
    """
    # Squeeze away extra singleton dims
    x_org = np.squeeze(x_org)
    x_cf  = np.squeeze(x_cf)
    x_rec = np.squeeze(x_rec)

    # Ensure first dimension is batch
    N = x_org.shape[0]
    x_org_flat = x_org.reshape(N, -1)
    x_cf_flat  = x_cf.reshape(N, -1)
    x_rec_flat = x_rec.reshape(N, -1)

    num = np.linalg.norm(x_cf_flat - x_org_flat, axis=1)
    den = np.linalg.norm(x_rec_flat - x_org_flat, axis=1) + eps

    rel_L2 = num / den
    return float(np.mean(rel_L2))

def find_and_extract_files(folder_path):
    pattern = re.compile(r"datab([\d-]+)g([\d-]+)(.+)_1000(?:_new)?\.pkl")

    extracted_data = []

    for filename in os.listdir(folder_path):
        match = pattern.match(filename)
        if match:
            beta_str = match.group(1).replace('-', '.')
            gamma_str = match.group(2).replace('-', '.')
            model_name = match.group(3)

            beta = float(beta_str)
            gamma = float(gamma_str)

            extracted_data.append((filename, model_name, beta, gamma))

    return extracted_data


def _normalize_for_hash(x):
    """
    Convert param values (including lists, tuples, dicts, numpy arrays)
    into something hashable so we can safely put them in a set.
    """
    if isinstance(x, (list, tuple)):
        return tuple(_normalize_for_hash(v) for v in x)
    if isinstance(x, dict):
        return tuple(sorted((k, _normalize_for_hash(v)) for k, v in x.items()))
    if isinstance(x, np.ndarray):
        return tuple(x.flatten().tolist())
    return x


def performance_to_latex_summary(all_results_clf, all_results_rec, all_params):
    def sci_format(x):
        return np.round(x, decimals=3)

    # --------- figure out which params differ across models ----------
    all_param_keys = set()
    for params in all_params.values():
        all_param_keys.update(params.keys())

    model_names = list(all_params.keys())
    differing_params = set()
    for key in all_param_keys:
        values = set()
        for name in model_names:
            if key in all_params[name]:
                values.add(_normalize_for_hash(all_params[name][key]))
        if len(values) > 1:
            differing_params.add(key)

    df_clf = pd.DataFrame(all_results_clf)
    df_rec = pd.DataFrame(all_results_rec)

    datasets = df_clf['dataset'].unique()
    splits = df_clf['split'].unique()

    for dataset in datasets:
        for split in splits:
            print(dataset, split)
            print()
            df_clf_filter = df_clf[(df_clf['dataset'] == dataset) & (df_clf['split'] == split)]
            df_rec_filter = df_rec[(df_rec['dataset'] == dataset) & (df_rec['split'] == split)]

            # ---- aggregate metrics ----
            summary_df_clf = df_clf_filter.groupby(["model_name", "clf"]).agg(
                Accuracy=("Accuracy", "mean"),
                Entropy=("Entropy", "mean"),
                AUC=("AUC", "mean"),
                Precision=("Precision", 'mean'),
                Recall=("Recall", 'mean')
            ).reset_index()

            summary_df_rec = df_rec_filter.groupby(["model_name"]).agg(
                MAE=("MAE", "mean"),
                MSE=("MSE", "mean"),
                KLD=("KLD", "mean")
            ).reset_index()

            summary_df_clf = summary_df_clf.sort_values(by=['model_name'])

            fixed_columns_clf = ["Accuracy", "Entropy", "AUC", "Precision", "Recall"]
            fixed_columns_rec = ['MAE', 'MSE', 'KLD']

            # We print two tables: one for clf metrics, one for rec metrics
            for df, fixed_columns in zip(
                [summary_df_clf, summary_df_rec],
                [fixed_columns_clf, fixed_columns_rec]
            ):
                if 'model_name' not in df.columns:
                    continue

                # ---- attach differing hyperparameters as columns ----
                for param in differing_params:
                    df[param] = df['model_name'].map(
                        lambda model: all_params.get(model, {}).get(param, 'N/A')
                    )

                # drop identifiers & non-useful param columns
                df.drop(columns=['model_name'], inplace=True)
                df.drop(columns=['save_path'], inplace=True, errors='ignore')

                # ---- special handling for loss_weights -> Rec w, KLD w, Clf w ----
                if 'loss_weights' in df.columns:
                    # loss_weights often looks like [[10.0, 0.001, 5.0]]; flatten that
                    def _flatten_loss_w(v):
                        if isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (list, tuple)):
                            return list(v[0])
                        return v

                    df['loss_weights'] = df['loss_weights'].apply(_flatten_loss_w)

                    df[['Rec w', 'KLD w', 'Clf w']] = pd.DataFrame(
                        df['loss_weights'].tolist(),
                        index=df.index
                    )

                    varying_columns = [
                        col for col in ['Rec w', 'KLD w', 'Clf w']
                        if df[col].nunique() > 1
                    ]
                    print('col_names,varying_columns', list(df.columns), varying_columns)

                    df = df.drop(
                        columns=['loss_weights'] +
                        [col for col in ['Rec w', 'KLD w', 'Clf w'] if col not in varying_columns]
                    )

                # ---- reorder columns: hyperparams (and others) first, then fixed metric columns ----
                other_columns = [col for col in df.columns if col not in fixed_columns]
                df = df[other_columns + fixed_columns]

                # ---- format and print LaTeX ----
                col_format = "l" + "c" * (len(df.columns) - 1)
                df = df.applymap(
                    lambda x: sci_format(x) if isinstance(x, (float, np.floating)) else x
                )
                latex_table = df.to_latex(index=False, column_format=col_format, escape=False)
                print(latex_table)


def cf_performance_to_latex_summary(all_results_clf, all_params):
    def sci_format(x):
        return np.round(x, decimals=2)

    # Determine parameter differences across models
    all_param_keys = set()
    for params in all_params.values():
        all_param_keys.update(params.keys())

    model_names = list(all_params.keys())
    differing_params = set()
    for key in all_param_keys:
        values = {
            _normalize_for_hash(all_params[model].get(key, None))
            for model in model_names
        }
        if len(values) > 1:
            differing_params.add(key)

    df_clf = pd.DataFrame(all_results_clf)
    datasets = df_clf['dataset'].unique()

    for dataset in datasets:
        print(dataset)
        df_clf_filter = df_clf[df_clf['dataset'] == dataset]

        # Add parameter differences to summary dataframe
        summary_df_clf = df_clf_filter.groupby(["vae_name", "beta", "gamma", "model_name"]).agg(
            L2=("L2", "mean"),
            validity=("validity", "mean"),
            FID=("FID", "mean"),
            Switch_epoch=("Switch epoch", 'mean')
        ).reset_index()

        summary_df_clf = summary_df_clf.sort_values(by=['vae_name', 'model_name'])

        for df in [summary_df_clf]:
            if 'vae_name' not in df.columns:
                continue

            # attach differing params
            for param in differing_params:
                df[param] = df['vae_name'].map(
                    lambda model: all_params.get(model, {}).get(param, 'N/A')
                )

            df.drop(columns=['vae_name'], inplace=True)
            df.drop(columns=['save_path'], inplace=True, errors='ignore')

            # Handle loss_weights if present
            if 'loss_weights' in df.columns:
                def _flatten_loss_w(v):
                    if isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (list, tuple)):
                        return list(v[0])
                    return v

                df['loss_weights'] = df['loss_weights'].apply(_flatten_loss_w)
                df[['Rec w', 'KLD w', 'Clf w']] = pd.DataFrame(
                    df['loss_weights'].tolist(), index=df.index
                )

                varying_columns = [col for col in ['Rec w', 'KLD w', 'Clf w'] if df[col].nunique() > 1]
                print('col_names,varying_columns', list(df.columns), varying_columns)

                df = df.drop(
                    columns=['loss_weights'] +
                    [col for col in ['Rec w', 'KLD w', 'Clf w'] if col not in varying_columns]
                )

            fixed_columns = ['model_name', 'beta', 'gamma', 'validity', 'L2', 'FID', 'Switch_epoch']
            other_columns = [col for col in df.columns if col not in fixed_columns]
            df = df[other_columns + fixed_columns]

            col_format = "l" + "c" * (len(df.columns) - 1)
            summ_df = df.applymap(lambda x: sci_format(x) if isinstance(x, (float, np.floating)) else x)
            latex_table = summ_df.to_latex(index=False, column_format=col_format, escape=False)
            print(latex_table)


def z_eval_cf_performance_to_latex_summary(all_results_clf, all_params):
    def sci_format(x):
        return np.round(x, decimals=2)

    # Determine parameter differences across models
    all_param_keys = set()
    for params in all_params.values():
        all_param_keys.update(params.keys())

    model_names = list(all_params.keys())
    differing_params = set()
    for key in all_param_keys:
        values = {
            _normalize_for_hash(all_params[model].get(key, None))
            for model in model_names
        }
        if len(values) > 1:
            differing_params.add(key)

    df_clf = pd.DataFrame(all_results_clf)
    datasets = df_clf['dataset'].unique()

    for dataset in datasets:
        print(dataset)
        df_clf_filter = df_clf[df_clf['dataset'] == dataset]

        summary_df_clf = df_clf_filter.groupby(["vae_name", "beta", "gamma", "model_name"]).agg(
            MaxIdx=("Max idx", "mean"),
            MaxFreq=('Max freq', 'mean'),
            MedianIdx=("Median idx", "mean"),
            MedianFreq=('Median freq', 'mean'),
            MinIdx=("Min idx", "mean"),
            MinFreq=('Min freq', 'mean'),
            DivFreq=("Diversity freq.", "mean"),
        ).reset_index()

        summary_df_clf = summary_df_clf.sort_values(by=['vae_name', 'model_name'])

        for df in [summary_df_clf]:
            if 'vae_name' not in df.columns:
                continue

            for param in differing_params:
                df[param] = df['vae_name'].map(
                    lambda model: all_params.get(model, {}).get(param, 'N/A')
                )

            df.drop(columns=['vae_name'], inplace=True)
            df.drop(columns=['save_path'], inplace=True, errors='ignore')

            if 'loss_weights' in df.columns:
                def _flatten_loss_w(v):
                    if isinstance(v, (list, tuple)) and len(v) == 1 and isinstance(v[0], (list, tuple)):
                        return list(v[0])
                    return v

                df['loss_weights'] = df['loss_weights'].apply(_flatten_loss_w)
                df[['Rec w', 'KLD w', 'Clf w']] = pd.DataFrame(
                    df['loss_weights'].tolist(), index=df.index
                )

                varying_columns = [col for col in ['Rec w', 'KLD w', 'Clf w'] if df[col].nunique() > 1]
                print('col_names,varying_columns', list(df.columns), varying_columns)

                df = df.drop(
                    columns=['loss_weights'] +
                    [col for col in ['Rec w', 'KLD w', 'Clf w'] if col not in varying_columns]
                )

            fixed_columns = [
                'model_name', 'beta', 'gamma',
                'MaxIdx', 'MaxFreq',
                'MedianIdx', 'MedianFreq',
                'MinIdx', 'MinFreq',
                'DivFreq'
            ]
            other_columns = [col for col in df.columns if col not in fixed_columns]
            df = df[other_columns + fixed_columns]

            col_format = "l" + "c" * (len(df.columns) - 1)
            summ_df = df.applymap(lambda x: sci_format(x) if isinstance(x, (float, np.floating)) else x)
            latex_table = summ_df.to_latex(index=False, column_format=col_format, escape=False)
            print(latex_table)


def df_to_latex_summary(df):
    """Summarizes the DataFrame and converts it to a LaTeX table."""

    # Group by model configs and compute mean & std
    summary_df = df.groupby(["vae_name", "beta", "gamma", "model_name"]).agg(
        L2_mean=("L2", "mean"),
        validity_mean=("validity", "mean"),
        FID_mean=("FID", "mean"),
        Switch_epoch=("Switch epoch", 'mean')
    ).reset_index()

    def sci_format(x):
        return np.round(x, decimals=4)

    summary_df = summary_df.applymap(lambda x: sci_format(x) if isinstance(x, float) else x)

    latex_table = summary_df.to_latex(index=False, column_format="llllcccc", escape=False)

    print(latex_table)
    return latex_table


def diff_counterfactual_z(rec_z, reconstructions):
    reconstructions_int = np.asarray(reconstructions)
    org_rec_z = np.asarray(rec_z)
    differences = org_rec_z - reconstructions_int
    differences_mean = np.mean(differences, axis=0)
    return differences_mean


def plot_cfs(dataset_name,
             data_path_grid,
             model_name,
             fold_idxs,
             possible_values, num_imgs=5):
    for fold_idx in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/' + 'counterfactual_imgs/'
        files_data = find_and_extract_files(data_path_fold)
        print(files_data)

        for image_num in range(num_imgs):
            for betas, gammas, model_n in zip([[0, 1], [0, 1]], [[0, 1], [0]], ['SPN', 'MLP']):
                save_file_path = data_path_fold + str(image_num) + 'image_no_w_' + model_n + '.png'
                save_file_path_raw = data_path_fold + str(image_num) + 'image_raw_' + model_n + '.png'

                make_individual_plot(
                    image_num, betas, gammas, data_path_fold, save_file_path,
                    model_n, diff_counterfactual_z, model_n, "mean(x-x')"
                )
                make_raw_plot(
                    image_num, betas, gammas, data_path_fold, save_file_path_raw,
                    model_n, diff_counterfactual_z, model_n, "mean(x-x')"
                )


def eval_counterfactuals(dataset_name,
                         data_path_grid,
                         vae_name,
                         fold_idxs,
                         possible_values, test):
    random.seed(0)
    test_idx = list(range(test[0].shape[0]))
    random.shuffle(test_idx)
    test_idx = test_idx[:1000]

    results = []
    results_z_eval = []
    for fold_idx in fold_idxs:
        data_path_fold = data_path_grid + 'fold_' + str(fold_idx) + '/' + 'counterfactual_imgs'
        files_data = find_and_extract_files(data_path_fold)
        print(files_data)

        for file_name, model_name, beta, gamma in files_data:
            if beta in possible_values and gamma in possible_values:
                print(f"File: {file_name} | Model: {model_name} | Beta: {beta} | Gamma: {gamma}", flush=True)
                all_data_list = pkl.load(open(data_path_fold + '/' + file_name, 'rb'))
                all_data = []
                for entry in all_data_list:
                    all_data.extend(entry)

                [x_cf, x_rec, z_cf, z, title_info, distance, arg_max, loss,
                 log_pred, p_z, label_switch_step, y_cf_goal, coordinates,
                 x_org, mean_cls, y_org] = zip(*all_data)

                print(np.min(x_cf), np.max(x_cf),
                      np.min(x_org), np.max(x_org),
                      np.min(x_rec), np.max(x_rec))

                x_cf = np.asarray(x_cf)
                x_rec = np.asarray(x_rec)
                z = np.asarray(z)
                z_cf = np.asarray(z_cf)
                arg_max = np.asarray(arg_max)
                x_org = np.asarray(x_org)
                y_cf_goal = np.asarray(y_cf_goal)
                x_cf_mean = np.mean(x_cf, axis=1)
                x_org = np.mean(x_org, axis=1)
                log_pred = np.asarray(log_pred)
                loss = np.asarray(loss)
                x_rec_mean = np.mean(x_rec, axis=1)
                y_org = np.asarray(y_org)

                abs_change = np.abs(z_cf - z)
                arg_max_change = np.argmax(abs_change, axis=2)

                total_count = arg_max_change.shape[1]
                all_fractions = []
                all_diversity_vals = []

                max_indices = []
                median_indices = []
                min_indices = []

                for entry in arg_max_change:
                    counts = Counter(entry)
                    fractions = sorted(
                        ((num, count / total_count) for num, count in counts.items()),
                        key=lambda x: x[1],
                        reverse=True
                    )
                    all_fractions.append(fractions)

                    diversity_freq = 1
                    for freq in fractions:
                        diversity_freq *= freq[1]
                    all_diversity_vals.append(diversity_freq)

                    if len(fractions) > 0:
                        max_indices.append(fractions[0][0])
                        median_indices.append(fractions[len(fractions) // 2][0])
                        min_indices.append(fractions[-1][0])

                most_common_max_index = Counter(max_indices).most_common(1)[0][0]
                median_counter = Counter(median_indices)
                most_common_median_index = sorted(
                    list(median_counter.items()),
                    key=lambda x: x[1],
                    reverse=True
                )[int(len(median_counter) / 2)][0]
                most_common_min_index = sorted(
                    list(Counter(min_indices).items()),
                    key=lambda x: x[1],
                    reverse=True
                )[-1][0]

                def mean_fraction_for_index(index, all_fractions):
                    fractions = [dict(f).get(index, 0) for f in all_fractions]
                    return np.mean(fractions)

                mean_max_fraction = mean_fraction_for_index(most_common_max_index, all_fractions)
                mean_median_fraction = mean_fraction_for_index(most_common_median_index, all_fractions)
                mean_min_fraction = mean_fraction_for_index(most_common_min_index, all_fractions)

                all_diversity_vals = np.mean(np.asarray(all_diversity_vals))
                mean_lsp = mean_lsp_calculation(z_cf)

                for strategy_name, strategy, input_arr in zip(
                    ['weighted_loss', 'mean', 'min_loss'],
                    [weighted, mean_cf, min_loss],
                    [loss, None, loss]
                ):
                    x_cf_strat, arg_max_process = strategy(x_cf, input_arr, arg_max)

                    class_FID = compute_class_fid(
                        test[0][test_idx], x_cf_strat, x_rec_mean,
                        np.round(y_cf_goal),
                        test[1][test_idx, 0],
                        np.round(arg_max_process)
                    )
                    FID = compute_fid(x_org, x_cf_strat)
                    blop_info = get_importance_region_information(x_rec_mean, x_cf_strat, threshold=0.5)
                    MSE, L2 = compute_MSE(x_org, x_cf_strat)
                    MAE, L1 = compute_MAE(x_org, x_cf_strat)
                    validity = compute_validity(np.round(y_cf_goal), np.round(arg_max_process))
                    rel_L2 = compute_rel_L2(x_org, x_cf_strat, x_rec_mean)

                    results.append({
                        "dataset": dataset_name,
                        "vae_name": vae_name,
                        "beta": beta,
                        "gamma": gamma,
                        "fold_idx": fold_idx,
                        "strategy": strategy_name,
                        "model_name": model_name,
                        "MSE": MSE,
                        "MAE": MAE,
                        'L2': L2,
                        'L1': L1,
                        "validity": validity,
                        "FID": FID,
                        "RelFID": class_FID,
                        "RelL2": rel_L2,
                        "Pos region size": blop_info[0, 0],
                        'Pos number of regions': [0, 1],
                        "Neg region size": [1, 0],
                        'Neg number of regions': [1, 1],
                    })

                label_switch_step_mean = np.mean([e for e in label_switch_step if e >= 0])

                results_z_eval.append({
                    "dataset": dataset_name,
                    "vae_name": vae_name,
                    "beta": beta,
                    "gamma": gamma,
                    "fold_idx": fold_idx,
                    "model_name": model_name,
                    "Max idx": most_common_max_index,
                    "Max freq": mean_max_fraction,
                    "Median idx": most_common_median_index,
                    "Median freq": mean_median_fraction,
                    "Min idx": most_common_min_index,
                    "Min freq": mean_min_fraction,
                    'Diversity freq.': all_diversity_vals,
                    "Switch epoch": label_switch_step_mean,
                    "Mean LSP": mean_lsp,
                })

    return results, results_z_eval
