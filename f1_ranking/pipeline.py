from config import YEARS, ARTIFACT_ROOT, ensure_dirs
from data_preprocessing import load_all_laps_and_results
from data_preprocessing.feature_engineering import build_dataset
# import warnings
# warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
from model_training import (
    walk_forward_split,
    grid_baseline_predictions,
    train_extratrees,
    tune_xgboost,
    tune_tabnet,
    calibrate_tabnet,
    train_parity_booster,
    save_full_extratrees,
    save_full_xgboost,
    save_full_tabnet,
)
from evaluation import (
    evaluate_predictions,
    print_overall_summary,
    print_models_comparison,
    print_mae_by_position_group,
    plot_pred_vs_actual_scatter,
    plot_tabnet_training_curves,
    print_race_prediction_table,
    write_validation_report,
)
from shap_analysis import (
    plot_all_feature_importances,
    export_all_shap,
    evaluate_extratrees_pca,
)


def run_full_pipeline(years=YEARS,
                      use_cache: bool = True,
                      tune_models: bool = True,
                      run_shap: bool = True,
                      run_pca_ablation: bool = True,
                      save_full_models: bool = True,
                      write_report: bool = True):
    """Run the full pipeline.
    """
    ensure_dirs()


    print("STAGE 1 — DATA PREPROCESSING")

    all_laps, race_results, _, wet_race_ids = load_all_laps_and_results(years=years, use_cache=use_cache)


    print("STAGE 2 — FEATURE ENGINEERING")

    dataset, feature_cols = build_dataset(all_laps, race_results)

    print("STAGE 3 — TRAIN MODELS")

    n_unique = dataset["race_id"].nunique()
    if n_unique < 3:
        raise ValueError(f"Need at least 3 races; got {n_unique}.")
    min_train_races = max(2, min(4, n_unique - 1))
    folds = walk_forward_split(dataset, feature_cols,
                               min_train_races=min_train_races, race_col="race_id")
    if not folds:
        raise ValueError("No walk-forward folds were created.")
    print(f"Walk-forward folds: {len(folds)} (min_train_races={min_train_races})")

    print("\n--- Grid baseline ---")
    baseline_preds = grid_baseline_predictions(folds)
    baseline_eval = evaluate_predictions(baseline_preds)
    print_overall_summary(baseline_eval, model_name="Grid Baseline")

    print("\n--- ExtraTrees (best fixed params) ---")
    et_preds, et_model = train_extratrees(folds)
    et_eval = evaluate_predictions(et_preds)
    print_overall_summary(et_eval, model_name="ExtraTrees")

    best_xgb = None
    best_tabnet = None
    parity = None
    if tune_models:
        print("\n--- XGBoost (random search) ---")
        best_xgb = tune_xgboost(folds, evaluate_predictions, n_trials=30, seed=42)

        print("\n--- TabNet (random search) ---")
        best_tabnet = tune_tabnet(folds, evaluate_predictions, n_trials=12, seed=42)

        print("\n--- TabNet calibrated blend ---")
        cal_summary, cal_preds, _ = calibrate_tabnet(best_tabnet["predictions"], evaluate_predictions)
        print(f"Best calibrated TabNet alpha={cal_summary['alpha']:.2f} | "
              f"Spearman={cal_summary['spearman']:.4f} MAE={cal_summary['mae']:.4f}")

        print("\n--- TabNet parity booster ---")
        parity_best, parity_eval, parity_preds, _ = train_parity_booster(
            folds, dataset,
            tabnet_cfg=best_tabnet["params"],
            xgb_params=best_xgb["params"],
            evaluate_predictions=evaluate_predictions,
        )
        parity = {"summary": parity_best, "eval": parity_eval, "predictions": parity_preds}
        print(f"Parity booster: mix_teacher={parity_best['mix_teacher']:.2f} "
              f"mix_grid={parity_best['mix_grid']:.2f} | "
              f"Spearman={parity_best['spearman']:.4f} MAE={parity_best['mae']:.4f}")

    print("\n" + "#" * 60)
    print("#  STAGE 4 — EVALUATION")
    print("#" * 60)
    eval_dfs = {"Grid Baseline (lower bound)": baseline_eval, "ExtraTrees": et_eval}
    if best_xgb is not None:
        eval_dfs["XGBoost"] = best_xgb["eval_df"]
    if parity is not None:
        eval_dfs["TabNet (parity)"] = parity["eval"]
    print_models_comparison(eval_dfs)

    print_mae_by_position_group(et_preds)
    plot_pred_vs_actual_scatter(et_preds,
                                save_path=ARTIFACT_ROOT / "pred_vs_actual_2022_2025.png",
                                title="ExtraTrees: Predicted vs Actual (Dry Races)")

    print_race_prediction_table(et_preds, year=2025, event_keyword="Abu Dhabi",
                                save_dir=ARTIFACT_ROOT / "race_views")

    if best_tabnet is not None:
        plot_tabnet_training_curves(best_tabnet["model"],
                                    save_path=ARTIFACT_ROOT / "tabnet_training_curves.png")

    print("\n" + "#" * 60)
    print("#  STAGE 5 — SHAP / FEATURE IMPORTANCE")
    print("#" * 60)

    importance_models = {"ExtraTrees": et_model}
    if best_xgb is not None:
        importance_models["XGBoost"] = best_xgb["model"]
    if best_tabnet is not None:
        importance_models["TabNet"] = best_tabnet["model"]
    plot_all_feature_importances(importance_models, feature_cols,
                                 save_dir=ARTIFACT_ROOT / "shap")

    if run_shap:
        models_for_shap = {"ExtraTrees": et_model}
        if best_xgb is not None:
            models_for_shap["XGBoost"] = best_xgb["model"]
        export_all_shap(models_for_shap, dataset, feature_cols)

    pca_eval = None
    n_comp = None
    if run_pca_ablation:
        print("\n--- PCA ablation (ExtraTrees, 95% variance) ---")
        _, pca_eval, n_comp = evaluate_extratrees_pca(folds, evaluate_predictions, variance=0.95)
        et_summary = {
            "spearman": float(et_eval["spearman_model"].mean()),
            "mae": float(et_eval["mae_model"].mean()),
            "ndcg": float(et_eval["ndcg_model"].mean()),
        }
        pca_summary = {
            "spearman": float(pca_eval["spearman_model"].mean()),
            "mae": float(pca_eval["mae_model"].mean()),
            "ndcg": float(pca_eval["ndcg_model"].mean()),
        }
        import numpy as np
        print(f"Avg components: {np.mean(n_comp):.1f} (min={min(n_comp)}, max={max(n_comp)})")
        print(f"Base Spearman={et_summary['spearman']:.4f}, PCA Spearman={pca_summary['spearman']:.4f}")
        print(f"Base MAE     ={et_summary['mae']:.4f}, PCA MAE     ={pca_summary['mae']:.4f}")

    saved_model_files: list[str] = []
    if save_full_models:
        print("\n" + "#" * 60)
        print("#  STAGE 6 — REFIT ON FULL DATA & SAVE WEIGHTS")
        print("#" * 60)
        saved_model_files += save_full_extratrees(dataset, feature_cols,
                                                  walkforward_model=et_model)
        xgb_params = best_xgb["params"] if best_xgb else None
        saved_model_files += save_full_xgboost(dataset, feature_cols, params=xgb_params)
        if best_tabnet is not None:
            saved_model_files += save_full_tabnet(
                dataset, feature_cols,
                tabnet_params=best_tabnet["params"],
                last_fold_model=best_tabnet.get("model"),
            )

    if write_report:
        write_validation_report(
            dataset=dataset,
            folds=folds,
            base_eval=et_eval,
            pca_eval=pca_eval,
            pca_components=n_comp,
            saved_model_files=saved_model_files,
        )

    return {
        "dataset": dataset,
        "feature_cols": feature_cols,
        "folds": folds,
        "baseline": {"predictions": baseline_preds, "eval": baseline_eval},
        "extratrees": {"predictions": et_preds, "eval": et_eval, "model": et_model},
        "xgboost": best_xgb,
        "tabnet": best_tabnet,
        "tabnet_parity": parity,
        "saved_models": saved_model_files,
    }


if __name__ == "__main__":
    run_full_pipeline()
