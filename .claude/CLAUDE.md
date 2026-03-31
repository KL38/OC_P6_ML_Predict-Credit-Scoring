# Projet OC_P6 — Scoring de crédit (Home Credit Default Risk)

## Contexte
Projet OpenClassrooms de scoring de crédit. L'objectif est de prédire la probabilité de défaut de remboursement d'un client (TARGET=1 = défaut).

## Stack
- **Python** via `uv` (voir `pyproject.toml`)
- **MLflow** pour le tracking des expériences (serveur local : `http://127.0.0.1:5000/`)
- **LightGBM, XGBoost, sklearn (LR, RF, MLP)** pour la modélisation
- **Notebooks Jupyter** dans `notebooks/`

## Structure
```
data/                    # Données brutes Home Credit (ne pas modifier)
notebooks/
  EDA.ipynb              # Feature engineering + merge des 8 sources de données
  Modelisation.ipynb     # Entraînement, CV, tracking MLflow — tous modèles
  Optimisation.ipynb     # Optimisation LGBM avec Optuna — experiment "OC_P6_optimisation"
mlartifacts/             # Artefacts MLflow
mlflow.db                # Base MLflow locale
presentation.md          # Support de présentation projet (à compléter avec métriques full dataset)
```

## Données
8 sources fusionnées au niveau client (`SK_ID_CURR`) :
- `application_train/test.csv` — données principales
- `bureau.csv` + `bureau_balance.csv`
- `previous_application.csv`
- `POS_CASH_balance.csv`
- `installments_payments.csv`
- `credit_card_balance.csv`

Dataset final : ~770 colonnes après feature engineering.
Déséquilibre de classes : ~8% de défauts (TARGET=1).

Deux datasets disponibles :
- `df.csv` — dataset complet
- `df_red.csv` — dataset réduit (10k lignes) pour tests rapides

### Features clés
- **`EXT_SOURCE_1/2/3`** — scores de crédit externes fournis par des tiers (bureaux de crédit partenaires de Home Credit). Normalisés entre 0 et 1, plus élevé = moins risqué. Ce sont les 3 features les plus importantes du modèle LGBM. EXT_SOURCE_1 a ~50% de NaN. Home Credit ne documente pas les sources précises.

## Versioning des notebooks

Quand des modifications sont demandées sur le notebook de modélisation, le fichier cible est toujours **`notebooks/Modelisation.ipynb`**.
Les fichiers `Modelisation_v0.ipynb`, `Modelisation_v1.ipynb`, etc. sont des snapshots de versioning personnel — **ne jamais les modifier**.

## Notebook Modelisation.ipynb — Architecture

### Fonctions principales
- **`delete_run_if_exists(experiment_name, run_name)`** — supprime une run MLflow existante avant d'en créer une nouvelle (pattern "delete-before-create" pour éviter les doublons)
- **`make_sklearn_pipeline(model)`** — wrape un modèle sklearn avec `SimpleImputer(median)` + `StandardScaler` (nécessaire pour LR, RF qui ne gèrent pas les NaN)
- **`run_cv(model, model_name, X, y, n_splits=5, tags=None, use_sample_weights=False, optimize_threshold=False)`** — cross-validation stratifiée 5 folds avec logging MLflow automatique. `use_sample_weights=True` active `compute_sample_weight("balanced")` à chaque fold. `optimize_threshold=True` cherche le seuil optimal (max Fbeta β=1.5) sur chaque val fold via `find_best_threshold`, l'utilise pour les métriques du fold, et logge `best_threshold_mean/std` dans MLflow.
- **`find_best_threshold(y_true, y_proba, beta=FBETA, n_steps=100)`** — scanne 100 seuils entre 0.01 et 0.99, retourne celui qui maximise le Fbeta.
- **`cost_score(y_true, y_pred)`** — coût métier : `-(10*FN + FP)`. `scorer = make_scorer(cost_score)` prêt pour GridSearchCV.

### Métriques loggées
- `roc_auc_mean` / `roc_auc_std`
- `recall_mean` / `recall_std`
- `precision_mean` / `precision_std`
- `f1_mean` / `f1_std`
- `fbeta_mean` / `fbeta_std` (β=1.5, recall favorisé)
- `cost_mean` / `cost_std` — coût métier : `-(10*FN + FP)`, FN coûte 10× plus qu'un FP
- `best_threshold_mean` / `best_threshold_std` — seuil optimal CV (si `optimize_threshold=True`)

### Constantes
- `FBETA = 1.5`
- `EXPERIMENT_NAME = "First trial"`

### Logging MLflow
- `log_model_fn` : dict qui mappe `LGBMClassifier` et `XGBClassifier` vers leurs fonctions MLflow. CatBoost complètement retiré (imports + `log_model_fn`).
- Fallback sur `mlflow.sklearn.log_model` pour tout modèle non listé (Pipeline, LR, RF, MLP, etc.)
- Tous les hyperparamètres sont loggués automatiquement via `model.get_params()`
- `model_name` loggué en paramètre pour faciliter le filtrage dans l'UI
- Tag `model_family` = `type(base).__name__` où `base = model.steps[-1][1] if isinstance(model, Pipeline) else model` → donne `LGBMClassifier`, `RandomForestClassifier`, etc.
- Toutes les métriques arrondies à **2 décimales** via `round(float(...), 2)` dans les deux notebooks

### Points d'attention

**MLflow**
- Les tags MLflow doivent être des **strings** — `{"fbeta": str(FBETA)}` et non `{"fbeta": FBETA}` → `TypeError: bad argument type for built-in operation`
- `mlflow.log_params()` nécessite des valeurs convertibles en string → toujours utiliser `{k: str(v) for k, v in model.get_params().items()}`
- `delete_run_if_exists` fait une **suppression douce** (soft delete) — les runs supprimées peuvent encore apparaître dans l'onglet "Models" de l'UI, c'est normal
- Le serveur MLflow doit être lancé **avant** d'exécuter `run_cv`, sinon les runs sont perdues silencieusement ou erreur de connexion
- `FBETA` et `EXPERIMENT_NAME` sont définis dans la cellule `run_cv` — si le kernel redémarre, relancer cette cellule avant tout
- Pour rouvrir une run existante et y ajouter des artefacts : `mlflow.start_run(run_id=run_id)` — récupérer le `run_id` via `client.search_runs()`
- Les métriques sont arrondies à **2 décimales** dans les deux notebooks via `round(float(...), 2)`

**Données**
- Le dataset complet `df.csv` contient des valeurs `inf` (divisions par zéro dans le feature engineering de l'EDA — ex: `AMT_CREDIT=0`). XGBoost les refuse avec `XGBoostError: Input data contains inf`. Ajouter `X = X.replace([np.inf, -np.inf], np.nan)` dans la cellule de préparation des données
- `df_red.csv` (10k lignes) ne contient pas forcément les mêmes cas limites que le dataset complet — un bug peut passer inaperçu sur le réduit
- Le flag `use_reduced = True` est facile à oublier lors du passage au dataset complet

**Modèles**
- Les modèles sklearn (`LogisticRegression`, `RandomForestClassifier`) ne gèrent **pas** les NaN nativement → toujours les passer via `make_sklearn_pipeline(model)`, jamais directement à `run_cv`
- Les modèles tree-based (LGBM, XGB) gèrent les NaN nativement — **pas** besoin de pipeline
- `log_model_fn` utilise `.get(type(model), mlflow.sklearn.log_model)` comme fallback — si on revenait à `log_model_fn[type(model)]`, un `Pipeline` lèverait un `KeyError`
- RF avec `class_weight="balanced"` peut encore prédire 0 positifs au seuil 0.5 → recall=0, precision=0, f1=0. C'est un comportement attendu, pas un bug. Essayer `class_weight="balanced_subsample"` ou optimiser le seuil
- RF a un threshold optimal (~0.11) bien inférieur aux autres modèles (~0.58) : **comportement attendu, pas un bug**. RF produit des probabilités mal calibrées (probas positifs naturellement basses car `predict_proba` = fraction de trees). Le sample_weight fonctionne bien (confirmé : precision 0→0.20 entre baseline et sw), mais n'augmente pas dramatiquement les probas de sortie. LR optimise directement la log-loss pondérée → probas bien calibrées → threshold cohérent avec LGBM/XGB.
- `MLPClassifier` **ne supporte pas `sample_weight`** dans `fit()` — exclure MLP de toute expérience `use_sample_weights=True`
- `sw_key` dans `run_cv` s'adapte automatiquement : `"model__sample_weight"` pour un `Pipeline`, `"sample_weight"` pour un modèle direct
- `LogisticRegression(n_jobs=-1)` lève un `FutureWarning` (sklearn ≥1.8, paramètre supprimé) — ne pas passer `n_jobs` à LR

**Métriques**
- Recall/precision/f1/fbeta très faibles (~1-5%) avec seuil 0.5 sont **attendus** vu le déséquilibre (~8% positifs) — le ROC AUC correct (~0.73-0.75) confirme que les modèles discriminent bien
- `precision_score` retourne 0 avec un `UndefinedMetricWarning` quand aucun positif n'est prédit — ne fait pas crasher mais fausse les moyennes
- `model.fit(X, y)` est appelé **deux fois** dans `run_cv` : une fois par fold (pour le CV) et une fois sur tout le train set à la fin (pour l'artefact sauvegardé dans MLflow). C'est voulu

## Notebook Optimisation.ipynb — Architecture

Dédié uniquement à LGBM. Experiment MLflow : `OC_P6_optimisation`.

### Différences clés vs Modelisation.ipynb
- `cost_score` retourne `+10*FN + FP` (positif, à **minimiser**) — dans Modelisation c'est `-(10*FN+FP)` (négatif, à maximiser). Cohérent avec `np.argmin` dans `find_best_threshold`.
- `find_best_threshold` optimise le **coût métier** (pas le Fbeta comme dans Modelisation)
- `pr_auc` (`average_precision_score`) ajouté aux métriques
- `log_model_fn` limité à `LGBMClassifier` uniquement
- `N_TRIALS = 50` (réduit) / `100+` (full)

### Flow post-Optuna
1. `run_cv(best_model, "lgbm_optuna_best", ...)` — métriques CV + modèle sauvegardé
2. Prédictions OOF sur X_train → courbe coût vs seuil (X_test intact)
3. Feature importances top 10 (plotly)
4. Évaluation sur X_test avec `best_thresh` OOF → métriques `test_*` + confusion matrix
5. Réouverture de la run → log des 5 artefacts HTML + métriques test
6. `mlflow.register_model(...)` → registry `lgbm_credit_scoring`

### Pour le run full dataset
- `use_reduced = False`
- Run name : `lgbm_optuna_best_full` (tag `dataset=full`)
- `N_TRIALS = 100`
- Filtre MLflow à adapter : `"tags.mlflow.runName = 'lgbm_optuna_best_full'"`
- `mlflow.register_model(...)` → crée automatiquement **Version 2** de `lgbm_credit_scoring`

## Roadmap / Prochaines étapes

1. ~~**Sample weights**~~ — **Fait** : `use_sample_weights=True` implémenté dans `run_cv`. Runs : `lgbm_sw_full`, `xgb_sw_full`, `log_reg_sw_full`, `rf_sw_full` (tag `phase=sample_weights`)
2. ~~**Threshold optimisation**~~ — **Fait** : `optimize_threshold=True` dans `run_cv`, seuil trouvé par fold sur val set (CV-aware, pas de leakage X_test). Runs : `lgbm_thr_full`, `xgb_thr_full`, `log_reg_thr_full`, `rf_thr_full`, `mlp_thr_full` (tag `phase=threshold_opt`). MLP inclus mais sans sample_weight.
3. ~~**Optuna sur dataset réduit**~~ — **Fait** : `notebooks/Optimisation.ipynb`, 50 trials, experiment `OC_P6_optimisation`. Run `lgbm_ini` (baseline) puis `lgbm_optuna_best` (meilleure trial + 5 artefacts HTML + métriques X_test + modèle enregistré sous `lgbm_credit_scoring` dans le Model Registry).
4. **Optuna sur full dataset** — relancer avec `use_reduced=False` et `N_TRIALS=100+` pour les résultats définitifs.

## Commandes utiles
```bash
# Lancer le serveur MLflow
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

# Lancer Jupyter
uv run jupyter notebook
```
