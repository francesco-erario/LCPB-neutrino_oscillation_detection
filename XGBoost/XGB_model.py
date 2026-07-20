import numpy as np
import xgboost as xgb


def build_and_train_model(
    # ── Data ────────────────────────────────────────────────────────────────
    X_train,
    y_train,
    X_val,
    y_val,

    # ── Hyperparameters XGBoost ───────────────────────────────────────────────
    n_estimators:          int   = 50,
    max_depth:             int   = 3,
    learning_rate:         float = 0.1,
    subsample:             float = 0.8,
    colsample_bytree:      float = 0.8,
    min_child_weight:      int   = 1,
    reg_lambda:            float = 1.0,
    reg_alpha:             float = 0.0,
    gamma:                 float = 0.0,
    max_delta_step:        int   = 0,
    colsample_bylevel:     float = 1.0,
    colsample_bynode:      float = 1.0,
    scale_pos_weight:      float = 1.0,
    tree_method:           str   = "hist",
    eval_metric:           str   = "logloss",
    early_stopping_rounds: int   = 15,
    random_state:          int   = 42,

    # ── General options ────────────────────────────────────────────────────
    sample_weight:         np.ndarray | None = None,  # manual weights
    verbose:               int               = 50,    # 0 = silent
) -> xgb.XGBClassifier:
    """
    Allena un XGBClassifier standard (senza adversarial training).

    Parametri
    ----------
    X_train, y_train : features e label di training
    X_val, y_val     : features e label di validazione (per early stopping)
    n_estimators     : numero massimo di alberi
    max_depth        : profondità massima di ogni albero
    learning_rate    : step size dello shrinkage (eta)
    subsample        : frazione di campioni per albero
    colsample_bytree : frazione di feature campionate per albero
    colsample_bylevel: frazione di feature campionate per livello
    colsample_bynode : frazione di feature campionate per nodo
    min_child_weight : somma minima dei pesi nel nodo foglia
    reg_lambda       : regolarizzazione L2 sui pesi delle foglie
    reg_alpha        : regolarizzazione L1 sui pesi delle foglie
    gamma            : riduzione minima della loss per uno split
    max_delta_step   : vincolo sul passo di aggiornamento (utile per classi sbilanciate)
    scale_pos_weight : peso della classe positiva (utile per dataset sbilanciati)
    tree_method      : algoritmo di costruzione degli alberi ("hist" o "exact")
    eval_metric      : metrica di valutazione ("logloss", "auc", "error", ...)
    early_stopping_rounds : stop se val_metric non migliora per N round
    random_state     : seed per riproducibilità
    sample_weight    : array opzionale di pesi per campione
    verbose          : ogni quanti round stampare le metriche (0 = mai)

    Restituisce
    -----------
    model : XGBClassifier allenato
    """

    params = dict(
        n_estimators          = n_estimators,
        max_depth             = max_depth,
        learning_rate         = learning_rate,
        subsample             = subsample,
        colsample_bytree      = colsample_bytree,
        colsample_bylevel     = colsample_bylevel,
        colsample_bynode      = colsample_bynode,
        min_child_weight      = min_child_weight,
        reg_lambda            = reg_lambda,
        reg_alpha             = reg_alpha,
        gamma                 = gamma,
        max_delta_step        = max_delta_step,
        scale_pos_weight      = scale_pos_weight,
        tree_method           = tree_method,
        eval_metric           = eval_metric,
        early_stopping_rounds = early_stopping_rounds,
        random_state          = random_state,
    )

    model = xgb.XGBClassifier(**params)
    model.fit(
        X_train, y_train,
        sample_weight = sample_weight,
        eval_set      = [(X_train, y_train), (X_val, y_val)],
        verbose       = verbose,
    )

    return model