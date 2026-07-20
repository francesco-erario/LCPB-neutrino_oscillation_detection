import pandas as pd
from sklearn.model_selection import train_test_split


def dataset_creator(path_set_events, path_set_background, target='is_event',
                    drop=None, analysis_cols=None):
    """
    Crea train/test set bilanciati da due file parquet (eventi e background).

    Modifiche rispetto all'originale
    --------------------------------
    1. ``drop is None`` al posto di ``drop == None`` (PEP 8).
    2. Se ``drop`` è fornito ma non contiene ``target``, il target viene
       comunque rimosso dalle feature.
    3. Nuovo parametro ``analysis_cols``: lista di colonne da conservare
       (ad es. 'Ep') allineate ai rispettivi split, anche se droppate da X.
       Se specificato la funzione restituisce due DataFrame aggiuntivi
       (A_train, A_test).
    """

    events = pd.read_parquet(path_set_events)
    background = pd.read_parquet(path_set_background)

    size = min(events.shape[0], background.shape[0])

    df = pd.concat(
        [events[:size], background[:size]],
        ignore_index=True
    )

    df = df.sample(
        frac=1.0,
        random_state=42
    ).reset_index(drop=True)

    # --- Determina le colonne da rimuovere da X ---
    if drop is None:
        drop_cols = [target]
    else:
        drop_cols = list(drop)
        if target not in drop_cols:
            drop_cols.append(target)

    # --- Estrai colonne di analisi prima del drop ---
    analysis_data = None
    if analysis_cols is not None:
        analysis_data = df[analysis_cols].copy()

    X = df.drop(columns=drop_cols)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    if analysis_data is not None:
        A_train = analysis_data.loc[X_train.index].reset_index(drop=True)
        A_test  = analysis_data.loc[X_test.index].reset_index(drop=True)
        X_train = X_train.reset_index(drop=True)
        X_test  = X_test.reset_index(drop=True)
        y_train = y_train.reset_index(drop=True)
        y_test  = y_test.reset_index(drop=True)
        return X_train, X_test, y_train, y_test, A_train, A_test

    return X_train, X_test, y_train, y_test