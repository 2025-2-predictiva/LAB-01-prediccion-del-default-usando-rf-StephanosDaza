# flake8: noqa: E501

import gzip
import json
import os
import pickle
import zipfile
from glob import glob
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def get_project_root():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    return project_root

def leer_zip_a_dfs(directorio_relativo: str) -> list[pd.DataFrame]:
    root = get_project_root()
    ruta_completa = os.path.join(root, directorio_relativo)
    
    dataframes = []
    
    for zip_path in glob(os.path.join(ruta_completa, "*")):
        with zipfile.ZipFile(zip_path, "r") as zf:
            for miembro in zf.namelist():
                with zf.open(miembro) as fh:
                    dataframes.append(pd.read_csv(fh, sep=",", index_col=0))
    return dataframes

def reiniciar_directorio(ruta_relativa: str) -> None:
    root = get_project_root()
    ruta = os.path.join(root, ruta_relativa)
    
    if os.path.exists(ruta):
        for f in glob(os.path.join(ruta, "*")):
            try:
                os.remove(f)
            except IsADirectoryError:
                pass
            except OSError:
                pass
        try:
            os.rmdir(ruta)
        except OSError:
            pass
    os.makedirs(ruta, exist_ok=True)

def guardar_modelo_gz(ruta_relativa: str, objeto) -> None:
    root = get_project_root()
    ruta_completa = os.path.join(root, ruta_relativa)
    
    
    os.makedirs(os.path.dirname(ruta_completa), exist_ok=True)
    
    with gzip.open(ruta_completa, "wb") as fh:
        pickle.dump(objeto, fh)
    
    print(f"Modelo guardado en: {ruta_completa}")

def depurar(df: pd.DataFrame) -> pd.DataFrame:
    tmp = df.copy()
    if "default payment next month" in tmp.columns:
        tmp = tmp.rename(columns={"default payment next month": "default"})

    
    tmp = tmp.loc[tmp["MARRIAGE"] != 0]
    tmp = tmp.loc[tmp["EDUCATION"] != 0]

    tmp["EDUCATION"] = tmp["EDUCATION"].apply(lambda v: 4 if v >= 4 else v)

    return tmp.dropna()

def separar_xy(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=["default"])
    y = df["default"]
    return X, y

def ensamblar_busqueda() -> GridSearchCV:
    cat_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    ohe = OneHotEncoder(handle_unknown="ignore")
    ct = ColumnTransformer(
        transformers=[("cat", ohe, cat_cols)],
        remainder="passthrough",
    )

    clf = RandomForestClassifier(random_state=42)
    pipe = Pipeline(
        steps=[
            ("prep", ct),
            ("rf", clf),
        ]
    )

    grid_params = {
        "rf__n_estimators": [100, 200, 500],
        "rf__max_depth": [None, 5, 10],
        "rf__min_samples_split": [2, 5],
        "rf__min_samples_leaf": [1, 2],
    }

    gs = GridSearchCV(
        estimator=pipe,
        param_grid=grid_params,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
        verbose=2,
    )
    return gs

def empaquetar_metricas(etiqueta: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics",
        "dataset": etiqueta,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

def empaquetar_matriz_conf(etiqueta: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": etiqueta,
        "true_0": {"predicted_0": int(cm[0][0]), "predicted_1": int(cm[0][1])},
        "true_1": {"predicted_0": int(cm[1][0]), "predicted_1": int(cm[1][1])},
    }

def main() -> None:
    # Paso 1. Carga y Limpieza
    raw_dfs = leer_zip_a_dfs(os.path.join("files", "input"))
    
    df_list = [depurar(d) for d in raw_dfs]
    
    
    if len(df_list[0]) < len(df_list[1]):
        test_df, train_df = df_list[0], df_list[1]
    else:
        train_df, test_df = df_list[0], df_list[1]

    X_tr, y_tr = separar_xy(train_df)
    X_te, y_te = separar_xy(test_df)

    # Paso 2. Entrenamiento
    buscador = ensamblar_busqueda()
    buscador.fit(X_tr, y_tr)

    # Paso 3. Guardar Modelo
    guardar_modelo_gz(os.path.join("files", "models", "model.pkl.gz"), buscador)

    # Paso 4. Métricas
    yhat_test = buscador.predict(X_te)
    yhat_train = buscador.predict(X_tr)

    m_test = empaquetar_metricas("test", y_te, yhat_test)
    m_train = empaquetar_metricas("train", y_tr, yhat_train)

    cm_test = empaquetar_matriz_conf("test", y_te, yhat_test)
    cm_train = empaquetar_matriz_conf("train", y_tr, yhat_train)

    # Paso 5. Guardar Métricas
    root = get_project_root()
    output_dir = os.path.join(root, "files", "output")
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as fh:
        fh.write(json.dumps(m_train) + "\n")
        fh.write(json.dumps(m_test) + "\n")
        fh.write(json.dumps(cm_train) + "\n")
        fh.write(json.dumps(cm_test) + "\n")
    
    print("Métricas generadas correctamente.")

if __name__ == "__main__":
    main()