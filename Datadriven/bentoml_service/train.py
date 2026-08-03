import os

import pandas as pd
import numpy as np
import bentoml
import joblib
from sklearn.ensemble import RandomForestClassifier

# N=window size to train on (matches what service.py's {"values": [...]} expects),
# which single sensor location to train on, and which BentoML tag to publish under.
WINDOW_SIZE = int(os.environ.get("BENTO_WINDOW_SIZE", 50))
LOCATION = os.environ.get("BENTO_LOCATION", "Out")
BENTO_MODEL_TAG = os.environ.get("BENTO_MODEL_TAG", "rfc_class")


def build_windows(df, window_size):
    """Non-overlapping windows of `window_size` consecutive readings, in
    chronological order. A window's label is temp_warn of its own last
    reading (same convention as Autoencoders/iiot_compression_experiment.py)."""
    temps = df['temp'].to_numpy(dtype='float32')
    labels = df['temp_warn'].to_numpy(dtype='int64')
    n_windows = len(temps) // window_size
    windows = np.stack([temps[i * window_size:(i + 1) * window_size] for i in range(n_windows)])
    window_labels = np.array([labels[(i + 1) * window_size - 1] for i in range(n_windows)])
    return windows, window_labels


if __name__ == "__main__":

    # Cargamos los datos: una sola ubicación, orden cronológico, sin lecturas duplicadas
    # (el CSV repite ~60% de las filas tal cual, un problema conocido de este dataset).
    df = pd.read_csv('dataset/IIOT-temp-warn-max.csv')
    df = df[df['out/in'] == LOCATION].drop_duplicates(subset=['noted_date', 'temp', 'out/in'])
    df['noted_date'] = pd.to_datetime(df['noted_date'], format='%d-%m-%Y %H:%M')
    df = df.sort_values('noted_date')

    windows, y = build_windows(df, WINDOW_SIZE)
    columns = [f"v{i}" for i in range(WINDOW_SIZE)]  # must match service.py's column convention
    X = pd.DataFrame(windows, columns=columns)

    # Entrenar el modelo
    model = RandomForestClassifier(random_state=0)
    model.fit(X, y)

    # Guardar el modelo entrenado con pickle
    joblib.dump(model, 'rfc_model.pkl')

    bento_model = bentoml.sklearn.save_model(BENTO_MODEL_TAG, model)
    print(f"Model saved: {bento_model}")

    # Test running inference with BentoML runner
    sample = pd.DataFrame([windows[0]], columns=columns)
    test_runner = bentoml.sklearn.get(f"{BENTO_MODEL_TAG}:latest").to_runner()
    test_runner.init_local()
    assert test_runner.predict.run(sample) == model.predict(sample)