import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bentoml
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
from torchprofile import profile_macs

from autoencoder import AsymmetricAutoencoder
from decoder import Decoder
from encoder import Encoder
from train import TrainConfig, train

# ------------------------------ Paths ---------------------------------
IIOT_CSV_PATH = Path(__file__).resolve().parent.parent / "Datadriven" / "bentoml_service" / "dataset" / "IIOT-temp-warn-max.csv"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# ------------------------------ Sweep parameters ---------------------------------
# Test "ratio": vary window size, at each fixed target compression ratio
# (latent_dim = round(input_dim / ratio), so latent_dim itself varies with input_dim).
INPUT_DIMS_RATIO_SWEEP = [50, 60, 70, 80, 90, 100]
COMPRESSION_RATIOS = [2, 5, 10, 15, 20]

# Test "latent": fixed window size, vary the latent dimension directly.
INPUT_DIM_LATENT_SWEEP = 50
LATENT_DIMS_LATENT_SWEEP = [2, 4, 8, 16, 32]

N_ITERATIONS = 30                  # seeds 0..N_ITERATIONS-1, shared across both scenarios
HIDDEN_LAYERS = 2                  # AE hidden-layer count, fixed (not a swept dimension)
ASYMMETRIC_OPTIONS = (False, True)  # both symmetric and asymmetric AEs are run at every point

# ------------------------------ AE training parameters ---------------------------------
ACTIVATION_FUNCTION = "elu"
LATENT_ACTIVATION = "selu"
DECODER_OUTPUT_ACTIVATION = "sigmoid"   # expects inputs scaled to ~[0,1], see scale()
EPOCHS, PATIENCE = 300, 15

# ------------------------------ Data pipeline parameters ---------------------------------
TRAIN_STRIDE = [10, 27, 33]        # multi-stride overlap for train windows
TRAIN_FRAC, VAL_FRAC = 0.7, 0.1    # test = remainder (0.2); split is chronological, per location
LOCATIONS = ("In", "Out")          # windows never mix locations
JSON_FLOAT_DECIMALS = 4            # rounding applied before json.dumps, for a realistic payload
BENTO_TAG_PREFIX = "iiot_sweep"

# Overheating label: one-sided z-score of a window's own last reading against
# the mean/std of its own preceding window_size-1 readings (Shewhart-style
# control-chart rule, k_sigma=2
K_SIGMA = 2
Z_SCORE_EPSILON = 1e-6             # guards baseline_std == 0 (a perfectly flat preceding window)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SMOKE_TEST = False


# ------------------------------ Generic window/AE utilities ---------------------------------

def get_window_starts(length: int, window_size: int, strides: list) -> list:
    """Get start index of all windows"""
    starts = list()
    for stride in strides:
        if stride <= 0:
            stride = window_size
        aux = list(range(1, length, stride))
        while aux[-1] + window_size > length:
            aux.pop()
        starts += aux
    starts = list(dict.fromkeys(starts))
    np.random.shuffle(starts)
    return starts


def read_window(temps: np.ndarray, start: int, window_size: int, extra_prev: bool = False) -> np.ndarray:
    """Read only CSV rows needed for a window instead of loading the entire file into memory."""
    if extra_prev:
        return temps[start - 2: start - 2 + window_size + 1]
    return temps[start - 1: start - 1 + window_size]


def build_hidden_layers(input_dim, latent_dim, n_hidden):
    """Return the list of hidden layer sizes for a given input and latent dimension, and number of hidden layers."""
    if n_hidden <= 0:
        return []
    dims = np.geomspace(input_dim, latent_dim, n_hidden + 2)  # incluye los dos extremos
    hidden_dims = [round(d) for d in dims[1:-1]]              # descarta los extremos (input/latent_dim)
    return hidden_dims


def save_csv(rows: List[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    print(f"\nSaved {len(rows)} rows to {path}")


# ------------------------------ Data loading & windowing ---------------------------------

def load_and_prepare(csv_path: Path) -> Dict[str, pd.DataFrame]:
    """Load the CSV, drop duplicate readings, and split into one chronologically
    ascending per-location stream (the file is stored newest-row-first)."""
    df = pd.read_csv(csv_path)
    dedup_cols = ["room_id/id", "noted_date", "temp", "out/in", "Month"]
    df = df.drop_duplicates(subset=dedup_cols)
    df["noted_date"] = pd.to_datetime(df["noted_date"], format="%d-%m-%Y %H:%M")

    result = {}
    for location in LOCATIONS:
        loc_df = df[df["out/in"] == location].sort_values("noted_date")
        result[location] = loc_df[["temp"]].reset_index(drop=True)
    return result


def chronological_pools(df_loc: pd.DataFrame, train_frac: float = TRAIN_FRAC, val_frac: float = VAL_FRAC) -> Dict[str, np.ndarray]:
    """Split one location's ascending-time temperature stream into
    train/val/test pools by row position (chronological, no leakage)."""
    n = len(df_loc)
    n_train = round(train_frac * n)
    n_val = round(val_frac * n)

    temps = df_loc["temp"].to_numpy(dtype=np.float32)

    return {
        "train": temps[:n_train],
        "val": temps[n_train:n_train + n_val],
        "test": temps[n_train + n_val:],
    }


def global_minmax(pools: Dict[str, Dict[str, np.ndarray]]) -> Tuple[float, float]:
    """Single scalar (min, max), fit only on the pooled TRAIN split of both locations."""
    train_temps = np.concatenate([pools[loc]["train"] for loc in LOCATIONS])
    return float(train_temps.min()), float(train_temps.max())


def scale(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((x - lo) / (hi - lo), 0.0, 1.0)


def build_windows(temps: np.ndarray, window_size: int, strides: List[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Raw windows of `window_size` consecutive readings. 
    The window's label is the overheating flag: its OWN last reading is compared against the
    mean/std of its OWN preceding window_size-1 readings (its local baseline) via a one-sided z-score; 
    label=1 if that reading sits more than K_SIGMA standard deviations ABOVE its own recent baseline (Shewhart-style control-chart rule).
     The baseline excludes the reading being judged, so it never contaminates its own reference."""
    if window_size >= len(temps):
        raise ValueError(f"window_size={window_size} too large for a pool of {len(temps)} readings")

    starts = get_window_starts(len(temps), window_size, strides)
    windows = np.empty((len(starts), window_size), dtype=np.float32)
    window_labels = np.empty(len(starts), dtype=np.int64)
    for i, start in enumerate(starts):
        window = read_window(temps, start, window_size)
        windows[i] = window
        # Here is where the overheating flag is calculated
        baseline, current = window[:-1], window[-1]
        z = (current - baseline.mean()) / (baseline.std() + Z_SCORE_EPSILON)
        window_labels[i] = int(z > K_SIGMA)
    return windows, window_labels


def build_pooled_windows(pools: Dict[str, Dict[str, np.ndarray]], split: str, window_size: int, strides: List[int]) -> Tuple[np.ndarray, np.ndarray, Dict[str, int]]:
    """build_windows() per location for `split`, pooled together (windows
    never mix locations, matching a real sensor fixed at one location)."""
    all_windows, all_labels, counts = [], [], {}
    for location in LOCATIONS:
        temps = pools[location][split]
        windows, window_labels = build_windows(temps, window_size, strides)
        all_windows.append(windows)
        all_labels.append(window_labels)
        counts[location.lower()] = len(windows)
    return np.concatenate(all_windows), np.concatenate(all_labels), counts


# ------------------------------ AE training & encoding ---------------------------------

def train_ae_config(X_train: np.ndarray, X_val: np.ndarray, input_dim: int, latent_dim: int, n_layers: int, asymmetric: bool, epochs: int, patience: int, seed: int) -> AsymmetricAutoencoder:
    """Train one AE config on globally-scaled raw temperature windows."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    if not asymmetric:
        encoder_hidden_layers = decoder_hidden_layers = n_layers
    else:
        encoder_hidden_layers = (n_layers - 1) // 2 if n_layers > 0 else 0
        decoder_hidden_layers = n_layers - encoder_hidden_layers

    encoder = Encoder(
        input_dim=input_dim, latent_dim=latent_dim,
        hidden_dims=build_hidden_layers(input_dim, latent_dim, encoder_hidden_layers),
        activation=ACTIVATION_FUNCTION, latent_activation=LATENT_ACTIVATION,
    )
    decoder = Decoder(
        latent_dim=latent_dim, output_dim=input_dim,
        hidden_dims=build_hidden_layers(latent_dim, input_dim, decoder_hidden_layers),
        activation=ACTIVATION_FUNCTION, output_activation=DECODER_OUTPUT_ACTIVATION,
    )
    model = AsymmetricAutoencoder(encoder, decoder).to(DEVICE)
    optimizer = Adam(model.parameters(), lr=2e-3)
    cfg = TrainConfig(
        epochs=epochs, loss="mse", early_stopping_patience=patience,
        early_stopping_min_delta=1e-7, log_every=0, device=str(DEVICE),
    )

    train_loader = DataLoader(TensorDataset(torch.tensor(X_train, dtype=torch.float32)), batch_size=128, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.tensor(X_val, dtype=torch.float32)), batch_size=256)

    train(model, train_loader, optimizer, cfg, val_loader)
    return model


def encode(model: AsymmetricAutoencoder, X_scaled: np.ndarray) -> np.ndarray:
    """Run only the encoder half """
    model.eval()
    with torch.no_grad():
        z = model.encode(torch.tensor(X_scaled, dtype=torch.float32).to(DEVICE))
    return z.cpu().numpy()


def encoder_decoder_complexity(model: AsymmetricAutoencoder, input_dim: int, latent_dim: int) -> Dict[str, float]:
    """Architecture-only cost figures (independent of trained weights):
    should come out identical across every seed for a fixed
    (input_dim, latent_dim, symmetric) combination."""
    return {
        "encoder_params": sum(p.numel() for p in model.encoder.parameters()),
        "decoder_params": sum(p.numel() for p in model.decoder.parameters()),
        "encoder_macs": profile_macs(model.encoder, torch.randn(1, input_dim).to(DEVICE)),
        "decoder_macs": profile_macs(model.decoder, torch.randn(1, latent_dim).to(DEVICE)),
    }


# ------------------------------ Classifier & BentoML glue ---------------------------------

def feature_columns(width: int) -> List[str]:
    """Column-naming convention -- must stay textually identical to the one
    used in Datadriven/bentoml_service/service.py's predict()."""
    return [f"v{i}" for i in range(width)]


def train_classifier(X: np.ndarray, y: np.ndarray, seed: int) -> RandomForestClassifier:
    """Train random forest classifier."""
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(pd.DataFrame(X, columns=feature_columns(X.shape[1])), y)
    return clf


def bento_save_and_runner(model: RandomForestClassifier, tag: str):
    """Saves the trained model and run it for the inferetions"""
    bentoml.sklearn.save_model(tag, model)
    runner = bentoml.sklearn.get(f"{tag}:latest").to_runner()
    runner.init_local()
    return runner


def bento_predict(runner, X: np.ndarray) -> np.ndarray:
    """Predicts with test data by using the previously trained and loaded runner"""
    df = pd.DataFrame(X, columns=feature_columns(X.shape[1]))
    return np.asarray(runner.predict.run(df))


def json_payload_bytes_mean(X: np.ndarray, decimals: int = JSON_FLOAT_DECIMALS) -> float:
    """Mean size, in bytes, of the {'values': [...]} JSON payload that would
    be POSTed to the BentoML /predict endpoint for each row."""
    sizes = [
        len(json.dumps({"values": [round(float(v), decimals) for v in row]}).encode("utf-8"))
        for row in X
    ]
    return float(np.mean(sizes))


def positive_class_proba(clf: RandomForestClassifier, X: np.ndarray) -> Optional[np.ndarray]:
    """None if class 1 isn't present in clf.classes_ -- can happen on tiny
    or heavily imbalanced folds (e.g. --smoke-test)."""
    classes = list(clf.classes_)
    if 1 not in classes:
        return None
    df = pd.DataFrame(X, columns=feature_columns(X.shape[1]))
    return clf.predict_proba(df)[:, classes.index(1)]


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray]) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None and len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    else:
        metrics["roc_auc"] = float("nan")  # e.g. a test fold with no positives
    return metrics


# ------------------------------ Sweep ---------------------------------

def run_one_input_dim(pools: Dict[str, Dict[str, np.ndarray]], g_min: float, g_max: float,
                       input_dim: int, latent_dims: List[int],
                       target_ratio_by_latent: Optional[Dict[int, float]] = None) -> List[dict]:
    """Runs the `raw` scenario once per seed, and the `latent` scenario for
    every (latent_dim, asymmetric) combo in `latent_dims`, for this
    input_dim, across all N_ITERATIONS seeds. `target_ratio_by_latent`
    optionally records which target compression ratio produced each
    latent_dim (ratio sweep only) -- the realized ratio input_dim/latent_dim
    is always recorded too, since rounding can make them differ slightly."""
    target_ratio_by_latent = target_ratio_by_latent or {}
    rows: List[dict] = []

    for seed in range(N_ITERATIONS):
        np.random.seed(seed)
        X_tr, y_tr, tr_cnt = build_pooled_windows(pools, "train", input_dim, TRAIN_STRIDE)
        X_va, _, _ = build_pooled_windows(pools, "val", input_dim, [input_dim])
        X_te, y_te, te_cnt = build_pooled_windows(pools, "test", input_dim, [input_dim])
        test_pos_rate = float(y_te.mean()) if len(y_te) else float("nan")

        base_row = {
            "input_dim": input_dim, "seed": seed,
            "n_train": len(X_tr), "n_train_in": tr_cnt["in"], "n_train_out": tr_cnt["out"],
            "n_val": len(X_va),
            "n_test": len(X_te), "n_test_in": te_cnt["in"], "n_test_out": te_cnt["out"],
            "test_pos_rate": test_pos_rate,
        }

        # ---- raw scenario: once per (input_dim, seed) -- independent of latent_dim ----
        # Trains bentoml with raw splitted windows
        t0 = time.time()
        raw_clf = train_classifier(X_tr, y_tr, seed)
        raw_runner = bento_save_and_runner(raw_clf, f"{BENTO_TAG_PREFIX}_raw_{input_dim}_{seed}")
        raw_preds = bento_predict(raw_runner, X_te)
        raw_metrics = classification_metrics(y_te, raw_preds, positive_class_proba(raw_clf, X_te))
        rows.append({
            **base_row, "scenario": "raw",
            "latent_dim": None, "symmetric": None, "hidden_layers": None,
            "target_ratio": None, "compression_ratio": None,
            **raw_metrics, "agreement_with_raw": None,
            "payload_bytes_theoretical": input_dim * 4,
            "payload_bytes_json_mean": json_payload_bytes_mean(X_te),
            "encoder_params": None, "encoder_macs": None, "decoder_params": None, "decoder_macs": None,
        })
        print(f"[input_dim={input_dim} seed={seed}] raw: acc={raw_metrics['accuracy']:.3f} "
              f"f1={raw_metrics['f1']:.3f} ({time.time() - t0:.1f}s)")
        # min max scale for the autoencoder
        X_tr_s = scale(X_tr, g_min, g_max)
        X_va_s = scale(X_va, g_min, g_max)
        X_te_s = scale(X_te, g_min, g_max)

        for latent_dim in latent_dims:
            if latent_dim >= input_dim or latent_dim < 1:
                continue
            for asymmetric in ASYMMETRIC_OPTIONS:
                t1 = time.time()
                # Trains ae model
                ae = train_ae_config(X_tr_s, X_va_s, input_dim, latent_dim, HIDDEN_LAYERS,
                                      asymmetric, EPOCHS, PATIENCE, seed)
                # We need to encode also train data in order to train the bentoml
                Z_tr, Z_te = encode(ae, X_tr_s), encode(ae, X_te_s)
                # Train and save bentoml model
                ae_clf = train_classifier(Z_tr, y_tr, seed)
                arch = "asym" if asymmetric else "sym"
                ae_runner = bento_save_and_runner(
                    ae_clf, f"{BENTO_TAG_PREFIX}_{arch}_{input_dim}_{latent_dim}_{seed}"
                )
                ae_preds = bento_predict(ae_runner, Z_te)
                ae_metrics = classification_metrics(y_te, ae_preds, positive_class_proba(ae_clf, Z_te))
                agreement = float(np.mean(ae_preds == raw_preds))
                rows.append({
                    **base_row, "scenario": "latent",
                    "latent_dim": latent_dim, "symmetric": not asymmetric, "hidden_layers": HIDDEN_LAYERS,
                    "target_ratio": target_ratio_by_latent.get(latent_dim),
                    "compression_ratio": input_dim / latent_dim,
                    **ae_metrics, "agreement_with_raw": agreement,
                    "payload_bytes_theoretical": latent_dim * 4,
                    "payload_bytes_json_mean": json_payload_bytes_mean(Z_te),
                    **encoder_decoder_complexity(ae, input_dim, latent_dim),
                })
                print(f"[input_dim={input_dim} seed={seed}] latent_dim={latent_dim} {arch}: "
                      f"acc={ae_metrics['accuracy']:.3f} f1={ae_metrics['f1']:.3f} "
                      f"agree={agreement:.3f} ({time.time() - t1:.1f}s)")

    return rows


def run_ratio_sweep(pools: Dict[str, Dict[str, np.ndarray]], g_min: float, g_max: float) -> None:
    """Test 1: window size in INPUT_DIMS_RATIO_SWEEP, at each target
    compression ratio in COMPRESSION_RATIOS (latent_dim derived per pair)."""
    print("=" * 70)
    print(f"RATIO SWEEP -- input_dim in {INPUT_DIMS_RATIO_SWEEP}, ratio in {COMPRESSION_RATIOS}")
    print("=" * 70)

    rows: List[dict] = []
    for input_dim in INPUT_DIMS_RATIO_SWEEP:
        target_ratio_by_latent: Dict[int, float] = {}
        for ratio in COMPRESSION_RATIOS:
            latent_dim = max(1, round(input_dim / ratio))
            target_ratio_by_latent.setdefault(latent_dim, ratio)  # first ratio wins on rounding collisions
        # Runs one input dim for each combination of input dim and ratio
        rows.extend(run_one_input_dim(pools, g_min, g_max, input_dim, list(target_ratio_by_latent), target_ratio_by_latent))

    out_name = "iiot_ratio_sweep_smoketest.csv" if SMOKE_TEST else "iiot_ratio_sweep.csv"
    save_csv(rows, RESULTS_DIR / out_name)


def run_latent_sweep(pools: Dict[str, Dict[str, np.ndarray]], g_min: float, g_max: float) -> None:
    """Test 2: fixed window size (INPUT_DIM_LATENT_SWEEP), latent_dim swept
    directly over LATENT_DIMS_LATENT_SWEEP."""
    print("=" * 70)
    print(f"LATENT SWEEP -- input_dim={INPUT_DIM_LATENT_SWEEP}, latent_dim in {LATENT_DIMS_LATENT_SWEEP}")
    print("=" * 70)

    rows = run_one_input_dim(pools, g_min, g_max, INPUT_DIM_LATENT_SWEEP, LATENT_DIMS_LATENT_SWEEP)

    out_name = "iiot_latent_sweep_smoketest.csv" if SMOKE_TEST else "iiot_latent_sweep.csv"
    save_csv(rows, RESULTS_DIR / out_name)


def apply_smoke_test_overrides() -> None:
    """Shrinks both sweeps to a ~1-2 minute run each (6 rows apiece) to
    sanity-check the pipeline end-to-end before committing to the real runs."""
    global INPUT_DIMS_RATIO_SWEEP, COMPRESSION_RATIOS, INPUT_DIM_LATENT_SWEEP, LATENT_DIMS_LATENT_SWEEP
    global N_ITERATIONS, EPOCHS, PATIENCE, TRAIN_STRIDE
    INPUT_DIMS_RATIO_SWEEP = [20]
    COMPRESSION_RATIOS = [5]
    INPUT_DIM_LATENT_SWEEP = 20
    LATENT_DIMS_LATENT_SWEEP = [4]
    N_ITERATIONS = 2
    EPOCHS, PATIENCE = 5, 3
    TRAIN_STRIDE = [7]


def main() -> None:
    global SMOKE_TEST
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny, fast run to sanity-check the pipeline end-to-end.")
    parser.add_argument("--sweep", choices=["ratio", "latent", "both"], default="both",
                         help="'ratio' = Test 1 (input_dim x compression ratio), "
                              "'latent' = Test 2 (fixed input_dim x latent_dim), "
                              "'both' = run both (default).")
    args = parser.parse_args()
    SMOKE_TEST = args.smoke_test
    if SMOKE_TEST:
        apply_smoke_test_overrides()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    pools = {loc: chronological_pools(df) for loc, df in load_and_prepare(IIOT_CSV_PATH).items()}
    g_min, g_max = global_minmax(pools)
    print(f"Global temperature range used for AE scaling: [{g_min}, {g_max}]")

    if args.sweep in ("ratio", "both"):
        run_ratio_sweep(pools, g_min, g_max)
    if args.sweep in ("latent", "both"):
        run_latent_sweep(pools, g_min, g_max)


if __name__ == "__main__":
    main()
