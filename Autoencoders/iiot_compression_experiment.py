import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import bentoml
import numpy as np
import pandas as pd
import torch
from sklearn.covariance import EllipticEnvelope
from sklearn.ensemble import IsolationForest, RandomForestClassifier
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
INPUT_DIM_LATENT_SWEEP = 100
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
# control-chart rule). K_SIGMA_SWEEP is processed in order by main(), one
# full results subdirectory per value (see run_ratio_sweep/run_latent_sweep
# output_dir). K_SIGMA itself is mutated per iteration -- read as a plain
# global inside build_windows(), same pattern as apply_smoke_test_overrides.
K_SIGMA_SWEEP = [2, 1.5]
K_SIGMA = K_SIGMA_SWEEP[0]
Z_SCORE_EPSILON = 1e-6             # guards baseline_std == 0 (a perfectly flat preceding window)

# ------------------------------ Classifier / detector parameters ---------------------------------
# Selected via --classifier {rf,if,ee} in main() (mutates CLASSIFIER_TYPE).
#   "rf": supervised RandomForestClassifier, trained on ALL train windows
#         (normal + anomalous) with their z-score labels.
#   "if": unsupervised IsolationForest, fit only on the NORMAL (label=0)
#         train windows -- labels are only used to size `contamination`,
#         never as a training target. Isolates points via random axis-aligned
#         splits -- doesn't account for correlation between dimensions, which
#         degrades on the "raw" scenario's 50-100 highly autocorrelated
#         consecutive-reading dimensions (see conversation).
#   "ee": unsupervised EllipticEnvelope (robust covariance / Mahalanobis
#         distance), also fit on NORMAL-only windows. Unlike "if", the
#         covariance matrix explicitly captures correlation between
#         dimensions, which should suit the "raw" scenario's redundant
#         dimensions better -- and it assumes local Gaussianity, the same
#         assumption K_SIGMA's z-score label itself already makes.
CLASSIFIER_TYPE = "rf"
UNSUPERVISED_CLASSIFIERS = {"if", "ee"}  # share the fit-on-normal-only / -1,1 predict() convention
RF_CLASS_WEIGHT = "balanced"       # compensates the imbalanced dataset
IF_CONTAMINATION_BOUNDS = (1e-3, 0.5)  # sklearn requires contamination in (0, 0.5]; shared by "if"/"ee"

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


def build_val_split(pools: Dict[str, Dict[str, np.ndarray]], window_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, int]]:
    """Builds the val windows (non-overlapping, like test), split into:
      - X_val_normal: label=0 windows, the only ones the AE ever trains on
        (its early stopping is unsupervised anyway, so val's labels were
        never used for anything else).
      - X_val_anomalous / y_val_anomalous: the label=1 windows, recycled
        into the test set by the caller -- test is where positives are
        scarcest, so these would otherwise go completely unused.
    Returns (X_val_normal, X_val_anomalous, y_val_anomalous, counts), counts
    being the total (normal+anomalous) window count per location."""
    normal_parts, anomalous_parts, anomalous_labels, counts = [], [], [], {}
    for location in LOCATIONS:
        windows, labels = build_windows(pools[location]["val"], window_size, [window_size])
        normal_parts.append(windows[labels == 0])
        anomalous_parts.append(windows[labels == 1])
        anomalous_labels.append(labels[labels == 1])
        counts[location.lower()] = len(windows)
    return (np.concatenate(normal_parts), np.concatenate(anomalous_parts), np.concatenate(anomalous_labels), counts)


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


def fit_detector(X: np.ndarray, y: np.ndarray, seed: int):
    """Trains the classifier/detector selected by CLASSIFIER_TYPE (set from
    --classifier in main()):
      - "rf": supervised RandomForestClassifier(class_weight=RF_CLASS_WEIGHT),
              trained on ALL of X/y (normal + anomalous).
      - "if"/"ee": unsupervised IsolationForest / EllipticEnvelope, fit only
              on the NORMAL (y==0) rows of X -- y is only used to size
              `contamination` (the proportion of outliers assumed), never as
              a training target."""
    columns = feature_columns(X.shape[1])
    if CLASSIFIER_TYPE in UNSUPERVISED_CLASSIFIERS:
        contamination = float(np.mean(y)) if len(y) else IF_CONTAMINATION_BOUNDS[0]
        contamination = min(max(contamination, IF_CONTAMINATION_BOUNDS[0]), IF_CONTAMINATION_BOUNDS[1])
        if CLASSIFIER_TYPE == "if":
            clf = IsolationForest(random_state=seed, contamination=contamination)
        else:
            clf = EllipticEnvelope(random_state=seed, contamination=contamination)
        clf.fit(pd.DataFrame(X[y == 0], columns=columns))
        return clf

    clf = RandomForestClassifier(random_state=seed, class_weight=RF_CLASS_WEIGHT)
    clf.fit(pd.DataFrame(X, columns=columns), y)
    return clf


def bento_save_and_runner(model, tag: str):
    """Saves the trained model and run it for the inferetions"""
    bentoml.sklearn.save_model(tag, model)
    runner = bentoml.sklearn.get(f"{tag}:latest").to_runner()
    runner.init_local()
    return runner


def bento_cleanup(tag: str) -> None:
    """Deletes `tag` from BentoML's local model store right after its one
    to ensure it does not crash due to lack of space"""
    try:
        bentoml.models.delete(tag)
    except Exception as exc:  # keep the sweep going even if cleanup itself fails
        print(f"  (warning: could not delete bentoml model '{tag}': {exc})")


def bento_predict(runner, X: np.ndarray) -> np.ndarray:
    """Predicts with test data by using the previously trained and loaded runner.
    IsolationForest/EllipticEnvelope's raw output is -1 (anomaly) / 1 (normal),
    the shared sklearn outlier-detector convention; normalized here to the
    same 0/1 (normal/anomaly) convention used everywhere else."""
    df = pd.DataFrame(X, columns=feature_columns(X.shape[1]))
    preds = np.asarray(runner.predict.run(df))
    if CLASSIFIER_TYPE in UNSUPERVISED_CLASSIFIERS:
        return (preds == -1).astype(np.int64)
    return preds


def json_payload_bytes_mean(X: np.ndarray, decimals: int = JSON_FLOAT_DECIMALS) -> float:
    """Mean size, in bytes, of the {'values': [...]} JSON payload that would
    be POSTed to the BentoML /predict endpoint for each row."""
    sizes = [
        len(json.dumps({"values": [round(float(v), decimals) for v in row]}).encode("utf-8"))
        for row in X
    ]
    return float(np.mean(sizes))


def positive_class_score(clf, X: np.ndarray) -> Optional[np.ndarray]:
    """Score used for ROC-AUC -- higher must mean 'more likely anomalous'.
    For "if"/"ee", decision_function() is higher for NORMAL points (shared
    sklearn outlier-detector convention), so it's negated. For "rf", None if
    class 1 isn't present in clf.classes_ -- can happen on tiny or heavily
    imbalanced folds (e.g. --smoke-test)."""
    df = pd.DataFrame(X, columns=feature_columns(X.shape[1]))
    if CLASSIFIER_TYPE in UNSUPERVISED_CLASSIFIERS:
        return -clf.decision_function(df)
    classes = list(clf.classes_)
    if 1 not in classes:
        return None
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

    tag_prefix = f"{BENTO_TAG_PREFIX}_{CLASSIFIER_TYPE}_k{str(K_SIGMA).replace('.', '_')}"

    for seed in range(N_ITERATIONS):
        np.random.seed(seed)
        X_tr, y_tr, tr_cnt = build_pooled_windows(pools, "train", input_dim, TRAIN_STRIDE)
        X_va_normal, X_va_recycled, y_va_recycled, _ = build_val_split(pools, input_dim)
        X_te, y_te, te_cnt = build_pooled_windows(pools, "test", input_dim, [input_dim])

        # Val's labels were never used for anything else (its only role is
        # unsupervised AE early stopping), so its anomalous windows are
        # recycled into test, which is where positives are scarcest.
        X_te = np.concatenate([X_te, X_va_recycled])
        y_te = np.concatenate([y_te, y_va_recycled])
        test_pos_rate = float(y_te.mean()) if len(y_te) else float("nan")

        base_row = {
            "classifier": CLASSIFIER_TYPE, "k_sigma": K_SIGMA,
            "input_dim": input_dim, "seed": seed,
            "n_train": len(X_tr), "n_train_in": tr_cnt["in"], "n_train_out": tr_cnt["out"],
            "n_val": len(X_va_normal),
            "n_test": len(X_te), "n_test_in": te_cnt["in"], "n_test_out": te_cnt["out"],
            "n_test_recycled_from_val": len(X_va_recycled),
            "test_pos_rate": test_pos_rate,
        }

        # ---- raw scenario: once per (input_dim, seed) -- independent of latent_dim ----
        # Trains bentoml with raw splitted windows (fit_detector filters to
        # normal-only internally for "if"; "rf" always sees the full set)
        t0 = time.time()
        raw_clf = fit_detector(X_tr, y_tr, seed)
        raw_tag = f"{tag_prefix}_raw_{input_dim}_{seed}"
        raw_runner = bento_save_and_runner(raw_clf, raw_tag)
        raw_preds = bento_predict(raw_runner, X_te)
        bento_cleanup(raw_tag)
        raw_metrics = classification_metrics(y_te, raw_preds, positive_class_score(raw_clf, X_te))
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

        # min max scale for the autoencoder. X_tr_s (full) feeds encode() to
        # get Z_tr for the downstream detector;
        # Select only normal subset for training in order to maximize detection of anomalies in AE
        # (X_tr_normal_s) actually trains the AE's weights, so it learns to
        # compress "normal" behaviour well and genuine overheating windows
        # fall in a worse-modelled, more separable region of latent space.
        X_tr_s = scale(X_tr, g_min, g_max)
        X_tr_normal_s = X_tr_s[y_tr == 0]
        X_va_normal_s = scale(X_va_normal, g_min, g_max)
        X_te_s = scale(X_te, g_min, g_max)

        for latent_dim in latent_dims:
            if latent_dim >= input_dim or latent_dim < 1:
                continue
            for asymmetric in ASYMMETRIC_OPTIONS:
                t1 = time.time()
                # Trains ae model -- normal-only train/val
                ae = train_ae_config(X_tr_normal_s, X_va_normal_s, input_dim, latent_dim, HIDDEN_LAYERS, asymmetric, EPOCHS, PATIENCE, seed)
                # Encode ALL train windows (normal + anomalous) so the
                # downstream detector can be fit/evaluated on labeled data
                Z_tr, Z_te = encode(ae, X_tr_s), encode(ae, X_te_s)
                ae_clf = fit_detector(Z_tr, y_tr, seed)
                arch = "asym" if asymmetric else "sym"
                ae_tag = f"{tag_prefix}_{arch}_{input_dim}_{latent_dim}_{seed}"
                ae_runner = bento_save_and_runner(ae_clf, ae_tag)
                ae_preds = bento_predict(ae_runner, Z_te)
                bento_cleanup(ae_tag)
                ae_metrics = classification_metrics(y_te, ae_preds, positive_class_score(ae_clf, Z_te))
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


def run_ratio_sweep(pools: Dict[str, Dict[str, np.ndarray]], g_min: float, g_max: float, output_dir: Path) -> None:
    """Test 1: window size in INPUT_DIMS_RATIO_SWEEP, at each target
    compression ratio in COMPRESSION_RATIOS (latent_dim derived per pair)."""
    print("=" * 70)
    print(f"RATIO SWEEP -- classifier={CLASSIFIER_TYPE} K_SIGMA={K_SIGMA} "
          f"input_dim in {INPUT_DIMS_RATIO_SWEEP}, ratio in {COMPRESSION_RATIOS}")
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
    save_csv(rows, output_dir / out_name)


def run_latent_sweep(pools: Dict[str, Dict[str, np.ndarray]], g_min: float, g_max: float, output_dir: Path) -> None:
    """Test 2: fixed window size (INPUT_DIM_LATENT_SWEEP), latent_dim swept
    directly over LATENT_DIMS_LATENT_SWEEP."""
    print("=" * 70)
    print(f"LATENT SWEEP -- classifier={CLASSIFIER_TYPE} K_SIGMA={K_SIGMA} "
          f"input_dim={INPUT_DIM_LATENT_SWEEP}, latent_dim in {LATENT_DIMS_LATENT_SWEEP}")
    print("=" * 70)

    rows = run_one_input_dim(pools, g_min, g_max, INPUT_DIM_LATENT_SWEEP, LATENT_DIMS_LATENT_SWEEP)

    out_name = "iiot_latent_sweep_smoketest.csv" if SMOKE_TEST else "iiot_latent_sweep.csv"
    save_csv(rows, output_dir / out_name)


def apply_smoke_test_overrides() -> None:
    """Shrinks both sweeps to a ~1-2 minute run each (6 rows apiece) to
    sanity-check the pipeline end-to-end before committing to the real runs.
    K_SIGMA_SWEEP and --classifier are left untouched so the smoke test still
    exercises the multi-k / multi-classifier looping and directory logic."""
    global INPUT_DIMS_RATIO_SWEEP, COMPRESSION_RATIOS, INPUT_DIM_LATENT_SWEEP, LATENT_DIMS_LATENT_SWEEP
    global N_ITERATIONS, EPOCHS, PATIENCE, TRAIN_STRIDE
    INPUT_DIMS_RATIO_SWEEP = [20]
    COMPRESSION_RATIOS = [5]
    INPUT_DIM_LATENT_SWEEP = 20
    LATENT_DIMS_LATENT_SWEEP = [4]
    N_ITERATIONS = 2
    EPOCHS, PATIENCE = 5, 3
    TRAIN_STRIDE = [7]


def k_sigma_dir_suffix(k: float) -> str:
    return str(k).replace(".", "_")


def main() -> None:
    global SMOKE_TEST, CLASSIFIER_TYPE, K_SIGMA
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Tiny, fast run to sanity-check the pipeline end-to-end.")
    parser.add_argument("--sweep", choices=["ratio", "latent", "both"], default="both",
                         help="'ratio' = Test 1 (input_dim x compression ratio), "
                              "'latent' = Test 2 (fixed input_dim x latent_dim), "
                              "'both' = run both (default).")
    parser.add_argument("--classifier", choices=["rf", "if", "ee"], default="rf",
                         help="'rf' = supervised RandomForestClassifier(class_weight=balanced). "
                              "'if' = unsupervised IsolationForest, fit on normal-only windows. "
                              "'ee' = unsupervised EllipticEnvelope (robust covariance / Mahalanobis "
                              "distance), also fit on normal-only windows (default: rf).")
    args = parser.parse_args()
    SMOKE_TEST = args.smoke_test
    CLASSIFIER_TYPE = args.classifier
    if SMOKE_TEST:
        apply_smoke_test_overrides()

    pools = {loc: chronological_pools(df) for loc, df in load_and_prepare(IIOT_CSV_PATH).items()}
    g_min, g_max = global_minmax(pools)
    print(f"Global temperature range used for AE scaling: [{g_min}, {g_max}]")

    # K_SIGMA_SWEEP is processed in order, each value fully (both requested
    # sweeps) before moving to the next, into its own results subdirectory --
    # pools/g_min/g_max don't depend on K_SIGMA so they're built only once.
    for k in K_SIGMA_SWEEP:
        K_SIGMA = k
        output_dir = RESULTS_DIR / f"results_{CLASSIFIER_TYPE}_k_{k_sigma_dir_suffix(k)}"
        output_dir.mkdir(parents=True, exist_ok=True)
        print("#" * 70)
        print(f"# classifier={CLASSIFIER_TYPE}  K_SIGMA={k}  ->  {output_dir}")
        print("#" * 70)

        if args.sweep in ("ratio", "both"):
            run_ratio_sweep(pools, g_min, g_max, output_dir)
        if args.sweep in ("latent", "both"):
            run_latent_sweep(pools, g_min, g_max, output_dir)


if __name__ == "__main__":
    main()
