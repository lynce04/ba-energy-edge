import os
import numpy as np
import pandas as pd

DATA_DIR = r"CLEAN_REFIT_081116"


def load_house_clean(house_id):
    """Lädt und bereinigt die Daten eines Haushaltes."""
    file_path = os.path.join(DATA_DIR, f"CLEAN_House{house_id}.csv")
    if not os.path.exists(file_path):
        print(f"Datei nicht gefunden: {file_path}")
        return None

    print(f"\n--- Lade House {house_id} ---")
    df = pd.read_csv(file_path)
    df["Time"] = pd.to_datetime(df["Time"])
    df = df.sort_values("Time").reset_index(drop=True)

    # Negative Werte / Ausreißer abfangen
    df["Aggregate"] = df["Aggregate"].clip(lower=0)
    return df


def extract_event_windows(
    df, delta_threshold=100, window_before=3, window_after=3
):
    """Aufgabe 1: Identifiziert Leistungssprünge (Delta P) im Aggregat und extrahiert Ereignis-Fenster.

    Window-Größe: (window_before + 1 + window_after) Messwerte (~8 Sek. pro Schritt)
    """
    agg = df["Aggregate"].values
    diff = np.diff(agg, prepend=agg[0])

    # Indizes mit signifikantem Leistungssprung
    event_indices = np.where(np.abs(diff) >= delta_threshold)[0]

    windows = []
    labels_delta = []

    total_len = len(agg)
    for idx in event_indices:
        start = idx - window_before
        end = idx + window_after + 1

        if start >= 0 and end <= total_len:
            windows.append(agg[start:end])
            labels_delta.append(diff[idx])

    windows = np.array(windows)
    labels_delta = np.array(labels_delta)

    print(f"  Ereigniserkennung (Delta P >= {delta_threshold} W):")
    print(f"    Gefundene Schaltevents: {len(windows):,}")
    print(f"    Fensterform: {windows.shape} (Events x Punkte)")

    return windows, labels_delta


def extract_profile_windows(
    df, window_minutes=60, step_minutes=15, aggregate_col="Aggregate"
):
    """Aufgabe 2: Erstellt rollierende 60-Minuten-Profilfenster auf 1-Minuten-Raster."""
    df_resampled = (
        df.set_index("Time")[aggregate_col].resample("1min").mean().fillna(0)
    )

    values = df_resampled.values
    window_size = window_minutes
    step_size = step_minutes

    windows = []
    for i in range(0, len(values) - window_size + 1, step_size):
        windows.append(values[i : i + window_size])

    windows = np.array(windows)
    print(f"  Profil-Analyse ({window_minutes} Min Fenster, {step_minutes} Min Schritt):")
    print(f"    Generierte Profil-Fenster: {windows.shape}")

    return windows


if __name__ == "__main__":
    # Test an House 1
    df1 = load_house_clean(1)
    if df1 is not None:
        # 1. Schaltevents extrahieren (z. B. Schwelle 100 Watt)
        event_win, deltas = extract_event_windows(df1, delta_threshold=100)

        # 2. Profilfenster extrahieren (60 Minuten)
        profile_win = extract_profile_windows(
            df1, window_minutes=60, step_minutes=15
        )