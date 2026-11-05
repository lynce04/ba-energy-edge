import os
import numpy as np
import pandas as pd

DATA_DIR = r"CLEAN_REFIT_081116"
OUTPUT_DIR = r"processed_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HOUSE_MAPPINGS = {
    1: {
        "Appliance1": "Fridge-Freezer",
        "Appliance2": "Freezer",
        "Appliance3": "Computer",
        "Appliance4": "TV",
        "Appliance5": "Hi-Fi",
    },
    10: {
        "Appliance1": "Magimix",
        "Appliance3": "Washing Machine",
        "Appliance4": "Chest Freezer",
    },
    16: {"Appliance1": "Fridge-Freezer", "Appliance2": "Electric Heater"},
    18: {"Appliance1": "Fridge-Freezer", "Appliance5": "Washer Dryer"},
    20: {"Appliance1": "Fridge-Freezer", "Appliance4": "Dishwasher"},
}


def build_event_dataset(house_ids=[1], delta_threshold=100):
    """Erstellt den Datensatz für Aufgabe 1: Ereigniserkennung (Short-Window ~56s)."""
    all_windows = []
    all_labels = []

    for h_id in house_ids:
        file_path = os.path.join(DATA_DIR, f"CLEAN_House{h_id}.csv")
        if not os.path.exists(file_path):
            continue

        print(f"Verarbeite Events aus House {h_id}...")
        df = pd.read_csv(file_path)
        agg = df["Aggregate"].values
        diff_agg = np.diff(agg, prepend=agg[0])

        mapping = HOUSE_MAPPINGS.get(h_id, {})
        valid_apps = [
            col
            for col in df.columns
            if col.startswith("Appliance") and col in mapping
        ]
        app_diffs = {
            col: np.diff(df[col].values, prepend=df[col].values[0])
            for col in valid_apps
        }

        event_indices = np.where(np.abs(diff_agg) >= delta_threshold)[0]

        for idx in event_indices:
            start, end = idx - 3, idx + 4
            if start >= 0 and end <= len(agg):
                best_app = None
                max_change = 0

                for col in valid_apps:
                    change = app_diffs[col][idx]
                    if abs(change) > max_change and abs(change) >= (
                        delta_threshold * 0.5
                    ):
                        max_change = abs(change)
                        best_app = col

                if best_app is not None:
                    device_name = mapping[best_app]
                    action = "EIN" if app_diffs[best_app][idx] > 0 else "AUS"
                    all_windows.append(agg[start:end])
                    all_labels.append(f"{device_name}_{action}")

    X = np.array(all_windows, dtype=np.float32)
    y = np.array(all_labels)

    # Z-Score Normalisierung pro Fenster
    X_mean = np.mean(X, axis=1, keepdims=True)
    X_std = np.std(X, axis=1, keepdims=True) + 1e-8
    X_norm = (X - X_mean) / X_std

    save_path = os.path.join(OUTPUT_DIR, "task1_events.npz")
    np.savez_compressed(save_path, X=X_norm, y=y)
    print(
        f"-> Aufgabe 1 Datensatz gespeichert: {X_norm.shape} | Labels: {len(y)}"
    )


def build_profile_dataset(house_ids=[1]):
    """Erstellt den Datensatz für Aufgabe 2: Profilanalyse (60-Minuten rollierend)."""
    all_profiles = []

    for h_id in house_ids:
        file_path = os.path.join(DATA_DIR, f"CLEAN_House{h_id}.csv")
        if not os.path.exists(file_path):
            continue

        print(f"Verarbeite Normal-Profile aus House {h_id}...")
        df = pd.read_csv(file_path)
        df["Time"] = pd.to_datetime(df["Time"])

        # Auf 1-Minuten-Raster verdichten
        df_resampled = (
            df.set_index("Time")["Aggregate"].resample("1min").mean().fillna(0)
        )
        values = df_resampled.values

        window_size = 60
        step_size = 15
        for i in range(0, len(values) - window_size + 1, step_size):
            all_profiles.append(values[i : i + window_size])

    X = np.array(all_profiles, dtype=np.float32)

    # Min-Max Normalisierung auf [0, 1] für Rekonstruktions-Modelle
    max_val = np.max(X) if np.max(X) > 0 else 1.0
    X_norm = X / max_val

    save_path = os.path.join(OUTPUT_DIR, "task2_profiles.npz")
    np.savez_compressed(save_path, X=X_norm, max_val=max_val)
    print(f"-> Aufgabe 2 Datensatz gespeichert: {X_norm.shape}")


if __name__ == "__main__":
    build_event_dataset(house_ids=[1])
    build_profile_dataset(house_ids=[1])