import os
import numpy as np
import pandas as pd

DATA_DIR = r"CLEAN_REFIT_081116"

# Namenszuordnung für House 1 laut REFIT Metadata
HOUSE1_MAPPING = {
    "Appliance1": "Fridge-Freezer",
    "Appliance2": "Freezer",
    "Appliance3": "Computer",
    "Appliance4": "TV",
    "Appliance5": "Hi-Fi",
    "Appliance6": "Aerosol",
    "Appliance7": "N/A",
    "Appliance8": "N/A",
    "Appliance9": "N/A",
}


def extract_events_with_labels(
    df, delta_threshold=100, window_before=3, window_after=3
):
    """Extrahiert Leistungssprünge und ordnet sie dem verursachenden Einzelgerät zu."""
    agg = df["Aggregate"].values
    diff_agg = np.diff(agg, prepend=agg[0])

    # Einzelgeräte-Differenzen berechnen
    app_cols = [c for c in df.columns if c.startswith("Appliance")]
    app_diffs = {col: np.diff(df[col].values, prepend=df[col].values[0]) for col in app_cols}

    event_indices = np.where(np.abs(diff_agg) >= delta_threshold)[0]

    windows = []
    labels_appliance = []
    labels_action = []  # ON oder OFF

    total_len = len(agg)
    for idx in event_indices:
        start = idx - window_before
        end = idx + window_after + 1

        if start >= 0 and end <= total_len:
            # Welches Gerät hat im selben Zeitschritt den größten Sprung gemacht?
            best_app = None
            max_change = 0

            for col in app_cols:
                change = app_diffs[col][idx]
                if abs(change) > max_change and abs(change) >= (delta_threshold * 0.5):
                    max_change = abs(change)
                    best_app = col

            if best_app is not None:
                windows.append(agg[start:end])
                device_name = HOUSE1_MAPPING.get(best_app, best_app)
                action = "EIN" if app_diffs[best_app][idx] > 0 else "AUS"

                labels_appliance.append(device_name)
                labels_action.append(f"{device_name}_{action}")

    windows = np.array(windows)
    df_labels = pd.DataFrame(
        {"Appliance": labels_appliance, "Action": labels_action}
    )

    print("\n  Extrahierte Events mit eindeutiger Zuordnung:")
    print(df_labels["Action"].value_counts())

    return windows, df_labels


if __name__ == "__main__":
    file_path = os.path.join(DATA_DIR, "CLEAN_House1.csv")
    if os.path.exists(file_path):
        print("Lade House 1 für Label-Zuordnung...")
        df1 = pd.read_csv(file_path)

        windows, labels = extract_events_with_labels(
            df1, delta_threshold=100
        )