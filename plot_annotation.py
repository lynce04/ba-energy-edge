import os
import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = r"processed_data"
REFIT_DIR = r"CLEAN_REFIT_081116"


def plot_daily_profile_with_annotations(house_id=1, date_str="2014-03-10"):
    """Visualisiert das Tagesprofil eines Haushalts und hebt auffällige Uhrzeiten hervor."""
    file_path = os.path.join(REFIT_DIR, f"CLEAN_House{house_id}.csv")
    if not os.path.exists(file_path):
        print("REFIT-Datei nicht gefunden!")
        return

    print(f"Lade Daten für House {house_id} am {date_str}...")
    df = pd.read_csv(file_path)
    df["Time"] = pd.to_datetime(df["Time"])

    # Filtere auf den gewünschten Tag
    day_df = df[df["Time"].dt.strftime("%Y-%m-%d") == date_str].copy()

    if day_df.empty:
        print(f"Keine Daten für das Datum {date_str} gefunden.")
        return

    # Visualisierung erstellen
    plt.figure(figsize=(14, 6))
    plt.plot(
        day_df["Time"],
        day_df["Aggregate"],
        label="Gesamtleistung P(t) [Watt]",
        color="#1f77b4",
        alpha=0.8,
    )

    # Betreuer-Fokus: Spezifische Uhrzeiten hervorheben (z. B. Nachts um 02:00 vs. Morgens um 09:00)
    time_02am = pd.to_datetime(f"{date_str} 02:00:00")
    time_09am = pd.to_datetime(f"{date_str} 09:00:00")

    plt.axvline(
        x=time_02am,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label="02:00 Uhr (Nacht-Fokus / Anomalie-Prüfung)",
    )
    plt.axvline(
        x=time_09am,
        color="green",
        linestyle="--",
        linewidth=1.5,
        label="09:00 Uhr (Morgen-Fokus / Echte Nutzung)",
    )

    plt.title(
        f"Lastprofil Visualisierung & Annotation - House {house_id} ({date_str})"
    )
    plt.xlabel("Uhrzeit")
    plt.ylabel("Wirkleistung [Watt]")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right")
    plt.tight_layout()

    # Grafik speichern für die Bachelorarbeit / Präsentation
    output_img = f"plot_house{house_id}_{date_str}.png"
    plt.savefig(output_img, dpi=300)
    print(f"-> Grafik erfolgreich gespeichert unter: {output_img}")
    plt.show()


if __name__ == "__main__":
    # Test-Visualisierung für einen Beispiel-Tag
    plot_daily_profile_with_annotations(house_id=1, date_str="2014-03-10")