import os
import numpy as np
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split

DATA_DIR = r"processed_data"


def evaluate_task1_baseline():
    """Baseline für Aufgabe 1: Einfacher Nächster-Nachbar / Rule Classifier auf Delta P."""
    print("\n==========================================")
    print("AUFGABE 1: Baseline Evaluation (Schaltevents)")
    print("==========================================")

    data_path = os.path.join(DATA_DIR, "task1_events.npz")
    if not os.path.exists(data_path):
        print("Datensatz nicht gefunden!")
        return

    data = np.load(data_path)
    X, y = data["X"], data["y"]

    # Simple Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Rule Baseline: Klassifikation basierend auf dem Mittelwert des Sprungs
    # Wir berechnen den Mittelwert der Trajektorie pro Klasse im Trainingssatz
    classes = np.unique(y_train)
    class_means = {
        c: np.mean(X_train[y_train == c], axis=0) for c in classes
    }

    # Vorhersage: Welcher Klassenmittelwert ist dem Testbeispiel am nächsten (Euklidischer Abstand)?
    y_pred = []
    for sample in X_test:
        distances = {
            c: np.linalg.norm(sample - class_means[c]) for c in classes
        }
        best_class = min(distances, key=distances.get)
        y_pred.append(best_class)

    y_pred = np.array(y_pred)

    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print(f"Baseline Macro F1-Score: {macro_f1:.4f}\n")
    print("Detaillierter Bericht:")
    print(classification_report(y_test, y_pred))


def evaluate_task2_baseline():
    """Baseline für Aufgabe 2: Z-Score / Moving Average Anomaly Detection."""
    print("\n==========================================")
    print("AUFGABE 2: Baseline Evaluation (Profil-Anomalien)")
    print("==========================================")

    data_path = os.path.join(DATA_DIR, "task2_profiles.npz")
    if not os.path.exists(data_path):
        print("Datensatz nicht gefunden!")
        return

    data = np.load(data_path)
    X = data["X"]

    # Berechne Mittelwert und Standardabweichung über den Trainingsanteil
    train_size = int(len(X) * 0.8)
    X_train = X[:train_size]
    X_val = X[train_size:]

    mean_profile = np.mean(X_train, axis=0)
    std_profile = np.std(X_train, axis=0) + 1e-8

    # Z-Score Fehlerberechnung auf Validierungsdaten
    z_scores = np.abs((X_val - mean_profile) / std_profile)
    anomaly_scores = np.mean(z_scores, axis=1)

    threshold = np.percentile(anomaly_scores, 95)
    print(f"Normal-Profil Referenz erstellt aus {len(X_train):,} Fenstern.")
    print(f"Z-Score Anomalie-Schwelle (95. Perzentil): {threshold:.4f}")
    print(
        f"Falsch-Alarm-Rate auf Validierungsdaten: {np.mean(anomaly_scores > threshold)*100:.2f}%"
    )


if __name__ == "__main__":
    evaluate_task1_baseline()
    evaluate_task2_baseline()