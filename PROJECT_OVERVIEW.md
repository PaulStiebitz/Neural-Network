# Neural Network from Scratch - Projektübersicht

## Projektbeschreibung

Dieses Projekt implementiert ein neuronales Netzwerk von Grund auf (from scratch), das auf der MNIST-Datenbank trainiert wird. Das Ziel ist es, handgeschriebene Ziffern (0-9) aus Bilddaten zu erkennen. Die Implementierung erfolgt weitgehend manuell, wobei nur grundlegende Bibliotheken wie PyTorch für Ableitungsberechnungen erlaubt sind.

## Ordnerstruktur

```
Neural_Network/
│
├── data/
│   ├── raw/                    # Rohe MNIST-Daten
│   ├── processed/              # Vorverarbeitete Daten
│   └── data_loader.py          # Datenlade- und Vorverarbeitungsfunktionen
│
├── models/
│   ├── neural_network.py       # Hauptklasse des neuronalen Netzwerks
│   ├── layers.py               # Implementierung verschiedener Layer-Typen
│   └── activations.py          # Aktivierungsfunktionen (ReLU, Sigmoid, etc.)
│
├── training/
│   ├── trainer.py              # Trainingslogik und -schleife
│   ├── optimizer.py            # Optimierungsalgorithmen (SGD, Adam, etc.)
│   └── loss_functions.py       # Verlustfunktionen (Cross-Entropy, MSE, etc.)
│
├── utils/
│   ├── metrics.py              # Evaluierungsmetriken (Accuracy, Precision, etc.)
│   ├── visualization.py        # Visualisierungsfunktionen für Training und Ergebnisse
│   └── helpers.py              # Hilfsfunktionen
│
├── experiments/
│   ├── configs/                # Konfigurationsdateien für verschiedene Experimente
│   └── results/                # Gespeicherte Modelle und Trainingsergebnisse
│
├── notebooks/
│   ├── exploration.ipynb       # Datenexploration und -analyse
│   └── training_demo.ipynb     # Demonstrations-Notebook für Training
│
├── tests/
│   ├── test_layers.py          # Unit-Tests für Layer
│   ├── test_activations.py     # Unit-Tests für Aktivierungsfunktionen
│   └── test_training.py        # Tests für Trainingslogik
│
├── main.py                     # Haupteinstiegspunkt für Training
├── requirements.txt            # Python-Abhängigkeiten
└── README.md                   # Projektdokumentation
```

## Komponenten und ihre Aufgaben

### 1. Data (`data/`)

**Zweck:** Verwaltung und Vorverarbeitung der MNIST-Bilddaten

- **`raw/`**: Speichert die originalen MNIST-Datensätze (Trainings- und Testbilder)
- **`processed/`**: Enthält vorverarbeitete und normalisierte Daten für effizientes Training
- **`data_loader.py`**: 
  - Lädt MNIST-Daten herunter oder aus lokalen Dateien
  - Implementiert Daten-Normalisierung (z.B. Pixel-Werte von 0-255 auf 0-1)
  - Erstellt Batches für das Training
  - Implementiert Data Augmentation (optional)

### 2. Models (`models/`)

**Zweck:** Kernimplementierung des neuronalen Netzwerks

- **`neural_network.py`**: 
  - Hauptklasse, die das gesamte Netzwerk zusammenfügt
  - Verwaltet Forward- und Backward-Pass
  - Koordiniert die verschiedenen Layer
  
- **`layers.py`**: 
  - **Dense/Fully Connected Layer**: Implementiert Matrixmultiplikation und Bias
  - **Convolutional Layer** (optional): Für fortgeschrittene Bildverarbeitung
  - Jeder Layer implementiert:
    - `forward()`: Berechnung der Ausgabe
    - `backward()`: Berechnung der Gradienten (Backpropagation)
    - Parameter-Updates
    
- **`activations.py`**: 
  - **ReLU**: Rectified Linear Unit für versteckte Layer
  - **Sigmoid**: Für binäre Klassifikation
  - **Softmax**: Für Multi-Klassen-Klassifikation (MNIST: 10 Klassen)
  - **Tanh**: Alternative Aktivierungsfunktion
  - Jede Funktion implementiert Forward- und Backward-Pass

### 3. Training (`training/`)

**Zweck:** Trainingslogik und Optimierung

- **`trainer.py`**: 
  - Haupttrainingsschleife (Epochen und Batches)
  - Koordiniert Forward-Pass, Loss-Berechnung und Backward-Pass
  - Implementiert Validierung während des Trainings
  - Speichert Checkpoints und beste Modelle
  - Logging von Trainingsmetriken
  
- **`optimizer.py`**: 
  - **SGD** (Stochastic Gradient Descent): Basis-Optimierer
  - **SGD mit Momentum**: Verbesserte Konvergenz
  - **Adam**: Adaptiver Lernraten-Optimierer
  - Implementiert Parameter-Updates basierend auf Gradienten
  
- **`loss_functions.py`**: 
  - **Cross-Entropy Loss**: Standard für Klassifikation
  - **Mean Squared Error**: Alternative Verlustfunktion
  - Berechnet Loss und dessen Gradienten

### 4. Utils (`utils/`)

**Zweck:** Hilfsfunktionen und Evaluierung

- **`metrics.py`**: 
  - **Accuracy**: Prozentsatz korrekter Vorhersagen
  - **Precision/Recall**: Detaillierte Klassifikationsmetriken
  - **Confusion Matrix**: Visualisierung der Klassifikationsergebnisse
  - **F1-Score**: Harmonisches Mittel von Precision und Recall
  
- **`visualization.py`**: 
  - Plottet Trainings- und Validierungsverlust
  - Visualisiert Accuracy über Epochen
  - Zeigt Beispielbilder mit Vorhersagen
  - Visualisiert Gewichte und Aktivierungen
  
- **`helpers.py`**: 
  - Seed-Setting für Reproduzierbarkeit
  - Datei-I/O-Funktionen
  - Konfigurationsmanagement

### 5. Experiments (`experiments/`)

**Zweck:** Verwaltung verschiedener Experimente und Ergebnisse

- **`configs/`**: 
  - JSON/YAML-Dateien mit Hyperparametern
  - Verschiedene Netzwerkarchitekturen
  - Trainingseinstellungen
  
- **`results/`**: 
  - Gespeicherte Modellgewichte
  - Trainingshistorie (Loss, Accuracy)
  - Evaluierungsergebnisse
  - Visualisierungen

### 6. Notebooks (`notebooks/`)

**Zweck:** Interaktive Exploration und Demonstration

- **`exploration.ipynb`**: 
  - Analyse der MNIST-Daten
  - Visualisierung von Beispielbildern
  - Statistiken über die Datenverteilung
  
- **`training_demo.ipynb`**: 
  - Schritt-für-Schritt-Training
  - Interaktive Visualisierung des Lernprozesses
  - Experimentieren mit Hyperparametern

### 7. Tests (`tests/`)

**Zweck:** Qualitätssicherung durch Unit-Tests

- **`test_layers.py`**: 
  - Testet Forward- und Backward-Pass der Layer
  - Überprüft Gradientenberechnungen
  - Validiert Dimensionen
  
- **`test_activations.py`**: 
  - Testet Aktivierungsfunktionen
  - Überprüft mathematische Korrektheit
  
- **`test_training.py`**: 
  - Testet Trainingslogik
  - Überprüft Konvergenz auf einfachen Beispielen

### 8. Hauptdateien

- **`main.py`**: 
  - Einstiegspunkt für das Training
  - Lädt Konfiguration
  - Initialisiert Modell, Daten und Trainer
  - Startet Training und Evaluierung
  
- **`requirements.txt`**: 
  - Liste aller Python-Abhängigkeiten
  - Versionen für Reproduzierbarkeit
  
- **`README.md`**: 
  - Projektbeschreibung
  - Installation und Setup
  - Verwendungsbeispiele
  - Ergebnisse und Benchmarks

## Implementierungsreihenfolge (Empfohlen)

1. **Datenverarbeitung**: `data_loader.py` - MNIST laden und vorbereiten
2. **Aktivierungsfunktionen**: `activations.py` - Grundlegende Funktionen
3. **Layer**: `layers.py` - Dense Layer mit Forward/Backward
4. **Verlustfunktionen**: `loss_functions.py` - Cross-Entropy
5. **Netzwerk**: `neural_network.py` - Zusammenführung der Komponenten
6. **Optimierer**: `optimizer.py` - SGD implementieren
7. **Trainer**: `trainer.py` - Trainingsschleife
8. **Metriken**: `metrics.py` - Evaluierung
9. **Visualisierung**: `visualization.py` - Ergebnisse darstellen
10. **Main**: `main.py` - Alles zusammenführen

## Technische Details

### Erlaubte Bibliotheken
- **NumPy**: Für Matrixoperationen
- **PyTorch** (nur für Autograd): Automatische Ableitungsberechnung
- **Matplotlib/Seaborn**: Visualisierung
- **Pandas**: Datenmanipulation (optional)

### Nicht erlaubt
- Fertige neuronale Netzwerk-Implementierungen (torch.nn.Module, etc.)
- High-Level-APIs für Training

### Kernkonzepte

1. **Forward Propagation**: Eingabe → Layer 1 → Aktivierung → Layer 2 → ... → Ausgabe
2. **Backpropagation**: Berechnung der Gradienten durch Kettenregel
3. **Gradient Descent**: Aktualisierung der Gewichte basierend auf Gradienten
4. **Batch Processing**: Verarbeitung mehrerer Beispiele gleichzeitig
5. **Regularisierung**: Vermeidung von Overfitting (optional: Dropout, L2)

## Erwartete Ergebnisse

- **Trainingsgenauigkeit**: ~95-98% auf MNIST
- **Testgenauigkeit**: ~93-97% auf MNIST
- **Trainingszeit**: Abhängig von Hardware und Netzwerkgröße
- **Modellgröße**: Wenige MB für einfache Architekturen

## Nächste Schritte

1. Projektstruktur erstellen
2. Datenverarbeitung implementieren
3. Einfaches 2-Layer-Netzwerk bauen
4. Training auf kleinem Datensatz testen
5. Schrittweise erweitern und optimieren