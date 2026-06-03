# Neural Network Projekt - Struktur und Übersicht

## Projektbeschreibung
Dieses Projekt implementiert ein neuronales Netzwerk von Grund auf mit Python und NumPy. Es enthält vollständige Implementierungen von Backpropagation, verschiedenen Aktivierungsfunktionen (ReLU, Softmax), Loss-Funktionen und Optimierern.

## Dateistruktur

### Kernkomponenten

#### `neural_network.py`
Hauptklasse des neuronalen Netzwerks
- **NeuralNetwork**: Orchestriert alle Komponenten
  - Verwaltet Layer-Stack
  - Führt Forward/Backward Pass durch
  - Koordiniert Training und Evaluation

#### `layers.py`
Implementierung verschiedener Layer-Typen
- **DenseLayer**: Fully Connected Layer mit Gewichten und Bias
- **DropoutLayer**: Regularisierung durch zufälliges Deaktivieren von Neuronen
- **BatchNormalization**: Normalisierung für stabileres Training

#### `activation_functions.py`
Aktivierungsfunktionen mit Forward und Backward Pass
- **ReLU**: Rectified Linear Unit (f(x) = max(0, x))
- **Softmax**: Für Mehrklassen-Klassifikation (Ausgabeschicht)

#### `loss_functions.py`
Loss-Funktionen für verschiedene Aufgaben
- **CategoricalCrossentropy**: Für Klassifikation
- **SoftmaxCategoricalCrossentropy**: Kombinierte Softmax+Loss für Effizienz
- **MeanSquaredError**: Für Regression

#### `optimizers.py`
Verschiedene Optimierungsalgorithmen
- **SGD**: Stochastic Gradient Descent (mit Momentum)
- **Adam**: Adaptive Moment Estimation
- **RMSprop**: Root Mean Square Propagation

### Hilfsfunktionen

#### `utils.py`
Utility-Funktionen für Training und Evaluation
- Accuracy-Berechnung (Klassifikation und Regression)
- Daten-Preprocessing (Normalisierung, One-Hot Encoding)
- Train/Test Split
- Model Save/Load
- Confusion Matrix und Metriken (Precision, Recall, F1)

#### `data_loader.py`
Datenverarbeitung und -generierung
- **DataLoader**: Effizientes Batch-Loading
- **DataGenerator**: Synthetische Daten für Tests
- **ImageDataLoader**: Spezialisiert für Bilddaten

### Anwendung

#### `main.py`
Einstiegspunkt mit Beispielen
- Verschiedene Anwendungsbeispiele
- Demonstrations-Code für alle Features
- Interaktives Menü

## Implementierungsreihenfolge (Empfehlung)

### Phase 1: Grundlagen
1. **layers.py - DenseLayer**
   - Forward Pass: Matrix-Multiplikation
   - Backward Pass: Gradienten berechnen
   
2. **activation_functions.py - ReLU**
   - Forward: max(0, x)
   - Backward: Gradient = 1 für x > 0, sonst 0

3. **activation_functions.py - Softmax**
   - Forward: Exponential normalisiert
   - Backward: Jacobi-Matrix

### Phase 2: Loss und Optimierung
4. **loss_functions.py - CategoricalCrossentropy**
   - Forward: -sum(y_true * log(y_pred))
   - Backward: Gradient berechnen

5. **optimizers.py - SGD**
   - Einfachster Optimizer
   - Parameter Update: w -= lr * gradient

### Phase 3: Netzwerk
6. **neural_network.py - Basis**
   - Layer-Management
   - Forward Pass durch alle Layers
   - Backward Pass durch alle Layers

7. **neural_network.py - Training**
   - train_step implementieren
   - fit Methode mit Epochen und Batches

### Phase 4: Erweiterte Features
8. **optimizers.py - Adam**
   - Momentum und adaptive Learning Rate
   
9. **layers.py - Dropout**
   - Regularisierung implementieren

10. **layers.py - BatchNormalization**
    - Komplexeste Layer-Implementierung

### Phase 5: Utilities
11. **utils.py**
    - Hilfsfunktionen nach Bedarf

12. **data_loader.py**
    - DataLoader für effizientes Training

13. **main.py**
    - Beispiele und Tests

## Wichtige Konzepte

### Forward Pass
```
Input -> Dense -> ReLU -> Dense -> Softmax -> Output
```

### Backward Pass (Backpropagation)
```
Loss Gradient <- Dense <- ReLU <- Dense <- Softmax <- Loss
```

### Gradient Flow
- Jede Layer berechnet:
  - Gradient bezüglich ihrer Eingaben (für vorherige Layer)
  - Gradient bezüglich ihrer Parameter (für Optimizer)

### Parameter Update
```
weights = weights - learning_rate * gradient_weights
bias = bias - learning_rate * gradient_bias
```

## Testing-Strategie

1. **Unit Tests**: Jede Komponente einzeln testen
   - Forward Pass mit bekannten Eingaben
   - Backward Pass mit Gradient Checking
   
2. **Integration Tests**: Komponenten zusammen testen
   - Einfaches Netzwerk auf einfachen Daten
   - Überprüfen ob Loss sinkt
   
3. **End-to-End Tests**: Vollständiges Training
   - Spiral-Daten klassifizieren
   - Erwartete Accuracy erreichen

## Debugging-Tipps

1. **Gradient Checking**: Numerische vs. analytische Gradienten vergleichen
2. **Shape Debugging**: Print statements für Array-Formen
3. **Loss Monitoring**: Loss sollte kontinuierlich sinken
4. **Learning Rate**: Zu hoch = instabil, zu niedrig = langsam
5. **Overfitting**: Training Accuracy >> Test Accuracy

## Nächste Schritte

Nach Implementierung der Basis-Funktionalität:
- Convolutional Layers für Bildverarbeitung
- Recurrent Layers für Sequenzen
- Attention Mechanismen
- Weitere Optimierungen und Features