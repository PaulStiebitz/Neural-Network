import numpy as np


class DenseLayer:
    """
    Fully Connected (Dense) Layer für neuronale Netze
    """
    
    def __init__(self, n_inputs, n_neurons):
        """
        Initialisiert die Dense Layer.
        
        Aufgabe:
        - Initialisiert Gewichte mit kleinen Zufallswerten (z.B. He-Initialisierung)
        - Initialisiert Bias mit Nullen
        
        Parameter:
        - n_inputs: Anzahl der Eingabe-Features
        - n_neurons: Anzahl der Neuronen in dieser Schicht
        """
        pass
    
    def forward(self, inputs):
        """
        Führt den Forward Pass der Dense Layer durch.
        
        Aufgabe:
        - Berechnet: output = inputs @ weights + bias
        - Speichert Eingaben für den Backward Pass
        
        Parameter:
        - inputs: numpy array der Form (batch_size, n_inputs)
        
        Ausgabeformat:
        - numpy array der Form (batch_size, n_neurons)
        """
        pass
    
    def backward(self, dvalues):
        """
        Führt den Backward Pass der Dense Layer durch.
        
        Aufgabe:
        - Berechnet Gradienten bezüglich Gewichte: dweights = inputs.T @ dvalues
        - Berechnet Gradienten bezüglich Bias: dbias = sum(dvalues, axis=0)
        - Berechnet Gradienten bezüglich Eingaben: dinputs = dvalues @ weights.T
        - Speichert Gradienten für Optimizer
        
        Parameter:
        - dvalues: Gradient aus der nächsten Schicht (numpy array der Form (batch_size, n_neurons))
        
        Ausgabeformat:
        - numpy array mit Gradienten bezüglich der Eingaben (Form: (batch_size, n_inputs))
        """
        pass


class DropoutLayer:
    """
    Dropout Layer für Regularisierung während des Trainings
    """
    
    def __init__(self, rate):
        """
        Initialisiert die Dropout Layer.
        
        Aufgabe:
        - Speichert die Dropout-Rate (Anteil der zu deaktivierenden Neuronen)
        
        Parameter:
        - rate: float zwischen 0 und 1 (z.B. 0.2 für 20% Dropout)
        """
        pass
    
    def forward(self, inputs, training=True):
        """
        Führt den Forward Pass der Dropout Layer durch.
        
        Aufgabe:
        - Im Training-Modus: Deaktiviert zufällig Neuronen und skaliert verbleibende
        - Im Inference-Modus: Gibt Eingaben unverändert zurück
        - Speichert Dropout-Maske für Backward Pass
        
        Parameter:
        - inputs: numpy array mit beliebiger Form
        - training: bool, ob Training-Modus aktiv ist
        
        Ausgabeformat:
        - numpy array mit gleicher Form wie inputs
        """
        pass
    
    def backward(self, dvalues):
        """
        Führt den Backward Pass der Dropout Layer durch.
        
        Aufgabe:
        - Wendet die gespeicherte Dropout-Maske auf Gradienten an
        
        Parameter:
        - dvalues: Gradient aus der nächsten Schicht (numpy array)
        
        Ausgabeformat:
        - numpy array mit Gradienten (gleiche Form wie dvalues)
        """
        pass


class BatchNormalization:
    """
    Batch Normalization Layer für stabileres Training
    """
    
    def __init__(self, n_inputs, momentum=0.9, epsilon=1e-5):
        """
        Initialisiert die Batch Normalization Layer.
        
        Aufgabe:
        - Initialisiert Gamma (Skalierung) mit Einsen
        - Initialisiert Beta (Verschiebung) mit Nullen
        - Initialisiert laufende Mittelwerte und Varianzen
        
        Parameter:
        - n_inputs: Anzahl der Features
        - momentum: Momentum für laufende Statistiken (default: 0.9)
        - epsilon: Kleine Konstante für numerische Stabilität (default: 1e-5)
        """
        pass
    
    def forward(self, inputs, training=True):
        """
        Führt den Forward Pass der Batch Normalization durch.
        
        Aufgabe:
        - Im Training: Normalisiert mit Batch-Statistiken, aktualisiert laufende Statistiken
        - Im Inference: Normalisiert mit laufenden Statistiken
        - Wendet Gamma und Beta an
        - Speichert Werte für Backward Pass
        
        Parameter:
        - inputs: numpy array der Form (batch_size, n_inputs)
        - training: bool, ob Training-Modus aktiv ist
        
        Ausgabeformat:
        - numpy array der Form (batch_size, n_inputs) mit normalisierten Werten
        """
        pass
    
    def backward(self, dvalues):
        """
        Führt den Backward Pass der Batch Normalization durch.
        
        Aufgabe:
        - Berechnet Gradienten bezüglich Gamma und Beta
        - Berechnet Gradienten bezüglich der Eingaben
        - Berücksichtigt die Normalisierung im Gradienten
        
        Parameter:
        - dvalues: Gradient aus der nächsten Schicht (numpy array der Form (batch_size, n_inputs))
        
        Ausgabeformat:
        - numpy array mit Gradienten bezüglich der Eingaben (Form: (batch_size, n_inputs))
        """
        pass

# Made with Bob
