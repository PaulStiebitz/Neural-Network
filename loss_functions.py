import numpy as np


class CategoricalCrossentropy:
    """
    Categorical Crossentropy Loss Funktion für Mehrklassen-Klassifikation
    """
    
    def forward(self, y_pred, y_true):
        """
        Berechnet den Categorical Crossentropy Loss.
        
        Aufgabe:
        - Berechnet den Loss: -sum(y_true * log(y_pred))
        - Clippt Vorhersagen um log(0) zu vermeiden
        - Unterstützt sowohl One-Hot als auch Sparse Labels
        
        Parameter:
        - y_pred: Vorhergesagte Wahrscheinlichkeiten (numpy array der Form (batch_size, num_classes))
        - y_true: Wahre Labels (numpy array der Form (batch_size, num_classes) für One-Hot 
                  oder (batch_size,) für Sparse Labels)
        
        Ausgabeformat:
        - float: Durchschnittlicher Loss über den Batch
        """
        pass
    
    def backward(self, y_pred, y_true):
        """
        Berechnet den Gradienten des Categorical Crossentropy Loss.
        
        Aufgabe:
        - Berechnet dL/dy_pred
        - Normalisiert durch Batch-Größe
        - Konvertiert Sparse Labels zu One-Hot falls nötig
        
        Parameter:
        - y_pred: Vorhergesagte Wahrscheinlichkeiten (numpy array der Form (batch_size, num_classes))
        - y_true: Wahre Labels (numpy array)
        
        Ausgabeformat:
        - numpy array der Form (batch_size, num_classes) mit Gradienten bezüglich y_pred
        """
        pass


class SoftmaxCategoricalCrossentropy:
    """
    Kombinierte Softmax und Categorical Crossentropy für effizienteren Backward Pass
    """
    
    def forward(self, inputs, y_true):
        """
        Führt Softmax und Loss-Berechnung in einem Schritt durch.
        
        Aufgabe:
        - Wendet Softmax auf Eingaben an
        - Berechnet Categorical Crossentropy Loss
        - Speichert Ausgaben für Backward Pass
        
        Parameter:
        - inputs: Logits vor Softmax (numpy array der Form (batch_size, num_classes))
        - y_true: Wahre Labels (numpy array)
        
        Ausgabeformat:
        - float: Durchschnittlicher Loss über den Batch
        """
        pass
    
    def backward(self, y_pred, y_true):
        """
        Berechnet den kombinierten Gradienten von Softmax und Crossentropy.
        
        Aufgabe:
        - Nutzt die vereinfachte Ableitung: y_pred - y_true
        - Normalisiert durch Batch-Größe
        - Konvertiert Sparse Labels zu One-Hot falls nötig
        
        Parameter:
        - y_pred: Softmax Ausgaben (numpy array der Form (batch_size, num_classes))
        - y_true: Wahre Labels (numpy array)
        
        Ausgabeformat:
        - numpy array der Form (batch_size, num_classes) mit Gradienten bezüglich der Eingaben
        """
        pass


class MeanSquaredError:
    """
    Mean Squared Error Loss Funktion für Regression
    """
    
    def forward(self, y_pred, y_true):
        """
        Berechnet den Mean Squared Error Loss.
        
        Aufgabe:
        - Berechnet MSE: mean((y_pred - y_true)^2)
        
        Parameter:
        - y_pred: Vorhergesagte Werte (numpy array)
        - y_true: Wahre Werte (numpy array mit gleicher Form wie y_pred)
        
        Ausgabeformat:
        - float: Durchschnittlicher MSE Loss
        """
        pass
    
    def backward(self, y_pred, y_true):
        """
        Berechnet den Gradienten des MSE Loss.
        
        Aufgabe:
        - Berechnet dL/dy_pred = 2 * (y_pred - y_true) / n
        
        Parameter:
        - y_pred: Vorhergesagte Werte (numpy array)
        - y_true: Wahre Werte (numpy array)
        
        Ausgabeformat:
        - numpy array mit Gradienten bezüglich y_pred (gleiche Form wie y_pred)
        """
        pass

# Made with Bob
