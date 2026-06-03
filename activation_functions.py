import numpy as np


class ReLU:
    """
    Rectified Linear Unit (ReLU) Aktivierungsfunktion
    """
    
    def forward(self, inputs):
        """
        Führt den Forward Pass der ReLU Funktion durch.
        
        Aufgabe:
        - Wendet ReLU auf die Eingaben an: f(x) = max(0, x)
        - Speichert die Eingaben für den Backward Pass
        
        Parameter:
        - inputs: numpy array mit beliebiger Form
        
        Ausgabeformat:
        - numpy array mit gleicher Form wie inputs, wobei negative Werte auf 0 gesetzt sind
        """
        pass
    
    def backward(self, dvalues):
        """
        Führt den Backward Pass der ReLU Funktion durch.
        
        Aufgabe:
        - Berechnet den Gradienten bezüglich der Eingaben
        - Gradient ist 1 für positive Eingaben, 0 für negative
        
        Parameter:
        - dvalues: Gradient aus der nächsten Schicht (numpy array)
        
        Ausgabeformat:
        - numpy array mit Gradienten bezüglich der Eingaben (gleiche Form wie dvalues)
        """
        pass


class Softmax:
    """
    Softmax Aktivierungsfunktion für die Ausgabeschicht
    """
    
    def forward(self, inputs):
        """
        Führt den Forward Pass der Softmax Funktion durch.
        
        Aufgabe:
        - Berechnet Softmax: exp(x_i) / sum(exp(x_j)) für alle j
        - Subtrahiert Maximum für numerische Stabilität
        - Speichert die Ausgabe für den Backward Pass
        
        Parameter:
        - inputs: numpy array der Form (batch_size, num_classes)
        
        Ausgabeformat:
        - numpy array der Form (batch_size, num_classes) mit Wahrscheinlichkeiten (Summe = 1 pro Zeile)
        """
        pass
    
    def backward(self, dvalues):
        """
        Führt den Backward Pass der Softmax Funktion durch.
        
        Aufgabe:
        - Berechnet den Gradienten der Softmax Funktion
        - Verwendet die Jacobi-Matrix für die Ableitung
        
        Parameter:
        - dvalues: Gradient aus der nächsten Schicht (numpy array der Form (batch_size, num_classes))
        
        Ausgabeformat:
        - numpy array mit Gradienten bezüglich der Eingaben (Form: (batch_size, num_classes))
        """
        pass

# Made with Bob
