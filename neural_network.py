import numpy as np
from layers import DenseLayer, DropoutLayer, BatchNormalization
from activation_functions import ReLU, Softmax
from loss_functions import CategoricalCrossentropy, SoftmaxCategoricalCrossentropy, MeanSquaredError
from optimizers import SGD, Adam, RMSprop
from utils import accuracy_categorical, accuracy_regression


class NeuralNetwork:
    """
    Hauptklasse für das neuronale Netzwerk
    """
    
    def __init__(self):
        """
        Initialisiert das neuronale Netzwerk.
        
        Aufgabe:
        - Initialisiert leere Listen für Layers
        - Setzt Standardwerte für Optimizer und Loss
        """
        pass
    
    def add(self, layer):
        """
        Fügt eine Layer zum Netzwerk hinzu.
        
        Aufgabe:
        - Fügt Layer zur internen Liste hinzu
        - Unterstützt Dense, Activation, Dropout, BatchNorm Layers
        
        Parameter:
        - layer: Layer-Objekt (z.B. DenseLayer, ReLU, etc.)
        """
        pass
    
    def set_loss(self, loss):
        """
        Setzt die Loss-Funktion.
        
        Aufgabe:
        - Speichert Loss-Funktion für Training
        
        Parameter:
        - loss: Loss-Objekt (z.B. CategoricalCrossentropy)
        """
        pass
    
    def set_optimizer(self, optimizer):
        """
        Setzt den Optimizer.
        
        Aufgabe:
        - Speichert Optimizer für Parameter-Updates
        
        Parameter:
        - optimizer: Optimizer-Objekt (z.B. Adam, SGD)
        """
        pass
    
    def forward(self, X, training=True):
        """
        Führt Forward Pass durch das gesamte Netzwerk durch.
        
        Aufgabe:
        - Propagiert Eingaben durch alle Layers
        - Berücksichtigt Training-Modus für Dropout/BatchNorm
        - Speichert Ausgaben jeder Layer
        
        Parameter:
        - X: numpy array mit Eingabedaten
        - training: bool, ob Training-Modus aktiv ist (default: True)
        
        Ausgabeformat:
        - numpy array mit finalen Ausgaben des Netzwerks
        """
        pass
    
    def backward(self, output, y_true):
        """
        Führt Backward Pass durch das gesamte Netzwerk durch.
        
        Aufgabe:
        - Berechnet Loss-Gradienten
        - Propagiert Gradienten rückwärts durch alle Layers
        - Speichert Gradienten in jeder Layer
        
        Parameter:
        - output: numpy array mit Netzwerk-Ausgaben
        - y_true: numpy array mit wahren Labels
        """
        pass
    
    def update_params(self):
        """
        Aktualisiert alle trainierbaren Parameter.
        
        Aufgabe:
        - Ruft Optimizer für jede trainierbare Layer auf
        - Aktualisiert Gewichte und Bias
        """
        pass
    
    def train_step(self, X_batch, y_batch):
        """
        Führt einen einzelnen Trainingsschritt durch.
        
        Aufgabe:
        - Forward Pass
        - Loss-Berechnung
        - Backward Pass
        - Parameter-Update
        
        Parameter:
        - X_batch: numpy array mit Batch von Eingabedaten
        - y_batch: numpy array mit Batch von Labels
        
        Ausgabeformat:
        - tuple: (loss, accuracy) für diesen Batch
        """
        pass
    
    def fit(self, X_train, y_train, epochs, batch_size=32, validation_data=None, verbose=True):
        """
        Trainiert das Netzwerk.
        
        Aufgabe:
        - Iteriert über Epochen
        - Teilt Daten in Batches
        - Führt Training Steps durch
        - Berechnet Validierungs-Metriken falls gegeben
        - Gibt Trainingsfortschritt aus
        
        Parameter:
        - X_train: numpy array mit Trainingsdaten
        - y_train: numpy array mit Training-Labels
        - epochs: int, Anzahl der Trainingsepochen
        - batch_size: int, Größe der Batches (default: 32)
        - validation_data: tuple (X_val, y_val) oder None (default: None)
        - verbose: bool, ob Trainingsfortschritt ausgegeben werden soll (default: True)
        
        Ausgabeformat:
        - dict: History mit 'loss', 'accuracy' und optional 'val_loss', 'val_accuracy' Listen
        """
        pass
    
    def predict(self, X):
        """
        Macht Vorhersagen für neue Daten.
        
        Aufgabe:
        - Führt Forward Pass im Inference-Modus durch
        - Gibt Vorhersagen zurück
        
        Parameter:
        - X: numpy array mit Eingabedaten
        
        Ausgabeformat:
        - numpy array mit Vorhersagen
        """
        pass
    
    def evaluate(self, X, y):
        """
        Evaluiert das Netzwerk auf gegebenen Daten.
        
        Aufgabe:
        - Macht Vorhersagen
        - Berechnet Loss und Accuracy
        
        Parameter:
        - X: numpy array mit Eingabedaten
        - y: numpy array mit Labels
        
        Ausgabeformat:
        - tuple: (loss, accuracy)
        """
        pass
    
    def get_params(self):
        """
        Gibt alle trainierbaren Parameter zurück.
        
        Aufgabe:
        - Sammelt Gewichte und Bias aus allen Layers
        
        Ausgabeformat:
        - list von dicts mit 'weights' und 'bias' für jede Layer
        """
        pass
    
    def set_params(self, params):
        """
        Setzt alle trainierbaren Parameter.
        
        Aufgabe:
        - Lädt Gewichte und Bias in alle Layers
        
        Parameter:
        - params: list von dicts mit 'weights' und 'bias'
        """
        pass
    
    def summary(self):
        """
        Gibt eine Zusammenfassung der Netzwerk-Architektur aus.
        
        Aufgabe:
        - Zeigt Layer-Typen und Formen
        - Berechnet Gesamtanzahl der Parameter
        - Formatiert Ausgabe übersichtlich
        
        Ausgabeformat:
        - Gibt formatierte Zusammenfassung auf Konsole aus
        """
        pass

# Made with Bob
