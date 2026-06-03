import numpy as np


class SGD:
    """
    Stochastic Gradient Descent Optimizer
    """
    
    def __init__(self, learning_rate=1.0, decay=0.0, momentum=0.0):
        """
        Initialisiert den SGD Optimizer.
        
        Aufgabe:
        - Speichert Hyperparameter (Learning Rate, Decay, Momentum)
        - Initialisiert Iterationszähler
        
        Parameter:
        - learning_rate: Lernrate (default: 1.0)
        - decay: Learning Rate Decay (default: 0.0)
        - momentum: Momentum-Faktor (default: 0.0)
        """
        pass
    
    def pre_update_params(self):
        """
        Wird vor dem Update der Parameter aufgerufen.
        
        Aufgabe:
        - Aktualisiert die aktuelle Lernrate basierend auf Decay
        - Inkrementiert Iterationszähler
        """
        pass
    
    def update_params(self, layer):
        """
        Aktualisiert die Parameter einer Layer.
        
        Aufgabe:
        - Berechnet Momentum-Updates falls aktiviert
        - Aktualisiert Gewichte: weights -= learning_rate * dweights
        - Aktualisiert Bias: bias -= learning_rate * dbias
        
        Parameter:
        - layer: Layer-Objekt mit weights, bias, dweights, dbias Attributen
        """
        pass
    
    def post_update_params(self):
        """
        Wird nach dem Update aller Parameter aufgerufen.
        
        Aufgabe:
        - Inkrementiert Iterationszähler (falls nicht in pre_update_params)
        """
        pass


class Adam:
    """
    Adam (Adaptive Moment Estimation) Optimizer
    """
    
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, beta_1=0.9, beta_2=0.999):
        """
        Initialisiert den Adam Optimizer.
        
        Aufgabe:
        - Speichert Hyperparameter
        - Initialisiert Iterationszähler
        
        Parameter:
        - learning_rate: Lernrate (default: 0.001)
        - decay: Learning Rate Decay (default: 0.0)
        - epsilon: Kleine Konstante für numerische Stabilität (default: 1e-7)
        - beta_1: Exponentieller Decay für ersten Moment (default: 0.9)
        - beta_2: Exponentieller Decay für zweiten Moment (default: 0.999)
        """
        pass
    
    def pre_update_params(self):
        """
        Wird vor dem Update der Parameter aufgerufen.
        
        Aufgabe:
        - Aktualisiert die aktuelle Lernrate basierend auf Decay
        """
        pass
    
    def update_params(self, layer):
        """
        Aktualisiert die Parameter einer Layer mit Adam.
        
        Aufgabe:
        - Initialisiert Momentum-Caches falls nicht vorhanden
        - Aktualisiert ersten Moment (Mittelwert der Gradienten)
        - Aktualisiert zweiten Moment (Mittelwert der quadrierten Gradienten)
        - Korrigiert Bias in den Momenten
        - Aktualisiert Parameter: param -= lr * m_corrected / (sqrt(v_corrected) + epsilon)
        
        Parameter:
        - layer: Layer-Objekt mit weights, bias, dweights, dbias Attributen
        """
        pass
    
    def post_update_params(self):
        """
        Wird nach dem Update aller Parameter aufgerufen.
        
        Aufgabe:
        - Inkrementiert Iterationszähler
        """
        pass


class RMSprop:
    """
    RMSprop (Root Mean Square Propagation) Optimizer
    """
    
    def __init__(self, learning_rate=0.001, decay=0.0, epsilon=1e-7, rho=0.9):
        """
        Initialisiert den RMSprop Optimizer.
        
        Aufgabe:
        - Speichert Hyperparameter
        - Initialisiert Iterationszähler
        
        Parameter:
        - learning_rate: Lernrate (default: 0.001)
        - decay: Learning Rate Decay (default: 0.0)
        - epsilon: Kleine Konstante für numerische Stabilität (default: 1e-7)
        - rho: Decay-Rate für laufenden Durchschnitt (default: 0.9)
        """
        pass
    
    def pre_update_params(self):
        """
        Wird vor dem Update der Parameter aufgerufen.
        
        Aufgabe:
        - Aktualisiert die aktuelle Lernrate basierend auf Decay
        """
        pass
    
    def update_params(self, layer):
        """
        Aktualisiert die Parameter einer Layer mit RMSprop.
        
        Aufgabe:
        - Initialisiert Cache falls nicht vorhanden
        - Aktualisiert Cache: cache = rho * cache + (1 - rho) * gradient^2
        - Aktualisiert Parameter: param -= lr * gradient / (sqrt(cache) + epsilon)
        
        Parameter:
        - layer: Layer-Objekt mit weights, bias, dweights, dbias Attributen
        """
        pass
    
    def post_update_params(self):
        """
        Wird nach dem Update aller Parameter aufgerufen.
        
        Aufgabe:
        - Inkrementiert Iterationszähler
        """
        pass

# Made with Bob
