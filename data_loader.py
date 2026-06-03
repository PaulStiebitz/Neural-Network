import numpy as np


class DataLoader:
    """
    DataLoader für effizientes Laden und Verarbeiten von Daten
    """
    
    def __init__(self, X, y, batch_size=32, shuffle=True):
        """
        Initialisiert den DataLoader.
        
        Aufgabe:
        - Speichert Daten und Konfiguration
        - Berechnet Anzahl der Batches
        
        Parameter:
        - X: numpy array mit Features
        - y: numpy array mit Labels
        - batch_size: int, Größe der Batches (default: 32)
        - shuffle: bool, ob Daten gemischt werden sollen (default: True)
        """
        pass
    
    def __len__(self):
        """
        Gibt die Anzahl der Batches zurück.
        
        Ausgabeformat:
        - int: Anzahl der Batches
        """
        pass
    
    def __iter__(self):
        """
        Macht den DataLoader iterierbar.
        
        Aufgabe:
        - Mischt Daten falls shuffle=True
        - Setzt Batch-Index zurück
        
        Ausgabeformat:
        - self
        """
        pass
    
    def __next__(self):
        """
        Gibt den nächsten Batch zurück.
        
        Aufgabe:
        - Gibt nächsten Batch von X und y zurück
        - Wirft StopIteration wenn alle Batches durchlaufen wurden
        
        Ausgabeformat:
        - tuple: (X_batch, y_batch)
        """
        pass


class DataGenerator:
    """
    Generator für synthetische Daten (für Tests und Demos)
    """
    
    @staticmethod
    def generate_spiral_data(samples, classes):
        """
        Generiert Spiral-Daten für Klassifikation.
        
        Aufgabe:
        - Erstellt spiralförmige Datenpunkte für jede Klasse
        - Fügt Rauschen hinzu
        
        Parameter:
        - samples: int, Anzahl der Samples pro Klasse
        - classes: int, Anzahl der Klassen
        
        Ausgabeformat:
        - tuple: (X, y) wobei X Form (samples*classes, 2) und y Form (samples*classes,) hat
        """
        pass
    
    @staticmethod
    def generate_linear_data(samples, features):
        """
        Generiert lineare Regressionsdaten.
        
        Aufgabe:
        - Erstellt Features und Zielwerte mit linearer Beziehung
        - Fügt Rauschen hinzu
        
        Parameter:
        - samples: int, Anzahl der Samples
        - features: int, Anzahl der Features
        
        Ausgabeformat:
        - tuple: (X, y) wobei X Form (samples, features) und y Form (samples,) hat
        """
        pass
    
    @staticmethod
    def generate_sine_data(samples):
        """
        Generiert Sinus-Daten für Regression.
        
        Aufgabe:
        - Erstellt Datenpunkte entlang einer Sinuskurve
        - Fügt Rauschen hinzu
        
        Parameter:
        - samples: int, Anzahl der Samples
        
        Ausgabeformat:
        - tuple: (X, y) wobei X Form (samples, 1) und y Form (samples,) hat
        """
        pass
    
    @staticmethod
    def generate_classification_data(samples, features, classes, separable=True):
        """
        Generiert allgemeine Klassifikationsdaten.
        
        Aufgabe:
        - Erstellt Cluster für jede Klasse
        - Kontrolliert Trennbarkeit der Klassen
        
        Parameter:
        - samples: int, Anzahl der Samples pro Klasse
        - features: int, Anzahl der Features
        - classes: int, Anzahl der Klassen
        - separable: bool, ob Klassen gut trennbar sein sollen (default: True)
        
        Ausgabeformat:
        - tuple: (X, y) wobei X Form (samples*classes, features) und y Form (samples*classes,) hat
        """
        pass


class ImageDataLoader:
    """
    Spezialisierter DataLoader für Bilddaten
    """
    
    def __init__(self, image_paths, labels, batch_size=32, shuffle=True, augment=False):
        """
        Initialisiert den ImageDataLoader.
        
        Aufgabe:
        - Speichert Bildpfade und Labels
        - Konfiguriert Augmentierung
        
        Parameter:
        - image_paths: list von Strings mit Bildpfaden
        - labels: numpy array mit Labels
        - batch_size: int, Größe der Batches (default: 32)
        - shuffle: bool, ob Daten gemischt werden sollen (default: True)
        - augment: bool, ob Data Augmentation angewendet werden soll (default: False)
        """
        pass
    
    def load_image(self, path):
        """
        Lädt ein einzelnes Bild.
        
        Aufgabe:
        - Lädt Bild von Pfad
        - Konvertiert zu numpy array
        - Normalisiert Pixelwerte
        
        Parameter:
        - path: str, Pfad zum Bild
        
        Ausgabeformat:
        - numpy array mit Bilddaten
        """
        pass
    
    def augment_image(self, image):
        """
        Wendet Data Augmentation auf ein Bild an.
        
        Aufgabe:
        - Zufällige Transformationen (Rotation, Flip, etc.)
        - Erhöht Datenvariabilität
        
        Parameter:
        - image: numpy array mit Bilddaten
        
        Ausgabeformat:
        - numpy array mit augmentiertem Bild
        """
        pass
    
    def __iter__(self):
        """
        Macht den ImageDataLoader iterierbar.
        
        Aufgabe:
        - Mischt Daten falls shuffle=True
        - Setzt Batch-Index zurück
        
        Ausgabeformat:
        - self
        """
        pass
    
    def __next__(self):
        """
        Gibt den nächsten Batch von Bildern zurück.
        
        Aufgabe:
        - Lädt Bilder für aktuellen Batch
        - Wendet Augmentation an falls aktiviert
        - Gibt Batch zurück
        
        Ausgabeformat:
        - tuple: (images_batch, labels_batch)
        """
        pass

# Made with Bob
