"""
Nullmodell für Hypothesentest
=============================

Shuffle-basiertes Nullmodell zur statistischen Validierung.

Die Idee: Wenn zwei Kanäle unabhängig sind, sollte das Shufflen
eines Kanals die MI nicht ändern.
"""

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, List

from solar_seed.mutual_info import mutual_information


def shuffle_array(arr: NDArray[np.float64], seed: int | None = None) -> NDArray[np.float64]:
    """
    Shuffelt ein Array (flach).
    
    Args:
        arr: Input Array
        seed: Random Seed für Reproduzierbarkeit
        
    Returns:
        Gesuffletes Array mit gleicher Shape
    """
    rng = np.random.default_rng(seed)
    flat = arr.ravel().copy()
    rng.shuffle(flat)
    return flat.reshape(arr.shape)


def compute_null_distribution(
    x: NDArray[np.float64], 
    y: NDArray[np.float64],
    n_shuffles: int = 100,
    bins: int = 64,
    seed: int | None = 42,
    verbose: bool = False
) -> Tuple[float, float, List[float]]:
    """
    Berechnet Nullverteilung durch wiederholtes Shufflen.
    
    Args:
        x: Erstes Array (bleibt unverändert)
        y: Zweites Array (wird geshufflet)
        n_shuffles: Anzahl Shuffle-Durchläufe
        bins: Bins für MI-Berechnung
        seed: Base-Seed für Reproduzierbarkeit
        verbose: Fortschrittsausgabe
        
    Returns:
        mean: Mittlere MI unter Nullhypothese
        std: Standardabweichung
        distribution: Liste aller MI-Werte
    """
    mi_values: List[float] = []
    
    base_rng = np.random.default_rng(seed)
    
    for i in range(n_shuffles):
        # Generiere neuen Seed für diesen Durchlauf
        shuffle_seed = base_rng.integers(0, 2**31)
        
        y_shuffled = shuffle_array(y, seed=shuffle_seed)
        mi = mutual_information(x, y_shuffled, bins)
        mi_values.append(mi)
        
        if verbose and (i + 1) % 20 == 0:
            print(f"     Shuffle {i+1}/{n_shuffles}...")
    
    return float(np.mean(mi_values)), float(np.std(mi_values)), mi_values


def compute_z_score(
    mi_real: float, 
    mi_null_mean: float, 
    mi_null_std: float
) -> float:
    """
    Berechnet Z-Score.
    
    Z = (MI_real - MI_null_mean) / MI_null_std
    
    Args:
        mi_real: Beobachtete MI
        mi_null_mean: Mittlere MI aus Nullmodell
        mi_null_std: Standardabweichung aus Nullmodell
        
    Returns:
        Z-Score (Standardabweichungen über Nullmodell)
    """
    if mi_null_std < 1e-10:
        return float('inf') if mi_real > mi_null_mean else 0.0
    
    return (mi_real - mi_null_mean) / mi_null_std


def compute_p_value(
    mi_real: float, 
    mi_null_distribution: List[float]
) -> float:
    """
    Berechnet empirischen p-Wert.
    
    p = Anteil der Null-MI-Werte >= MI_real
    
    Args:
        mi_real: Beobachtete MI
        mi_null_distribution: Liste der MI-Werte aus Nullmodell
        
    Returns:
        p-Wert (1-seitig, rechtsseitig)
    """
    if not mi_null_distribution:
        return 1.0
    
    n_greater = sum(1 for mi in mi_null_distribution if mi >= mi_real)
    return n_greater / len(mi_null_distribution)


def interpret_result(
    z_score: float, 
    p_value: float
) -> Tuple[str, str]:
    """
    Interpretiert statistische Ergebnisse.
    
    Returns:
        status: Kurzer Status-String
        interpretation: Längere Erklärung
    """
    if z_score > 3 and p_value < 0.01:
        return (
            "✓ SIGNIFIKANT",
            "MI ist hochsignifikant höher als erwartet (p < 0.01). "
            "Die Kanäle teilen Information, die nicht durch Zufall erklärbar ist."
        )
    elif z_score > 2 and p_value < 0.05:
        return (
            "~ TENDENZIELL",
            "MI ist tendenziell höher als erwartet (p < 0.05). "
            "Weitere Untersuchung empfohlen."
        )
    elif z_score < -2:
        return (
            "? UNERWARTET NIEDRIG",
            "MI ist niedriger als erwartet. "
            "Dies könnte auf Anti-Korrelation oder Datenartefakte hindeuten."
        )
    else:
        return (
            "✗ NICHT SIGNIFIKANT",
            "MI liegt im erwarteten Bereich des Nullmodells. "
            "Keine Evidenz für zusätzliche Struktur."
        )
