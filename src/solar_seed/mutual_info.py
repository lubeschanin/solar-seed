"""
Mutual Information Berechnung
=============================

Pure NumPy Implementation der Mutual Information.

MI(X,Y) = H(X) + H(Y) - H(X,Y)

wobei H die Shannon-Entropie ist.
"""

import numpy as np
from numpy.typing import NDArray


def compute_histogram_2d(
    x: NDArray[np.float64], 
    y: NDArray[np.float64], 
    bins: int
) -> NDArray[np.float64]:
    """
    Berechnet 2D-Histogramm für zwei Arrays.
    
    Args:
        x: Erstes Array (wird geflattened)
        y: Zweites Array (wird geflattened)
        bins: Anzahl der Bins pro Dimension
        
    Returns:
        2D-Histogramm der Größe (bins, bins)
    """
    # Normalisiere auf [0, bins-1]
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    
    x_norm = (x - x_min) / (x_max - x_min + 1e-10)
    y_norm = (y - y_min) / (y_max - y_min + 1e-10)
    
    x_bins = np.clip((x_norm * bins).astype(int), 0, bins - 1)
    y_bins = np.clip((y_norm * bins).astype(int), 0, bins - 1)
    
    # 2D Histogramm (schnellere Variante mit bincount)
    hist = np.zeros((bins, bins), dtype=np.float64)
    indices = x_bins.ravel() * bins + y_bins.ravel()
    counts = np.bincount(indices, minlength=bins * bins)
    hist = counts.reshape((bins, bins)).astype(np.float64)
    
    return hist


def entropy(p: NDArray[np.float64]) -> float:
    """
    Berechnet Shannon-Entropie.
    
    Args:
        p: Wahrscheinlichkeitsverteilung (muss sich zu 1 summieren)
        
    Returns:
        Entropie in Bits
    """
    p = p[p > 0]
    if len(p) == 0:
        return 0.0
    return -float(np.sum(p * np.log2(p)))


def mutual_information(
    x: NDArray[np.float64], 
    y: NDArray[np.float64], 
    bins: int = 64
) -> float:
    """
    Berechnet Mutual Information zwischen zwei Arrays.
    
    MI(X,Y) = H(X) + H(Y) - H(X,Y)
    
    Args:
        x: Erstes Array (beliebige Shape, wird geflattened)
        y: Zweites Array (beliebige Shape, wird geflattened)
        bins: Anzahl der Bins für Histogramm
        
    Returns:
        Mutual Information in Bits
        
    Example:
        >>> x = np.random.randn(100, 100)
        >>> y = x + np.random.randn(100, 100) * 0.1  # Stark korreliert
        >>> mi = mutual_information(x, y)
        >>> mi > 1.0  # Sollte hohe MI zeigen
        True
    """
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)
    
    # Entferne NaN/Inf
    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    
    if len(x_flat) < 100:
        return 0.0
    
    # 2D Histogramm
    hist_2d = compute_histogram_2d(x_flat, y_flat, bins)
    
    # Marginale Histogramme
    hist_x = hist_2d.sum(axis=1)
    hist_y = hist_2d.sum(axis=0)
    
    # Normalisiere zu Wahrscheinlichkeiten
    n = hist_2d.sum()
    if n == 0:
        return 0.0
    
    p_xy = hist_2d / n
    p_x = hist_x / n
    p_y = hist_y / n
    
    # Entropien
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.ravel())
    
    # MI = H(X) + H(Y) - H(X,Y)
    mi = h_x + h_y - h_xy
    
    return max(0.0, mi)


def normalized_mutual_information(
    x: NDArray[np.float64], 
    y: NDArray[np.float64], 
    bins: int = 64
) -> float:
    """
    Normalisierte Mutual Information.
    
    NMI = 2 * MI(X,Y) / (H(X) + H(Y))
    
    Args:
        x: Erstes Array
        y: Zweites Array
        bins: Anzahl der Bins
        
    Returns:
        NMI im Bereich [0, 1], wobei 1 = perfekte Korrelation
    """
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)
    
    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    
    if len(x_flat) < 100:
        return 0.0
    
    hist_2d = compute_histogram_2d(x_flat, y_flat, bins)
    hist_x = hist_2d.sum(axis=1)
    hist_y = hist_2d.sum(axis=0)
    
    n = hist_2d.sum()
    if n == 0:
        return 0.0
    
    p_xy = hist_2d / n
    p_x = hist_x / n
    p_y = hist_y / n
    
    h_x = entropy(p_x)
    h_y = entropy(p_y)
    h_xy = entropy(p_xy.ravel())
    
    mi = h_x + h_y - h_xy
    
    if h_x + h_y == 0:
        return 0.0
    
    nmi = 2 * mi / (h_x + h_y)
    return max(0.0, min(1.0, nmi))


def conditional_entropy(
    x: NDArray[np.float64], 
    y: NDArray[np.float64], 
    bins: int = 64
) -> float:
    """
    Bedingte Entropie H(X|Y).
    
    H(X|Y) = H(X,Y) - H(Y) = H(X) - MI(X,Y)
    
    Returns:
        H(X|Y) in Bits
    """
    x_flat = x.ravel().astype(np.float64)
    y_flat = y.ravel().astype(np.float64)
    
    valid = np.isfinite(x_flat) & np.isfinite(y_flat)
    x_flat = x_flat[valid]
    y_flat = y_flat[valid]
    
    if len(x_flat) < 100:
        return 0.0
    
    hist_2d = compute_histogram_2d(x_flat, y_flat, bins)
    hist_y = hist_2d.sum(axis=0)
    
    n = hist_2d.sum()
    if n == 0:
        return 0.0
    
    p_xy = hist_2d / n
    p_y = hist_y / n
    
    h_xy = entropy(p_xy.ravel())
    h_y = entropy(p_y)
    
    return max(0.0, h_xy - h_y)
