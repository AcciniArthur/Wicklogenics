"""
Fonctions de correction pour les feuilles (leaves) de l'arbre phylogénétique.
Détecte la ligne verticale des feuilles et corrige les positions.
"""

from typing import List, Tuple, Optional
import numpy as np

Point = Tuple[int, int, float]


def estimate_leaf_vertical_x(
    leaves: List[Point],
    img_width: int,
    bin_size: int = 6,
    inlier_window_px: Optional[int] = None,
    min_inliers: int = 3,
) -> Tuple[Optional[int], List[Point]]:
    """
    Estime la coordonnée X dominante de la ligne verticale des feuilles.
    
    Stratégie:
    - Crée un histogramme des positions X pour trouver le cluster dominant
    - Garde les inliers dans une fenêtre autour du pic
    - Retourne la ligne affinée comme médiane X des inliers
    
    Retourne: (x_line ou None, feuilles_inliers_utilisées)
    """
    if not leaves:
        return None, []

    xs = np.array([p[0] for p in leaves], dtype=np.int32)
    xs = np.clip(xs, 0, img_width - 1)

    if inlier_window_px is None:
        inlier_window_px = int(np.clip(round(img_width * 0.02), 12, 40))

    nbins = max(1, int(np.ceil(img_width / bin_size)))
    bins = np.linspace(0, img_width, nbins + 1)
    hist, edges = np.histogram(xs, bins=bins)

    peak_bin = int(np.argmax(hist))
    peak_center = int((edges[peak_bin] + edges[peak_bin + 1]) / 2)

    mask = np.abs(xs - peak_center) <= inlier_window_px
    inlier_leaves = [leaves[i] for i, ok in enumerate(mask) if ok]

    if len(inlier_leaves) < min_inliers:
        return int(np.median(xs)), leaves

    x_line = int(np.median([p[0] for p in inlier_leaves]))
    return x_line, inlier_leaves


def apply_leaf_line_filter(
    nodes: List[Point],
    corners: List[Point],
    leaves: List[Point],
    x_line: Optional[int],
    left_margin_px: int = 0,
    right_margin_px: int = 0,
) -> Tuple[List[Point], List[Point], List[Point]]:
    """
    Applique le filtre basé sur la ligne verticale des feuilles:
    - Retire TOUS les points (nodes/corners/leaves) à droite de x_line + right_margin_px
    - Retire les FEUILLES à gauche de x_line - left_margin_px
    
    Si x_line est None: retourne les listes inchangées.
    """
    if x_line is None:
        return list(nodes), list(corners), list(leaves)

    left_cut = int(x_line - left_margin_px)
    right_cut = int(x_line + right_margin_px)

    nodes_f = [p for p in nodes if p[0] <= right_cut]
    corners_f = [p for p in corners if p[0] <= right_cut]
    leaves_tmp = [p for p in leaves if p[0] <= right_cut]

    leaves_f = [p for p in leaves_tmp if p[0] >= left_cut]

    return nodes_f, corners_f, leaves_f


def remove_non_leaves_near_leaf_line(
    nodes: List[Point],
    corners: List[Point],
    x_line: Optional[int],
    tol_px: int = 4,
) -> Tuple[List[Point], List[Point]]:
    """
    Retire les nodes et corners trop proches de la ligne verticale des feuilles.
    Les feuilles ne sont pas touchées (traitées ailleurs).
    """
    if x_line is None:
        return nodes, corners

    x0 = int(x_line)
    nodes2 = [p for p in nodes if abs(p[0] - x0) > tol_px]
    corners2 = [p for p in corners if abs(p[0] - x0) > tol_px]
    return nodes2, corners2


def leaf_line_correction_step1(
    nodes: List[Point],
    corners: List[Point],
    leaves: List[Point],
    img_width: int,
    bin_size: int = 6,
    inlier_window_px: Optional[int] = None,
    min_inliers: int = 3,
    left_margin_px: int = 0,
    right_margin_px: int = 0,
    inside_margin_px: int = 0,
) -> Tuple[List[Point], List[Point], List[Point], Optional[int]]:
    """
    Étape complète de correction de la ligne des feuilles:
    1. Estime la ligne verticale des feuilles (x_line)
    2. Applique le filtre en utilisant les marges
    
    Retourne: nodes2, corners2, leaves2, x_line
    """
    x_line, _inliers = estimate_leaf_vertical_x(
        leaves=leaves,
        img_width=img_width,
        bin_size=bin_size,
        inlier_window_px=inlier_window_px,
        min_inliers=min_inliers,
    )

    nodes2, corners2, leaves2 = apply_leaf_line_filter(
        nodes=nodes,
        corners=corners,
        leaves=leaves,
        x_line=x_line,
        left_margin_px=left_margin_px,
        right_margin_px=right_margin_px,
    )

    nodes2, corners2 = remove_non_leaves_near_leaf_line(
        nodes2, corners2,
        x_line=x_line,
        tol_px=inside_margin_px
    )

    return nodes2, corners2, leaves2, x_line

def fix_leaves_step(
    leaves: List[Point],
    step_px: Optional[float] = None,
    tol_px: Optional[float] = None,
    x_ref: Optional[int] = None,
    score_new: float = -0.5,
    max_add_per_gap: int = 50,
) -> Tuple[List[Point], int, int, Optional[int], Optional[float]]:
    """
    Corrige les feuilles en détectant et comblant les gaps.
    
    Version qui:
    - Estime le pas vertical (step) si non fourni
    - Ne supprime aucune feuille
    - Détecte les gaps via le step
    - Ajoute les feuilles manquantes par interpolation locale
    
    Retourne: (nouvelles_feuilles, nb_retirees=0, nb_ajoutees, x_ref_utilisé, step_utilisé)
    """
    if not leaves:
        return [], 0, 0, x_ref, step_px

    leaves_sorted = sorted(leaves, key=lambda p: p[1])
    ys = np.array([p[1] for p in leaves_sorted], dtype=np.float32)
    xs = np.array([p[0] for p in leaves_sorted], dtype=np.float32)

    if x_ref is None:
        x_ref = int(np.median(xs))

    # Estimation du pas vertical
    if step_px is None:
        if len(leaves_sorted) < 2:
            return leaves_sorted, 0, 0, x_ref, None

        dys = np.diff(ys)
        dys = dys[dys > 1.0]
        if dys.size == 0:
            return leaves_sorted, 0, 0, x_ref, None

        cap = np.percentile(dys, 60)
        cand = dys[dys <= cap] if cap > 0 else dys
        if cand.size == 0:
            cand = dys
        step_px = float(np.median(cand))

    if step_px is None or step_px <= 1.0:
        return leaves_sorted, 0, 0, x_ref, step_px

    step = float(step_px)

    # Ajout des feuilles manquantes par interpolation locale dans chaque gap
    added: List[Point] = []
    added_count = 0

    for i in range(len(leaves_sorted) - 1):
        y1 = float(leaves_sorted[i][1])
        y2 = float(leaves_sorted[i + 1][1])
        gap = y2 - y1
        if gap <= 1.0:
            continue

        k = int(round(gap / step))
        if k <= 1:
            continue

        missing = k - 1
        to_add = min(missing, max_add_per_gap)

        denom = float(missing + 1)
        for j in range(1, to_add + 1):
            y_new = y1 + (gap * (float(j) / denom))
            if y1 < y_new < y2:
                added.append((int(x_ref), int(round(y_new)), float(score_new)))
                added_count += 1

    new_leaves = list(leaves_sorted) + added
    new_leaves.sort(key=lambda p: p[1])

    removed_count = 0
    return new_leaves, removed_count, added_count, x_ref, step_px