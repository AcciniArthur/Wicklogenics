"""
Correction des points détectés par l'IA dans un arbre phylogénétique.
Corrige les nodes, corners et leaves manquants.
"""

from typing import List, Tuple

from itpt._data.models.v0.postprocess.corrector_leaves import leaf_line_correction_step1, fix_leaves_step
from itpt._data.models.v0.postprocess.corrector_vote import (
    apply_rules_one_pass,
    consolidate_future_and_select_points,
    print_future_list,
    add_missing_root_node
)

Point = Tuple[int, int, float]

def correction(
    nodes: List[Point],
    corners: List[Point],
    leaves: List[Point],
    img_width: int,
    printlog: bool = False
) -> Tuple[List[Point], List[Point], List[Point]]:
    """
    Applique les corrections sur les points détectés par l'IA.
    Retourne les listes corrigées de nodes, corners et leaves.
    """

    # Étape 1: Correction de la ligne des feuilles
    nodes2, corners2, leaves2, x_line = leaf_line_correction_step1(
        nodes, corners, leaves,
        img_width=img_width,
        bin_size=8,
        left_margin_px=6,
        right_margin_px=6,
        inside_margin_px=6,
    )

    # Étape 2: Correction des feuilles (step en x)
    leaves2, rem, add, xref, step = fix_leaves_step(leaves2, tol_px=5)
    #print("step =       leaves removed:", rem, "leaves added:", add, "step:", step, "xref:", xref)

    # Étape 3: Application des règles de vote avec scores élevés
    scores = [2.5, 2]
    for score in scores:
        futur, menfou = apply_rules_one_pass(nodes2, corners2, leaves2, delta=6)
        if printlog:
            print(f"------------- {score}")
            print_future_list(futur)

        nodes_to_add, corners_to_add, modified = consolidate_future_and_select_points(
            futur, score_threshold=score, delta=6
        )
        nodes2.extend(nodes_to_add)
        corners2.extend(corners_to_add)

    # Étape 4: Application des règles de vote avec scores plus bas
    scores = [1.5, 1.5]
    for score in scores:
        futur, menfou = apply_rules_one_pass(nodes2, corners2, leaves2, r2c=0.5, delta=6)
        if printlog:
            print(f"------------- {score}")
            print_future_list(futur)

        nodes_to_add, corners_to_add, modified = consolidate_future_and_select_points(
            futur, score_threshold=score, delta=6
        )
        nodes2.extend(nodes_to_add)
        corners2.extend(corners_to_add)

    # Étape 5: Ajout du nœud racine manquant si nécessaire
    nodes2, modified_root = add_missing_root_node(nodes2, corners2)

    return nodes2, corners2, leaves2
