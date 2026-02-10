#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
newick_clean.py

Construit un Newick (avec longueurs de branches RELATIVES) à partir de 3 listes de points :
- leaves    : [(x,y), ...]
- internals : [(x,y), ...]
- corners   : [(x,y), ...]

Hypothèses :
- arbre vertical
- y sert uniquement à regrouper les enfants via des "intervals" définis par les corners
- x sert à calculer les longueurs de branches (horizontales)

Longueurs :
- length(child->parent) = abs(x_child - x_parent)
  (tu peux ensuite multiplier toutes les longueurs par une constante pour récupérer la temporalité réelle)

Sortie :
- Newick terminé par ';'
"""

from typing import List, Tuple, Optional


def build_newick(
    leaves: List[Tuple[int, int]],
    internals: List[Tuple[int, int]],
    corners: List[Tuple[int, int]],
    x_tol: float = 8.0,
    y_tol: float = 8.0,
    leaf_names: Optional[List[str]] = None,
    decimals: int = 9
) -> Optional[str]:
    """
    Args:
        leaves, internals, corners: list of (x,y) en pixels (int)
        x_tol: tolérance en x pour associer 2 corners à un internal
        y_tol: marge autour de l'intervalle [y_corner1, y_corner2] pour capturer les clusters
        leaf_names: noms des feuilles (sinon A,B,C...)
        decimals: nombre de décimales dans les longueurs

    Returns:
        Newick string avec longueurs, ou None si impossible.
    """
    if not leaves or not internals or not corners:
        return None

    def fmt_len(L: float) -> str:
        return f"{L:.{decimals}f}"

    def branch_len(x_child: int, x_parent: int) -> float:
        return float(abs(int(x_child) - int(x_parent)))

    if not leaves or not internals or not corners:
        return None

    # --------------------------------------------------------
    # Ordonner et nommer les feuilles
    # --------------------------------------------------------
    # Tri par y décroissant (haut -> bas)
    leaves_sorted = sorted(leaves, key=lambda p: p[1], reverse=True)
    n_leaves = len(leaves_sorted)

    def gen_name_stream():
        """Génère A,B,C... puis L26,L27... (infini)."""
        i = 0
        while True:
            if i < 26:
                yield chr(ord("A") + i)
            else:
                yield f"L{i}"
            i += 1

    if leaf_names is None:
        leaf_names = []
        g = gen_name_stream()
        for _ in range(n_leaves):
            leaf_names.append(next(g))
    else:
        leaf_names = list(leaf_names)

        if len(leaf_names) < n_leaves:
            used = set(leaf_names)
            g = gen_name_stream()
            while len(leaf_names) < n_leaves:
                cand = next(g)
                if cand in used:
                    continue
                leaf_names.append(cand)
                used.add(cand)
        elif len(leaf_names) > n_leaves:
            leaf_names = leaf_names[:n_leaves]

    xs = sorted([x for (x, _y) in leaves_sorted])
    leaf_tip_x = int(xs[len(xs) // 2])  # médiane (int)

    clusters = []
    for name, (_x, y) in zip(leaf_names, leaves_sorted):
        clusters.append({"y": int(y), "x": int(leaf_tip_x), "nw": str(name)})

    # --------------------------------------------------------
    # Préparer corners
    # --------------------------------------------------------
    corners_int = corners

    def find_two_corners_for_node(xn: int, yn: int):
        cand = [(x, y) for (x, y) in corners_int if abs(int(x) - int(xn)) <= x_tol]
        if len(cand) < 2:
            return None

        above = [c for c in cand if c[1] <= yn]
        below = [c for c in cand if c[1] >= yn]

        if above and below:
            c1 = min(above, key=lambda c: abs(c[1] - yn))
            c2 = min(below, key=lambda c: abs(c[1] - yn))
            if c1 != c2:
                return c1, c2

        cand.sort(key=lambda c: abs(c[1] - yn))
        return cand[0], cand[1]

    # --------------------------------------------------------
    # Fusion bottom-up
    # --------------------------------------------------------
    # Tri par x décroissant (droite -> gauche) pour remonter
    internals_sorted = sorted(internals, key=lambda p: p[0], reverse=True)

    for xn, yn in internals_sorted:
        found = find_two_corners_for_node(int(xn), int(yn))
        if found is None:
            continue

        (x1, y1), (x2, y2) = found
        y_low, y_high = (y1, y2) if y1 <= y2 else (y2, y1)

        low = int(y_low - y_tol)
        high = int(y_high + y_tol)

        inside_idx = [i for i, c in enumerate(clusters) if low <= int(c["y"]) <= high]
        if len(inside_idx) < 2:
            continue

        inside = [clusters[i] for i in inside_idx]
        inside.sort(key=lambda c: c["y"], reverse=True)

        parts = []
        for c in inside:
            L = branch_len(int(c["x"]), int(xn))
            parts.append(f"{c['nw']}:{fmt_len(L)}")

        merged_nw = "(" + ",".join(parts) + ")"
        merged_cluster = {"y": int(yn), "x": int(xn), "nw": merged_nw}

        clusters = [c for j, c in enumerate(clusters) if j not in inside_idx]
        clusters.append(merged_cluster)
        clusters.sort(key=lambda c: c["y"], reverse=True)

    # --------------------------------------------------------
    # Finalisation
    # --------------------------------------------------------
    if len(clusters) == 1:
        return clusters[0]["nw"] + ";"

    root_x = min([x for (x, _y) in internals_sorted] + [x for (x, _y) in corners_int])

    clusters.sort(key=lambda c: c["y"], reverse=True)
    parts = []
    for c in clusters:
        L = branch_len(int(c["x"]), int(root_x))
        parts.append(f"{c['nw']}:{fmt_len(L)}")

    return "(" + ",".join(parts) + ");"


# ============================================================
# MAIN
# ============================================================
def main():
    import json
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--in-json", help="Input JSON file (list of dicts)", default="../heatmap/train_dataset/points_dataset.json")
    ap.add_argument("--out-txt", help="Output TXT (1 JSON per line)", default="./newick.txt")
    ap.add_argument("--x-tol", type=float, default=18.0)
    ap.add_argument("--y-tol", type=float, default=14.0)
    ap.add_argument("--decimals", type=int, default=6)
    args = ap.parse_args()

    with open(args.in_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Input JSON must be a list")

    with open(args.out_txt, "w", encoding="utf-8") as fout:
        for item in data:
            image = item.get("image", None)

            leaves = [(int(round(float(x))), int(round(float(y)))) for x, y in item.get("leaf", [])]
            internals = [(int(round(float(x))), int(round(float(y)))) for x, y in item.get("node", [])]
            corners = [(int(round(float(x))), int(round(float(y)))) for x, y in item.get("corner", [])]

            newick = build_newick(
                leaves,
                internals,
                corners,
                x_tol=args.x_tol,
                y_tol=args.y_tol,
                decimals=args.decimals,
                # leaf_names=...  (si tu veux le passer depuis ailleurs)
            )

            line = {
                "image": image,
                "newick": newick,
            }

            fout.write(json.dumps(line, ensure_ascii=False) + "\n")

    print(f"Done. Wrote {len(data)} lines to {args.out_txt}")


if __name__ == "__main__":
    main()
