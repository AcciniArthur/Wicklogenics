"""
Système de vote pour corriger les nodes et corners manquants.
Applique des règles géométriques et consolide les votes pour détecter les points manquants.
"""

from typing import List, Tuple, Optional, Dict, Literal

Point = Tuple[int, int, float]
Side = Literal["gauche", "droite", "sym", "milieu", ""]
FutureItem = Tuple[str, int, int, float, Side, int]

def _abs(a: int) -> int:
    """Valeur absolue."""
    return -a if a < 0 else a


def _near(a: int, b: int, delta: int) -> bool:
    """Vérifie si deux valeurs sont proches (distance <= delta)."""
    return _abs(a - b) <= delta

def _exists_leaf_or_node_right_of_y(
    leaves: List[Point],
    nodes: List[Point],
    corners: List[Point],
    ref_x: int,
    y: int,
    delta: int
) -> bool:
    """
    Vérifie la structure à droite d'un corner sur la ligne horizontale y.
    Prend le point le PLUS PROCHE à droite et valide s'il s'agit d'une feuille ou d'un node.
    """
    candidates = []

    for px, py, _ in leaves:
        if px > ref_x and _near(py, y, delta):
            candidates.append(("leaf", px))

    for px, py, _ in nodes:
        if px > ref_x and _near(py, y, delta):
            candidates.append(("node", px))

    for px, py, _ in corners:
        if px > ref_x and _near(py, y, delta):
            candidates.append(("corner", px))

    if not candidates:
        return False

    candidates.sort(key=lambda t: t[1])
    first_type, _ = candidates[0]

    return first_type in ("leaf", "node")


def _pt_key(x: int, y: int, delta: int) -> Tuple[int, int]:
    """Crée une clé de bucket pour regrouper les points proches."""
    d = max(1, delta)
    return (int(round(x / d)), int(round(y / d)))


def _exists_any_node_between(nodes: List[Point], x: int, y1: int, y2: int, delta: int) -> bool:
    """Vérifie si un node existe sur la même colonne x entre y1 et y2 (intervalle ouvert)."""
    ylo, yhi = (y1, y2) if y1 <= y2 else (y2, y1)
    for nx, ny, _ in nodes:
        if abs(nx - x) <= delta and (ylo + delta) < ny < (yhi - delta):
            return True
    return False


def _bucket_x(x: int, delta: int) -> int:
    """Calcule le bucket x pour regroupement."""
    return int(round(int(x) / max(1, delta)))


def _find_node_near_in_list(nodes_list: List[Point], x: int, y: int, delta: int) -> Optional[Point]:
    """Trouve un node proche de (x, y) dans la liste."""
    for nx, ny, ns in nodes_list:
        if abs(int(nx) - int(x)) <= delta and abs(int(ny) - int(y)) <= delta:
            return (nx, ny, ns)
    return None


def _is_symmetric_y(ny: int, y_top: int, y_bot: int, delta: int) -> bool:
    """Vérifie si ny est symétrique entre y_top et y_bot."""
    return abs((ny - y_top) - (y_bot - ny)) <= delta


def _nearest_left_on_same_y(
    corners: List[Point],
    nodes: List[Point],
    ref_x: int,
    y: int,
    delta: int,
) -> Tuple[str, Optional[Point]]:
    """
    Retourne le type et le point le plus proche à gauche sur la même horizontale y±delta.
    Type possible: "corner", "node", "none"
    """
    best_type = "none"
    best_pt = None
    best_dx = None

    for cx, cy, cs in corners:
        if cx < ref_x and _near(cy, y, delta):
            dx = ref_x - cx
            if best_dx is None or dx < best_dx:
                best_dx = dx
                best_type = "corner"
                best_pt = (cx, cy, cs)

    for nx, ny, ns in nodes:
        if nx < ref_x and _near(ny, y, delta):
            dx = ref_x - nx
            if best_dx is None or dx < best_dx:
                best_dx = dx
                best_type = "node"
                best_pt = (nx, ny, ns)

    return best_type, best_pt


def _nearest_left_on_same_y_for_leaf(
    corners: List[Point],
    nodes: List[Point],
    ref_x: int,
    y: int,
    delta: int,
) -> Tuple[str, Optional[Point]]:
    """
    Retourne le type et le point le plus proche à gauche sur la même horizontale y±delta (pour feuille).
    Type possible: "corner", "node", "none"
    """
    best_type = "none"
    best_pt = None
    best_dx = None

    for cx, cy, cs in corners:
        if cx < ref_x and _near(cy, y, delta):
            dx = ref_x - cx
            if best_dx is None or dx < best_dx:
                best_dx = dx
                best_type = "corner"
                best_pt = (cx, cy, cs)

    for nx, ny, ns in nodes:
        if nx < ref_x and _near(ny, y, delta):
            dx = ref_x - nx
            if best_dx is None or dx < best_dx:
                best_dx = dx
                best_type = "node"
                best_pt = (nx, ny, ns)

    return best_type, best_pt


def apply_rules_one_pass(
    nodes: List[Point],
    corners: List[Point],
    leaves: List[Point],
    score_threshold: float = 2.0,
    delta: int = 5,
    r2c: float = 0,
) -> Tuple[List[FutureItem], bool]:
    """
    Applique les règles géométriques en une seule passe.
    
    Règles appliquées:
    - R1: À gauche de chaque feuille doit se trouver un corner
    - R2: Détection des trios symétriques (node au centre de 2 corners)
    - R3: À gauche de chaque node doit se trouver un corner
    - R4: À droite de chaque corner doit se trouver une feuille ou un node
    
    Retourne: (liste_propositions, modification_possible)
    """
    future: List[FutureItem] = []

    # R1: À gauche de chaque feuille = corner
    for lx, ly, _ in leaves:
        t, pt = _nearest_left_on_same_y_for_leaf(corners, nodes, ref_x=lx, y=ly, delta=delta)

        if t == "corner":
            continue

        future.append(("corner", -1, int(ly), 1.5, "gauche", int(lx)))

    # R3: À gauche d'un node = corner
    for nx, ny, _ in nodes:
        t, pt = _nearest_left_on_same_y(corners, nodes, ref_x=nx, y=ny, delta=delta)

        if t == "corner":
            continue

        future.append(("corner", -1, int(ny), 1.0, "gauche", int(nx)))

    # R4: À droite d'un corner = leaf ou node
    for cx, cy, _ in corners:
        if not _exists_leaf_or_node_right_of_y(leaves, nodes, corners, ref_x=cx, y=cy, delta=delta):
            future.append(("node", -1, int(cy), 1.0, "droite", int(cx)))
    
    # R2: Détection des trios symétriques (node au milieu de 2 corners)
    corner_buckets: Dict[int, List[Point]] = {}
    for cx, cy, cs in corners:
        corner_buckets.setdefault(_bucket_x(cx, delta), []).append((cx, cy, cs))

    node_buckets: Dict[int, List[Point]] = {}
    for nx, ny, ns in nodes:
        node_buckets.setdefault(_bucket_x(nx, delta), []).append((nx, ny, ns))

    trios: List[Tuple[int, int, int, int, int, int]] = []
    remove_corners = set()
    remove_nodes = set()


    for kx, cgroup in corner_buckets.items():
        c_sorted = sorted(cgroup, key=lambda p: p[1])
        ngroup = node_buckets.get(kx, [])
        if len(c_sorted) < 2 or not ngroup:
            continue

        # nodes list for this x bucket
        n_list = list(ngroup)

        # scan corners adjacents -> check node au milieu + symétrie
        for i in range(len(c_sorted) - 1):
            c1 = c_sorted[i]
            c2 = c_sorted[i + 1]

            y_top = int(c1[1])
            y_bot = int(c2[1])
            x_mid = int(round((c1[0] + c2[0]) / 2))
            y_mid = int(round((y_top + y_bot) / 2))

            nmid = _find_node_near_in_list(n_list, x_mid, y_mid, delta)
            if nmid is None:
                continue

            # trio valide si symétrique (et corners adjacents dans la liste => top->bottom)
            if not _is_symmetric_y(int(nmid[1]), y_top, y_bot, delta):
                continue

            # enregistrer trio + marquer à retirer
            trios.append((int(nmid[0]), int(nmid[1]), int(c1[0]), int(c2[0]), y_top, y_bot))

            remove_corners.add((_bucket_x(int(c1[0]), delta), _pt_key(int(c1[0]), y_top, delta)[1]))
            remove_corners.add((_bucket_x(int(c2[0]), delta), _pt_key(int(c2[0]), y_bot, delta)[1]))
            remove_nodes.add((_bucket_x(int(nmid[0]), delta), _pt_key(int(nmid[0]), int(nmid[1]), delta)[1]))

    # --- 2) créer listes restantes (sans les trios)
    corners_remain: List[Point] = []
    for cx, cy, cs in corners:
        k = (_bucket_x(int(cx), delta), _pt_key(int(cx), int(cy), delta)[1])
        if k in remove_corners:
            continue
        corners_remain.append((cx, cy, cs))

    nodes_remain: List[Point] = []
    for nx, ny, ns in nodes:
        k = (_bucket_x(int(nx), delta), _pt_key(int(nx), int(ny), delta)[1])
        if k in remove_nodes:
            continue
        nodes_remain.append((nx, ny, ns))

    corners = corners_remain
    nodes = nodes_remain

    # --- 3) buckets "restants" + trios par x bucket (obstacles)
    corner_buckets = {}
    for cx, cy, cs in corners:
        corner_buckets.setdefault(_bucket_x(cx, delta), []).append((cx, cy, cs))

    node_buckets = {}
    for nx, ny, ns in nodes:
        node_buckets.setdefault(_bucket_x(nx, delta), []).append((nx, ny, ns))

    trio_buckets: Dict[int, List[Tuple[int, int, int, int, int, int]]] = {}
    for x_node, y_node, x_top, x_bot, y_top, y_bot in trios:
        trio_buckets.setdefault(_bucket_x(x_node, delta), []).append((x_node, y_node, x_top, x_bot, y_top, y_bot))

    def _build_tokens(kx: int) -> List[Tuple]:
        """Construit une liste de tokens triée par y, incluant les trios comme obstacles."""
        toks: List[Tuple] = []
        for cx, cy, cs in sorted(corner_buckets.get(kx, []), key=lambda p: p[1]):
            toks.append(("corner", int(cx), int(cy)))
        for nx, ny, ns in sorted(node_buckets.get(kx, []), key=lambda p: p[1]):
            toks.append(("node", int(nx), int(ny)))
        for x_node, y_node, x_top, x_bot, y_top, y_bot in trio_buckets.get(kx, []):
            toks.append(("trio", int(x_node), int(y_node), int(y_top), int(y_bot)))
        toks.sort(key=lambda t: t[2])
        return toks

    # R2a: Détection des corners symétriques manquants
    for kx in set(list(corner_buckets.keys()) + list(node_buckets.keys()) + list(trio_buckets.keys())):
        tokens = _build_tokens(kx)
        if len(tokens) < 2:
            continue

        for i in range(len(tokens) - 1):
            t1 = tokens[i]
            t2 = tokens[i + 1]

            if t1[0] == "node" and t2[0] == "corner":
                nx, ny = t1[1], t1[2]
                cx, cy = t2[1], t2[2]
                y_sym = int(2 * ny - cy)
                future.append(("corner", int(nx), int(y_sym), 1.0, "sym", int(nx)))
                continue

            if t1[0] == "corner" and t2[0] == "node":
                cx, cy = t1[1], t1[2]
                nx, ny = t2[1], t2[2]
                y_sym = int(2 * ny - cy)

                nxt = tokens[i + 2] if (i + 2) < len(tokens) else None
                if nxt is None or nxt[0] in ("node", "trio"):
                    future.append(("corner", int(nx), int(y_sym), 1.0, "sym", int(nx)))
                    continue

                if nxt[0] == "corner":
                    c3y = int(nxt[2])
                    if abs(c3y - y_sym) > delta:
                        future.append(("corner", int(nx), int(y_sym), 1.0, "sym", int(nx)))

    # R2b: Détection des nodes centraux manquants entre deux corners adjacents
    for kx in set(list(corner_buckets.keys()) + list(trio_buckets.keys()) + list(node_buckets.keys())):
        tokens = _build_tokens(kx)
        if len(tokens) < 2:
            continue

        for i in range(len(tokens) - 1):
            a = tokens[i]
            b = tokens[i + 1]

            if not (a[0] == "corner" and b[0] == "corner"):
                continue

            prev_t = tokens[i - 1][0] if i - 1 >= 0 else None
            next_t = tokens[i + 2][0] if i + 2 < len(tokens) else None

            if prev_t not in (None, "trio", "node"):
                continue
            if next_t not in (None, "trio", "node"):
                continue

            y1, y2 = int(a[2]), int(b[2])
            x_ref = int(round((a[1] + b[1]) / 2))
            if _exists_any_node_between(nodes, x=x_ref, y1=y1, y2=y2, delta=delta):
                continue

            x_mid = int(round((a[1] + b[1]) / 2))
            y_mid = int(round((a[2] + b[2]) / 2))
            future.append(("node", x_mid, y_mid, 1.0, "milieu", x_mid))

    # R2c: Détection des points isolés
    if r2c != 0:
        for kx in set(list(corner_buckets.keys()) + list(node_buckets.keys()) + list(trio_buckets.keys())):
            tokens = _build_tokens(kx)
            if not tokens:
                continue

            picked = False
            for i in range(len(tokens)):
                if tokens[i][0] not in ("corner", "node"):
                    continue

                prev_t = tokens[i - 1][0] if i - 1 >= 0 else None
                next_t = tokens[i + 1][0] if i + 1 < len(tokens) else None

                prev_ok = (prev_t is None) or (prev_t == "trio")
                next_ok = (next_t is None) or (next_t == "trio")

                if not (prev_ok and next_ok):
                    continue

                t, x, y = tokens[i][0], tokens[i][1], tokens[i][2]

                if t == "corner":
                    future.append(("corner", int(x), -1, r2c, "", int(x)))
                    future.append(("node", int(x), -1, r2c, "", int(x)))
                else:
                    future.append(("corner", int(x), -1, r2c, "", int(x)))

                picked = True
                break

            if picked:
                continue

    would_modify = any(v >= score_threshold for (_, _, _, v, _, _) in future)
    return future, would_modify


def print_future_list(future: List[FutureItem], limit: Optional[int] = None) -> None:
    """Affiche la liste des propositions futures pour debug."""
    print("=" * 80)
    print(f"FUTURE LIST  (n={len(future)})")
    print("format: (type, x, y, vote, side, ref_x)")
    print("-" * 80)

    n = len(future) if limit is None else min(len(future), limit)
    for i in range(n):
        ptype, x, y, vote, side, ref_x = future[i]
        print(f"{i:04d} | {ptype:6s} | x={x:5d} y={y:5d} | vote={vote:4.1f} | side={side:6s} | ref_x={ref_x:5d}")

    if limit is not None and len(future) > limit:
        print(f"... ({len(future) - limit} more)")
    print("=" * 80)

def _bucket(v: int, delta: int) -> int:
    """Calcule le bucket pour une valeur (évite division par 0)."""
    d = max(1, delta)
    return int(round(v / d))


def consolidate_future_and_select_points(
    future: List[FutureItem],
    score_threshold: float = 2.0,
    delta: int = 5,
    score_new: float = -0.70,
) -> Tuple[List[Point], List[Point], bool]:
    """
    Consolide les propositions futures et sélectionne les points à ajouter.
    
    - Agrège les votes sur des points précis (x et y connus)
    - Gère les entrées "axe" (x=-1, y connu) et "colonne" (y=-1, x connu)
    - Crée des intersections axe×colonne pour générer de nouveaux points
    - Sélectionne uniquement les points dont le score total >= score_threshold
    
    Retourne: (nodes_à_ajouter, corners_à_ajouter, modifié)
    """
    # Séparation en 3 catégories
    precise_xy: List[FutureItem] = [it for it in future if it[1] != -1 and it[2] != -1]
    axis: List[FutureItem] = [it for it in future if it[1] == -1 and it[2] != -1]
    col: List[FutureItem] = [it for it in future if it[1] != -1 and it[2] == -1]

    # Agrégation des points précis
    agg: Dict[Tuple[str, int, int], Dict[str, object]] = {}

    def _add_to_agg(ptype: str, x: int, y: int, vote: float, src: str, side: str, ref_x: int):
        """Ajoute ou fusionne un point dans l'agrégation."""
        k = (ptype, _bucket(x, delta), _bucket(y, delta))
        if k not in agg:
            agg[k] = {
                "ptype": ptype,
                "x": int(x),
                "y": int(y),
                "vote": float(vote),
                "support": [(src, side, ref_x, float(vote))],
            }
        else:
            d = agg[k]
            d["vote"] = float(d["vote"]) + float(vote)
            d["x"] = int(round((int(d["x"]) + int(x)) / 2))
            d["y"] = int(round((int(d["y"]) + int(y)) / 2))
            d["support"].append((src, side, ref_x, float(vote)))

    for ptype, x, y, vote, side, ref_x in precise_xy:
        _add_to_agg(ptype, int(x), int(y), float(vote), "precise", side, int(ref_x))

    # Intersection axe × colonne pour créer des points précis
    for ptype_a, _, y_axis, vote_a, side_a, ref_x_a in axis:
        if side_a not in ("gauche", "droite"):
            continue
        for ptype_c, x_col, _, vote_c, _, ref_x_c in col:
            if ptype_c != ptype_a:
                continue

            x_col = int(x_col)
            ref_x_a = int(ref_x_a)

            if side_a == "gauche" and not (x_col < ref_x_a):
                continue
            if side_a == "droite" and not (x_col > ref_x_a):
                continue

            _add_to_agg(
                ptype=str(ptype_a),
                x=int(x_col),
                y=int(y_axis),
                vote=0.0,
                src="intersect_seed",
                side=side_a,
                ref_x=ref_x_a
            )

    # Boost du meilleur candidat précis pour chaque axe
    for ptype, _, y, vote, side, ref_x in axis:
        if side not in ("droite", "gauche"):
            continue

        best_key = None
        best_score = None
        for k, d in agg.items():
            if d["ptype"] != ptype:
                continue
            cx = int(d["x"])
            cy = int(d["y"])
            if not _near(cy, y, delta):
                continue
            if side == "droite" and not (cx > ref_x):
                continue
            if side == "gauche" and not (cx < ref_x):
                continue

            sc = _abs(cy - y) * 10000 + _abs(cx - ref_x)
            if best_score is None or sc < best_score:
                best_score = sc
                best_key = k

        if best_key is not None:
            agg[best_key]["vote"] = float(agg[best_key]["vote"]) + float(vote)
            agg[best_key]["support"].append(("axis", side, int(ref_x), float(vote)))

    # Boost du meilleur candidat précis pour chaque colonne
    for ptype, x, _, vote, side, ref_x in col:
        x_col = int(x)

        best_key = None
        best_score = None
        for k, d in agg.items():
            if d["ptype"] != ptype:
                continue
            cx = int(d["x"])
            if not _near(cx, x_col, delta):
                continue
            sc = _abs(cx - x_col) * 10000 - float(d["vote"])
            if best_score is None or sc < best_score:
                best_score = sc
                best_key = k

        if best_key is not None:
            agg[best_key]["vote"] = float(agg[best_key]["vote"]) + float(vote)
            agg[best_key]["support"].append(("column", "", int(x_col), float(vote)))

    # Sélection des points avec vote >= threshold
    nodes_to_add: List[Point] = []
    corners_to_add: List[Point] = []

    for d in agg.values():
        total = float(d["vote"])
        if total < score_threshold:
            continue
        ptype = str(d["ptype"])
        x = int(d["x"])
        y = int(d["y"])
        if ptype == "node":
            nodes_to_add.append((x, y, float(score_new)))
        elif ptype == "corner":
            corners_to_add.append((x, y, float(score_new)))

    modified = (len(nodes_to_add) + len(corners_to_add)) > 0
    return nodes_to_add, corners_to_add, modified


def add_missing_root_node(
    nodes: List[Point],
    corners: List[Point],
    delta: int = 5,
    score_new: float = -0,
) -> Tuple[List[Point], bool]:
    """
    Ajoute le node racine manquant si:
    - Les 2 corners les plus à gauche sont alignés verticalement (même x ±delta)
    - Aucun node n'existe déjà près de leur milieu
    
    Retourne: (nodes_mis_à_jour, modifié)
    """
    if len(corners) < 2:
        return nodes, False

    corners_sorted = sorted(corners, key=lambda p: p[0])
    c1 = corners_sorted[0]

    c2 = None
    for cand in corners_sorted[1:]:
        if abs(cand[1] - c1[1]) > delta:
            c2 = cand
            break
    if c2 is None:
        return nodes, False

    x1, y1, _ = c1
    x2, y2, _ = c2

    if abs(x1 - x2) > delta:
        return nodes, False

    x_root = int(round((x1 + x2) / 2))
    y_root = int(round((y1 + y2) / 2))

    for nx, ny, _ in nodes:
        if abs(nx - x_root) <= delta and abs(ny - y_root) <= delta:
            return nodes, False

    nodes2 = nodes[:] + [(x_root, y_root, float(score_new))]
    return nodes2, True