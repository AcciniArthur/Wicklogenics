import os
import customtkinter as ctk
from tkinter import filedialog, messagebox

from src.core.scripts.models import PointType, POINT_COLORS, Point
from src.core.scripts.pipeline import run_pipeline
from src.core.scripts.postprocessing.newick import build_newick  # <-- NEW: on utilise ton build_newick du fichier newick.py
from src.gui.widgets.image_viewer import ImageViewer
from src.core.scripts.newick import compute_newick



class HomePage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)

        self.image_path: str | None = None

        # layout
        self.grid_rowconfigure(1, weight=1)
        self.grid_columnconfigure(0, weight=1)

        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, sticky="ew", padx=14, pady=12)
        header.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(header, text="Home", font=ctk.CTkFont(size=18, weight="bold")).grid(row=0, column=0, sticky="w")

        btns = ctk.CTkFrame(header, fg_color="transparent")
        btns.grid(row=0, column=2, sticky="e")

        ctk.CTkButton(btns, text="Charger l'image", command=self.load_image_dialog).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Run pipeline", command=self.run_pipeline_and_show_points).pack(side="left", padx=6)
        ctk.CTkButton(btns, text="Calculer le Newick", command=self.compute_and_show_newick).pack(side="left", padx=6)

        main = ctk.CTkFrame(self)
        main.grid(row=1, column=0, sticky="nsew", padx=14, pady=(0, 14))
        main.grid_rowconfigure(0, weight=1)
        main.grid_columnconfigure(0, weight=1)
        main.grid_columnconfigure(1, weight=0)
        main.grid_columnconfigure(2, weight=0)

        left = ctk.CTkFrame(main)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12), pady=12)
        left.grid_rowconfigure(1, weight=1)
        left.grid_columnconfigure(0, weight=1)

        self.drop_zone = ctk.CTkLabel(left, text="Drag & Drop une image ici (optionnel)\nou utilise 'Charger l'image'", height=60)
        self.drop_zone.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 10))

        self.viewer = ImageViewer(left)
        self.viewer.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 10))

        # Panel "Noms des feuilles"
        leaves_panel = ctk.CTkFrame(main, width=180)
        leaves_panel.grid(row=0, column=1, sticky="ns", padx=(0, 12), pady=12)
        leaves_panel.grid_propagate(False)

        ctk.CTkLabel(leaves_panel, text="Noms des feuilles", font=ctk.CTkFont(size=14, weight="bold")).pack(
            anchor="w", padx=12, pady=(12, 8)
        )

        self.leaves_list = ctk.CTkScrollableFrame(leaves_panel, width=160, height=500)
        self.leaves_list.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        right = ctk.CTkFrame(main, width=340)
        right.grid(row=0, column=2, sticky="ns", padx=(0, 12), pady=12)
        right.grid_propagate(False)

        ctk.CTkLabel(right, text="Édition des points", font=ctk.CTkFont(size=14, weight="bold")).pack(anchor="w", padx=12, pady=(12, 10))

        self.mode_var = ctk.StringVar(value="move")
        mode_box = ctk.CTkFrame(right)
        mode_box.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(mode_box, text="Mode").pack(anchor="w", padx=10, pady=(10, 6))
        ctk.CTkRadioButton(mode_box, text="Déplacer", variable=self.mode_var, value="move", command=self._apply_mode).pack(anchor="w", padx=10, pady=4)
        ctk.CTkRadioButton(mode_box, text="Ajouter", variable=self.mode_var, value="add", command=self._apply_mode).pack(anchor="w", padx=10, pady=(0, 10))

        self.ptype_var = ctk.StringVar(value=PointType.NODE.value)
        type_box = ctk.CTkFrame(right)
        type_box.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(type_box, text="Type de point à ajouter").pack(anchor="w", padx=10, pady=(10, 6))

        for pt in [PointType.ROOT, PointType.NODE, PointType.CORNER, PointType.TIP]:
            ctk.CTkRadioButton(type_box, text=f"{pt.value} ({POINT_COLORS[pt]})", variable=self.ptype_var, value=pt.value).pack(anchor="w", padx=10, pady=3)
        ctk.CTkFrame(type_box, height=6, fg_color="transparent").pack()

        actions = ctk.CTkFrame(right)
        actions.pack(fill="x", padx=12, pady=(0, 10))
        ctk.CTkLabel(actions, text="Actions").pack(anchor="w", padx=10, pady=(10, 6))
        ctk.CTkButton(actions, text="Supprimer point sélectionné (Del)", command=self.viewer.delete_selected_point).pack(fill="x", padx=10, pady=6)
        ctk.CTkButton(actions, text="Vider tous les points", command=self.viewer.clear_points).pack(fill="x", padx=10, pady=(0, 10))

        helpbox = ctk.CTkFrame(right)
        helpbox.pack(fill="x", padx=12, pady=(0, 12))
        ctk.CTkLabel(helpbox, text="Raccourcis", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 6))
        ctk.CTkLabel(
            helpbox,
            text="• ← ↑ ↓ → / Clic droit + Drag: Déplacement\n• Zoom: molette\n• Pan: clic milieu / ou Shift + clic gauche\n• Déplacer point: clic + drag\n• Supprimer: Del\n ",
            justify="left"
        ).pack(anchor="w", padx=10, pady=(0, 10))

        self.viewer.set_home_page(self)
        self._setup_dnd_if_available()
        self._apply_mode()

        self.refresh_leaf_panel()

    def _apply_mode(self):
        self.viewer.set_mode(self.mode_var.get())
        
        

    def _show_drop_zone(self):
        self.drop_zone.grid()
        self.viewer.grid_configure(pady=(0, 10))

    def _hide_drop_zone(self):
        self.drop_zone.grid_remove()
        self.viewer.grid_configure(pady=(10, 10))

    def _setup_dnd_if_available(self):
        try:
            self.drop_zone.drop_target_register("DND_Files")  # type: ignore
            self.drop_zone.dnd_bind("<<Drop>>", self._on_drop)  # type: ignore
            self.drop_zone.configure(text="Drag & Drop une image ici\nou utilise 'Charger l'image'")
        except Exception:
            self.drop_zone.configure(text="Drag & Drop indisponible (installe tkinterdnd2)\nUtilise 'Charger l'image'")

    def _on_drop(self, event):
        data = event.data.strip()
        if data.startswith("{") and data.endswith("}"):
            data = data[1:-1]
        path = data.split()[0]
        self.load_image(path)

    def load_image_dialog(self):
        path = filedialog.askopenfilename(
            title="Choisir une image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"), ("All files", "*.*")],
        )
        if path:
            self.load_image(path)

    def load_image(self, path: str):
        if not os.path.exists(path):
            messagebox.showerror("Erreur", "Fichier introuvable.")
            return
        self.image_path = path
        try:
            self.viewer.load_image(path)
            self._hide_drop_zone()
            self.viewer.clear_points()
            self.refresh_leaf_panel()
        except Exception as e:
            messagebox.showerror("Erreur", f"Impossible de charger l'image:\n{e}")

    def run_pipeline_and_show_points(self):
        if not self.image_path:
            messagebox.showwarning("Info", "Charge une image d'abord.")
            return
        try:
            # run_pipeline doit retourner une liste[Point]
            points = run_pipeline(self.image_path)
            self.viewer.set_points(points)
            self.refresh_leaf_panel()
        except Exception as e:
            messagebox.showerror("Erreur pipeline", str(e))

    # ---------------------------
    # NEWICK
    # ---------------------------



    def compute_and_show_newick(self):
        pts = self.viewer.get_points()
        if not pts:
            messagebox.showwarning("Info", "Aucun point. Lance la pipeline ou ajoute des points.")
            return
        try:
            newick_str = compute_newick(pts)
        except Exception as e:
            messagebox.showerror("Erreur Newick", str(e))
            return
        NewickWindow(self, newick_str)


    # ---------------------------
    # Leaf panel
    # ---------------------------

    def refresh_leaf_panel(self):
                for child in self.leaves_list.winfo_children():
                    child.destroy()

                points = self.viewer.get_points()
                tip_indices = [i for i, p in enumerate(points) if p.ptype.value == "tip"]

                if not tip_indices:
                    ctk.CTkLabel(self.leaves_list, text="(aucune)", text_color="gray").pack(anchor="w", padx=6, pady=4)
                    return

                for n, idx in enumerate(tip_indices, start=1):
                    p = points[idx]

                    row = ctk.CTkFrame(self.leaves_list)
                    row.pack(fill="x", padx=6, pady=4)

                    ctk.CTkLabel(row, text=f"{n}.", width=28).pack(side="left", padx=(6, 6))

                    entry = ctk.CTkEntry(row)
                    entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
                    entry.insert(0, p.label or f"tip{n}")

                    def commit_label(tip_index=idx, widget=entry):
                        new_name = widget.get().strip()
                        if not new_name:
                            new_name = "tip"
                            widget.delete(0, "end")
                            widget.insert(0, new_name)
                        self.viewer.set_point_label(tip_index, new_name)

                    entry.bind("<Return>", lambda e, f=commit_label: f())
                    entry.bind("<FocusOut>", lambda e, f=commit_label: f())


class NewickWindow(ctk.CTkToplevel):
    def __init__(self, master, newick: str):
        super().__init__(master)
        self.title("Newick")
        self.geometry("720x320")
        self.minsize(520, 240)

        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(0, weight=1)

        self.textbox = ctk.CTkTextbox(self)
        self.textbox.grid(row=0, column=0, sticky="nsew", padx=12, pady=12)
        self.textbox.insert("1.0", newick)

        bottom = ctk.CTkFrame(self)
        bottom.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 12))
        bottom.grid_columnconfigure(0, weight=1)

        ctk.CTkButton(bottom, text="Sauvegarder (.nwk / .newick)", command=self.save).pack(side="right")

    def save(self):
        content = self.textbox.get("1.0", "end").strip()
        if not content:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".nwk",
            filetypes=[("Newick", "*.nwk *.newick *.tree"), ("All files", "*.*")],
            title="Sauvegarder le Newick",
        )
        if not path:
            return
        with open(path, "w", encoding="utf-8") as f:
            f.write(content + ("\n" if not content.endswith("\n") else ""))
