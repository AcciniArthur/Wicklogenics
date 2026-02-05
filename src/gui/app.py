from __future__ import annotations
import tkinter as tk

def _make_root():
    """
    DnD optionnel:
    - Si tkinterdnd2 est dispo, on crée une fenêtre Tk compatible drop.
    - Sinon, Tk classique.
    """
    try:
        from tkinterdnd2 import TkinterDnD  # type: ignore
        return TkinterDnD.Tk()
    except Exception:
        return tk.Tk()

def run_app():
    from src.gui.app import WicklogenicsApp
    root = _make_root()
    app = WicklogenicsApp(root)
    root.mainloop()

class WicklogenicsApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Wicklogenics")
        self.root.geometry("1200x750")
        self.root.minsize(1000, 650)

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(1, weight=1)

        