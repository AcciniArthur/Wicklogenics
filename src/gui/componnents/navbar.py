import customtkinter as ctk

class Navbar(ctk.CTkFrame):
    def __init__(self, master, on_navigate):
        super().__init__(master, width=220, corner_radius=0)
        self.on_navigate = on_navigate
        self.grid_propagate(False)

        self._active = None
        self.buttons = {}

        title = ctk.CTkLabel(self, text="Wicklogenics", font=ctk.CTkFont(size=16, weight="bold"))
        title.pack(pady=(18, 16), padx=12, anchor="w")

        self._add_btn("home", "Home")
        self._add_btn("page2", "Page 2")

        ctk.CTkFrame(self, fg_color="transparent").pack(expand=True, fill="both")

        self._add_btn("settings", "Settings", bottom=True)

    def _add_btn(self, key: str, text: str, bottom: bool = False):
        btn = ctk.CTkButton(self, text=text, anchor="w", command=lambda: self.on_navigate(key))
        if bottom:
            btn.pack(fill="x", padx=10, pady=(0, 14))
        else:
            btn.pack(fill="x", padx=10, pady=6)
        self.buttons[key] = btn

    def set_active(self, key: str):
        # simple visuel: désactive/active (tu peux raffiner plus tard)
        for k, b in self.buttons.items():
            if k == key:
                b.configure(state="disabled")
            else:
                b.configure(state="normal")
        self._active = key
