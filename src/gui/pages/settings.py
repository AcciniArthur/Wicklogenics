import customtkinter as ctk

class SettingsPage(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)

        ctk.CTkLabel(self, text="Settings (placeholder)", font=ctk.CTkFont(size=18, weight="bold")).pack(padx=20, pady=(20, 10))

        box = ctk.CTkFrame(self)
        box.pack(padx=20, pady=10, fill="x")

        ctk.CTkLabel(
            box,
            text="Idées:\n• taille des points\n• raccourcis\n• thème\n• export auto\n• paramètres pipeline",
            justify="left",
        ).pack(padx=14, pady=14, anchor="w")
