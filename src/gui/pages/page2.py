import customtkinter as ctk

class Page2(ctk.CTkFrame):
    def __init__(self, master):
        super().__init__(master, corner_radius=0)
        ctk.CTkLabel(self, text="Page 2 (placeholder)", font=ctk.CTkFont(size=18, weight="bold")).pack(padx=20, pady=20)
