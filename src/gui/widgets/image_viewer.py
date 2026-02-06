import math
import tkinter as tk
import customtkinter as ctk
from PIL import Image, ImageTk

from src.core.scripts.models import Point, PointType, POINT_COLORS

class ImageViewer(ctk.CTkFrame):
    """
    Viewer robuste:
    - image centrée automatiquement
    - pan = translation du top-left
    - zoom garde le point sous la souris
    - coords canvas<->image exactes
    """
    def __init__(self, master):
        super().__init__(master)

        self.canvas = tk.Canvas(self, bg="#111111", highlightthickness=0)
        self.canvas.pack(fill="both", expand=True)

        self._img_pil: Image.Image | None = None
        self._img_tk: ImageTk.PhotoImage | None = None
        self._img_id: int | None = None

        self.scale = 1.0
        self.min_scale = 0.1
        self.max_scale = 8.0

        # Pan offsets (in canvas pixels) applied on top of "center image"
        self.pan_x = 0.0
        self.pan_y = 0.0

        # Top-left of rendered image in canvas coords (computed)
        self.img_x0 = 0.0
        self.img_y0 = 0.0

        self.points: list[Point] = []
        self.point_radius = 5
        self._draw_ids: list[int] = []

        self.mode = "move"
        self.selected_index: int | None = None
        self._drag_point_index: int | None = None

        self._panning = False
        self._pan_last = (0, 0)

        self._home_page = None
        self._bind_events()
        self.canvas.configure(takefocus=True)


    def set_home_page(self, home_page):
        self._home_page = home_page

    def set_mode(self, mode: str):
        self.mode = mode

    def load_image(self, path: str):
        self._img_pil = Image.open(path).convert("RGBA")
        self.scale = 1.0
        self.pan_x = 0.0
        self.pan_y = 0.0
        self.selected_index = None
        # force layout sizes updated
        self.update_idletasks()
        self.canvas.focus_set()
        self._render_image_and_points()

    def set_points(self, points: list[Point]):
        self.points = list(points)
        self.selected_index = None
        self._redraw_points()

    def get_points(self) -> list[Point]:
        return list(self.points)

    def clear_points(self):
        self.points = []
        self.selected_index = None
        self._redraw_points()

    def delete_selected_point(self):
        if self.selected_index is None:
            return
        if 0 <= self.selected_index < len(self.points):
            self.points.pop(self.selected_index)
        self.selected_index = None
        self._redraw_points()

    def _bind_events(self):
        self.canvas.bind("<Configure>", lambda e: self._render_image_and_points())

        self.canvas.bind("<Button-1>", lambda e: (self.canvas.focus_set(), self._on_left_down(e)))
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)

        self.canvas.bind("<Button-2>", lambda e: (self.canvas.focus_set(), self._on_middle_down(e)))
        self.canvas.bind("<B2-Motion>", self._on_middle_drag)
        self.canvas.bind("<ButtonRelease-2>", self._on_middle_up)

        self.canvas.bind("<Shift-Button-1>", self._on_middle_down)
        self.canvas.bind("<Shift-B1-Motion>", self._on_middle_drag)
        self.canvas.bind("<Shift-ButtonRelease-1>", self._on_middle_up)

        self.canvas.bind("<MouseWheel>", self._on_mousewheel)      # Windows
        self.canvas.bind("<Button-4>", self._on_mousewheel_linux)  # Linux
        self.canvas.bind("<Button-5>", self._on_mousewheel_linux)

        # Flèches = pan
        self.canvas.bind("<Left>", self._on_arrow_key)
        self.canvas.bind("<Right>", self._on_arrow_key)
        self.canvas.bind("<Up>", self._on_arrow_key)
        self.canvas.bind("<Down>", self._on_arrow_key)

        self.canvas.bind_all("<Delete>", lambda e: self.delete_selected_point())

    # ---------- Coordinate mapping (robuste) ----------
    def _compute_img_top_left(self):
        if not self._img_pil:
            self.img_x0 = 0.0
            self.img_y0 = 0.0
            return

        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())

        sw = self._img_pil.width * self.scale
        sh = self._img_pil.height * self.scale

        # center + pan
        self.img_x0 = (cw - sw) / 2.0 + self.pan_x
        self.img_y0 = (ch - sh) / 2.0 + self.pan_y

    def image_to_canvas(self, x: float, y: float):
        return (self.img_x0 + x * self.scale, self.img_y0 + y * self.scale)

    def canvas_to_image(self, x: float, y: float):
        return ((x - self.img_x0) / self.scale, (y - self.img_y0) / self.scale)

    # ---------- Rendering ----------
    def _render_image_and_points(self):
        self._render_image()
        self._redraw_points()

    def _render_image(self):
        if not self._img_pil:
            self.canvas.delete("IMG")
            return

        self._compute_img_top_left()

        w = max(1, int(self._img_pil.width * self.scale))
        h = max(1, int(self._img_pil.height * self.scale))
        resized = self._img_pil.resize((w, h), Image.Resampling.LANCZOS)
        self._img_tk = ImageTk.PhotoImage(resized)

        self.canvas.delete("IMG")
        # draw image using top-left anchor
        self._img_id = self.canvas.create_image(self.img_x0, self.img_y0, image=self._img_tk, anchor="nw", tags=("IMG",))

    def _redraw_points(self):
        for did in self._draw_ids:
            self.canvas.delete(did)
        self._draw_ids = []

        if not self._img_pil:
            return

        for i, p in enumerate(self.points):
            cx, cy = self.image_to_canvas(p.x, p.y)
            r = self.point_radius + (2 if i == self.selected_index else 0)
            color = POINT_COLORS[p.ptype]

            oid = self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r, outline="black", width=1, fill=color)
            self._draw_ids.append(oid)

            if p.label:
                tid = self.canvas.create_text(cx + 10, cy - 10, text=p.label, fill="black", font=("Segoe UI", 10), anchor="nw")
                self._draw_ids.append(tid)

    # ---------- Interaction ----------
    def _on_left_down(self, event):
        if not self._img_pil:
            return

        if self.mode == "add":
            ix, iy = self.canvas_to_image(event.x, event.y)
            if 0 <= ix <= self._img_pil.width and 0 <= iy <= self._img_pil.height:
                ptype = PointType.NODE
                if self._home_page is not None:
                    ptype = PointType(self._home_page.ptype_var.get())

                label = None
                if ptype == PointType.TIP:
                    label = f"tip{sum(1 for p in self.points if p.ptype == PointType.TIP) + 1}"

                self.points.append(Point(ix, iy, ptype, label))
                self.selected_index = len(self.points) - 1
                self._redraw_points()
            return

        idx = self._hit_test(event.x, event.y, threshold=10)
        self.selected_index = idx
        self._drag_point_index = idx
        self._redraw_points()

    def _on_left_drag(self, event):
        if not self._img_pil:
            return
        if self._drag_point_index is None:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)
        ix = max(0, min(ix, self._img_pil.width))
        iy = max(0, min(iy, self._img_pil.height))

        p = self.points[self._drag_point_index]
        p.x, p.y = ix, iy
        self._redraw_points()

    def _on_left_up(self, event):
        self._drag_point_index = None

    def _on_middle_down(self, event):
        self._panning = True
        self._pan_last = (event.x, event.y)

    def _on_middle_drag(self, event):
        if not self._panning:
            return
        dx = event.x - self._pan_last[0]
        dy = event.y - self._pan_last[1]
        self._pan_last = (event.x, event.y)

        self.pan_x += dx
        self.pan_y += dy
        self._render_image_and_points()

    def _on_middle_up(self, event):
        self._panning = False

    def _on_mousewheel(self, event):
        if not self._img_pil:
            return
        factor = 1.1 if event.delta > 0 else 1 / 1.1
        self._zoom_at(event.x, event.y, factor)

    def _on_mousewheel_linux(self, event):
        if not self._img_pil:
            return
        factor = 1.1 if event.num == 4 else 1 / 1.1
        self._zoom_at(event.x, event.y, factor)

    def _zoom_at(self, canvas_x: float, canvas_y: float, factor: float):
        old_scale = self.scale
        new_scale = max(self.min_scale, min(self.max_scale, old_scale * factor))
        if math.isclose(new_scale, old_scale):
            return

        # image coords under cursor before zoom
        self._compute_img_top_left()
        ix, iy = self.canvas_to_image(canvas_x, canvas_y)

        self.scale = new_scale

        # After scale change, adjust pan so that (ix, iy) stays under cursor:
        # canvas_x = img_x0_new + ix*scale
        # img_x0_new = center_x0_new + pan_x_new
        # => pan_x_new = canvas_x - center_x0_new - ix*scale
        cw = max(1, self.canvas.winfo_width())
        ch = max(1, self.canvas.winfo_height())
        sw = self._img_pil.width * self.scale
        sh = self._img_pil.height * self.scale
        center_x0_new = (cw - sw) / 2.0
        center_y0_new = (ch - sh) / 2.0

        self.pan_x = canvas_x - center_x0_new - ix * self.scale
        self.pan_y = canvas_y - center_y0_new - iy * self.scale

        self._render_image_and_points()

    def _hit_test(self, canvas_x: float, canvas_y: float, threshold: float = 10.0):
        best_i = None
        best_d = 1e18
        for i, p in enumerate(self.points):
            px, py = self.image_to_canvas(p.x, p.y)
            d = (px - canvas_x) ** 2 + (py - canvas_y) ** 2
            if d < best_d:
                best_d = d
                best_i = i
        if best_i is None:
            return None
        if best_d <= threshold ** 2:
            return best_i
        return None
    
    def _on_arrow_key(self, event):
        if not self._img_pil:
            return

        # vitesse de déplacement
        step = 50
        # Shift = plus rapide
        if event.state & 0x0001:
            step = 60
        # Ctrl = plus précis
        if event.state & 0x0004:
            step = 5

        if event.keysym == "Left":
            self.pan_x += step
        elif event.keysym == "Right":
            self.pan_x -= step
        elif event.keysym == "Up":
            self.pan_y += step
        elif event.keysym == "Down":
            self.pan_y -= step

        self._render_image_and_points()

