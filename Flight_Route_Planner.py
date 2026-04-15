"""
╔══════════════════════════════════════════════════════════════════╗
║        FLIGHT ROUTE PLANNER  —  Full Algorithm Visualizer        ║
║  BFS · DFS · Dijkstra · A* · Floyd-Warshall · Prim · Kruskal     ║
║  ✈ Animated plane · Edge weights · Bottom results panel          ║
╚══════════════════════════════════════════════════════════════════╝
Layout:
  ┌─────────┬────────────────────────────────────┐
  │  LEFT   │          MAP  CANVAS               │
  │ SIDEBAR │  (fills all available space)        │
  │         ├────────────────────────────────────┤
  │         │      BOTTOM  RESULTS  PANEL        │
  └─────────┴────────────────────────────────────┘

Requirements:  pip install matplotlib numpy
"""

import tkinter as tk
from tkinter import ttk, messagebox
import math, time, heapq, collections, datetime

try:
    import matplotlib
    matplotlib.use("TkAgg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# ══════════════════════════════════════════════════════════════════
# AIRPORTS  (code → (full name, lat, lon))
# ══════════════════════════════════════════════════════════════════
AIRPORTS = {
    "JFK": ("New York JFK",          40.64,  -73.78),
    "LAX": ("Los Angeles",           33.94, -118.41),
    "LHR": ("London Heathrow",       51.48,   -0.45),
    "CDG": ("Paris CDG",             49.01,    2.55),
    "DXB": ("Dubai Intl",            25.25,   55.36),
    "SIN": ("Singapore Changi",       1.36,  103.99),
    "NRT": ("Tokyo Narita",          35.77,  140.39),
    "SYD": ("Sydney Kingsford",     -33.95,  151.18),
    "GRU": ("Sao Paulo Guarulhos",  -23.43,  -46.47),
    "ORD": ("Chicago O'Hare",        41.98,  -87.91),
    "ATL": ("Atlanta Hartsfield",    33.64,  -84.43),
    "FRA": ("Frankfurt Main",        50.03,    8.57),
    "AMS": ("Amsterdam Schiphol",    52.31,    4.77),
    "ICN": ("Seoul Incheon",         37.46,  126.44),
    "PEK": ("Beijing Capital",       40.08,  116.58),
    "DEL": ("Delhi Indira Gandhi",   28.56,   77.10),
    "BOM": ("Mumbai Chhatrapati",    19.09,   72.87),
    "CAI": ("Cairo Intl",            30.12,   31.41),
    "JNB": ("Johannesburg OR Tambo", -26.13,   28.24),
    "YYZ": ("Toronto Pearson",       43.68,  -79.63),
    "MEX": ("Mexico City Intl",      19.44,  -99.07),
    "BOG": ("Bogota El Dorado",       4.70,  -74.15),
    "SCL": ("Santiago Arturo",      -33.39,  -70.79),
    "CPT": ("Cape Town Intl",       -33.96,   18.60),
    "DOH": ("Doha Hamad",            25.27,   51.61),
    "KUL": ("Kuala Lumpur Intl",      2.74,  101.71),
    "BKK": ("Bangkok Suvarnabhumi",  13.69,  100.75),
    "IST": ("Istanbul Ataturk",      40.98,   28.82),
    "SVO": ("Moscow Sheremetyevo",   55.97,   37.41),
    "MAD": ("Madrid Barajas",        40.47,   -3.57),
    "BCN": ("Barcelona El Prat",     41.30,    2.08),
    "MXP": ("Milan Malpensa",        45.63,    8.73),
    "ZRH": ("Zurich Intl",           47.46,    8.55),
    "VIE": ("Vienna Intl",           48.11,   16.57),
    "MNL": ("Manila Ninoy Aquino",   14.51,  121.02),
    "CGK": ("Jakarta Soekarno",      -6.13,  106.65),
    "HND": ("Tokyo Haneda",          35.55,  139.78),
    "PVG": ("Shanghai Pudong",       31.14,  121.81),
    "SFO": ("San Francisco Intl",    37.62, -122.38),
    "DFW": ("Dallas Ft Worth",       32.90,  -97.04),
}

# ══════════════════════════════════════════════════════════════════
# GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════
def _haversine(la1, lo1, la2, lo2):
    R = 6371
    p1, p2 = math.radians(la1), math.radians(la2)
    a = (math.sin(math.radians(la2-la1)/2)**2 +
         math.cos(p1)*math.cos(p2)*math.sin(math.radians(lo2-lo1)/2)**2)
    return int(R * 2 * math.asin(math.sqrt(a)))

def build_graph(max_km=9000):
    codes = list(AIRPORTS); g = {c: {} for c in codes}
    for i, a in enumerate(codes):
        la, lo = AIRPORTS[a][1], AIRPORTS[a][2]
        for b in codes[i+1:]:
            d = _haversine(la, lo, AIRPORTS[b][1], AIRPORTS[b][2])
            if d <= max_km: g[a][b] = d; g[b][a] = d
    return g

GRAPH = build_graph()
CODES = list(AIRPORTS)
INF   = float('inf')

# ══════════════════════════════════════════════════════════════════
# ALGORITHMS
# ══════════════════════════════════════════════════════════════════
def _cost(g, path):
    return sum(g[path[i]].get(path[i+1], 0) for i in range(len(path)-1))

def bfs(g, s, e):
    q, vis = collections.deque([[s]]), {s}
    while q:
        p = q.popleft(); n = p[-1]
        if n == e: return p, _cost(g, p)
        for nb in g[n]:
            if nb not in vis: vis.add(nb); q.append(p+[nb])
    return [], 0

def dfs(g, s, e):
    stk, vis = [[s]], set()
    while stk:
        p = stk.pop(); n = p[-1]
        if n in vis: continue
        vis.add(n)
        if n == e: return p, _cost(g, p)
        for nb in g[n]:
            if nb not in vis: stk.append(p+[nb])
    return [], 0

def dijkstra(g, s, e):
    dist = {n: INF for n in g}; dist[s] = 0
    prev = {n: None for n in g}; vis = set(); pq = [(0, s)]
    while pq:
        d, u = heapq.heappop(pq)
        if u in vis: continue
        vis.add(u)
        if u == e: break
        for v, w in g[u].items():
            if d+w < dist[v]:
                dist[v] = d+w; prev[v] = u; heapq.heappush(pq, (dist[v], v))
    path, cur = [], e
    while cur: path.append(cur); cur = prev[cur]
    path.reverse()
    if not path or path[0] != s: return [], 0
    return path, dist[e]

def _heur(a, b):
    return _haversine(AIRPORTS[a][1], AIRPORTS[a][2], AIRPORTS[b][1], AIRPORTS[b][2])

def astar(g, s, e):
    ost, vis = [(_heur(s,e), 0, s, [s])], {}
    while ost:
        f, gv, u, path = heapq.heappop(ost)
        if u == e: return path, gv
        if u in vis and vis[u] <= gv: continue
        vis[u] = gv
        for v, w in g[u].items():
            ng = gv+w; heapq.heappush(ost, (ng+_heur(v,e), ng, v, path+[v]))
    return [], 0

def floyd_warshall(g):
    nodes = list(g); n = len(nodes); idx = {c: i for i,c in enumerate(nodes)}
    dist  = [[INF]*n for _ in range(n)]; nxt = [[None]*n for _ in range(n)]
    for i in range(n): dist[i][i] = 0
    for u in g:
        for v, w in g[u].items():
            i, j = idx[u], idx[v]; dist[i][j] = w; nxt[i][j] = v
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][k]+dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k]+dist[k][j]; nxt[i][j] = nxt[i][k]
    def reconstruct(s, e):
        si, ei = idx[s], idx[e]
        if dist[si][ei] == INF: return [], 0
        path = [s]; cur = s
        while cur != e:
            nc = nxt[idx[cur]][ei]
            if nc is None: return [], 0
            path.append(nc); cur = nc
        return path, dist[si][ei]
    return reconstruct, dist, idx, nodes

def prim_mst(g):
    s = next(iter(g)); vis = {s}
    edges = [(w, s, v) for v, w in g[s].items()]; heapq.heapify(edges); mst = []
    while edges and len(vis) < len(g):
        w, u, v = heapq.heappop(edges)
        if v in vis: continue
        vis.add(v); mst.append((u, v, w))
        for nb, ew in g[v].items():
            if nb not in vis: heapq.heappush(edges, (ew, v, nb))
    return mst

def kruskal_mst(g):
    par, rnk = {n: n for n in g}, {n: 0 for n in g}
    def find(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx == ry: return False
        if rnk[rx] < rnk[ry]: rx, ry = ry, rx
        par[ry] = rx
        if rnk[rx] == rnk[ry]: rnk[rx] += 1
        return True
    mst = []
    for w, u, v in sorted({(g[u][v], u, v) for u in g for v in g[u] if u < v}):
        if union(u, v): mst.append((u, v, w))
    return mst

# ══════════════════════════════════════════════════════════════════
# THEME & ALGO META
# ══════════════════════════════════════════════════════════════════
C = {
    "bg":     "#07090f", "panel":  "#0c1220", "panel2": "#111c32",
    "border": "#1a2d4a", "accent": "#00d4ff", "text":   "#dde8f0",
    "sub":    "#4a6a88", "green":  "#00ff9d", "yellow": "#ffd166",
    "red":    "#ff4560", "orange": "#ff6b35", "purple": "#b388ff",
    "pink":   "#f72585",
}

ALGO_META = {
    "BFS":            {"color": "#00d4ff", "arc": 0.05, "lw": 2.2},
    "DFS":            {"color": "#ff6b35", "arc": 0.16, "lw": 2.0},
    "Dijkstra":       {"color": "#00ff9d", "arc": 0.10, "lw": 2.4},
    "A*":             {"color": "#ffd166", "arc": 0.03, "lw": 2.2},
    "Floyd-Warshall": {"color": "#f72585", "arc": 0.13, "lw": 2.0},
    "Prim MST":       {"color": "#b388ff", "arc": 0.00, "lw": 1.6},
    "Kruskal MST":    {"color": "#ff4560", "arc": 0.00, "lw": 1.6},
}
ALGOS = list(ALGO_META)

CONTINENTS = [
    [(-168,71),(-141,60),(-130,54),(-125,48),(-118,33),(-117,32),(-88,16),
     (-77,8),(-75,10),(-65,18),(-65,45),(-52,47),(-53,55),(-79,62),(-95,74),
     (-120,72),(-140,72),(-168,71)],
    [(-81,8),(-78,2),(-74,-4),(-70,-18),(-65,-25),(-70,-55),(-65,-55),
     (-57,-39),(-48,-27),(-35,-7),(-50,2),(-62,10),(-75,10),(-81,8)],
    [(2,51),(15,55),(28,70),(30,60),(25,65),(30,70),(28,72),(20,72),(15,70),
     (10,63),(5,62),(0,58),(-5,58),(-10,52),(-9,39),(-6,37),(5,43),(15,47),
     (20,46),(28,46),(32,42),(28,40),(20,37),(15,38),(12,44),(10,47),(5,47),(2,51)],
    [(-17,14),(-15,10),(-18,5),(-10,-1),(8,-5),(12,-18),(30,-30),(35,-35),
     (26,-35),(18,-28),(12,-17),(10,-5),(8,4),(2,6),(-2,5),(-6,5),(-15,10),(-17,14)],
    [(28,72),(50,73),(80,73),(100,72),(140,72),(166,68),(166,58),(140,47),
     (130,33),(122,32),(110,20),(100,3),(103,1),(100,5),(100,13),(88,22),
     (80,28),(72,22),(60,22),(50,28),(40,36),(35,37),(30,42),(28,46),(35,46),
     (40,45),(50,50),(60,54),(70,58),(80,65),(100,70),(120,72),(130,73),
     (140,73),(140,65),(120,62),(100,65),(80,70),(60,68),(40,68),(28,72)],
    [(114,-22),(120,-34),(130,-34),(138,-36),(146,-39),(148,-38),(152,-28),
     (152,-23),(144,-14),(138,-12),(132,-12),(124,-16),(114,-22)],
]

# ══════════════════════════════════════════════════════════════════
# APPLICATION
# ══════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("✈  Flight Route Planner  —  Multi-Algorithm Visualizer")
        self.configure(bg=C["bg"])
        self.state("zoomed")

        self._anim_job  = None
        self._plane_job = None
        self._plane_obj = None
        self._algo_times = {}

        self._style_ttk()
        self._build_ui()
        self._draw_base_map()

    # ── TTK theme ─────────────────────────────────────────────────
    def _style_ttk(self):
        s = ttk.Style(self); s.theme_use("clam")
        s.configure("TCombobox",
                    fieldbackground=C["panel2"], background=C["panel2"],
                    foreground=C["text"], selectbackground=C["border"],
                    selectforeground=C["accent"], arrowcolor=C["accent"],
                    borderwidth=0, relief="flat", font=("Consolas", 11))
        s.map("TCombobox", fieldbackground=[("readonly", C["panel2"])])
        s.configure("Vertical.TScrollbar",
                    background=C["panel2"], troughcolor=C["panel"],
                    arrowcolor=C["accent"])

    # ══════════════════════════════════════════════════════════════
    # TOP-LEVEL LAYOUT
    # ══════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── header ──────────────────────────────────────────────
        hdr = tk.Frame(self, bg="#040609", height=56)
        hdr.pack(fill="x"); hdr.pack_propagate(False)

        tk.Label(hdr, text="✈  FLIGHT ROUTE PLANNER",
                 font=("Consolas", 20, "bold"),
                 fg=C["accent"], bg="#040609").pack(side="left", padx=20, pady=10)
        tk.Label(hdr, text="|  Multi-Algorithm Aviation Optimizer",
                 font=("Consolas", 11), fg=C["sub"],
                 bg="#040609").pack(side="left")

        self._clock_var = tk.StringVar()
        tk.Label(hdr, textvariable=self._clock_var,
                 font=("Consolas", 10), fg=C["sub"],
                 bg="#040609").pack(side="right", padx=18)
        self._tick_clock()

        # ── body (horizontal split: left sidebar | right column) ─
        body = tk.Frame(self, bg=C["bg"])
        body.pack(fill="both", expand=True, padx=4, pady=4)

        # LEFT SIDEBAR — fixed width, full height
        self._left = tk.Frame(body, bg=C["panel"], width=320)
        self._left.pack(side="left", fill="y", padx=(0, 4))
        self._left.pack_propagate(False)

        # RIGHT COLUMN — map on top, bottom panel below
        self._col = tk.Frame(body, bg=C["bg"])
        self._col.pack(side="left", fill="both", expand=True)

        # Map frame — expands vertically
        self._map_frame = tk.Frame(self._col, bg=C["bg"])
        self._map_frame.pack(fill="both", expand=True)

        # Bottom results panel — fixed height
        self._bot = tk.Frame(self._col, bg=C["panel"], height=230)
        self._bot.pack(fill="x", pady=(4, 0))
        self._bot.pack_propagate(False)

        self._build_left()
        self._build_map()
        self._build_bottom()

    def _tick_clock(self):
        self._clock_var.set(datetime.datetime.now().strftime("  %a %d %b %Y   %H:%M:%S  "))
        self.after(1000, self._tick_clock)

    # ══════════════════════════════════════════════════════════════
    # LEFT SIDEBAR
    # ══════════════════════════════════════════════════════════════
    def _build_left(self):
        p = self._left

        # ── Route config ──
        self._sec(p, "ROUTE CONFIGURATION")

        tk.Label(p, text="  Origin Airport", font=("Consolas", 10),
                 fg=C["sub"], bg=C["panel"]).pack(anchor="w", padx=14, pady=(8,2))
        self.v_orig = tk.StringVar(value="JFK")
        cb_o = ttk.Combobox(p, textvariable=self.v_orig, values=sorted(CODES),
                            state="readonly", font=("Consolas", 11))
        cb_o.pack(fill="x", padx=14, pady=(0,4))
        cb_o.bind("<<ComboboxSelected>>", lambda e: self._preview())

        self._lbl_orig = tk.Label(p, text="", font=("Consolas", 8),
                                  fg=C["accent"], bg=C["panel"], anchor="w")
        self._lbl_orig.pack(fill="x", padx=16, pady=(0,6))

        tk.Label(p, text="  Destination Airport", font=("Consolas", 10),
                 fg=C["sub"], bg=C["panel"]).pack(anchor="w", padx=14, pady=(2,2))
        self.v_dest = tk.StringVar(value="DXB")
        cb_d = ttk.Combobox(p, textvariable=self.v_dest, values=sorted(CODES),
                            state="readonly", font=("Consolas", 11))
        cb_d.pack(fill="x", padx=14, pady=(0,4))
        cb_d.bind("<<ComboboxSelected>>", lambda e: self._preview())

        self._lbl_dest = tk.Label(p, text="", font=("Consolas", 8),
                                  fg=C["yellow"], bg=C["panel"], anchor="w")
        self._lbl_dest.pack(fill="x", padx=16, pady=(0,2))

        self._lbl_dist = tk.Label(p, text="", font=("Consolas", 9, "bold"),
                                  fg=C["orange"], bg=C["panel"], anchor="w")
        self._lbl_dist.pack(fill="x", padx=16, pady=(0,4))
        self._preview()

        # ── Algorithm selector ──
        self._sec(p, "SELECT ALGORITHM")
        self.v_algo = tk.StringVar(value="Dijkstra")
        for algo in ALGOS:
            col = ALGO_META[algo]["color"]
            row = tk.Frame(p, bg=C["panel"])
            row.pack(fill="x", padx=10, pady=1)
            tk.Frame(row, bg=col, width=4, height=26).pack(side="left")
            tk.Radiobutton(row, text=f"  {algo}", variable=self.v_algo,
                           value=algo, font=("Consolas", 11, "bold"),
                           fg=col, bg=C["panel"], selectcolor="#182840",
                           activebackground=C["panel"], relief="flat", bd=0
                           ).pack(side="left", anchor="w", pady=2)

        tk.Label(p, text="  ★ Prim & Kruskal → full MST overlay",
                 font=("Consolas", 8), fg=C["sub"], bg=C["panel"]
                 ).pack(anchor="w", padx=14, pady=(4,0))

        # ── Options ──
        self._sec(p, "DISPLAY OPTIONS")
        self.v_anim     = tk.BooleanVar(value=True)
        self.v_plane    = tk.BooleanVar(value=True)
        self.v_weights  = tk.BooleanVar(value=True)
        self.v_all_edge = tk.BooleanVar(value=False)
        for var, txt in [
            (self.v_anim,    "  ✈  Animate path drawing"),
            (self.v_plane,   "  ✈  Show flying plane marker"),
            (self.v_weights, "  ⊡  Show edge weights (km)"),
            (self.v_all_edge,"  ⊞  Show all graph edges"),
        ]:
            tk.Checkbutton(p, text=txt, variable=var,
                           font=("Consolas", 10), fg=C["text"],
                           bg=C["panel"], selectcolor="#182840",
                           activebackground=C["panel"]
                           ).pack(anchor="w", padx=14, pady=2)

        # ── Buttons ──
        tk.Frame(p, bg=C["panel"], height=6).pack()
        self._btn(p, "▶   FIND ROUTE",              C["accent"], "#000",  self._run_algorithm, pady=11, sz=13)
        tk.Frame(p, bg=C["panel"], height=3).pack()
        self._btn(p, "⊞   COMPARE ALL ALGORITHMS",  C["yellow"], "#000",  self._compare_all,   pady=9,  sz=11)
        tk.Frame(p, bg=C["panel"], height=3).pack()
        self._btn(p, "⟳   FLOYD-WARSHALL (all-pairs)",C["pink"], "#fff",  self._run_fw,        pady=8,  sz=10)
        tk.Frame(p, bg=C["panel"], height=3).pack()
        self._btn(p, "✕   CLEAR MAP",               C["border"], C["sub"], self._clear_map,     pady=8,  sz=10)

        # ── Graph stats ──
        self._sec(p, "GRAPH STATISTICS")
        edges = sum(len(v) for v in GRAPH.values())//2
        for lbl, val in [
            ("Airports (nodes)",      str(len(GRAPH))),
            ("Flight routes (edges)", str(edges)),
            ("Max route length",      "9,000 km"),
            ("Algorithms available",  str(len(ALGOS))),
        ]:
            row = tk.Frame(p, bg=C["panel"])
            row.pack(fill="x", padx=14, pady=1)
            tk.Label(row, text=lbl, font=("Consolas", 8),
                     fg=C["sub"], bg=C["panel"]).pack(side="left")
            tk.Label(row, text=val, font=("Consolas", 9, "bold"),
                     fg=C["text"], bg=C["panel"]).pack(side="right", padx=4)

    def _preview(self):
        o, d = self.v_orig.get(), self.v_dest.get()
        if o in AIRPORTS:
            nm, la, lo = AIRPORTS[o]
            self._lbl_orig.configure(text=f"  {nm}  ({la:.2f}°, {lo:.2f}°)")
        if d in AIRPORTS:
            nm, la, lo = AIRPORTS[d]
            self._lbl_dest.configure(text=f"  {nm}  ({la:.2f}°, {lo:.2f}°)")
        if o in AIRPORTS and d in AIRPORTS and o != d:
            gc = _haversine(AIRPORTS[o][1], AIRPORTS[o][2],
                            AIRPORTS[d][1], AIRPORTS[d][2])
            self._lbl_dist.configure(text=f"  ✈ Great-circle: {gc:,} km")
        else:
            self._lbl_dist.configure(text="")

    # ══════════════════════════════════════════════════════════════
    # MAP CANVAS  (fills the top portion of the right column)
    # ══════════════════════════════════════════════════════════════
    def _build_map(self):
        p = self._map_frame
        if not HAS_MPL:
            tk.Label(p, text="matplotlib not found.\npip install matplotlib numpy",
                     fg=C["red"], bg=C["bg"], font=("Consolas",14)).pack(expand=True)
            return

        self.fig, self.ax = plt.subplots(figsize=(16, 8), facecolor=C["bg"])
        self.ax.set_facecolor(C["bg"])
        self.mpl_canvas = FigureCanvasTkAgg(self.fig, master=p)
        self.mpl_canvas.get_tk_widget().pack(fill="both", expand=True)
        self.mpl_canvas.mpl_connect("button_press_event", self._map_click)

        self.v_status = tk.StringVar(
            value="  Ready — select airports & algorithm, then click FIND ROUTE")
        tk.Label(p, textvariable=self.v_status,
                 font=("Consolas", 9), fg=C["accent"],
                 bg="#040609", anchor="w", padx=12, pady=3
                 ).pack(fill="x", side="bottom")

    # ══════════════════════════════════════════════════════════════
    # BOTTOM RESULTS PANEL  (horizontal strip, 7 columns)
    # ══════════════════════════════════════════════════════════════
    def _build_bottom(self):
        p = self._bot

        # thin top accent line
        tk.Frame(p, bg=C["accent"], height=2).pack(fill="x")

        inner = tk.Frame(p, bg=C["panel"])
        inner.pack(fill="both", expand=True, padx=0, pady=0)

        # ── Column 1-7 : time cards (one per algo) ──────────────
        self._tcards = {}
        for algo in ALGOS:
            col   = ALGO_META[algo]["color"]
            frame = tk.Frame(inner, bg=C["panel2"],
                             highlightbackground=C["border"], highlightthickness=1)
            frame.pack(side="left", fill="both", expand=True,
                       padx=3, pady=6)

            # coloured top bar
            tk.Frame(frame, bg=col, height=4).pack(fill="x")

            body = tk.Frame(frame, bg=C["panel2"])
            body.pack(fill="both", expand=True, padx=8, pady=6)

            tk.Label(body, text=algo, font=("Consolas", 9, "bold"),
                     fg=col, bg=C["panel2"], anchor="w").pack(fill="x")

            val = tk.Label(body, text="—", font=("Consolas", 14, "bold"),
                           fg=C["text"], bg=C["panel2"], anchor="w")
            val.pack(fill="x")

            # mini sub-label for dist/hops
            sub = tk.Label(body, text="", font=("Consolas", 7),
                           fg=C["sub"], bg=C["panel2"], anchor="w", wraplength=120)
            sub.pack(fill="x")

            self._tcards[algo] = (val, sub)

        # ── Column 8: benchmark bar chart ───────────────────────
        chart_frame = tk.Frame(inner, bg=C["panel2"],
                               highlightbackground=C["border"], highlightthickness=1)
        chart_frame.pack(side="left", fill="both",
                         padx=(3, 6), pady=6, ipadx=0)
        chart_frame.pack_propagate(False)
        chart_frame.configure(width=260)

        tk.Frame(chart_frame, bg=C["pink"], height=4).pack(fill="x")
        tk.Label(chart_frame, text="BENCHMARK (ms)",
                 font=("Consolas", 8, "bold"),
                 fg=C["sub"], bg=C["panel2"]).pack(pady=(4,0))

        self.bench_fig, self.bench_ax = plt.subplots(
            figsize=(2.6, 1.55), facecolor=C["panel2"])
        self.bench_ax.set_facecolor("#040609")
        self.bench_cvs = FigureCanvasTkAgg(self.bench_fig, master=chart_frame)
        self.bench_cvs.get_tk_widget().pack(fill="both", expand=True,
                                            padx=4, pady=(0,4))
        self._empty_bench()

        # ── Column 9: route detail text ──────────────────────────
        detail_frame = tk.Frame(inner, bg=C["panel2"],
                                highlightbackground=C["border"], highlightthickness=1)
        detail_frame.pack(side="left", fill="both",
                          padx=(0, 3), pady=6)
        detail_frame.pack_propagate(False)
        detail_frame.configure(width=310)

        tk.Frame(detail_frame, bg=C["green"], height=4).pack(fill="x")
        tk.Label(detail_frame, text="ROUTE & WEIGHT DETAILS",
                 font=("Consolas", 8, "bold"),
                 fg=C["sub"], bg=C["panel2"]).pack(anchor="w", padx=8, pady=(4,2))

        txt_wrap = tk.Frame(detail_frame, bg=C["panel2"])
        txt_wrap.pack(fill="both", expand=True, padx=4, pady=(0,4))
        sb = tk.Scrollbar(txt_wrap, orient="vertical")
        self.rtxt = tk.Text(txt_wrap, font=("Consolas", 8),
                            fg=C["text"], bg="#040609", relief="flat",
                            wrap="word", state="disabled",
                            yscrollcommand=sb.set)
        sb.configure(command=self.rtxt.yview)
        sb.pack(side="right", fill="y")
        self.rtxt.pack(fill="both", expand=True)

        # colour tags
        self.rtxt.tag_configure("head",   foreground=C["accent"],  font=("Consolas",9,"bold"))
        self.rtxt.tag_configure("green",  foreground=C["green"])
        self.rtxt.tag_configure("yellow", foreground=C["yellow"])
        self.rtxt.tag_configure("sub",    foreground=C["sub"],     font=("Consolas",7))
        self.rtxt.tag_configure("hop",    foreground=C["orange"],  font=("Consolas",8))
        for algo in ALGOS:
            self.rtxt.tag_configure(algo, foreground=ALGO_META[algo]["color"],
                                    font=("Consolas",8,"bold"))

    # ══════════════════════════════════════════════════════════════
    # MAP RENDERING
    # ══════════════════════════════════════════════════════════════
    def _draw_base_map(self):
        if not HAS_MPL: return
        ax = self.ax; ax.clear()
        ax.set_facecolor(C["bg"]); ax.set_xlim(-180,180); ax.set_ylim(-90,90)
        ax.set_aspect("equal")

        ax.add_patch(mpatches.Rectangle((-180,-90),360,180,
                     facecolor="#0a1628", edgecolor="none", zorder=0))

        for pts in CONTINENTS:
            xs=[p[0] for p in pts]; ys=[p[1] for p in pts]
            ax.fill(xs, ys, color="#162440", zorder=1)
            ax.plot(xs+[xs[0]], ys+[ys[0]], color="#1e3860", lw=0.7, zorder=2)

        for lon in range(-180,181,30): ax.axvline(lon, color="#0f1e36", lw=0.45, zorder=1)
        for lat in range(-90,91,30):  ax.axhline(lat, color="#0f1e36", lw=0.45, zorder=1)

        if self.v_all_edge.get():
            for a, nb in GRAPH.items():
                x1,y1 = AIRPORTS[a][2], AIRPORTS[a][1]
                for b in nb:
                    x2,y2 = AIRPORTS[b][2], AIRPORTS[b][1]
                    ax.plot([x1,x2],[y1,y2], color="#182a44", lw=0.3, zorder=2)

        # Bigger glowing airport nodes
        for code, (name, lat, lon) in AIRPORTS.items():
            ax.plot(lon, lat, "o", color=C["accent"], ms=16, zorder=4, alpha=0.10, markeredgewidth=0)
            ax.plot(lon, lat, "o", color=C["accent"], ms=11, zorder=4, alpha=0.20, markeredgewidth=0)
            ax.plot(lon, lat, "o", color=C["accent"], ms=7,  zorder=5,
                    markeredgecolor="#040810", markeredgewidth=0.9)
            ax.text(lon+1.3, lat+1.5, code, color=C["accent"],
                    fontsize=6.8, zorder=6, fontfamily="monospace",
                    fontweight="bold", alpha=0.92)

        ax.set_xlabel("Longitude", color=C["sub"], fontsize=9)
        ax.set_ylabel("Latitude",  color=C["sub"], fontsize=9)
        ax.tick_params(colors=C["sub"], labelsize=8)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        self.fig.tight_layout(pad=0.3)
        self.mpl_canvas.draw()

    # ── Bézier arc + arrowhead ────────────────────────────────────
    def _bezier(self, ax, x1, y1, x2, y2, arc=0.09,
                color="#00d4ff", lw=2.2, alpha=0.9, zorder=8):
        mx = (x1+x2)/2; my = (y1+y2)/2 + abs(x2-x1)*arc
        t  = np.linspace(0,1,60)
        bx = (1-t)**2*x1 + 2*(1-t)*t*mx + t**2*x2
        by = (1-t)**2*y1 + 2*(1-t)*t*my + t**2*y2
        ax.plot(bx, by, color=color, lw=lw, alpha=alpha,
                zorder=zorder, solid_capstyle="round")
        ax.annotate("", xy=(bx[-1],by[-1]), xytext=(bx[-6],by[-6]),
                    arrowprops=dict(arrowstyle="-|>", color=color,
                                   lw=lw*0.55, mutation_scale=11), zorder=zorder+1)
        return bx, by

    def _weight_label(self, ax, x1, y1, x2, y2, w, color, arc=0.09):
        mx=(x1+x2)/2; my=(y1+y2)/2+abs(x2-x1)*arc; t=0.5
        px=(1-t)**2*x1+2*(1-t)*t*mx+t**2*x2; py=(1-t)**2*y1+2*(1-t)*t*my+t**2*y2
        ax.text(px, py, f"{w:,}", fontsize=5.8, color=color,
                fontfamily="monospace", zorder=15, ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.17", fc=C["bg"],
                          ec=color, alpha=0.85, lw=0.55))

    def _draw_path(self, path, algo, lw=2.4, alpha=0.88,
                   arc=0.09, wts=True, redraw=True):
        ax=self.ax; color=ALGO_META[algo]["color"]
        xs=[AIRPORTS[c][2] for c in path]; ys=[AIRPORTS[c][1] for c in path]

        for i in range(len(path)-1):
            x1,y1=AIRPORTS[path[i]][2],   AIRPORTS[path[i]][1]
            x2,y2=AIRPORTS[path[i+1]][2], AIRPORTS[path[i+1]][1]
            self._bezier(ax,x1,y1,x2,y2,arc=arc,color=color,lw=lw,alpha=alpha)
            if wts and self.v_weights.get():
                w=GRAPH.get(path[i],{}).get(path[i+1],0)
                if w: self._weight_label(ax,x1,y1,x2,y2,w,color,arc)

        for c in path:
            ax.plot(AIRPORTS[c][2],AIRPORTS[c][1],"o",color=color,
                    ms=10,zorder=10,markeredgecolor="white",markeredgewidth=0.9)

        ax.plot(xs[0],ys[0],"*",color=C["green"],ms=18,zorder=11,
                markeredgecolor="white",markeredgewidth=0.5)
        ax.plot(xs[-1],ys[-1],"D",color=C["red"],ms=11,zorder=11,
                markeredgecolor="white",markeredgewidth=0.5)

        off_y = ALGOS.index(algo)*4.5
        ax.text(xs[0]+1.5,ys[0]+2.5+off_y, algo, color=color,
                fontsize=7.5, fontweight="bold", zorder=12, fontfamily="monospace",
                bbox=dict(boxstyle="round,pad=0.25",fc=C["bg"],alpha=0.7,ec="none"))
        if redraw: self.mpl_canvas.draw()

    def _draw_mst(self, mst_edges, algo, alpha=0.55, wts=True):
        ax=self.ax; color=ALGO_META[algo]["color"]
        for u,v,w in mst_edges:
            x1,y1=AIRPORTS[u][2],AIRPORTS[u][1]; x2,y2=AIRPORTS[v][2],AIRPORTS[v][1]
            ax.plot([x1,x2],[y1,y2],color=color,lw=1.5,alpha=alpha,zorder=7)
            if wts and self.v_weights.get():
                mx,my=(x1+x2)/2,(y1+y2)/2
                ax.text(mx,my,f"{w:,}",fontsize=5.2,color=color,
                        fontfamily="monospace",zorder=14,ha="center",va="center",
                        bbox=dict(boxstyle="round,pad=0.13",fc=C["bg"],ec=color,alpha=0.8,lw=0.4))
        self.mpl_canvas.draw()

    # ── Segment-by-segment animation ─────────────────────────────
    def _animate_path(self, path, algo, on_done=None):
        if self._anim_job: self.after_cancel(self._anim_job)
        ax=self.ax; color=ALGO_META[algo]["color"]; arc=ALGO_META[algo]["arc"]

        def step(i):
            if i >= len(path)-1:
                ax.plot(AIRPORTS[path[0]][2],AIRPORTS[path[0]][1],
                        "*",color=C["green"],ms=18,zorder=11,
                        markeredgecolor="white",markeredgewidth=0.5)
                ax.plot(AIRPORTS[path[-1]][2],AIRPORTS[path[-1]][1],
                        "D",color=C["red"],ms=11,zorder=11,
                        markeredgecolor="white",markeredgewidth=0.5)
                self.mpl_canvas.draw()
                if on_done: on_done()
                return
            x1,y1=AIRPORTS[path[i]][2],AIRPORTS[path[i]][1]
            x2,y2=AIRPORTS[path[i+1]][2],AIRPORTS[path[i+1]][1]
            self._bezier(ax,x1,y1,x2,y2,arc=arc,color=color,lw=2.4,alpha=0.9)
            ax.plot(x1,y1,"o",color=color,ms=10,zorder=10,
                    markeredgecolor="white",markeredgewidth=0.9)
            if self.v_weights.get():
                w=GRAPH.get(path[i],{}).get(path[i+1],0)
                if w: self._weight_label(ax,x1,y1,x2,y2,w,color,arc)
            self.mpl_canvas.draw()
            self._anim_job = self.after(210, step, i+1)
        step(0)

    # ── ✈ Flying plane along Bézier ──────────────────────────────
    def _fly_plane(self, path, algo):
        if not self.v_plane.get(): return
        if self._plane_job: self.after_cancel(self._plane_job)
        if self._plane_obj:
            try: self._plane_obj.remove()
            except: pass
            self._plane_obj = None

        ax=self.ax; color=ALGO_META[algo]["color"]; arc=ALGO_META[algo]["arc"]
        all_bx, all_by = [], []
        for i in range(len(path)-1):
            x1,y1=AIRPORTS[path[i]][2],AIRPORTS[path[i]][1]
            x2,y2=AIRPORTS[path[i+1]][2],AIRPORTS[path[i+1]][1]
            mx=(x1+x2)/2; my=(y1+y2)/2+abs(x2-x1)*arc
            t=np.linspace(0,1,80)
            bx=(1-t)**2*x1+2*(1-t)*t*mx+t**2*x2
            by=(1-t)**2*y1+2*(1-t)*t*my+t**2*y2
            if all_bx: bx,by = bx[1:],by[1:]
            all_bx.extend(bx); all_by.extend(by)

        total=len(all_bx); pidx=[0]
        self._plane_obj, = ax.plot(all_bx[0],all_by[0],marker=(3,0,-30),
                                   ms=15,color=color,zorder=20,
                                   markeredgecolor="white",markeredgewidth=0.5,
                                   linestyle="none")
        def move():
            i=pidx[0]
            if i>=total: return
            if i<total-2:
                dx=all_bx[i+1]-all_bx[i]; dy=all_by[i+1]-all_by[i]
                ang=math.degrees(math.atan2(dy,dx))
            else: ang=0
            self._plane_obj.set_data([all_bx[i]],[all_by[i]])
            self._plane_obj.set_marker((3,0,ang-90))
            self.mpl_canvas.draw_idle()
            pidx[0]+=3
            self._plane_job=self.after(18,move)
        move()

    # ══════════════════════════════════════════════════════════════
    # RUN SINGLE ALGORITHM
    # ══════════════════════════════════════════════════════════════
    def _run_algorithm(self):
        orig=self.v_orig.get(); dest=self.v_dest.get(); algo=self.v_algo.get()
        if orig==dest:
            messagebox.showwarning("Same Airport","Pick different airports."); return

        self.v_status.set(f"  ⏳  Running {algo}   {orig} → {dest} …")
        self.update_idletasks()

        mst_e=None; t0=time.perf_counter()
        if   algo=="BFS":            path,cost=bfs(GRAPH,orig,dest)
        elif algo=="DFS":            path,cost=dfs(GRAPH,orig,dest)
        elif algo=="Dijkstra":       path,cost=dijkstra(GRAPH,orig,dest)
        elif algo=="A*":             path,cost=astar(GRAPH,orig,dest)
        elif algo=="Floyd-Warshall": path,cost=floyd_warshall(GRAPH)[0](orig,dest)
        elif algo=="Prim MST":       mst_e=prim_mst(GRAPH);path=None;cost=sum(w for _,_,w in mst_e)
        elif algo=="Kruskal MST":    mst_e=kruskal_mst(GRAPH);path=None;cost=sum(w for _,_,w in mst_e)
        else:                        path,cost=[],0
        elapsed=(time.perf_counter()-t0)*1000

        self._algo_times[algo]=elapsed
        self._update_card(algo, elapsed)
        self._update_bench(self._algo_times)
        self._draw_base_map()

        if mst_e is not None:
            self._draw_mst(mst_e,algo)
            self._card_sub(algo,f"{len(mst_e)} edges  {cost:,} km")
            self._show_mst_result(algo,mst_e,cost,elapsed)
            self.v_status.set(f"  ✔  {algo} MST — {len(mst_e)} edges — {cost:,} km — {elapsed:.3f} ms")
        elif path:
            arc=ALGO_META[algo]["arc"]
            if self.v_anim.get():
                self._animate_path(path,algo,on_done=lambda: self._fly_plane(path,algo))
            else:
                self._draw_path(path,algo,arc=arc)
                self._fly_plane(path,algo)
            self._card_sub(algo,f"{len(path)-1} hops  {cost:,} km")
            self._show_path_result(algo,path,cost,elapsed)
            self.v_status.set(
                f"  ✔  {algo}: {orig}→{dest}  —  {cost:,} km  —  {len(path)-1} hops  —  {elapsed:.3f} ms")
        else:
            self._write_result([(f"No path found from {orig} to {dest} using {algo}.\n","sub")])
            self.v_status.set(f"  ✘  No path found using {algo}")

    # ══════════════════════════════════════════════════════════════
    # COMPARE ALL
    # ══════════════════════════════════════════════════════════════
    def _compare_all(self):
        orig=self.v_orig.get(); dest=self.v_dest.get()
        if orig==dest:
            messagebox.showwarning("Same Airport","Pick different airports."); return

        self.v_status.set(f"  ⏳  Running all {len(ALGOS)} algorithms  {orig} → {dest} …")
        self.update_idletasks()

        times={}; results={}
        for algo in ALGOS:
            mst_e=None; t0=time.perf_counter()
            if   algo=="BFS":            p,c=bfs(GRAPH,orig,dest)
            elif algo=="DFS":            p,c=dfs(GRAPH,orig,dest)
            elif algo=="Dijkstra":       p,c=dijkstra(GRAPH,orig,dest)
            elif algo=="A*":             p,c=astar(GRAPH,orig,dest)
            elif algo=="Floyd-Warshall": p,c=floyd_warshall(GRAPH)[0](orig,dest)
            elif algo=="Prim MST":       mst_e=prim_mst(GRAPH);p=None;c=sum(w for _,_,w in mst_e)
            elif algo=="Kruskal MST":    mst_e=kruskal_mst(GRAPH);p=None;c=sum(w for _,_,w in mst_e)
            elapsed=(time.perf_counter()-t0)*1000
            times[algo]=elapsed; results[algo]=(p,c,mst_e)

        self._algo_times=dict(times)
        for algo,ms in times.items():
            p,c,mst_e=results[algo]
            self._update_card(algo,ms)
            if p:   self._card_sub(algo,f"{len(p)-1} hops  {c:,} km")
            elif mst_e: self._card_sub(algo,f"{len(mst_e)} edges (MST)")
        self._update_bench(times)

        self._draw_base_map()
        for algo,(p,c,mst_e) in results.items():
            if mst_e:    self._draw_mst(mst_e,algo,wts=False)
            elif p:      self._draw_path(p,algo,lw=1.9,alpha=0.80,
                                         arc=ALGO_META[algo]["arc"],
                                         wts=self.v_weights.get(),redraw=False)

        patches=[mpatches.Patch(color=ALGO_META[a]["color"],label=a) for a in ALGOS]
        self.ax.legend(handles=patches,loc="lower left",fontsize=8,
                       framealpha=0.3,facecolor="#040609",labelcolor="white",
                       edgecolor=C["border"])
        self.mpl_canvas.draw()

        lines=[("COMPARISON RESULTS\n","head"),
               (f"{'ALGO':<18}{'DIST':>10}{'HOPS':>6}{'TIME (ms)':>13}\n","sub"),
               ("─"*48+"\n","sub")]
        for algo in ALGOS:
            p,c,mst_e=results[algo]
            hops=str(len(p)-1) if p else "MST"
            dist=f"{c:,}" if c else "—"
            lines.append((f"{algo:<18}{dist:>10}{hops:>6}  {times[algo]:>11.4f}\n",algo))
        self._write_result(lines)
        self.v_status.set(
            f"  ✔  All {len(ALGOS)} algorithms compared — {orig}→{dest} — paths colour-coded on map")

    # ══════════════════════════════════════════════════════════════
    # FLOYD-WARSHALL  (dedicated button)
    # ══════════════════════════════════════════════════════════════
    def _run_fw(self):
        orig=self.v_orig.get(); dest=self.v_dest.get()
        if orig==dest:
            messagebox.showwarning("Same Airport","Pick different airports."); return

        self.v_status.set("  ⏳  Running Floyd-Warshall (all-pairs) …")
        self.update_idletasks()

        t0=time.perf_counter()
        recon,dist_mat,idx,nodes=floyd_warshall(GRAPH)
        path,cost=recon(orig,dest)
        elapsed=(time.perf_counter()-t0)*1000
        algo="Floyd-Warshall"

        self._algo_times[algo]=elapsed
        self._update_card(algo,elapsed)
        self._update_bench(self._algo_times)
        self._draw_base_map()

        if path:
            self._draw_path(path,algo,arc=ALGO_META[algo]["arc"])
            self._fly_plane(path,algo)
            self._card_sub(algo,f"{len(path)-1} hops  {cost:,} km")

            si=idx[orig]
            top5=sorted([(dist_mat[si][idx[d]],d)
                          for d in nodes if d!=orig and dist_mat[si][idx[d]]<INF],
                         key=lambda x:x[0])[:5]

            lines=[("FLOYD-WARSHALL RESULTS\n","head"),
                   (f"  Algorithm : Floyd-Warshall (All-Pairs SP)\n","sub"),
                   (f"  From      : {orig}  {AIRPORTS[orig][0]}\n","yellow"),
                   (f"  To        : {dest}  {AIRPORTS[dest][0]}\n","yellow"),
                   (f"  Distance  : {cost:,} km\n","green"),
                   (f"  Hops      : {len(path)-1}\n","green"),
                   (f"  Time      : {elapsed:.4f} ms\n\n","green"),
                   ("  PATH + WEIGHTS:\n","head")]
            cum=0
            for i in range(len(path)-1):
                a,b=path[i],path[i+1]; w=GRAPH.get(a,{}).get(b,0); cum+=w
                lines.append((f"  [{i+1}] {a} --({w:,} km)--> {b}  [total {cum:,}]\n","hop"))
            lines.append((f"\n  TOP-5 NEAREST FROM {orig}:\n","head"))
            for dv,dst in top5:
                lines.append((f"  {dst}: {AIRPORTS[dst][0][:20]}  {dv:,} km\n","sub"))
            self._write_result(lines)
            self.v_status.set(
                f"  ✔  Floyd-Warshall: {orig}→{dest} — {cost:,} km — {len(path)-1} hops — {elapsed:.3f} ms")
        else:
            self._write_result([("No path found by Floyd-Warshall.\n","sub")])
            self.v_status.set("  ✘  Floyd-Warshall: no path found")

    # ══════════════════════════════════════════════════════════════
    # BOTTOM PANEL HELPERS
    # ══════════════════════════════════════════════════════════════
    def _update_card(self, algo, ms):
        val, _ = self._tcards[algo]; val.configure(text=f"{ms:.4f} ms")

    def _card_sub(self, algo, txt):
        _, sub = self._tcards[algo]; sub.configure(text=txt)

    def _reset_cards(self):
        for val, sub in self._tcards.values():
            val.configure(text="—"); sub.configure(text="")

    def _write_result(self, tagged):
        self.rtxt.configure(state="normal"); self.rtxt.delete("1.0","end")
        for txt, tag in tagged: self.rtxt.insert("end", txt, tag)
        self.rtxt.configure(state="disabled")

    def _show_path_result(self, algo, path, cost, elapsed):
        lines=[
            (f"{algo}  ROUTE RESULT\n","head"),
            (f"  From : {path[0]}  {AIRPORTS[path[0]][0]}\n","yellow"),
            (f"  To   : {path[-1]}  {AIRPORTS[path[-1]][0]}\n","yellow"),
            (f"  Dist : {cost:,} km     Hops : {len(path)-1}     Time : {elapsed:.4f} ms\n\n","green"),
            ("  HOPS + WEIGHTS:\n","head"),
        ]
        cum=0
        for i in range(len(path)-1):
            a,b=path[i],path[i+1]; w=GRAPH.get(a,{}).get(b,0); cum+=w
            lines.append((f"  [{i+1}] {a} --({w:,})--> {b}  [Σ {cum:,} km]\n","hop"))
        self._write_result(lines)

    def _show_mst_result(self, algo, mst_edges, cost, elapsed):
        lines=[
            (f"{algo}  MST RESULT\n","head"),
            (f"  Total weight : {cost:,} km\n","green"),
            (f"  Edges in MST : {len(mst_edges)}\n","green"),
            (f"  Time Taken   : {elapsed:.4f} ms\n\n","green"),
            ("  MST EDGES (sorted by weight):\n","head"),
        ]
        for u,v,w in sorted(mst_edges,key=lambda x:x[2]):
            lines.append((f"  {u} --({w:,} km)-- {v}\n","hop"))
        self._write_result(lines)

    def _empty_bench(self):
        ax=self.bench_ax; ax.clear(); ax.set_facecolor("#040609")
        ax.text(0.5,0.5,"run algo\nto see",ha="center",va="center",
                color=C["sub"],fontsize=7,transform=ax.transAxes)
        ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_color(C["border"])
        self.bench_fig.tight_layout(pad=0.3); self.bench_cvs.draw()

    def _update_bench(self, times):
        ax=self.bench_ax; ax.clear(); ax.set_facecolor("#040609")
        algos=list(times); vals=[times[a] for a in algos]
        colors=[ALGO_META.get(a,{}).get("color",C["accent"]) for a in algos]
        bars=ax.barh(algos,vals,color=colors,height=0.55,edgecolor="none")
        mx=max(vals) if vals else 1
        for bar,v in zip(bars,vals):
            ax.text(v+mx*0.03,bar.get_y()+bar.get_height()/2,
                    f"{v:.2f}",va="center",ha="left",
                    color="white",fontsize=5.5,fontfamily="monospace")
        ax.set_xlabel("ms",color=C["sub"],fontsize=7)
        ax.tick_params(colors=C["sub"],labelsize=6)
        for sp in ax.spines.values(): sp.set_color(C["border"])
        ax.set_xlim(0,mx*1.4)
        self.bench_fig.tight_layout(pad=0.3); self.bench_cvs.draw()

    # ══════════════════════════════════════════════════════════════
    # MAP CLICK  (left-click = set origin, right-click = destination)
    # ══════════════════════════════════════════════════════════════
    def _map_click(self, event):
        if event.xdata is None: return
        lon,lat=event.xdata,event.ydata
        nearest=min(AIRPORTS,key=lambda c:(AIRPORTS[c][1]-lat)**2+(AIRPORTS[c][2]-lon)**2)
        if event.button==1: self.v_orig.set(nearest)
        else:               self.v_dest.set(nearest)
        self._preview()
        self.v_status.set(
            f"  Selected: {nearest}  —  {AIRPORTS[nearest][0]}"
            f"  ({AIRPORTS[nearest][1]:.2f}°, {AIRPORTS[nearest][2]:.2f}°)")

    # ══════════════════════════════════════════════════════════════
    # CLEAR
    # ══════════════════════════════════════════════════════════════
    def _clear_map(self):
        for j in (self._anim_job, self._plane_job):
            if j: self.after_cancel(j)
        if self._plane_obj:
            try: self._plane_obj.remove()
            except: pass
            self._plane_obj=None
        self._draw_base_map()
        self._algo_times.clear()
        self._reset_cards()
        self._empty_bench()
        self._write_result([])
        self.v_status.set("  Map cleared.")

    # ══════════════════════════════════════════════════════════════
    # GENERIC UI HELPERS
    # ══════════════════════════════════════════════════════════════
    def _btn(self, parent, text, bg, fg, cmd, pady=10, sz=11):
        tk.Button(parent, text=text, font=("Consolas",sz,"bold"),
                  bg=bg, fg=fg, activebackground=bg,
                  relief="flat", cursor="hand2", pady=pady,
                  command=cmd).pack(fill="x", padx=14)

    def _sec(self, parent, title):
        f=tk.Frame(parent, bg=C["panel"]); f.pack(fill="x", padx=14, pady=(12,3))
        tk.Frame(f, bg=C["accent"], height=1).pack(fill="x", pady=(0,3))
        tk.Label(f, text=title, font=("Consolas",8,"bold"),
                 fg=C["sub"], bg=C["panel"]).pack(anchor="w")


# ══════════════════════════════════════════════════════════════════
# ENTRY
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    if not HAS_MPL:
        print("ERROR: matplotlib required.\n  pip install matplotlib numpy")
    else:
        App().mainloop()