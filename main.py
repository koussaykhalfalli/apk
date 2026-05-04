"""
MEMBRANE CIRCULAIRE - Modes Propres & Analyse de Bessel
Application KivyMD - Version Mobile 2.1 (Android-ready)
Auteur : Koussay Khalfalli

scipy.sparse.linalg.eigs remplacé par numpy.linalg.eig
pour compatibilité Android / Buildozer.
"""

import os
os.environ.setdefault("KIVY_NO_ENV_CONFIG", "1")

import threading
import math
import numpy as np
from scipy.sparse import lil_matrix   # scipy.sparse OK pour construire A
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Circle
import io
from functools import partial

from kivy.config import Config
Config.set("input", "mouse", "mouse,multitouch_on_demand")
Config.set("graphics", "width",  "400")
Config.set("graphics", "height", "780")
Config.set("graphics", "resizable", "1")

from kivy.app import App
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage
from kivy.core.window import Window
from kivy.animation import Animation
from kivy.uix.screenmanager import ScreenManager, Screen, FadeTransition
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.gridlayout import GridLayout
from kivy.uix.scrollview import ScrollView
from kivy.uix.image import Image as KivyImage
from kivy.uix.widget import Widget
from kivy.graphics import Color, Rectangle, Line, Ellipse
from kivy.properties import ListProperty, NumericProperty
from kivy.metrics import dp, sp
from kivy.lang import Builder

from kivymd.app import MDApp
from kivymd.uix.label import MDLabel
from kivymd.uix.button import MDRaisedButton, MDFlatButton, MDIconButton
from kivymd.uix.textfield import MDTextField
from kivymd.uix.card import MDCard
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.snackbar import Snackbar
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.boxlayout import MDBoxLayout
from kivymd.uix.spinner import MDSpinner


# ─────────────────────────────────────────────────────────────────────────────
#  PALETTE
# ─────────────────────────────────────────────────────────────────────────────
C = {
    "bg_deep":    (0.05, 0.07, 0.12, 1),
    "bg_mid":     (0.07, 0.10, 0.17, 1),
    "bg_card":    (0.09, 0.13, 0.22, 1),
    "bg_input":   (0.06, 0.09, 0.16, 1),
    "cyan":       (0.27, 0.82, 0.92, 1),
    "teal":       (0.20, 0.70, 0.70, 1),
    "violet":     (0.55, 0.40, 0.95, 1),
    "ok_green":   (0.25, 0.88, 0.55, 1),
    "warn_amber": (1.00, 0.75, 0.10, 1),
    "err_red":    (0.95, 0.30, 0.30, 1),
    "text_hi":    (0.92, 0.95, 1.00, 1),
    "text_mid":   (0.65, 0.72, 0.85, 1),
    "text_lo":    (0.38, 0.45, 0.60, 1),
    "border":     (0.15, 0.22, 0.38, 1),
}

def rgb_mpl(key):
    return C[key][:3]


# ─────────────────────────────────────────────────────────────────────────────
#  KV
# ─────────────────────────────────────────────────────────────────────────────
KV = """
#:import dp kivy.metrics.dp
#:import C __main__.C
#:import FadeTransition kivy.uix.screenmanager.FadeTransition

<ScreenManager>:
    transition: FadeTransition(duration=0.3)

<SplashScreen>:
    name: 'splash'
    canvas.before:
        Color:
            rgba: C['bg_deep']
        Rectangle:
            pos: self.pos
            size: self.size

<ParamsScreen>:
    name: 'params'
    canvas.before:
        Color:
            rgba: C['bg_deep']
        Rectangle:
            pos: self.pos
            size: self.size

<ComputeScreen>:
    name: 'compute'
    canvas.before:
        Color:
            rgba: C['bg_deep']
        Rectangle:
            pos: self.pos
            size: self.size

<ResultsScreen>:
    name: 'results'
    canvas.before:
        Color:
            rgba: C['bg_deep']
        Rectangle:
            pos: self.pos
            size: self.size
"""


# ─────────────────────────────────────────────────────────────────────────────
#  BESSEL ZEROS
# ─────────────────────────────────────────────────────────────────────────────
BESSEL_ZEROS = {
    0: [2.4048, 5.5201, 8.6537, 11.7915, 14.9309],
    1: [3.8317, 7.0156, 10.1735, 13.3237, 16.4706],
    2: [5.1356, 8.4172, 11.6198, 14.7960, 17.9598],
    3: [6.3802, 9.7610, 13.0152, 16.2235, 19.4094],
    4: [7.5883, 11.0647, 14.3725, 17.6159, 20.8269],
    5: [8.7715, 12.3386, 15.7002, 18.9801, 22.2178],
}


# ─────────────────────────────────────────────────────────────────────────────
#  COMPUTE ENGINE  (numpy.linalg.eig — no scipy eigs)
# ─────────────────────────────────────────────────────────────────────────────
def compute_membrane(R, c, Nr, Ntheta, nb_modes, log_cb):
    results = {}
    dr     = R / Nr
    dtheta = 2 * math.pi / Ntheta
    N      = 1 + (Nr - 1) * Ntheta
    results.update(dr=dr, dtheta=dtheta, N=N, Nr=Nr, Ntheta=Ntheta, R=R, c=c)

    log_cb("[b][color=44d4eb]Parametres :[/color][/b]")
    log_cb(f"  Dr={dr:.5f}  Dtheta={dtheta:.5f}  N={N}")
    log_cb("-" * 46)

    def idx(i, j):
        return 0 if i == 0 else 1 + (i - 1) * Ntheta + j

    # ── Build sparse matrix A ────────────────────────────────────────────────
    log_cb("[b][color=44d4eb]Construction matrice A...[/color][/b]")
    A = lil_matrix((N, N))

    coeff_c = 4.0 / dr**2
    A[0, 0] = -coeff_c
    for j in range(Ntheta):
        A[0, idx(1, j)] += coeff_c / Ntheta

    r1 = dr
    for j in range(Ntheta):
        k = idx(1, j)
        A[k, k]             += -2.0/dr**2 - 2.0/(r1**2 * dtheta**2)
        A[k, idx(0, 0)]     += 1.0/dr**2 - 1.0/(2.0*r1*dr)
        if Nr - 1 >= 2:
            A[k, idx(2, j)] += 1.0/dr**2 + 1.0/(2.0*r1*dr)
        ct = 1.0 / (r1**2 * dtheta**2)
        A[k, idx(1, (j+1) % Ntheta)] += ct
        A[k, idx(1, (j-1) % Ntheta)] += ct

    for i in range(2, Nr - 1):
        r  = i * dr
        ct = 1.0 / (r**2 * dtheta**2)
        for j in range(Ntheta):
            k = idx(i, j)
            A[k, k]                     += -2.0/dr**2 - 2.0/(r**2 * dtheta**2)
            A[k, idx(i+1, j)]           += 1.0/dr**2 + 1.0/(2.0*r*dr)
            A[k, idx(i-1, j)]           += 1.0/dr**2 - 1.0/(2.0*r*dr)
            A[k, idx(i, (j+1)%Ntheta)] += ct
            A[k, idx(i, (j-1)%Ntheta)] += ct

    if Nr - 1 >= 1:
        i  = Nr - 1; r = i * dr
        ct = 1.0 / (r**2 * dtheta**2)
        for j in range(Ntheta):
            k = idx(i, j)
            A[k, k]                     += -2.0/dr**2 - 2.0/(r**2 * dtheta**2)
            A[k, idx(i-1, j)]           += 1.0/dr**2 - 1.0/(2.0*r*dr)
            A[k, idx(i, (j+1)%Ntheta)] += ct
            A[k, idx(i, (j-1)%Ntheta)] += ct

    A_csr = A.tocsr()
    nnz   = A_csr.nnz
    fill  = 100 * nnz / (N**2)
    results.update(nnz=nnz, fill=fill)
    log_cb(f"  {N}x{N}  |  {nnz} nz  |  {fill:.3f}%")
    log_cb("-" * 46)

    # ── Eigenvalues via numpy (Android-compatible) ───────────────────────────
    log_cb(f"[b][color=44d4eb]Calcul valeurs propres (numpy)...[/color][/b]")
    log_cb(f"  N={N} — conversion dense puis np.linalg.eig")
    log_cb(f"  [color=ffc107]Patience : peut prendre 1-2 min sur mobile[/color]")

    try:
        A_dense = A_csr.toarray()
        all_evals, all_evecs = np.linalg.eig(A_dense)

        # Keep only real negative eigenvalues
        mask = (np.abs(all_evals.imag) < 1e-6) & (all_evals.real < -1e-10)
        real_evals = all_evals[mask].real
        real_evecs = all_evecs[:, mask].real

        if len(real_evals) == 0:
            raise ValueError("Aucune valeur propre reelle negative trouvee")

        nb_modes = min(nb_modes, len(real_evals))
        order  = np.argsort(real_evals)[::-1][:nb_modes]
        evals  = real_evals[order]
        evecs  = real_evecs[:, order]
        freqs  = np.sqrt(-evals) * c / (2 * math.pi)

    except Exception as e:
        log_cb(f"[color=f24444]ERREUR : {e}[/color]")
        results["error"] = str(e)
        return results

    results.update(evals=evals, evecs=evecs, freqs=freqs, idx_fn=idx)

    log_cb("[b]Frequences propres :[/b]")
    for m in range(nb_modes):
        log_cb(f"  Mode {m+1}: [color=44d4eb]{freqs[m]:.5f} Hz[/color]")
    log_cb("-" * 46)

    # ── Compare with Bessel zeros ────────────────────────────────────────────
    theo_list = []
    for m_ord, zeros in BESSEL_ZEROS.items():
        for n_idx, z in enumerate(zeros):
            f_th = z * c / (2 * math.pi * R)
            theo_list.append({"m": m_ord, "n": n_idx+1, "z": z, "freq": f_th})
    theo_list.sort(key=lambda x: x["freq"])

    TOL   = 0.02
    comp  = []
    diffs = []

    log_cb("[b][color=44d4eb]Comparaison Bessel :[/color][/b]")
    for idx_m, f_num in enumerate(freqs):
        best, best_diff = None, float("inf")
        for th in theo_list:
            d = abs(f_num - th["freq"])
            if d < best_diff:
                best_diff, best = d, th
        ok = best_diff < TOL
        diffs.append(best_diff)
        comp.append({
            "mode": idx_m+1, "f_num": f_num, "f_theo": best["freq"],
            "diff": best_diff, "m": best["m"], "n": best["n"],
            "negligeable": ok,
        })
        tick = "[color=40e090]OK[/color]" if ok else "[color=f24444]!![/color]"
        log_cb(f"  M{idx_m+1} ({best['m']},{best['n']}) "
               f"{f_num:.4f}/{best['freq']:.4f} Hz {tick}")

    diffs_arr  = np.array(diffs)
    theo_f     = [theo_list[i]["freq"] for i in range(len(freqs))]
    diffs_rel  = [d/f*100 for d, f in zip(diffs, theo_f)]
    d_mean     = float(np.mean(diffs_arr))
    d_mean_rel = float(np.mean(diffs_rel))
    d_max      = float(np.max(diffs_arr))
    d_max_rel  = float(np.max(diffs_rel))
    all_ok     = all(c_["negligeable"] for c_ in comp)
    nb_bad     = sum(1 for c_ in comp if not c_["negligeable"])

    results.update(comp=comp, d_mean=d_mean, d_mean_rel=d_mean_rel,
                   d_max=d_max, d_max_rel=d_max_rel,
                   all_ok=all_ok, nb_bad=nb_bad, theo_list=theo_list)

    log_cb("-" * 46)
    log_cb(f"  Err. moy. : [b]{d_mean:.5f} Hz[/b]  ({d_mean_rel:.3f}%)")
    if all_ok:
        log_cb("[color=40e090][b]>> TOUS modes OK (Df < 0.02 Hz)[/b][/color]")
    else:
        log_cb(f"[color=f24444]!! {nb_bad} mode(s) hors tolerance[/color]")
    log_cb("=" * 46)
    log_cb("[b][color=44d4eb]  CALCUL TERMINE[/color][/b]")
    return results


# ─────────────────────────────────────────────────────────────────────────────
#  MATPLOTLIB RENDERS
# ─────────────────────────────────────────────────────────────────────────────
def _build_u_cart(results, mode_index, N_plot=60):
    R      = results["R"]
    Nr     = results["Nr"]
    Ntheta = results["Ntheta"]
    dr     = results["dr"]
    dtheta = results["dtheta"]
    evec   = results["evecs"][:, mode_index]
    idx_fn = results["idx_fn"]

    x_lin = np.linspace(-R, R, N_plot)
    y_lin = np.linspace(-R, R, N_plot)
    X, Y  = np.meshgrid(x_lin, y_lin)
    U0    = evec[0]
    U_vals = np.zeros((Nr, Ntheta))
    for i in range(1, Nr):
        for j in range(Ntheta):
            U_vals[i, j] = evec[idx_fn(i, j)]

    U_cart = np.full((N_plot, N_plot), np.nan)
    for ix in range(N_plot):
        for iy in range(N_plot):
            r = math.sqrt(X[ix, iy]**2 + Y[ix, iy]**2)
            if r <= R:
                if r < dr / 2:
                    U_cart[ix, iy] = U0
                else:
                    theta = math.atan2(Y[ix, iy], X[ix, iy])
                    if theta < 0: theta += 2 * math.pi
                    j = int(round(theta / dtheta)) % Ntheta
                    i = min(int(round(r / dr)), Nr - 1)
                    U_cart[ix, iy] = U0 if i == 0 else U_vals[i, j]
    return X, Y, U_cart


def render_mode_2d(results, mode_index, dpi=80):
    X, Y, U_cart = _build_u_cart(results, mode_index)
    freq = results["freqs"][mode_index]
    comp = results["comp"][mode_index]
    R    = results["R"]
    bg   = tuple(C["bg_card"][:3])

    fig, ax = plt.subplots(figsize=(3.6, 3.6), dpi=dpi, facecolor=bg)
    ax.set_facecolor(bg)
    masked = np.ma.masked_invalid(U_cart)
    vmax   = np.nanmax(np.abs(U_cart)) or 1.0
    cf = ax.contourf(X, Y, masked, levels=30, cmap="RdBu_r",
                     vmin=-vmax, vmax=vmax)
    ax.contour(X, Y, masked, levels=[0], colors=["white"],
               linewidths=0.6, linestyles="--", alpha=0.5)
    ax.add_patch(Circle((0,0), R, fill=False,
                         edgecolor=rgb_mpl("cyan"), linewidth=1.2))
    cb = fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.03)
    cb.ax.tick_params(colors=rgb_mpl("text_mid"), labelsize=6)
    mn = f"(m={comp['m']},n={comp['n']})"
    ax.set_title(f"Mode {mode_index+1} {mn}\nf={freq:.4f} Hz",
                 color=rgb_mpl("text_hi"), fontsize=8, pad=4)
    ax.set_aspect("equal"); ax.axis("off")
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=bg)
    plt.close(fig); buf.seek(0)
    return CoreImage(buf, ext="png")


def render_mode_3d(results, mode_index, dpi=80):
    X, Y, U_cart = _build_u_cart(results, mode_index, N_plot=50)
    freq = results["freqs"][mode_index]
    comp = results["comp"][mode_index]
    bg   = tuple(C["bg_card"][:3])

    fig = plt.figure(figsize=(3.8, 3.4), dpi=dpi, facecolor=bg)
    ax  = fig.add_subplot(111, projection="3d")
    ax.set_facecolor(bg); ax.patch.set_facecolor(bg)
    vmax = np.nanmax(np.abs(U_cart)) or 1.0
    norm = mcolors.Normalize(vmin=-vmax, vmax=vmax)
    ax.plot_surface(X, Y, U_cart, cmap="RdBu_r", norm=norm,
                    linewidth=0, antialiased=True, alpha=0.92,
                    rstride=2, cstride=2)
    mn = f"(m={comp['m']},n={comp['n']})"
    ax.set_title(f"Mode {mode_index+1} {mn}\nf={freq:.4f} Hz",
                 color=rgb_mpl("text_hi"), fontsize=8, pad=3)
    txt = rgb_mpl("text_lo")
    ax.tick_params(colors=txt, labelsize=6)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(rgb_mpl("border"))
    ax.set_xlabel("x", color=txt, fontsize=7)
    ax.set_ylabel("y", color=txt, fontsize=7)
    ax.set_zlabel("U", color=txt, fontsize=7)
    ax.view_init(elev=28, azim=-55)
    fig.tight_layout(pad=0.3)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=bg)
    plt.close(fig); buf.seek(0)
    return CoreImage(buf, ext="png")


def render_freq_chart(results, dpi=80):
    comp = results["comp"]
    bg   = tuple(C["bg_card"][:3])
    n    = len(comp)
    x    = np.arange(n)
    fig, ax = plt.subplots(figsize=(5.5, 2.6), dpi=dpi, facecolor=bg)
    ax.set_facecolor(bg)
    f_num  = [c_["f_num"]  for c_ in comp]
    f_theo = [c_["f_theo"] for c_ in comp]
    labels = [f"M{c_['mode']}\n({c_['m']},{c_['n']})" for c_ in comp]
    ax.bar(x-.18, f_num,  .34, color=rgb_mpl("cyan"),   label="Num.",   alpha=.9, zorder=3)
    ax.bar(x+.18, f_theo, .34, color=rgb_mpl("violet"), label="Bessel", alpha=.8, zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(labels, color=rgb_mpl("text_mid"), fontsize=6)
    ax.tick_params(axis="y", colors=rgb_mpl("text_mid"), labelsize=6)
    ax.set_ylabel("Hz", color=rgb_mpl("text_mid"), fontsize=7)
    ax.set_title("Num. vs Bessel", color=rgb_mpl("text_hi"), fontsize=8, pad=4)
    ax.legend(framealpha=.2, labelcolor=rgb_mpl("text_hi"), fontsize=6, facecolor=bg)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    for sp in ["bottom","left"]: ax.spines[sp].set_color(rgb_mpl("border"))
    ax.yaxis.grid(True, color=rgb_mpl("border"), linewidth=.5, zorder=0)
    fig.tight_layout(pad=0.4)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=bg)
    plt.close(fig); buf.seek(0)
    return CoreImage(buf, ext="png")


# ─────────────────────────────────────────────────────────────────────────────
#  HELPER WIDGET
# ─────────────────────────────────────────────────────────────────────────────
class GradientRect(Widget):
    top_color    = ListProperty([0.05, 0.07, 0.12, 1])
    bottom_color = ListProperty([0.07, 0.10, 0.17, 1])
    def __init__(self, **kw):
        super().__init__(**kw)
        self.bind(pos=self._draw, size=self._draw)
    def _draw(self, *_):
        self.canvas.before.clear()
        n = 32
        with self.canvas.before:
            for i in range(n):
                t = i/n; t2 = (i+1)/n
                def lerp(a,b,t): return a*(1-t)+b*t
                r  = lerp(self.top_color[0], self.bottom_color[0], (t+t2)/2)
                g  = lerp(self.top_color[1], self.bottom_color[1], (t+t2)/2)
                b  = lerp(self.top_color[2], self.bottom_color[2], (t+t2)/2)
                Color(r, g, b, 1)
                y = self.y + self.height*(1-t2)
                Rectangle(pos=(self.x, y), size=(self.width, self.height/n+1))


# ─────────────────────────────────────────────────────────────────────────────
#  SCREEN 1 — SPLASH
# ─────────────────────────────────────────────────────────────────────────────
class SplashScreen(Screen):
    def on_enter(self):
        self.clear_widgets()
        root = MDFloatLayout()
        root.add_widget(GradientRect(top_color=C["bg_deep"],
                                     bottom_color=C["bg_mid"], size_hint=(1,1)))
        box = MDBoxLayout(orientation="vertical",
                          size_hint=(None,None), size=(dp(300),dp(320)),
                          pos_hint={"center_x":.5,"center_y":.5},
                          spacing=dp(10), padding=dp(20))

        icon_w = Widget(size_hint=(None,None), size=(dp(70),dp(70)),
                        pos_hint={"center_x":.5})
        with icon_w.canvas:
            Color(*C["cyan"]);   Line(circle=(dp(35),dp(35),dp(28)), width=2.5)
            Color(*C["violet"]); Line(circle=(dp(35),dp(35),dp(18)), width=2)
            Color(*C["teal"]);   Line(circle=(dp(35),dp(35),dp(8)),  width=1.5)
        box.add_widget(icon_w)

        box.add_widget(MDLabel(
            text="[b]Membrane Circulaire[/b]", markup=True,
            halign="center", font_style="H5",
            theme_text_color="Custom", text_color=C["text_hi"],
            size_hint_y=None, height=dp(36),
        ))
        box.add_widget(MDLabel(
            text="Modes Propres & Bessel",
            halign="center", font_style="Subtitle2",
            theme_text_color="Custom", text_color=C["cyan"],
            size_hint_y=None, height=dp(24),
        ))
        box.add_widget(MDLabel(
            text="[b]Koussay Khalfalli[/b]", markup=True,
            halign="center", font_style="Caption",
            theme_text_color="Custom", text_color=C["violet"],
            size_hint_y=None, height=dp(20),
        ))
        prog = MDProgressBar(value=0, max=100,
                             size_hint=(.8,None), height=dp(4),
                             pos_hint={"center_x":.5})
        box.add_widget(prog)
        box.add_widget(MDLabel(
            text="v2.1 Android -- Differences Finies",
            halign="center", font_style="Caption",
            theme_text_color="Custom", text_color=C["text_lo"],
            size_hint_y=None, height=dp(18),
        ))
        root.add_widget(box)
        self.add_widget(root)

        def tick(dt):
            prog.value += 3
            if prog.value >= 100:
                Clock.unschedule(tick)
                Clock.schedule_once(
                    lambda *_: setattr(self.manager,"current","params"), 0.2)
        Clock.schedule_interval(tick, 0.04)


# ─────────────────────────────────────────────────────────────────────────────
#  SCREEN 2 — PARAMS
# ─────────────────────────────────────────────────────────────────────────────
class ParamsScreen(Screen):
    def on_enter(self):
        self.clear_widgets()
        root = MDFloatLayout()
        root.add_widget(GradientRect(top_color=C["bg_deep"],
                                     bottom_color=C["bg_mid"], size_hint=(1,1)))
        sv  = ScrollView(size_hint=(1,1))
        col = MDBoxLayout(orientation="vertical", size_hint=(1,None),
                          spacing=dp(12), padding=[dp(16),dp(20),dp(16),dp(20)])
        col.bind(minimum_height=col.setter("height"))

        col.add_widget(MDLabel(
            text="[b]Parametres[/b]", markup=True, halign="center",
            font_style="H5", theme_text_color="Custom", text_color=C["text_hi"],
            size_hint_y=None, height=dp(38),
        ))
        col.add_widget(MDLabel(
            text="[color=8866f5][b]Koussay Khalfalli[/b][/color]",
            markup=True, halign="center", font_style="Caption",
            theme_text_color="Custom", text_color=C["violet"],
            size_hint_y=None, height=dp(20),
        ))
        col.add_widget(Widget(size_hint_y=None, height=dp(6)))

        phys = self._card("Parametres Physiques", dp(186))
        self.field_R = self._field("Rayon R (m)", "1.0")
        self.field_c = self._field("Celerite c (m/s)", "1.0")
        phys.add_widget(self.field_R); phys.add_widget(self.field_c)
        col.add_widget(phys)

        num = self._card("Discretisation & Modes", dp(252))
        self.field_Nr     = self._field("Points en r (Nr)", "15")
        self.field_Ntheta = self._field("Points en theta (Ntheta)", "20")
        self.field_modes  = self._field("Nombre de modes", "6")
        num.add_widget(self.field_Nr)
        num.add_widget(self.field_Ntheta)
        num.add_widget(self.field_modes)
        col.add_widget(num)

        info = MDCard(elevation=0, radius=[dp(10)],
                      md_bg_color=(0.10,0.14,0.25,1),
                      padding=dp(10), size_hint_y=None, height=dp(66))
        info.add_widget(MDLabel(
            text=("[b][color=44d4eb]Conseil mobile :[/color][/b]\n"
                  "Nr=15, Nth=20 → rapide (~5s) precision ~1%\n"
                  "Nr=25, Nth=30 → lent (~60s) precision ~0.3%"),
            markup=True, font_style="Caption",
            theme_text_color="Custom", text_color=C["text_mid"],
        ))
        col.add_widget(info)

        btn = MDRaisedButton(
            text="  LANCER LE CALCUL",
            md_bg_color=C["cyan"], theme_text_color="Custom",
            text_color=C["bg_deep"], font_style="Button",
            size_hint=(1,None), height=dp(52), elevation=6,
        )
        btn.bind(on_release=self.launch)
        col.add_widget(Widget(size_hint_y=None, height=dp(6)))
        col.add_widget(btn)
        sv.add_widget(col); root.add_widget(sv)
        self.add_widget(root)

    def _card(self, title, height):
        card = MDCard(orientation="vertical", elevation=0,
                      radius=[dp(12)], md_bg_color=C["bg_card"],
                      padding=dp(14), spacing=dp(10),
                      size_hint_y=None, height=height)
        card.add_widget(MDLabel(
            text=f"[b]{title}[/b]", markup=True,
            theme_text_color="Custom", text_color=C["cyan"],
            font_style="Subtitle2", size_hint_y=None, height=dp(26),
        ))
        return card

    def _field(self, hint, default):
        return MDTextField(
            hint_text=hint, text=default, mode="rectangle",
            size_hint_y=None, height=dp(52),
            line_color_focus=C["cyan"], text_color_focus=C["text_hi"],
            hint_text_color_normal=C["text_lo"],
            fill_color_normal=C["bg_input"], fill_color_focus=C["bg_input"],
            input_type="number",
        )

    def _validate(self):
        errs = []
        try:
            R = float(self.field_R.text);      assert R > 0
        except: errs.append("R > 0"); R = 1.0
        try:
            c = float(self.field_c.text);      assert c > 0
        except: errs.append("c > 0"); c = 1.0
        try:
            Nr = int(self.field_Nr.text);      assert Nr >= 5
        except: errs.append("Nr >= 5"); Nr = 15
        try:
            Ntheta = int(self.field_Ntheta.text); assert Ntheta >= 8
        except: errs.append("Ntheta >= 8"); Ntheta = 20
        try:
            modes = int(self.field_modes.text); assert modes >= 1
        except: errs.append("Modes >= 1"); modes = 6
        return errs, R, c, Nr, Ntheta, modes

    def launch(self, *_):
        errs, R, c, Nr, Ntheta, modes = self._validate()
        if errs:
            Snackbar(text=" | ".join(errs),
                     snackbar_x=dp(10), snackbar_y=dp(20),
                     size_hint_x=.9).open()
            return
        cs = self.manager.get_screen("compute")
        cs.params = {"R":R,"c":c,"Nr":Nr,"Ntheta":Ntheta,"modes":modes}
        self.manager.current = "compute"


# ─────────────────────────────────────────────────────────────────────────────
#  SCREEN 3 — COMPUTE
# ─────────────────────────────────────────────────────────────────────────────
class ComputeScreen(Screen):
    params  = {}
    results = {}

    def on_enter(self):
        self.clear_widgets()
        root = MDFloatLayout()
        root.add_widget(GradientRect(top_color=C["bg_deep"],
                                     bottom_color=C["bg_mid"], size_hint=(1,1)))
        vbox = MDBoxLayout(orientation="vertical", size_hint=(.97,.96),
                           pos_hint={"center_x":.5,"center_y":.5},
                           spacing=dp(8), padding=dp(10))

        hdr = MDBoxLayout(size_hint_y=None, height=dp(44), spacing=dp(10))
        self._spinner = MDSpinner(size_hint=(None,None), size=(dp(32),dp(32)),
                                  active=True, color=C["cyan"])
        hdr.add_widget(self._spinner)
        self._title_lbl = MDLabel(
            text="[b]Calcul en cours...[/b]", markup=True,
            theme_text_color="Custom", text_color=C["text_hi"], font_style="H6")
        hdr.add_widget(self._title_lbl)
        vbox.add_widget(hdr)

        self._prog = MDProgressBar(value=0, max=100,
                                   size_hint_y=None, height=dp(4))
        vbox.add_widget(self._prog)

        log_card = MDCard(elevation=0, radius=[dp(10)],
                          md_bg_color=C["bg_card"], padding=dp(8), size_hint=(1,1))
        sv = ScrollView()
        self._log_layout = MDBoxLayout(orientation="vertical",
                                       size_hint=(1,None), spacing=dp(1))
        self._log_layout.bind(minimum_height=self._log_layout.setter("height"))
        sv.add_widget(self._log_layout)
        log_card.add_widget(sv)
        vbox.add_widget(log_card)
        self._sv = sv

        btn_row = MDBoxLayout(size_hint_y=None, height=dp(46), spacing=dp(8))
        btn_back = MDFlatButton(text="<- Params",
                                theme_text_color="Custom", text_color=C["cyan"],
                                size_hint_x=.4)
        btn_back.bind(on_release=lambda *_: setattr(
            self.manager,"current","params"))
        self._btn_results = MDRaisedButton(
            text="Voir Resultats", md_bg_color=C["cyan"],
            theme_text_color="Custom", text_color=C["bg_deep"],
            disabled=True, size_hint_x=.6)
        self._btn_results.bind(on_release=lambda *_: setattr(
            self.manager,"current","results"))
        btn_row.add_widget(btn_back); btn_row.add_widget(self._btn_results)
        vbox.add_widget(btn_row)
        root.add_widget(vbox); self.add_widget(root)
        threading.Thread(target=self._run, daemon=True).start()

    def _log(self, text):
        Clock.schedule_once(partial(self._append, text))

    def _append(self, text, *_):
        lbl = MDLabel(text=text, markup=True, theme_text_color="Custom",
                      text_color=C["text_mid"], font_style="Caption",
                      size_hint_y=None)
        lbl.bind(texture_size=lambda w,v: setattr(w,"height",v[1]+dp(2)))
        self._log_layout.add_widget(lbl)
        Clock.schedule_once(lambda *_: setattr(self._sv,"scroll_y",0), 0.05)

    def _run(self):
        p = self.params
        Clock.schedule_once(lambda *_: setattr(self._prog,"value",5))
        def log_cb(text):
            self._log(text)
            if self._prog.value < 85:
                Clock.schedule_once(lambda *_: setattr(
                    self._prog,"value",min(85,self._prog.value+2)))
        res = compute_membrane(p["R"],p["c"],p["Nr"],p["Ntheta"],p["modes"],log_cb)
        self.results = res
        Clock.schedule_once(self._on_done)

    def _on_done(self, *_):
        self._spinner.active = False
        if "error" in self.results:
            self._title_lbl.text = "[b][color=f24444]Erreur[/color][/b]"
        else:
            self._prog.value = 100
            self._title_lbl.text = "[b][color=40e090]Termine ![/color][/b]"
            self._btn_results.disabled = False
            self.manager.get_screen("results").results = self.results


# ─────────────────────────────────────────────────────────────────────────────
#  SCREEN 4 — RESULTS
# ─────────────────────────────────────────────────────────────────────────────
class ResultsScreen(Screen):
    results = {}

    def on_enter(self):
        self.clear_widgets()
        res = self.results
        if not res or "freqs" not in res:
            self.add_widget(MDLabel(text="Aucun resultat.", halign="center"))
            return

        root = MDFloatLayout()
        root.add_widget(GradientRect(top_color=C["bg_deep"],
                                     bottom_color=C["bg_mid"], size_hint=(1,1)))
        sv_main = ScrollView(size_hint=(1,1))
        col = MDBoxLayout(orientation="vertical", size_hint=(1,None),
                          spacing=dp(10),
                          padding=[dp(12),dp(10),dp(12),dp(16)])
        col.bind(minimum_height=col.setter("height"))

        # Nav
        nav = MDBoxLayout(size_hint_y=None, height=dp(40), spacing=dp(6))
        btn_back = MDFlatButton(text="<- Params",
                                theme_text_color="Custom", text_color=C["cyan"],
                                size_hint_x=None, width=dp(100))
        btn_back.bind(on_release=lambda *_: setattr(
            self.manager,"current","params"))
        nav.add_widget(btn_back)
        nav.add_widget(MDLabel(
            text="[b]Resultats[/b] · [color=8866f5]K. Khalfalli[/color]",
            markup=True, theme_text_color="Custom",
            text_color=C["text_hi"], font_style="Subtitle2"))
        col.add_widget(nav)

        # Metrics
        metr = MDBoxLayout(size_hint_y=None, height=dp(68), spacing=dp(6))
        metr.add_widget(self._metric("Df moy.",
                                     f"{res['d_mean']:.5f} Hz",
                                     res["d_mean"] < 0.01))
        metr.add_widget(self._metric("Err. rel.",
                                     f"{res['d_mean_rel']:.3f}%",
                                     res["d_mean_rel"] < 1.0))
        lp = ("EXCEL." if res["d_mean_rel"]<0.5
              else "BONNE" if res["d_mean_rel"]<1.0 else "MOY.")
        metr.add_widget(self._metric("Precision", lp, res["d_mean_rel"]<1.0))
        col.add_widget(metr)

        # Freq chart
        col.add_widget(self._section("Spectre de Frequences"))
        fc = MDCard(elevation=0, radius=[dp(10)], md_bg_color=C["bg_card"],
                    size_hint=(1,None), height=dp(180))
        self._freq_img = KivyImage(allow_stretch=True, keep_ratio=True)
        fc.add_widget(self._freq_img)
        col.add_widget(fc)

        # Mode viewer
        col.add_widget(self._section("Mode Propre"))
        nb_modes = len(res["freqs"])
        self._mode_idx = 0
        sel = MDBoxLayout(size_hint_y=None, height=dp(44), spacing=dp(4))
        bp = MDIconButton(icon="chevron-left",
                          theme_text_color="Custom", text_color=C["cyan"])
        bp.bind(on_release=self._prev_mode)
        bn = MDIconButton(icon="chevron-right",
                          theme_text_color="Custom", text_color=C["cyan"])
        bn.bind(on_release=self._next_mode)
        self._mode_lbl = MDLabel(
            text=f"Mode 1 / {nb_modes}",
            theme_text_color="Custom", text_color=C["text_hi"],
            font_style="Body2", halign="center")
        self._view_3d = False
        self._btn_toggle = MDRaisedButton(
            text="Vue 3D", md_bg_color=C["violet"],
            theme_text_color="Custom", text_color=(1,1,1,1),
            size_hint=(None,None), size=(dp(82),dp(36)))
        self._btn_toggle.bind(on_release=self._toggle_view)
        sel.add_widget(bp); sel.add_widget(self._mode_lbl)
        sel.add_widget(bn); sel.add_widget(self._btn_toggle)
        col.add_widget(sel)

        mc = MDCard(elevation=0, radius=[dp(10)], md_bg_color=C["bg_card"],
                    size_hint=(1,None), height=dp(280))
        self._mode_img = KivyImage(allow_stretch=True, keep_ratio=True)
        mc.add_widget(self._mode_img)
        col.add_widget(mc)

        # Table
        col.add_widget(self._section("Comparaison Numerique / Bessel"))
        col.add_widget(self._build_table(res["comp"]))

        # Matrix info
        ic = MDCard(elevation=0, radius=[dp(10)], md_bg_color=C["bg_card"],
                    padding=dp(10), size_hint_y=None, height=dp(56))
        ic.add_widget(MDLabel(
            text=(f"Matrice [b]{res['N']}x{res['N']}[/b] "
                  f"| [color=40e0b0]{res['nnz']}[/color] nz "
                  f"| {res['fill']:.3f}%\n"
                  f"Dr={res['dr']:.5f}  Dtheta={res['dtheta']:.5f}"),
            markup=True, theme_text_color="Custom",
            text_color=C["text_mid"], font_style="Caption"))
        col.add_widget(ic)

        sv_main.add_widget(col); root.add_widget(sv_main)
        self.add_widget(root)
        Clock.schedule_once(self._render_freq, 0.1)
        Clock.schedule_once(self._render_mode, 0.2)

    def _section(self, text):
        return MDLabel(text=f"[b]{text}[/b]", markup=True,
                       theme_text_color="Custom", text_color=C["cyan"],
                       font_style="Subtitle2",
                       size_hint_y=None, height=dp(24))

    def _metric(self, label, value, ok):
        card = MDCard(orientation="vertical", elevation=0,
                      radius=[dp(10)], md_bg_color=C["bg_card"],
                      padding=dp(6), size_hint_x=1)
        card.add_widget(MDLabel(
            text=value, theme_text_color="Custom",
            text_color=C["ok_green"] if ok else C["warn_amber"],
            font_style="H6", halign="center", bold=True))
        card.add_widget(MDLabel(
            text=label, theme_text_color="Custom",
            text_color=C["text_lo"], font_style="Caption", halign="center"))
        return card

    def _build_table(self, comp):
        grid = GridLayout(cols=6, size_hint=(1,None),
                          spacing=[dp(1),dp(2)], padding=[dp(2),dp(2)])
        grid.bind(minimum_height=grid.setter("height"))
        for h in ["#","f num","f theo","|Df|","(m,n)","OK"]:
            grid.add_widget(MDLabel(
                text=f"[b]{h}[/b]", markup=True,
                theme_text_color="Custom", text_color=C["cyan"],
                font_style="Caption", halign="center",
                size_hint_y=None, height=dp(24)))
        for c_ in comp:
            ok  = c_["negligeable"]
            row = [str(c_["mode"]), f"{c_['f_num']:.4f}",
                   f"{c_['f_theo']:.4f}", f"{c_['diff']:.4f}",
                   f"({c_['m']},{c_['n']})", "OK" if ok else "!!"]
            cols = [C["text_hi"], C["cyan"], C["text_mid"],
                    C["text_mid"] if ok else C["warn_amber"],
                    C["violet"], C["ok_green"] if ok else C["err_red"]]
            for txt, col in zip(row, cols):
                grid.add_widget(MDLabel(
                    text=txt, theme_text_color="Custom",
                    text_color=col, font_style="Caption",
                    halign="center", size_hint_y=None, height=dp(22)))
        return grid

    def _render_freq(self, *_):
        def _bg():
            img = render_freq_chart(self.results)
            Clock.schedule_once(partial(self._set_img, self._freq_img, img))
        threading.Thread(target=_bg, daemon=True).start()

    def _render_mode(self, *_):
        def _bg():
            fn  = render_mode_3d if self._view_3d else render_mode_2d
            img = fn(self.results, self._mode_idx)
            Clock.schedule_once(partial(self._set_img, self._mode_img, img))
        threading.Thread(target=_bg, daemon=True).start()

    def _set_img(self, widget, img, *_):
        widget.texture = img.texture

    def _prev_mode(self, *_):
        nb = len(self.results["freqs"])
        self._mode_idx = (self._mode_idx-1) % nb
        self._mode_lbl.text = f"Mode {self._mode_idx+1} / {nb}"
        self._render_mode()

    def _next_mode(self, *_):
        nb = len(self.results["freqs"])
        self._mode_idx = (self._mode_idx+1) % nb
        self._mode_lbl.text = f"Mode {self._mode_idx+1} / {nb}"
        self._render_mode()

    def _toggle_view(self, *_):
        self._view_3d = not self._view_3d
        self._btn_toggle.text = "Vue 2D" if self._view_3d else "Vue 3D"
        self._btn_toggle.md_bg_color = C["cyan"] if self._view_3d else C["violet"]
        self._btn_toggle.text_color  = (
            C["bg_deep"] if self._view_3d else (1,1,1,1))
        self._render_mode()


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN APP
# ─────────────────────────────────────────────────────────────────────────────
class MembraneApp(MDApp):
    def build(self):
        self.theme_cls.theme_style     = "Dark"
        self.theme_cls.primary_palette = "Cyan"
        self.theme_cls.accent_palette  = "DeepPurple"
        Builder.load_string(KV)
        Window.clearcolor = C["bg_deep"]
        sm = ScreenManager()
        sm.add_widget(SplashScreen(name="splash"))
        sm.add_widget(ParamsScreen(name="params"))
        sm.add_widget(ComputeScreen(name="compute"))
        sm.add_widget(ResultsScreen(name="results"))
        return sm

    def on_start(self):
        self.title = "Membrane Circulaire"


if __name__ == "__main__":
    MembraneApp().run()
