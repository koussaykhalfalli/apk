[app]

# ── Identity ─────────────────────────────────────────────────────────────────
title           = Membrane Circulaire
package.name    = membranecirculaire
package.domain  = org.khalfalli
version         = 2.1

# ── Source ───────────────────────────────────────────────────────────────────
source.dir      = .
source.include_exts = py,png,jpg,kv,atlas

# ── Entry point ──────────────────────────────────────────────────────────────
# Rename membrane_app_mobile.py → main.py in your project folder,
# OR set this to the actual filename:
entrypoint = main.py

# ── Requirements ─────────────────────────────────────────────────────────────
# All pure-Python or pre-built wheels available for Android via p4a recipes.
# scipy is NOT available as a p4a recipe → use kivy-garden or build yourself.
# Practical alternative: use numpy eigenvalue solver (numpy.linalg.eig)
# instead of scipy.sparse.linalg.eigs.  See note in README.
requirements = python3,kivy==2.3.0,kivymd==1.1.1,numpy,matplotlib,pillow

# ── Android settings ─────────────────────────────────────────────────────────
android.minapi  = 21
android.sdk     = 33
android.ndk     = 25b
android.arch    = arm64-v8a
android.permissions = INTERNET,WRITE_EXTERNAL_STORAGE,READ_EXTERNAL_STORAGE

# ── Orientation ──────────────────────────────────────────────────────────────
orientation = portrait

# ── Fullscreen / status bar ──────────────────────────────────────────────────
fullscreen = 0

# ── Icons / Presplash (place your own files here) ────────────────────────────
# icon.filename     = %(source.dir)s/icon.png
# presplash.filename = %(source.dir)s/presplash.png

# ── Build ────────────────────────────────────────────────────────────────────
[buildozer]
log_level = 2
warn_on_root = 1
