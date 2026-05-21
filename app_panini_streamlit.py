"""
🃏 CCD·UNAB — Generador de Tarjeta Panini FIFA World Cup 2026
Powered by DCGAN entrenada con rostros de futbolistas FIFA.

Dependencias (requirements.txt):
    streamlit
    tensorflow
    pillow
    numpy
    gdown
"""

import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import io
import os
import gdown
import math

# ─── Configuración de página ────────────────────────────────────────────────
st.set_page_config(
    page_title="Panini FIFA 2026 · CCD·UNAB",
    page_icon="🃏",
    layout="centered",
)

# ─── Constantes ─────────────────────────────────────────────────────────────
Z_DIM        = 100
IMG_SIZE     = 64
MODEL_PATH   = "generador_fifa_ccd.keras"

# URL pública del modelo en Google Drive o GitHub Releases.
# ⚠️  AJUSTA ESTA URL con el enlace real de tu modelo subido.
MODEL_URL = "https://drive.google.com/uc?id=REEMPLAZA_CON_TU_FILE_ID"

# ─── Carga del modelo (con caché) ───────────────────────────────────────────
@st.cache_resource(show_spinner="Cargando generador DCGAN…")
def cargar_modelo():
    if not os.path.exists(MODEL_PATH):
        st.info("📥 Descargando modelo desde la nube… (solo la primera vez)")
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    modelo = tf.keras.models.load_model(MODEL_PATH, compile=False)
    return modelo

# ─── Generación de imagen por semilla ───────────────────────────────────────
def generar_imagen(modelo, semilla: int) -> Image.Image:
    """Genera una cara de futbolista a partir de una semilla entera."""
    tf.random.set_seed(semilla)
    z = tf.random.normal([1, Z_DIM], seed=semilla)
    img_tensor = modelo(z, training=False)
    img_np = ((img_tensor[0].numpy() + 1.0) / 2.0)
    img_np = np.clip(img_np, 0, 1)
    img_uint8 = (img_np * 255).astype(np.uint8)
    return Image.fromarray(img_uint8)


# ─── Helpers de dibujo ──────────────────────────────────────────────────────

def _font(path: str, size: int) -> ImageFont.FreeTypeFont:
    """Carga fuente TrueType con fallback al default de Pillow."""
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

BOLD   = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
REGULAR = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"

def _center_x(draw, text, font, card_w, x_offset=0):
    """Devuelve la coordenada x para centrar texto horizontalmente."""
    bb = draw.textbbox((0, 0), text, font=font)
    return (card_w - (bb[2] - bb[0])) // 2 + x_offset

def _text_w(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[2] - bb[0]

def _text_h(draw, text, font):
    bb = draw.textbbox((0, 0), text, font=font)
    return bb[3] - bb[1]

def _draw_circle_flag(draw, cx, cy, r, color_main, color_secondary=(255,255,255)):
    """Dibuja un símbolo de bandera circular simple (círculo bicolor)."""
    draw.ellipse([cx - r, cy - r, cx + r, cy + r],
                 fill=color_secondary, outline=(30, 30, 30), width=2)
    draw.ellipse([cx - r + 4, cy - r + 4, cx + r - 4, cy + r - 4],
                 fill=color_main)

def _draw_trophy(draw, x, y, size, color=(255, 255, 255)):
    """Dibuja un trofeo FIFA simplificado con líneas."""
    s = size
    # Copa
    pts_cup = [
        x, y,
        x + s, y,
        x + s * 0.85, y + s * 0.6,
        x + s * 0.55, y + s * 0.75,
        x + s * 0.55, y + s * 0.9,
        x + s * 0.7, y + s * 0.9,
        x + s * 0.7, y + s,
        x + s * 0.3, y + s,
        x + s * 0.3, y + s * 0.9,
        x + s * 0.45, y + s * 0.9,
        x + s * 0.45, y + s * 0.75,
        x + s * 0.15, y + s * 0.6,
    ]
    draw.polygon(pts_cup, fill=color)
    # Asas izquierda
    draw.arc([x - s * 0.15, y + s * 0.05, x + s * 0.25, y + s * 0.5],
             start=180, end=360, fill=color, width=max(2, s // 10))
    # Asas derecha
    draw.arc([x + s * 0.75, y + s * 0.05, x + s * 1.15, y + s * 0.5],
             start=0, end=180, fill=color, width=max(2, s // 10))


# ─── Construcción de la tarjeta Panini ──────────────────────────────────────

def generar_tarjeta_panini(
    foto_pil: Image.Image,
    nombre_jugador: str = "TU NOMBRE",
    fecha_nacimiento: str = "DD-MM-YYYY",
    altura: str = "1,76 m",
    peso: str = "73 kg",
    club: str = "REAL MADRID CF (ESP)",
    pais_abrev: str = "BRA",
    color_fondo: tuple = (0, 200, 210),      # turquesa FIFA
    color_acento: tuple = (0, 160, 80),      # verde bandera
    card_w: int = 420,
    card_h: int = 560,
) -> Image.Image:
    """
    Genera una tarjeta coleccionable lo más fiel posible al diseño
    oficial Panini FIFA World Cup 2026.

    Estructura visual (de arriba a abajo):
      1. Fondo turquesa sólido
      2. Números "26" grandes semitransparentes (izq verde / der blanco)
      3. Logo FIFA + trofeo (esquina sup. der.)
      4. Foto del jugador (ocupa casi toda la carta)
      5. Franja lateral derecha: bandera + letras del país
      6. Panel inferior blanco: nombre / datos / club / PANINI
    """

    # ── Paleta ──────────────────────────────────────────────────────────────
    BLANCO  = (255, 255, 255)
    NEGRO   = (10,  10,  10)
    AMARILLO= (255, 210,   0)
    GRIS    = (80,  80,  80)

    COLORES_PAIS = {
        "green":  (0,  160,  80),
        "red":    (210,  30,  30),
        "blue":   (30,  80, 180),
        "yellow": (200, 160,   0),
        "orange": (220, 100,  20),
        "white":  (200, 200, 200),
    }

    # ── Lienzo base (fondo turquesa) ─────────────────────────────────────────
    card = Image.new("RGBA", (card_w, card_h), (*color_fondo, 255))
    draw = ImageDraw.Draw(card, "RGBA")

    # ── Fuentes ──────────────────────────────────────────────────────────────
    f_num   = _font(BOLD,    220)   # dígitos "2" y "6"
    f_fifa  = _font(BOLD,     18)   # "FIFA"
    f_wc    = _font(REGULAR,  11)   # "WORLD CUP"
    f_year  = _font(BOLD,     16)   # "2026"
    f_name  = _font(BOLD,     26)   # nombre jugador
    f_data  = _font(REGULAR,  15)   # datos biométricos
    f_club  = _font(BOLD,     16)   # club
    f_panini= _font(BOLD,     15)   # logo PANINI
    f_pais  = _font(BOLD,     20)   # letras país vertical
    f_ccd   = _font(REGULAR,  11)   # marca CCD·UNAB
    f_num_sm= _font(BOLD,     14)   # "26" en bloque derecho

    # ── 1. Números decorativos "2 6" grandes ────────────────────────────────
    # "2" verde izquierda  (muy transparente, sólo textura)
    overlay_num = Image.new("RGBA", (card_w, card_h), (0, 0, 0, 0))
    dn = ImageDraw.Draw(overlay_num, "RGBA")

    # "2" verde grande (izquierda) — igual al original
    dn.text((-30, -10), "2", font=f_num, fill=(*color_acento, 160))
    # "6" verde grande (izquierda, desplazado abajo)
    dn.text((50, 110), "6", font=f_num, fill=(*color_acento, 160))

    # "2" blanco derecha (más arriba)
    dn.text((card_w - 185, -10), "2", font=f_num, fill=(255, 255, 255, 60))
    # "6" blanco derecha
    dn.text((card_w - 120, 110), "6", font=f_num, fill=(255, 255, 255, 55))

    card = Image.alpha_composite(card, overlay_num)
    draw = ImageDraw.Draw(card, "RGBA")

    # ── 2. Logo FIFA + trofeo (esquina superior derecha) ────────────────────
    logo_x = card_w - 105
    logo_y = 10

    # Trofeo FIFA
    _draw_trophy(draw, logo_x + 10, logo_y + 2, 38, color=BLANCO)

    # Texto "FIFA"
    draw.text((logo_x + 3, logo_y + 45), "FIFA",
              font=f_fifa, fill=BLANCO)
    draw.text((logo_x, logo_y + 64), "WORLD CUP",
              font=f_wc, fill=BLANCO)
    draw.text((logo_x + 8, logo_y + 77), "2026",
              font=f_year, fill=AMARILLO)

    # ── 3. Foto del jugador ──────────────────────────────────────────────────
    # La foto ocupa prácticamente toda la carta excepto el panel inferior
    panel_h   = 118    # altura del panel inferior blanco
    franja_w  = 55     # ancho de la franja lateral derecha
    foto_w    = card_w - franja_w
    foto_h    = card_h - panel_h
    foto_x    = 0
    foto_y    = 0

    img_jugador = foto_pil.convert("RGB")

    if img_jugador.width <= 64:
        # Imagen GAN 64×64 → upscale con nearest para estilo pixel art nítido
        img_jugador = img_jugador.resize((foto_w, foto_h), Image.NEAREST)
    else:
        # Foto real: crop cuadrado centrado, luego resize
        w_j, h_j = img_jugador.size
        # Mantener más de la parte superior (cara) → crop desde arriba
        lado = min(w_j, h_j)
        top  = 0
        left = (w_j - lado) // 2
        img_jugador = img_jugador.crop((left, top, left + lado, top + lado))
        img_jugador = img_jugador.resize((foto_w, foto_h), Image.LANCZOS)

    # Realce de color y brillo
    img_jugador = ImageEnhance.Color(img_jugador).enhance(1.25)
    img_jugador = ImageEnhance.Brightness(img_jugador).enhance(1.05)
    img_jugador = ImageEnhance.Contrast(img_jugador).enhance(1.1)

    card.paste(img_jugador.convert("RGBA"), (foto_x, foto_y))
    draw = ImageDraw.Draw(card, "RGBA")

    # ── 4. Franja lateral derecha ────────────────────────────────────────────
    fx = card_w - franja_w
    fy = 0
    fh = card_h - panel_h

    draw.rectangle([fx, fy, card_w, fh], fill=(*color_acento, 245))

    # Bandera circular centrada en la franja, un tercio desde arriba
    flag_cx = fx + franja_w // 2
    flag_cy = fy + fh // 3
    flag_r  = 20

    # Círculo exterior blanco + interior del color del país
    draw.ellipse(
        [flag_cx - flag_r, flag_cy - flag_r,
         flag_cx + flag_r, flag_cy + flag_r],
        fill=BLANCO, outline=(20, 20, 20), width=2
    )
    draw.ellipse(
        [flag_cx - flag_r + 5, flag_cy - flag_r + 5,
         flag_cx + flag_r - 5, flag_cy + flag_r - 5],
        fill=color_acento
    )
    # Cruz/estrella interior blanca
    draw.ellipse(
        [flag_cx - 5, flag_cy - 5, flag_cx + 5, flag_cy + 5],
        fill=BLANCO
    )

    # Letras del país verticales (debajo de la bandera)
    letters_start_y = flag_cy + flag_r + 18
    for i, letra in enumerate(pais_abrev.upper()[:3]):
        lw = _text_w(draw, letra, f_pais)
        draw.text(
            (fx + (franja_w - lw) // 2, letters_start_y + i * 30),
            letra, font=f_pais, fill=BLANCO
        )

    # Números "26" en blanco pequeños al fondo de la franja
    draw.text((fx + 8, fh - 80), "2", font=_font(BOLD, 55),
              fill=(255, 255, 255, 80))
    draw.text((fx + 8, fh - 45), "6", font=_font(BOLD, 55),
              fill=(255, 255, 255, 70))

    # ── 5. Panel inferior blanco ─────────────────────────────────────────────
    panel_y = card_h - panel_h
    draw.rectangle([0, panel_y, card_w, card_h], fill=(*BLANCO, 255))

    # Línea separadora superior del panel
    draw.line([(0, panel_y), (card_w, panel_y)], fill=(200, 200, 200), width=1)

    # — Nombre del jugador —
    nombre_upper = nombre_jugador.upper()
    draw.text(
        (_center_x(draw, nombre_upper, f_name, card_w), panel_y + 10),
        nombre_upper, font=f_name, fill=NEGRO
    )

    # — Datos biométricos —
    datos_str = f"{fecha_nacimiento}  |  {altura}  |  {peso}"
    draw.text(
        (_center_x(draw, datos_str, f_data, card_w), panel_y + 44),
        datos_str, font=f_data, fill=GRIS
    )

    # — Separador horizontal delgado —
    sep_y = panel_y + 66
    draw.line([(12, sep_y), (card_w - 12, sep_y)], fill=(210, 210, 210), width=1)

    # — Fila de club + logo PANINI —
    club_y = sep_y + 8

    # Franja celeste de fondo para el club (igual al original)
    draw.rectangle([0, club_y - 2, card_w, club_y + 32],
                   fill=(*color_fondo, 30))

    # Nombre del club
    club_upper = club.upper()
    draw.text((16, club_y + 4), club_upper, font=f_club, fill=NEGRO)

    # Logo PANINI (caja amarilla con texto, a la derecha)
    panini_label = "PANINI"
    pw = _text_w(draw, panini_label, f_panini)
    px1 = card_w - pw - 28
    px2 = card_w - 8
    py1 = club_y
    py2 = club_y + 28

    draw.rounded_rectangle([px1 - 6, py1, px2, py2],
                            radius=4, fill=AMARILLO)
    draw.text((px1, py1 + 5), panini_label, font=f_panini, fill=NEGRO)

    # ── 6. Borde exterior mínimo ─────────────────────────────────────────────
    draw_border = ImageDraw.Draw(card)
    draw_border.rectangle([0, 0, card_w - 1, card_h - 1],
                           outline=(180, 180, 180), width=1)

    # ── 7. Marca CCD·UNAB (discreta) ────────────────────────────────────────
    marca = "✦ DCGAN  CCD·UNAB"
    draw.text((8, card_h - 14), marca, font=f_ccd,
              fill=(180, 230, 240, 200))

    return card.convert("RGB")


# ═══════════════════════════════════════════════════════════════════════════════
#   INTERFAZ STREAMLIT
# ═══════════════════════════════════════════════════════════════════════════════

st.title("🃏 Generador de Tarjeta Panini FIFA 2026")
st.caption("Centro de Competencias Digitales · UNAB · Powered by DCGAN")

st.markdown("""
> La imagen de tu carta es **generada por inteligencia artificial** usando una
> red neuronal generativa adversaria (DCGAN) entrenada con rostros de futbolistas FIFA.
> Cada **semilla** produce un rostro distinto y reproducible.
""")

# ── Carga del modelo ─────────────────────────────────────────────────────────
try:
    modelo = cargar_modelo()
    st.success("✅ Modelo DCGAN listo")
except Exception as e:
    st.error(f"❌ No se pudo cargar el modelo: {e}")
    st.info("Asegúrate de configurar `MODEL_URL` en el código con el enlace de tu modelo.")
    st.stop()

# ── Columnas: configuración | previsualización ───────────────────────────────
col_config, col_preview = st.columns([1, 1], gap="large")

# Mapa de colores de países
COLORES_MAP = {
    "green":  ((0,  160,  80),  (0, 200, 210)),   # (acento, fondo)
    "red":    ((210,  30,  30), (0, 200, 210)),
    "blue":   ((30,  80, 180),  (0, 200, 210)),
    "yellow": ((200, 160,   0), (0, 200, 210)),
    "orange": ((220, 100,  20), (0, 200, 210)),
}

with col_config:
    st.subheader("🎮 Tu tarjeta")

    nombre    = st.text_input("Nombre completo", value="TU NOMBRE")
    fecha_nac = st.text_input("Fecha de nacimiento (DD-MM-YYYY)", value="01-01-2000")
    altura    = st.text_input("Altura", value="1,75 m")
    peso      = st.text_input("Peso", value="70 kg")
    club      = st.text_input("Club / Universidad", value="UNAB FC (COL)")
    pais      = st.text_input("País (3 letras)", value="COL", max_chars=3)

    color_pais = st.selectbox(
        "Color acento del país",
        options=list(COLORES_MAP.keys()),
        format_func=lambda x: {
            "green":  "🟢 Verde",
            "red":    "🔴 Rojo",
            "blue":   "🔵 Azul",
            "yellow": "🟡 Amarillo",
            "orange": "🟠 Naranja",
        }[x]
    )

    st.markdown("---")
    st.subheader("🎲 Semilla del generador")
    st.markdown("Cada semilla produce un **rostro diferente** generado por la GAN.")

    semilla = st.slider("Semilla", min_value=0, max_value=9999, value=42, step=1)

    # Vista previa rápida del rostro GAN
    img_gan = generar_imagen(modelo, semilla)
    img_gan_big = img_gan.resize((128, 128), Image.NEAREST)
    st.image(img_gan_big, caption=f"Rostro GAN — semilla {semilla}", width=128)

    generar_btn = st.button("🃏 Generar mi tarjeta", type="primary", use_container_width=True)

with col_preview:
    st.subheader("📋 Vista previa")

    acento, fondo = COLORES_MAP[color_pais]

    if generar_btn or True:  # mostrar preview en tiempo real
        with st.spinner("Generando tarjeta…"):
            tarjeta = generar_tarjeta_panini(
                foto_pil         = img_gan,
                nombre_jugador   = nombre,
                fecha_nacimiento = fecha_nac,
                altura           = altura,
                peso             = peso,
                club             = club,
                pais_abrev       = pais.upper(),
                color_fondo      = fondo,
                color_acento     = acento,
            )

        st.image(tarjeta, use_column_width=True)

        # Botón de descarga (300 DPI, ×3)
        buf = io.BytesIO()
        tarjeta_hd = tarjeta.resize(
            (tarjeta.width * 3, tarjeta.height * 3), Image.LANCZOS
        )
        tarjeta_hd.save(buf, format="PNG", dpi=(300, 300))
        buf.seek(0)

        nombre_archivo = f"panini_{nombre.replace(' ', '_')}_s{semilla}.png"
        st.download_button(
            label="⬇️ Descargar tarjeta (300 DPI)",
            data=buf,
            file_name=nombre_archivo,
            mime="image/png",
            use_container_width=True,
        )

# ── Sección educativa ────────────────────────────────────────────────────────
with st.expander("💡 ¿Cómo funciona? (Conceptos clave)", expanded=False):
    st.markdown("""
    ### ¿Por qué usamos semilla en vez de foto?

    Esta app usa una **DCGAN** (*Deep Convolutional Generative Adversarial Network*).
    La GAN **genera** rostros desde cero — no transforma tu foto.

    El proceso:
    ```
    Semilla (número entero)
        ↓
    Ruido aleatorio z ∈ ℝ¹⁰⁰  (vector latente)
        ↓
    GENERADOR (red neuronal entrenada con futbolistas FIFA)
        ↓
    Imagen 64×64 RGB de un "futbolista sintético"
    ```

    - La misma **semilla** siempre produce la misma cara (reproducible).
    - Semillas diferentes producen caras distintas.
    - El modelo **no almacena** ninguna foto real — solo aprendió patrones de distribución.

    ### Arquitectura del Generador
    | Capa | Salida |
    |------|--------|
    | Dense → Reshape | 4 × 4 × 512 |
    | ConvTranspose + BN + ReLU | 8 × 8 × 256 |
    | ConvTranspose + BN + ReLU | 16 × 16 × 128 |
    | ConvTranspose + BN + ReLU | 32 × 32 × 64 |
    | ConvTranspose + Tanh | **64 × 64 × 3** |
    """)

st.markdown("---")
st.caption("🏫 Centro de Competencias Digitales · UNAB · Bucaramanga, Colombia")
