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
from PIL import Image, ImageDraw, ImageFont, ImageEnhance
import io
import os
import gdown

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
# Ejemplo Google Drive (enlace de descarga directa):
#   https://drive.google.com/uc?id=TU_FILE_ID
# Ejemplo GitHub Releases:
#   https://github.com/tu-usuario/tu-repo/releases/download/v1.0/generador_fifa_ccd.keras
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

# ─── Construcción de la tarjeta Panini ──────────────────────────────────────
def generar_tarjeta_panini(
    foto_pil: Image.Image,
    nombre_jugador: str = "TU NOMBRE",
    fecha_nacimiento: str = "DD-MM-YYYY",
    altura: str = "1,75m",
    peso: str = "70 kg",
    club: str = "UNAB FC",
    pais_abrev: str = "COL",
    color_pais: str = "green",
    card_w: int = 420,
    card_h: int = 560,
) -> Image.Image:
    """Genera una tarjeta coleccionable estilo Panini FIFA World Cup 2026."""

    # Paleta oficial FIFA
    CELESTE  = (0,   200, 215)
    VERDE    = (0,   166,  81)
    ROJO     = (232,  49,  42)
    AMARILLO = (255, 215,   0)
    BLANCO   = (255, 255, 255)
    GRIS_OSC = ( 40,  40,  40)
    AZUL_OSC = ( 15,  30,  80)

    COLORES_PAIS = {
        "green":  VERDE,
        "red":    ROJO,
        "blue":   (30,  80, 160),
        "yellow": (200, 160,   0),
        "orange": (220, 100,  20),
    }
    color_acento = COLORES_PAIS.get(color_pais, VERDE)

    # Lienzo base
    card = Image.new("RGB", (card_w, card_h), CELESTE)
    draw = ImageDraw.Draw(card)

    # Borde arco-iris FIFA
    border_colors = [ROJO, AMARILLO, VERDE, (100, 50, 200), CELESTE]
    for i, bc in enumerate(border_colors):
        b = i * 3
        draw.rectangle([b, b, card_w - b - 1, card_h - b - 1], outline=bc, width=3)

    # Fondo con degradado simulado (franjas horizontales)
    for y in range(card_h):
        t = y / card_h
        r   = int(CELESTE[0] * (1 - t) + (CELESTE[0] - 30) * t)
        g   = int(CELESTE[1] * (1 - t) + (CELESTE[1] - 20) * t)
        b_c = int(CELESTE[2] * (1 - t) + (CELESTE[2] + 10) * t)
        draw.line([(15, y), (card_w - 15, y)], fill=(r, g, b_c))

    # Cargar fuentes
    try:
        font_big = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 180)
        font_med = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 36)
        font_sm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        font_xs  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",      16)
        font_nm  = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except Exception:
        font_big = font_med = font_sm = font_xs = font_nm = ImageFont.load_default()

    # Números decorativos "26" (fondo)
    draw.text((10, 50), "2",  font=font_big, fill=(*VERDE,  120))
    draw.text((85, 110), "6", font=font_big, fill=(*ROJO,   100))
    draw.text((card_w - 130, 50),  "2", font=font_big, fill=(*BLANCO, 80))
    draw.text((card_w - 70,  110), "6", font=font_big, fill=(*BLANCO, 70))

    # Logo FIFA World Cup 2026
    draw.text((20, 20),      "FIFA",      font=font_sm, fill=BLANCO)
    draw.text((20, 46),      "WORLD CUP", font=font_xs, fill=BLANCO)
    draw.text((20, 64),      "2026",      font=font_sm, fill=AMARILLO)

    # Foto del jugador generada por GAN
    foto_w, foto_h = 260, 290
    foto_x = (card_w - foto_w) // 2
    foto_y = 60

    img_jugador = foto_pil.convert("RGB")
    # Si es 64x64 (GAN), hacer upscale con calidad
    if img_jugador.width <= 64:
        img_jugador = img_jugador.resize((foto_w, foto_h), Image.NEAREST)
    else:
        w_j, h_j = img_jugador.size
        lado_j = min(w_j, h_j)
        img_jugador = img_jugador.crop(((w_j - lado_j) // 2, (h_j - lado_j) // 2,
                                         (w_j + lado_j) // 2, (h_j + lado_j) // 2))
        img_jugador = img_jugador.resize((foto_w, foto_h), Image.LANCZOS)

    img_jugador = ImageEnhance.Color(img_jugador).enhance(1.3)
    img_jugador = ImageEnhance.Brightness(img_jugador).enhance(1.05)
    card.paste(img_jugador, (foto_x, foto_y))
    draw.rectangle([foto_x - 2, foto_y - 2, foto_x + foto_w + 2, foto_y + foto_h + 2],
                   outline=BLANCO, width=3)

    # Banda de país (derecha)
    banda_x = card_w - 60
    draw.rectangle([banda_x, foto_y, card_w - 18, foto_y + foto_h], fill=color_acento)
    flag_cx = banda_x + 21
    flag_cy = foto_y + 50
    draw.ellipse([flag_cx - 20, flag_cy - 20, flag_cx + 20, flag_cy + 20],
                 fill=BLANCO, outline=GRIS_OSC, width=2)
    draw.ellipse([flag_cx - 13, flag_cy - 13, flag_cx + 13, flag_cy + 13],
                 fill=color_acento)
    for i, letra in enumerate(pais_abrev[:3]):
        draw.text((banda_x + 8, foto_y + 90 + i * 28), letra, font=font_sm, fill=BLANCO)

    # Panel nombre
    info_y = foto_y + foto_h + 15
    draw.rectangle([18, info_y, card_w - 18, info_y + 48], fill=(*AZUL_OSC, 220))
    draw.rectangle([18, info_y, card_w - 18, info_y + 48], outline=color_acento, width=2)
    bbox_nm = draw.textbbox((0, 0), nombre_jugador.upper(), font=font_med)
    nm_w = bbox_nm[2] - bbox_nm[0]
    draw.text(((card_w - nm_w) // 2, info_y + 6), nombre_jugador.upper(),
              font=font_med, fill=BLANCO)

    # Panel datos biométricos
    datos_y = info_y + 55
    draw.rectangle([18, datos_y, card_w - 18, datos_y + 32], fill=(20, 20, 60))
    datos_str = f"{fecha_nacimiento}  |  {altura}  |  {peso}"
    bbox_d = draw.textbbox((0, 0), datos_str, font=font_xs)
    d_w = bbox_d[2] - bbox_d[0]
    draw.text(((card_w - d_w) // 2, datos_y + 8), datos_str,
              font=font_xs, fill=(180, 180, 220))

    # Panel club
    club_y = datos_y + 40
    draw.rectangle([18, club_y, card_w - 18, club_y + 32], fill=(10, 10, 40))
    draw.rectangle([18, club_y, card_w - 18, club_y + 32], outline=AMARILLO, width=1)
    bbox_c = draw.textbbox((0, 0), club.upper(), font=font_nm)
    c_w = bbox_c[2] - bbox_c[0]
    draw.text(((card_w - c_w) // 2, club_y + 6), club.upper(), font=font_nm, fill=AMARILLO)

    # Logo PANINI
    panini_y = club_y + 42
    draw.rectangle([card_w - 130, panini_y, card_w - 18, panini_y + 30], fill=AMARILLO)
    draw.text((card_w - 123, panini_y + 5), "PANINI", font=font_sm, fill=GRIS_OSC)

    # Marca CCD·UNAB
    draw.text((20, card_h - 30), "✦ DCGAN  CCD·UNAB",
              font=font_xs, fill=(200, 240, 255))

    return card


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

with col_config:
    st.subheader("🎮 Tu tarjeta")

    nombre    = st.text_input("Nombre completo", value="TU NOMBRE")
    fecha_nac = st.text_input("Fecha de nacimiento (DD-MM-YYYY)", value="01-01-2000")
    altura    = st.text_input("Altura", value="1,75m")
    peso      = st.text_input("Peso", value="70 kg")
    club      = st.text_input("Club / Universidad", value="UNAB FC (COL)")
    pais      = st.text_input("País (3 letras)", value="COL", max_chars=3)

    color_pais = st.selectbox(
        "Color acento del país",
        options=["green", "red", "blue", "yellow", "orange"],
        format_func=lambda x: {"green": "🟢 Verde", "red": "🔴 Rojo",
                                "blue": "🔵 Azul", "yellow": "🟡 Amarillo",
                                "orange": "🟠 Naranja"}[x]
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
                color_pais       = color_pais,
            )

        # Mostrar tarjeta en la app
        st.image(tarjeta, use_column_width=True)

        # Botón de descarga
        buf = io.BytesIO()
        # Alta resolución x3
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
