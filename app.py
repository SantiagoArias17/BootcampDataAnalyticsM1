"""Home: presentación e identidad de la aplicación."""
import streamlit as st

from core.ui import ARENA, CRUDO, LODO, esquema_pozo, html

# Datos del participante (edita aquí tus nombres y apellidos completos)
PARTICIPANTE = "Santiago Arias"
TITULO_APP = "Tres Pistas"
PROGRAMA = "Bootcamp Data Analytics for Oil & Gas"

# ---------- Portada: identidad a la izquierda, esquema de pozo interactivo a la derecha ----------
col_texto, col_pozo = st.columns([1.05, 1], gap="large")

with col_texto:
    html(f"""
    <div class="titular">
      <h1>{TITULO_APP}</h1>
      <p>Una calculadora de ingeniería de petróleos organizada como un registro de pozo:
      una pista para producción, otra para perforación y otra para reservorios.</p>
    </div>
    <div class="tarjeta">
      <h3>Propósito técnico</h3>
      <p>Resolver tres cálculos de uso diario en campo: la curva IPR compuesta de un reservorio
      subsaturado, la presión hidrostática del lodo frente a la presión de formación y el
      petróleo original en sitio por el método volumétrico. Cada cálculo valida sus datos,
      muestra el resultado con unidades y lo acompaña de un gráfico para interpretarlo.</p>
      <dl class="ficha">
        <dt>Participante</dt><dd>{PARTICIPANTE}</dd>
        <dt>Aplicación</dt><dd>{TITULO_APP}: calculadora de ingeniería de petróleos</dd>
        <dt>Programa</dt><dd>{PROGRAMA}</dd>
        <dt>Organiza</dt><dd>SPE Ecuador Section</dd>
      </dl>
    </div>
    """)

with col_pozo:
    esquema_pozo(altura=450)

# ---------- Las tres pistas ----------
st.markdown("### Qué calcula cada pista")

pistas = [
    (
        CRUDO, "Producción",
        "IPR compuesta: tramo lineal sobre el punto de burbuja y Vogel por debajo. "
        "Entrega qₒ, qᵦ, qₒ,max y la curva completa.",
        "qₒ = qᵦ + (J·Pᵦ/1.8)[1 − 0.2(Pwf/Pᵦ) − 0.8(Pwf/Pᵦ)²]",
    ),
    (
        LODO, "Perforación",
        "Gradiente y presión hidrostática del lodo a la TVD, diferencial contra la presión "
        "de formación y condición de balance del pozo.",
        "ΔP = 0.052 × MW × TVD − Pform",
    ),
    (
        ARENA, "Reservorios",
        "POES volumétrico con espesor neto, porosidad, saturación de agua y Boi, más el "
        "volumen recuperable según el factor de recobro.",
        "POES = 7758·A·hₙ·φ·(1 − Swi) / Boi",
    ),
]

columnas = st.columns(3, gap="medium")
for col, (color, nombre, descripcion, ecuacion) in zip(columnas, pistas):
    with col:
        html(f"""
        <div class="pista" style="--acento:{color}">
          <h3>{nombre}</h3>
          <p>{descripcion}</p>
          <code>{ecuacion}</code>
        </div>
        """)

st.write("")
st.page_link("vistas/ejercicios.py", label="Abrir los ejercicios", icon=":material/calculate:")

html("""
<div class="pie">
  <p>Referencias: Vogel (1968), JPT 20(1), 83–92; Bourgoyne et al. (1986), Applied Drilling Engineering;
  Ahmed (2010), Reservoir Engineering Handbook.</p>
  <p>Aplicación académica. Los resultados no reemplazan un estudio de ingeniería.</p>
</div>
""")



