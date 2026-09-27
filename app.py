"""Home: presentación e identidad de la aplicación."""
import streamlit as st

from core.ui import ARENA, CRUDO, LODO, esquema_pozo, html

# Datos del participante 
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

"""
Lógica de cálculo de ingeniería (sin dependencias de Streamlit).

Cada ejercicio tiene:
  - una función `validar_*` que devuelve la lista de errores de entrada (vacía si todo está bien)
  - una función `calcular_*` que asume entradas válidas y devuelve un dataclass con los resultados
"""
from dataclasses import dataclass

import numpy as np

# Constantes de campo
FACTOR_HIDROSTATICO = 0.052   # psi/ft por ppg
BBL_POR_ACRE_FT = 7758.0      # barriles por acre-ft
STB_POR_MMSTB = 1_000_000.0


# ---------------------------------------------------------------------------
# Ejercicio 1. Producción: IPR compuesta (lineal + Vogel bajo el punto de burbuja)
# ---------------------------------------------------------------------------
@dataclass
class ResultadoIPR:
    qo: float                 # caudal a la Pwf ingresada [STB/d]
    qb: float                 # caudal a la presión de burbuja [STB/d]
    qo_max: float             # caudal máximo teórico (Pwf = 0) [STB/d]
    sobre_burbuja: bool       # True si Pwf >= Pb (régimen lineal)
    abatimiento: float        # Pr - Pwf [psi]
    curva_pwf: np.ndarray     # presiones de la curva IPR [psi]
    curva_q: np.ndarray       # caudales de la curva IPR [STB/d]


def validar_ipr(pr: float, pb: float, j: float, pwf: float) -> list[str]:
    errores = []
    if pr <= 0:
        errores.append("La presión del reservorio Pr debe ser mayor que 0 psi.")
    if pb <= 0:
        errores.append("La presión de burbuja Pb debe ser mayor que 0 psi.")
    if j <= 0:
        errores.append("El índice de productividad J debe ser mayor que 0 STB/d/psi.")
    if pwf < 0:
        errores.append("La presión de fondo fluyente Pwf no puede ser negativa.")
    if pr > 0 and pb >= pr:
        errores.append(
            f"Pb ({pb:,.0f} psi) debe ser menor que Pr ({pr:,.0f} psi): "
            "el ejercicio asume un reservorio subsaturado."
        )
    if pr > 0 and pwf > pr:
        errores.append(
            f"Pwf ({pwf:,.0f} psi) no puede superar a Pr ({pr:,.0f} psi): "
            "no habría flujo hacia el pozo."
        )
    return errores


def caudal_ipr(pr: float, pb: float, j: float, pwf):
    """Caudal de petróleo para uno o varios valores de Pwf (acepta escalares o arrays)."""
    pwf = np.asarray(pwf, dtype=float)
    qb = j * (pr - pb)
    razon = pwf / pb
    q_lineal = j * (pr - pwf)
    q_vogel = qb + (j * pb / 1.8) * (1 - 0.2 * razon - 0.8 * razon**2)
    return np.where(pwf >= pb, q_lineal, q_vogel)


def calcular_ipr(pr: float, pb: float, j: float, pwf: float, puntos: int = 120) -> ResultadoIPR:
    qb = j * (pr - pb)
    qo_max = qb + j * pb / 1.8
    # se incluye Pb para que el tramo lineal y el de Vogel se unan en la curva
    curva_pwf = np.unique(np.append(np.linspace(0.0, pr, puntos), pb))
    return ResultadoIPR(
        qo=float(caudal_ipr(pr, pb, j, pwf)),
        qb=qb,
        qo_max=qo_max,
        sobre_burbuja=pwf >= pb,
        abatimiento=pr - pwf,
        curva_pwf=curva_pwf,
        curva_q=caudal_ipr(pr, pb, j, curva_pwf),
    )


# ---------------------------------------------------------------------------
# Ejercicio 2. Perforación: presión hidrostática del lodo
# ---------------------------------------------------------------------------
@dataclass
class ResultadoHidrostatica:
    gradiente: float          # Gh [psi/ft]
    presion_hidro: float      # Ph a la TVD [psi]
    diferencial: float        # ΔP = Ph - Pform [psi]
    condicion: str            # "sobrebalance" | "balance" | "bajo balance"
    emw_formacion: float      # peso de lodo equivalente de la formación [ppg]
    inclinacion_media: float  # ángulo medio aproximado a partir de TVD/MD [grados]


def validar_hidrostatica(mw: float, md: float, tvd: float, pform: float, tolerancia: float) -> list[str]:
    errores = []
    if mw <= 0:
        errores.append("El peso del lodo MW debe ser mayor que 0 ppg.")
    if md <= 0:
        errores.append("La profundidad medida MD debe ser mayor que 0 ft.")
    if tvd <= 0:
        errores.append("La profundidad vertical verdadera TVD debe ser mayor que 0 ft.")
    if md > 0 and tvd > md:
        errores.append(
            f"TVD ({tvd:,.0f} ft) no puede ser mayor que MD ({md:,.0f} ft): "
            "la profundidad vertical nunca supera la longitud del hoyo."
        )
    if pform < 0:
        errores.append("La presión de formación no puede ser negativa.")
    if tolerancia < 0:
        errores.append("La tolerancia de balance no puede ser negativa.")
    return errores


def calcular_hidrostatica(mw: float, md: float, tvd: float, pform: float, tolerancia: float) -> ResultadoHidrostatica:
    gradiente = FACTOR_HIDROSTATICO * mw
    presion_hidro = gradiente * tvd          # se usa TVD, no MD
    diferencial = presion_hidro - pform

    if abs(diferencial) <= tolerancia:
        condicion = "balance"
    elif diferencial > 0:
        condicion = "sobrebalance"
    else:
        condicion = "bajo balance"

    return ResultadoHidrostatica(
        gradiente=gradiente,
        presion_hidro=presion_hidro,
        diferencial=diferencial,
        condicion=condicion,
        emw_formacion=pform / (FACTOR_HIDROSTATICO * tvd),
        inclinacion_media=float(np.degrees(np.arccos(tvd / md))),
    )


# ---------------------------------------------------------------------------
# Ejercicio 3. Reservorios: POES volumétrico
# ---------------------------------------------------------------------------
@dataclass
class ResultadoPOES:
    espesor_neto: float       # hn [ft]
    vol_bruto: float          # 7758·A·h [bbl]
    vol_neto: float           # 7758·A·hn [bbl]
    vol_poroso: float         # ·φ [bbl]
    vol_hidrocarburo: float   # ·(1 - Swi) [rb]
    poes: float               # [STB]
    recuperable: float        # [STB]

    @property
    def poes_mm(self) -> float:
        return self.poes / STB_POR_MMSTB

    @property
    def recuperable_mm(self) -> float:
        return self.recuperable / STB_POR_MMSTB


def validar_poes(area, h, ntg, phi, swi, boi, fr) -> list[str]:
    errores = []
    if area <= 0:
        errores.append("El área A debe ser mayor que 0 acres.")
    if h <= 0:
        errores.append("El espesor bruto h debe ser mayor que 0 ft.")
    if not 0 < ntg <= 1:
        errores.append("NTG debe estar entre 0 y 1 (0 % y 100 %), sin incluir el 0.")
    if not 0 < phi < 1:
        errores.append("La porosidad φ debe estar entre 0 y 1 (0 % y 100 %).")
    if not 0 <= swi < 1:
        errores.append("La saturación de agua Swi debe estar entre 0 y 1; con Swi = 1 no hay petróleo.")
    if boi <= 0:
        errores.append("El factor volumétrico Boi debe ser mayor que 0 rb/STB.")
    if not 0 <= fr <= 1:
        errores.append("El factor de recobro FR debe estar entre 0 y 1 (0 % y 100 %).")
    return errores


def calcular_poes(area, h, ntg, phi, swi, boi, fr) -> ResultadoPOES:
    espesor_neto = h * ntg
    vol_bruto = BBL_POR_ACRE_FT * area * h
    vol_neto = BBL_POR_ACRE_FT * area * espesor_neto
    vol_poroso = vol_neto * phi
    vol_hidrocarburo = vol_poroso * (1 - swi)
    poes = vol_hidrocarburo / boi
    return ResultadoPOES(
        espesor_neto=espesor_neto,
        vol_bruto=vol_bruto,
        vol_neto=vol_neto,
        vol_poroso=vol_poroso,
        vol_hidrocarburo=vol_hidrocarburo,
        poes=poes,
        recuperable=poes * fr,
    )



