"""
Tres Pistas: calculadora de ingeniería de petróleos
Bootcamp Data Analytics for Oil & Gas (SPE Ecuador Section), Módulo 1.

Aplicación en un solo archivo, organizada en secciones:
  1. Lógica de cálculo y validaciones (sin Streamlit)
  2. Estilos CSS
  3. Componentes visuales HTML / JavaScript y estilo de gráficos
  4. Página Home
  5. Página Ejercicios (tabs Producción, Perforación y Reservorios)
  6. Configuración y navegación (solo Home y Ejercicios)
"""
import json
from dataclasses import dataclass
from html import escape

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components


# ===========================================================================
# 1. LÓGICA DE CÁLCULO
# ===========================================================================
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


# ===========================================================================
# 2. ESTILOS CSS
# ===========================================================================
ESTILOS_CSS = """
/* =====================================================================
   Tres Pistas: hoja de estilos
   Concepto: papel de registro de pozo (well log). Fondo claro con
   cuadrícula tenue, tinta azul petróleo y un color por disciplina.
   ===================================================================== */
@import url('https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap');

:root {
  --tinta: #12324A;          /* azul petróleo: texto y estructura */
  --tinta-suave: #4A6275;
  --papel: #EEF2F4;          /* fondo tipo papel de registro */
  --blanco: #FFFFFF;
  --reticula: rgba(18, 50, 74, 0.06);
  --crudo: #C9861A;          /* producción */
  --lodo: #1F7A74;           /* perforación */
  --arena: #B89A5B;          /* reservorios */
  --ladrillo: #B5452F;       /* alertas / bajo balance */
  --borde: rgba(18, 50, 74, 0.14);
}

/* ---------- Fondo general: cuadrícula de registro ---------- */
[data-testid="stAppViewContainer"] {
  background-color: var(--papel);
  background-image:
    linear-gradient(90deg, var(--reticula) 1px, transparent 1px),
    linear-gradient(0deg, var(--reticula) 1px, transparent 1px);
  background-size: 96px 24px;
}
[data-testid="stHeader"] { background: transparent; }

html, body, [class*="st-"], .stMarkdown, button, input, textarea {
  font-family: 'Archivo', system-ui, sans-serif;
}
.block-container { padding-top: 2.2rem; max-width: 1240px; }

h1, h2, h3, h4 {
  font-family: 'Archivo', sans-serif !important;
  color: var(--tinta) !important;
  font-stretch: 112%;
  letter-spacing: -0.01em;
}

/* ---------- Barra lateral ---------- */
[data-testid="stSidebar"] {
  background: var(--tinta);
  border-right: 4px solid var(--crudo);
}
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] small { color: #DCE6EC !important; }
[data-testid="stSidebarNavLink"] { border-radius: 4px; }
[data-testid="stSidebarNavLink"]:hover { background: rgba(255, 255, 255, 0.08); }
[data-testid="stSidebarNavLink"][aria-current="page"] { background: rgba(201, 134, 26, 0.22); }

.marca { padding: 0.4rem 0.2rem 1.2rem; border-bottom: 1px solid rgba(255,255,255,0.14); margin-bottom: 0.8rem; }
.marca-titulo { font-size: 1.6rem; font-weight: 800; font-stretch: 125%; color: #FFFFFF !important; line-height: 1; }
.marca-sub { font-size: 0.82rem; opacity: 0.8; margin-top: 0.35rem; }
.pistas-mini { display: flex; gap: 4px; margin-top: 0.8rem; }
.pistas-mini i { flex: 1; height: 6px; border-radius: 1px; }

/* ---------- Encabezados de página ---------- */
.titular { margin: 0 0 1.4rem; }
.titular h1 {
  font-size: clamp(2.2rem, 4.6vw, 3.6rem);
  font-weight: 800; font-stretch: 125%;
  line-height: 0.98; margin: 0; padding: 0;
}
.titular p { color: var(--tinta-suave); font-size: 1.08rem; max-width: 62ch; margin: 0.8rem 0 0; }

/* ---------- Tarjetas ---------- */
.tarjeta {
  background: var(--blanco);
  border: 1px solid var(--borde);
  border-radius: 6px;
  padding: 1.1rem 1.25rem;
  margin-bottom: 1rem;
}
.tarjeta h3 { font-size: 1.15rem; margin: 0 0 0.4rem; padding: 0; }
.tarjeta p { color: var(--tinta-suave); margin: 0; line-height: 1.5; }

/* Tarjeta de pista (Home): banda de color superior como encabezado de pista */
.pista {
  background: var(--blanco);
  border: 1px solid var(--borde);
  border-top: 6px solid var(--acento);
  border-radius: 2px 2px 6px 6px;
  padding: 1.1rem 1.2rem 1.2rem;
  height: 100%;
  transition: transform .18s ease, box-shadow .18s ease;
}
.pista:hover { transform: translateY(-2px); box-shadow: 0 8px 22px rgba(18, 50, 74, 0.10); }
.pista h3 { font-size: 1.25rem; margin: 0 0 0.35rem; padding: 0; }
.pista p { color: var(--tinta-suave); margin: 0 0 0.7rem; line-height: 1.5; }
.pista code, .formula code { background: none; color: var(--tinta); font-size: 0.92em; padding: 0; }

/* Ficha del participante */
.ficha { display: grid; grid-template-columns: auto 1fr; gap: 0.35rem 1.2rem; margin-top: 1.4rem; }
.ficha dt { color: var(--tinta-suave); font-size: 0.9rem; }
.ficha dd { margin: 0; font-weight: 600; color: var(--tinta); }

/* ---------- Fórmulas ---------- */
.formula {
  background: #F7F9FA;
  border-left: 4px solid var(--acento, var(--tinta));
  border-radius: 0 6px 6px 0;
  padding: 0.8rem 1rem;
  margin-bottom: 1rem;
  font-size: 0.98rem;
  line-height: 1.75;
  color: var(--tinta);
}
.formula .cond { color: var(--tinta-suave); font-size: 0.86rem; display: block; margin-top: 0.3rem; }

/* ---------- Indicadores de condición ---------- */
.estado {
  display: flex; align-items: center; gap: 0.9rem;
  border-radius: 6px; padding: 0.85rem 1.1rem; margin-bottom: 0.9rem;
  background: var(--fondo); color: var(--texto);
  border: 1px solid var(--texto);
}
.estado .punto { width: 14px; height: 14px; border-radius: 50%; background: var(--texto); flex: none; }
.estado b { font-size: 1.05rem; }
.estado span { display: block; font-size: 0.9rem; opacity: 0.9; }

.errores {
  background: #FBEDEA; border: 1px solid var(--ladrillo); color: #7A2A1C;
  border-radius: 6px; padding: 0.9rem 1.1rem; margin-bottom: 1rem;
}
.errores b { display: block; margin-bottom: 0.3rem; }
.errores ul { margin: 0; padding-left: 1.1rem; }

/* Barra de balance (perforación) */
.balance { margin: 0.2rem 0 1rem; }
.balance-pista { position: relative; height: 12px; border-radius: 6px;
  background: linear-gradient(90deg, #F3D6CF 0%, #EAF1EF 50%, #D3E8E5 100%); border: 1px solid var(--borde); }
.balance-marca { position: absolute; top: -6px; width: 4px; height: 22px; border-radius: 2px; transform: translateX(-2px); }
.balance-leyenda { display: flex; justify-content: space-between; font-size: 0.82rem; color: var(--tinta-suave); margin-top: 0.5rem; }
.balance-leyenda i { display: inline-block; width: 10px; height: 10px; border-radius: 2px; margin-right: 5px; vertical-align: -1px; }

/* ---------- Tabs: como encabezados de pista ---------- */
.stTabs [data-baseweb="tab-list"] { gap: 6px; border-bottom: 2px solid var(--borde); }
.stTabs [data-baseweb="tab"] {
  background: rgba(255,255,255,0.7);
  border: 1px solid var(--borde); border-bottom: none;
  border-radius: 6px 6px 0 0;
  padding: 0.55rem 1.3rem;
  height: auto;
}
.stTabs [data-baseweb="tab"] p { font-size: 1.02rem; font-weight: 600; font-stretch: 110%; }
.stTabs [aria-selected="true"] { background: var(--blanco); }
.stTabs [data-baseweb="tab-highlight"] { background-color: var(--crudo); height: 3px; }
.stTabs [data-baseweb="tab-panel"] { padding-top: 1.4rem; }

/* ---------- Entradas ---------- */
[data-testid="stVerticalBlockBorderWrapper"] { background: var(--blanco); border-color: var(--borde) !important; }
[data-testid="stNumberInput"] input { font-variant-numeric: tabular-nums; font-weight: 600; color: var(--tinta); }
[data-testid="stWidgetLabel"] p { font-weight: 600; color: var(--tinta); }

/* Pie de página */
.pie { color: var(--tinta-suave); font-size: 0.85rem; border-top: 1px solid var(--borde); padding-top: 0.9rem; margin-top: 2rem; }
.pie p { margin: 0 0 0.25rem; }

/* Accesibilidad */
a:focus-visible, button:focus-visible { outline: 3px solid var(--crudo); outline-offset: 2px; }
@media (prefers-reduced-motion: reduce) {
  .pista { transition: none; }
  .pista:hover { transform: none; }
}
@media (max-width: 640px) {
  .block-container { padding-left: 1rem; padding-right: 1rem; }
  .ficha { grid-template-columns: 1fr; gap: 0.1rem; }
  .ficha dd { margin-bottom: 0.5rem; }
}
"""


# ===========================================================================
# 3. COMPONENTES VISUALES (HTML, JAVASCRIPT, GRÁFICOS)
# ===========================================================================
# Paleta (debe coincidir con assets/styles.css)
TINTA = "#12324A"
TINTA_SUAVE = "#4A6275"
CRUDO = "#C9861A"
LODO = "#1F7A74"
ARENA = "#B89A5B"
LADRILLO = "#B5452F"
RETICULA = "#DCE3E8"

FUENTE_WEB = "https://fonts.googleapis.com/css2?family=Archivo:wdth,wght@62..125,400..800&display=swap"


# ---------------------------------------------------------------------------
# Utilidades HTML
# ---------------------------------------------------------------------------
def html(bloque: str) -> None:
    """Renderiza HTML en la página. Se quitan sangrías y líneas vacías para que
    Markdown no convierta el HTML en bloques de código visibles."""
    limpio = "\n".join(linea.strip() for linea in bloque.splitlines() if linea.strip())
    st.markdown(limpio, unsafe_allow_html=True)


def cargar_css() -> None:
    st.markdown(f"<style>{ESTILOS_CSS}</style>", unsafe_allow_html=True)


def barra_lateral() -> None:
    with st.sidebar:
        html(f"""
        <div class="marca">
          <div class="marca-titulo">Tres Pistas</div>
          <div class="marca-sub">Calculadora de ingeniería de petróleos</div>
          <div class="pistas-mini">
            <i style="background:{CRUDO}"></i><i style="background:{LODO}"></i><i style="background:{ARENA}"></i>
          </div>
        </div>
        """)


def titular(titulo: str, bajada: str) -> None:
    html(f'<div class="titular"><h1>{titulo}</h1><p>{bajada}</p></div>')


def formula(contenido: str, acento: str) -> None:
    html(f'<div class="formula" style="--acento:{acento}">{contenido}</div>')


def estado(titulo: str, detalle: str, texto: str, fondo: str) -> None:
    """Indicador de condición de operación (banda de color con punto)."""
    html(f"""
    <div class="estado" role="status" style="--texto:{texto};--fondo:{fondo}">
      <div class="punto"></div>
      <div><b>{titulo}</b><span>{detalle}</span></div>
    </div>
    """)


def errores(lista: list[str]) -> None:
    items = "".join(f"<li>{escape(e)}</li>" for e in lista)
    html(f"""
    <div class="errores" role="alert">
      <b>Revisa los datos de entrada antes de calcular:</b>
      <ul>{items}</ul>
    </div>
    """)


# ---------------------------------------------------------------------------
# Componente JavaScript 1: contadores animados de resultados
# ---------------------------------------------------------------------------
_PLANTILLA_CONTADORES = """
<link href="%%FUENTE%%" rel="stylesheet">
<style>
  body { margin: 0; font-family: 'Archivo', system-ui, sans-serif; background: transparent; }
  .fila { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px; }
  .c { background: #fff; border: 1px solid rgba(18,50,74,.14); border-left: 5px solid var(--acento);
       border-radius: 6px; padding: 12px 14px 10px; cursor: pointer; transition: box-shadow .15s ease; }
  .c:hover { box-shadow: 0 6px 16px rgba(18,50,74,.10); }
  .c:focus-visible { outline: 3px solid #C9861A; outline-offset: 2px; }
  .l { display: block; color: #4A6275; font-size: 13px; line-height: 1.25; min-height: 32px; }
  .v { display: block; color: #12324A; font-size: 28px; font-weight: 800; font-stretch: 112%;
       font-variant-numeric: tabular-nums; line-height: 1.15; margin-top: 4px; }
  .u { color: #4A6275; font-size: 13px; }
  .aviso { font-size: 11px; color: #1F7A74; height: 14px; }
</style>
<div class="fila">%%TARJETAS%%</div>
<script>
  const reducido = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const formato = (x, d) => x.toLocaleString('en-US', {minimumFractionDigits: d, maximumFractionDigits: d});

  // Animación de conteo desde 0 hasta el valor calculado
  function animar(el) {
    const fin = parseFloat(el.dataset.valor), dec = parseInt(el.dataset.dec);
    if (reducido) { el.textContent = formato(fin, dec); return; }
    const t0 = performance.now(), dur = 900;
    function paso(t) {
      const p = Math.min((t - t0) / dur, 1), e = 1 - Math.pow(1 - p, 3);
      el.textContent = formato(fin * e, dec);
      if (p < 1) requestAnimationFrame(paso);
    }
    requestAnimationFrame(paso);
  }
  document.querySelectorAll('.v').forEach(animar);

  // Clic en una tarjeta: copia el valor al portapapeles y muestra confirmación
  document.querySelectorAll('.c').forEach(card => {
    const accion = () => {
      const v = card.querySelector('.v'), aviso = card.querySelector('.aviso');
      const texto = v.dataset.valor + ' ' + card.dataset.unidad;
      const listo = () => { aviso.textContent = 'Valor copiado'; setTimeout(() => aviso.textContent = '', 1400); };
      if (navigator.clipboard) { navigator.clipboard.writeText(texto).then(listo, () => { aviso.textContent = texto; }); }
      else { aviso.textContent = texto; }
    };
    card.addEventListener('click', accion);
    card.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); accion(); } });
  });
</script>
"""


def contadores(items: list[dict], altura: int = 118) -> None:
    """items: dicts con etiqueta, valor, unidad, dec (decimales) y color."""
    tarjetas = "".join(
        f'<div class="c" tabindex="0" role="button" style="--acento:{it["color"]}" '
        f'data-unidad="{escape(it["unidad"])}" title="Clic para copiar el valor">'
        f'<span class="l">{escape(it["etiqueta"])}</span>'
        f'<span class="v" data-valor="{it["valor"]:.{it["dec"]}f}" data-dec="{it["dec"]}">0</span>'
        f'<span class="u">{escape(it["unidad"])}</span><div class="aviso"></div></div>'
        for it in items
    )
    codigo = _PLANTILLA_CONTADORES.replace("%%FUENTE%%", FUENTE_WEB).replace("%%TARJETAS%%", tarjetas)
    components.html(codigo, height=altura)


# ---------------------------------------------------------------------------
# Componente JavaScript 2: esquema de pozo interactivo (Home)
# ---------------------------------------------------------------------------
_ZONAS = {
    "perforacion": {
        "titulo": "Perforación",
        "color": LODO,
        "texto": "La columna de lodo ejerce presión sobre las paredes del hoyo. Si supera a la "
                 "presión de formación el pozo está en sobrebalance y se controla el influjo.",
        "formula": "Pₕ = 0.052 × MW × TVD",
    },
    "reservorio": {
        "titulo": "Reservorios",
        "color": ARENA,
        "texto": "La arena productora guarda el petróleo en sus poros. El método volumétrico "
                 "estima cuánto había originalmente a condiciones de tanque.",
        "formula": "POES = 7758 · A · hₙ · φ · (1 − Swi) / Boi",
    },
    "produccion": {
        "titulo": "Producción",
        "color": CRUDO,
        "texto": "El fluido entra al pozo por los punzados. Cuanto menor es la Pwf, mayor el caudal; "
                 "bajo el punto de burbuja la relación deja de ser lineal (Vogel).",
        "formula": "qₒ = J (Pᵣ − Pwf)   si Pwf ≥ Pᵦ",
    },
}

_PLANTILLA_POZO = """
<link href="%%FUENTE%%" rel="stylesheet">
<style>
  body { margin: 0; font-family: 'Archivo', system-ui, sans-serif; color: #12324A; background: transparent; }
  .marco { display: grid; grid-template-columns: 230px 1fr; gap: 18px; align-items: stretch;
           background: #fff; border: 1px solid rgba(18,50,74,.14); border-radius: 6px; padding: 14px; }
  svg { width: 100%; height: auto; display: block; }
  .zona { cursor: pointer; transition: opacity .2s ease; }
  .zona:focus { outline: none; }
  .zona:focus-visible .borde { stroke: #12324A; stroke-width: 3; }
  svg.activo .zona { opacity: .28; }
  svg.activo .zona.sel { opacity: 1; }
  .panel { display: flex; flex-direction: column; justify-content: center; }
  .panel h4 { margin: 0 0 6px; font-size: 22px; font-weight: 800; font-stretch: 118%; }
  .panel p { margin: 0 0 12px; color: #4A6275; line-height: 1.5; font-size: 15px; max-width: 42ch; }
  .panel .f { font-size: 15px; background: #F7F9FA; border-left: 4px solid var(--c, #12324A);
              padding: 8px 10px; border-radius: 0 4px 4px 0; }
  .botones { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 16px; }
  .botones button { font: inherit; font-size: 13px; font-weight: 600; border-radius: 4px; cursor: pointer;
                    border: 1px solid rgba(18,50,74,.25); background: #fff; color: #12324A; padding: 6px 10px; }
  .botones button[aria-pressed="true"] { background: var(--c); border-color: var(--c); color: #fff; }
  .botones button:focus-visible { outline: 3px solid #C9861A; outline-offset: 2px; }
  @media (max-width: 560px) { .marco { grid-template-columns: 1fr; } svg { max-width: 240px; margin: 0 auto; } }
</style>
<div class="marco">
  <svg id="pozo" viewBox="0 0 230 400" role="img" aria-label="Esquema de pozo con tres zonas seleccionables">
    <!-- escala de profundidad -->
    <g font-size="9" fill="#4A6275">
      <line x1="30" y1="10" x2="30" y2="392" stroke="#9FB0BC"/>
      %%TICKS%%
    </g>
    <!-- zona de perforación: formaciones superiores + columna de lodo -->
    <g class="zona" data-zona="perforacion" tabindex="0" role="button" aria-label="Perforación">
      <rect class="borde" x="40" y="10" width="180" height="190" fill="#E7ECEF" stroke="none"/>
      <path d="M40 70 H220 M40 130 H220" stroke="#C9D3DA" stroke-dasharray="4 4"/>
      <rect x="118" y="10" width="24" height="190" fill="#1F7A74" opacity=".85"/>
      <rect x="112" y="10" width="4" height="150" fill="#12324A"/><rect x="144" y="10" width="4" height="150" fill="#12324A"/>
      <path d="M126 40 v18 M134 80 v18 M126 120 v18 M134 160 v18" stroke="#fff" stroke-width="2" marker-end="url(#flecha)"/>
    </g>
    <!-- zona de reservorio: arena con poros de petróleo -->
    <g class="zona" data-zona="reservorio" tabindex="0" role="button" aria-label="Reservorios">
      <rect class="borde" x="40" y="200" width="180" height="120" fill="#EFE3C4"/>
      %%POROS%%
      <rect x="118" y="200" width="24" height="120" fill="#F7F9FA"/>
    </g>
    <!-- zona de producción: punzados y flujo hacia el pozo -->
    <g class="zona" data-zona="produccion" tabindex="0" role="button" aria-label="Producción">
      <rect class="borde" x="40" y="320" width="180" height="72" fill="#F5E6CC"/>
      <rect x="118" y="320" width="24" height="72" fill="#C9861A"/>
      <rect x="112" y="200" width="4" height="192" fill="#12324A"/><rect x="144" y="200" width="4" height="192" fill="#12324A"/>
      <path d="M70 340 H108 M70 368 H108 M190 340 H152 M190 368 H152" stroke="#C9861A" stroke-width="3" marker-end="url(#flecha2)"/>
      <path d="M130 386 V330" stroke="#fff" stroke-width="2" marker-end="url(#flecha)"/>
    </g>
    <defs>
      <marker id="flecha" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#fff"/></marker>
      <marker id="flecha2" viewBox="0 0 10 10" refX="5" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10z" fill="#C9861A"/></marker>
    </defs>
  </svg>
  <div class="panel" id="panel" aria-live="polite">
    <h4 id="t">Un pozo, tres cálculos</h4>
    <p id="d">Toca una zona del esquema para ver qué se calcula en cada tramo del pozo.</p>
    <div class="f" id="f" style="display:none"></div>
    <div class="botones" id="botones"></div>
  </div>
</div>
<script>
  const ZONAS = %%ZONAS%%;
  const svg = document.getElementById('pozo');
  const botones = document.getElementById('botones');

  function seleccionar(clave) {
    const z = ZONAS[clave];
    svg.classList.add('activo');
    svg.querySelectorAll('.zona').forEach(g => g.classList.toggle('sel', g.dataset.zona === clave));
    document.getElementById('t').textContent = z.titulo;
    document.getElementById('t').style.color = z.color;
    document.getElementById('d').textContent = z.texto;
    const f = document.getElementById('f');
    f.style.display = 'block'; f.style.setProperty('--c', z.color); f.textContent = z.formula;
    botones.querySelectorAll('button').forEach(b => b.setAttribute('aria-pressed', b.dataset.zona === clave));
  }

  Object.entries(ZONAS).forEach(([clave, z]) => {
    const b = document.createElement('button');
    b.textContent = z.titulo; b.dataset.zona = clave; b.style.setProperty('--c', z.color);
    b.setAttribute('aria-pressed', 'false');
    b.addEventListener('click', () => seleccionar(clave));
    botones.appendChild(b);
  });
  svg.querySelectorAll('.zona').forEach(g => {
    g.addEventListener('click', () => seleccionar(g.dataset.zona));
    g.addEventListener('keydown', e => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); seleccionar(g.dataset.zona); } });
  });
</script>
"""


def esquema_pozo(altura: int = 440) -> None:
    ticks = "".join(
        f'<line x1="26" y1="{y}" x2="34" y2="{y}" stroke="#9FB0BC"/>'
        f'<text x="22" y="{y + 3}" text-anchor="end">{prof}</text>'
        for y, prof in zip(range(10, 400, 76), range(0, 12000, 2000))
    )
    # poros distribuidos de forma regular (evita aleatoriedad entre recargas)
    poros = "".join(
        f'<circle cx="{48 + (i * 23) % 170}" cy="{210 + (i * 37) % 104}" r="{2 + i % 3}" fill="#8A5A12" opacity=".55"/>'
        for i in range(46)
    )
    codigo = (
        _PLANTILLA_POZO.replace("%%FUENTE%%", FUENTE_WEB)
        .replace("%%TICKS%%", ticks)
        .replace("%%POROS%%", poros)
        .replace("%%ZONAS%%", json.dumps(_ZONAS, ensure_ascii=False))
    )
    components.html(codigo, height=altura)


# ---------------------------------------------------------------------------
# Estilo común de gráficos Plotly
# ---------------------------------------------------------------------------
def estilo_grafico(fig, titulo: str, x: str, y: str, altura: int = 420):
    fig.update_layout(
        title=dict(text=titulo, font=dict(size=16, color=TINTA), x=0, xanchor="left"),
        font=dict(family="Archivo, sans-serif", color=TINTA, size=13),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        height=altura,
        margin=dict(l=60, r=20, t=56, b=50),
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", font=dict(family="Archivo, sans-serif", color=TINTA)),
    )
    fig.update_xaxes(title_text=x, gridcolor=RETICULA, zeroline=False, linecolor=TINTA_SUAVE, showline=True)
    fig.update_yaxes(title_text=y, gridcolor=RETICULA, zeroline=False, linecolor=TINTA_SUAVE, showline=True)
    return fig


# ===========================================================================
# 4. PÁGINA HOME
# ===========================================================================
# Datos del participante (edita aquí tus nombres y apellidos completos)
PARTICIPANTE = "Santiago Arias"
TITULO_APP = "Tres Pistas"
PROGRAMA = "Bootcamp Data Analytics for Oil & Gas"


def pagina_home() -> None:
    """Home: presentación e identidad de la aplicación."""
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
    st.page_link(PAGINA_EJERCICIOS, label="Abrir los ejercicios", icon=":material/calculate:")

    html("""
    <div class="pie">
      <p>Referencias: Vogel (1968), JPT 20(1), 83–92; Bourgoyne et al. (1986), Applied Drilling Engineering;
      Ahmed (2010), Reservoir Engineering Handbook.</p>
      <p>Aplicación académica. Los resultados no reemplazan un estudio de ingeniería.</p>
    </div>
    """)


# ===========================================================================
# 5. PÁGINA EJERCICIOS
# ===========================================================================
# Colores de los indicadores de condición: (texto/borde, fondo)
COLOR_ESTADO = {
    "sobre": (LODO, "#E3F1EF"),
    "bajo": (CRUDO, "#FBF1DF"),
    "sobrebalance": (LODO, "#E3F1EF"),
    "balance": ("#8A5A12", "#FBF1DF"),
    "bajo balance": (LADRILLO, "#FBEDEA"),
}


# ---------------------------------------------------------------------------
# Tab 1. Producción: IPR compuesta
# ---------------------------------------------------------------------------
def tab_produccion() -> None:
    col_datos, col_resultados = st.columns([1, 2.1], gap="large")

    # Entradas
    with col_datos:
        with st.container(border=True):
            st.markdown("#### Datos del pozo")
            pr = st.number_input("Pᵣ: presión promedio del reservorio [psi]", min_value=0.0, value=3000.0, step=50.0, key="ipr_pr")
            pb = st.number_input("Pᵦ: presión de burbuja [psi]", min_value=0.0, value=2000.0, step=50.0, key="ipr_pb")
            j = st.number_input("J: índice de productividad [STB/d/psi]", min_value=0.0, value=1.20, step=0.05, format="%.2f", key="ipr_j")
            pwf = st.number_input("Pwf: presión de fondo fluyente [psi]", min_value=0.0, value=1500.0, step=50.0, key="ipr_pwf")
        formula(
            "Pwf ≥ Pᵦ:  qₒ = J (Pᵣ − Pwf)<br>"
            "Pwf &lt; Pᵦ:  qₒ = qᵦ + (J Pᵦ / 1.8) [1 − 0.2(Pwf/Pᵦ) − 0.8(Pwf/Pᵦ)²]<br>"
            "qᵦ = J (Pᵣ − Pᵦ)   qₒ,max = qᵦ + J Pᵦ / 1.8"
            '<span class="cond">La app elige la expresión según el valor de Pwf.</span>',
            CRUDO,
        )

    # Validación, cálculo y resultados
    with col_resultados:
        lista_errores = validar_ipr(pr, pb, j, pwf)
        if lista_errores:
            errores(lista_errores)
            return

        r = calcular_ipr(pr, pb, j, pwf)

        if r.sobre_burbuja:
            texto, fondo = COLOR_ESTADO["sobre"]
            estado("Operando por encima de Pᵦ: flujo monofásico",
                   f"Pwf = {pwf:,.0f} psi ≥ Pᵦ = {pb:,.0f} psi. Se aplica la IPR lineal.", texto, fondo)
        else:
            texto, fondo = COLOR_ESTADO["bajo"]
            estado("Operando por debajo de Pᵦ: gas libre en el reservorio",
                   f"Pwf = {pwf:,.0f} psi < Pᵦ = {pb:,.0f} psi. Se aplica el tramo de Vogel.", texto, fondo)

        contadores([
            dict(etiqueta="Caudal de petróleo qₒ", valor=r.qo, unidad="STB/d", dec=1, color=TINTA),
            dict(etiqueta="Caudal a la presión de burbuja qᵦ", valor=r.qb, unidad="STB/d", dec=1, color=LODO),
            dict(etiqueta="Caudal máximo teórico qₒ,max", valor=r.qo_max, unidad="STB/d", dec=1, color=CRUDO),
            dict(etiqueta="Abatimiento Pᵣ − Pwf", valor=r.abatimiento, unidad="psi", dec=0, color=TINTA_SUAVE),
        ])

        # Curva IPR: tramo lineal y tramo de Vogel con el punto del usuario
        lineal = r.curva_pwf >= pb
        vogel = r.curva_pwf <= pb
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=r.curva_q[vogel], y=r.curva_pwf[vogel], mode="lines", name="Tramo de Vogel (Pwf < Pᵦ)",
            line=dict(color=CRUDO, width=3),
            hovertemplate="q = %{x:,.1f} STB/d<br>Pwf = %{y:,.0f} psi<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=r.curva_q[lineal], y=r.curva_pwf[lineal], mode="lines", name="Tramo lineal (Pwf ≥ Pᵦ)",
            line=dict(color=LODO, width=3),
            hovertemplate="q = %{x:,.1f} STB/d<br>Pwf = %{y:,.0f} psi<extra></extra>",
        ))
        fig.add_hline(y=pb, line=dict(color=TINTA_SUAVE, dash="dot", width=1),
                      annotation_text=f"Pᵦ = {pb:,.0f} psi", annotation_position="top right")
        fig.add_trace(go.Scatter(
            x=[r.qb, r.qo_max], y=[pb, 0], mode="markers", name="qᵦ y qₒ,max",
            marker=dict(color="#FFFFFF", size=10, symbol="diamond", line=dict(color=TINTA, width=2)),
            hovertemplate="q = %{x:,.1f} STB/d<br>Pwf = %{y:,.0f} psi<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[r.qo], y=[pwf], mode="markers+text", name="Punto calculado",
            marker=dict(color=TINTA, size=15, line=dict(color="#FFFFFF", width=2)),
            text=[f"  qₒ = {r.qo:,.1f} STB/d"], textposition="middle right",
            hovertemplate="qₒ = %{x:,.1f} STB/d<br>Pwf = %{y:,.0f} psi<extra></extra>",
        ))
        estilo_grafico(fig, "Curva IPR compuesta", "Caudal de petróleo qₒ [STB/d]", "Presión de fondo fluyente Pwf [psi]")
        fig.update_xaxes(range=[0, r.qo_max * 1.12])
        fig.update_yaxes(range=[0, pr * 1.05])
        st.plotly_chart(fig, width="stretch", key="graf_ipr")

        with st.expander("Ver la tabla de la curva IPR"):
            tabla = pd.DataFrame({"Pwf [psi]": r.curva_pwf[::-1].round(1), "qₒ [STB/d]": r.curva_q[::-1].round(2)})
            st.dataframe(tabla, hide_index=True, height=260)
            st.download_button("Descargar curva IPR (CSV)", tabla.to_csv(index=False).encode("utf-8"),
                               file_name="curva_ipr.csv", mime="text/csv")


# ---------------------------------------------------------------------------
# Tab 2. Perforación: presión hidrostática del lodo
# ---------------------------------------------------------------------------
def barra_balance(mw: float, emw: float) -> None:
    """Escala en ppg con la posición del lodo y del peso equivalente de la formación."""
    minimo = min(mw, emw) * 0.85
    maximo = max(mw, emw) * 1.15
    rango = maximo - minimo if maximo > minimo else 1.0
    pos_lodo = (mw - minimo) / rango * 100
    pos_form = (emw - minimo) / rango * 100
    html(f"""
    <div class="balance" aria-label="Comparación en ppg del lodo y la formación">
      <div class="balance-pista">
        <div class="balance-marca" style="left:{pos_form:.1f}%;background:{LADRILLO}"></div>
        <div class="balance-marca" style="left:{pos_lodo:.1f}%;background:{LODO}"></div>
      </div>
      <div class="balance-leyenda">
        <span><i style="background:{LADRILLO}"></i>Formación: {emw:.2f} ppg equivalentes</span>
        <span><i style="background:{LODO}"></i>Lodo: {mw:.2f} ppg</span>
      </div>
    </div>
    """)


def tab_perforacion() -> None:
    col_datos, col_resultados = st.columns([1, 2.1], gap="large")

    with col_datos:
        with st.container(border=True):
            st.markdown("#### Datos del pozo")
            mw = st.number_input("MW: peso del lodo [ppg]", min_value=0.0, value=10.5, step=0.1, format="%.2f", key="hid_mw")
            md = st.number_input("MD: profundidad medida [ft]", min_value=0.0, value=11500.0, step=100.0, key="hid_md")
            tvd = st.number_input("TVD: profundidad vertical verdadera [ft]", min_value=0.0, value=10000.0, step=100.0, key="hid_tvd")
            pform = st.number_input("Pform: presión de formación [psi]", min_value=0.0, value=5200.0, step=50.0, key="hid_pform")
            tolerancia = st.number_input("Tolerancia para considerar balance, |ΔP| ≤ [psi]", min_value=0.0, value=50.0, step=10.0, key="hid_tol")
        formula(
            "Gₕ = 0.052 × MW  [psi/ft]<br>"
            "Pₕ = 0.052 × MW × TVD  [psi]<br>"
            "ΔP = Pₕ − Pform  [psi]"
            '<span class="cond">Se usa TVD porque la presión depende de la altura vertical de la columna; '
            "MD solo se usa para validar que TVD ≤ MD.</span>",
            LODO,
        )

    with col_resultados:
        lista_errores = validar_hidrostatica(mw, md, tvd, pform, tolerancia)
        if lista_errores:
            errores(lista_errores)
            return

        r = calcular_hidrostatica(mw, md, tvd, pform, tolerancia)

        texto, fondo = COLOR_ESTADO[r.condicion]
        mensajes = {
            "sobrebalance": ("Sobrebalance (ΔP > 0)",
                             "La columna de lodo supera a la presión de formación: el influjo está controlado."),
            "balance": ("Balance aproximado (ΔP ≈ 0)",
                        f"El diferencial está dentro de ±{tolerancia:,.0f} psi: margen mínimo frente a un influjo."),
            "bajo balance": ("Bajo balance (ΔP < 0)",
                             "La formación supera a la columna de lodo: existe riesgo de influjo (kick)."),
        }
        titulo_estado, detalle_estado = mensajes[r.condicion]
        estado(titulo_estado, detalle_estado, texto, fondo)

        contadores([
            dict(etiqueta="Gradiente hidrostático Gₕ", valor=r.gradiente, unidad="psi/ft", dec=3, color=LODO),
            dict(etiqueta="Presión hidrostática Pₕ a la TVD", valor=r.presion_hidro, unidad="psi", dec=0, color=TINTA),
            dict(etiqueta="Diferencial ΔP = Pₕ − Pform", valor=r.diferencial, unidad="psi", dec=0, color=texto),
            dict(etiqueta="Peso equivalente de la formación", valor=r.emw_formacion, unidad="ppg", dec=2, color=LADRILLO),
        ])

        barra_balance(mw, r.emw_formacion)

        # Presión versus TVD (profundidad creciente hacia abajo)
        profundidades = np.linspace(0.0, tvd, 60)
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=FACTOR_HIDROSTATICO * mw * profundidades, y=profundidades, mode="lines",
            name=f"Presión hidrostática ({mw:.2f} ppg)", line=dict(color=LODO, width=3),
            hovertemplate="Pₕ = %{x:,.0f} psi<br>TVD = %{y:,.0f} ft<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[0, pform], y=[0, tvd], mode="lines", name="Referencia de formación (gradiente lineal)",
            line=dict(color=LADRILLO, width=2, dash="dash"),
            hovertemplate="P = %{x:,.0f} psi<br>TVD = %{y:,.0f} ft<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[pform], y=[tvd], mode="markers", name="Pform",
            marker=dict(color=LADRILLO, size=12, symbol="square", line=dict(color="#FFFFFF", width=2)),
            hovertemplate="Pform = %{x:,.0f} psi<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=[r.presion_hidro], y=[tvd], mode="markers", name="Pₕ a la TVD ingresada",
            marker=dict(color=TINTA, size=15, line=dict(color="#FFFFFF", width=2)),
            hovertemplate="Pₕ = %{x:,.0f} psi<br>TVD = %{y:,.0f} ft<extra></extra>",
        ))
        fig.add_annotation(
            x=max(r.presion_hidro, pform), y=tvd, xanchor="left", yanchor="bottom", showarrow=False,
            text=f"  ΔP = {r.diferencial:+,.0f} psi", font=dict(color=texto, size=14),
        )
        estilo_grafico(fig, "Presión versus profundidad vertical", "Presión [psi]", "TVD [ft]")
        fig.update_yaxes(autorange="reversed")
        fig.update_xaxes(range=[0, max(r.presion_hidro, pform) * 1.25])
        st.plotly_chart(fig, width="stretch", key="graf_hidro")

        st.caption(
            f"MD = {md:,.0f} ft y TVD = {tvd:,.0f} ft: el hoyo tiene {md - tvd:,.0f} ft más de longitud que de "
            f"profundidad vertical (inclinación media aproximada de {r.inclinacion_media:.1f}°). "
            f"Si se usara MD, Pₕ quedaría sobrestimada en {FACTOR_HIDROSTATICO * mw * (md - tvd):,.0f} psi."
        )


# ---------------------------------------------------------------------------
# Tab 3. Reservorios: POES volumétrico
# ---------------------------------------------------------------------------
def entrada_fraccion(etiqueta: str, valor: float, clave: str, en_porcentaje: bool) -> float:
    """Número entre 0 y 1; si el usuario trabaja en %, se convierte a fracción."""
    escala = 100.0 if en_porcentaje else 1.0
    unidad = "[%]" if en_porcentaje else "[fracción]"
    dato = st.number_input(
        f"{etiqueta} {unidad}", min_value=0.0, max_value=escala, value=valor * escala,
        step=0.01 * escala, format="%.1f" if en_porcentaje else "%.3f",
        key=f"{clave}_{'pct' if en_porcentaje else 'frac'}",
    )
    return dato / escala


def tab_reservorios() -> None:
    col_datos, col_resultados = st.columns([1, 2.1], gap="large")

    with col_datos:
        with st.container(border=True):
            st.markdown("#### Datos del reservorio")
            modo = st.radio("Porosidad, saturación, NTG y FR en", ["Fracción", "Porcentaje"],
                            horizontal=True, key="poes_modo")
            en_pct = modo == "Porcentaje"
            area = st.number_input("A: área del reservorio [acres]", min_value=0.0, value=640.0, step=10.0, key="poes_a")
            h = st.number_input("h: espesor bruto [ft]", min_value=0.0, value=60.0, step=1.0, key="poes_h")
            ntg = entrada_fraccion("NTG: relación net-to-gross", 0.80, "poes_ntg", en_pct)
            phi = entrada_fraccion("φ: porosidad efectiva", 0.20, "poes_phi", en_pct)
            swi = entrada_fraccion("Swi: saturación inicial de agua", 0.25, "poes_swi", en_pct)
            boi = st.number_input("Boi: factor volumétrico inicial [rb/STB]", min_value=0.0, value=1.25, step=0.01, format="%.3f", key="poes_boi")
            fr = entrada_fraccion("FR: factor de recobro", 0.30, "poes_fr", en_pct)
        formula(
            "hₙ = h × NTG<br>"
            "POES = 7758 × A × hₙ × φ × (1 − Swi) / Boi<br>"
            "Recuperable = POES × FR"
            '<span class="cond">7758 convierte acre-ft a barriles. El POES no es una reserva: '
            "solo la fracción FR se considera recuperable.</span>",
            ARENA,
        )

    with col_resultados:
        lista_errores = validar_poes(area, h, ntg, phi, swi, boi, fr)
        if lista_errores:
            errores(lista_errores)
            return

        r = calcular_poes(area, h, ntg, phi, swi, boi, fr)

        if boi < 1:
            texto, fondo = COLOR_ESTADO["balance"]
            estado("Boi menor que 1 rb/STB",
                   "Es poco común: el petróleo suele expandirse en el yacimiento por el gas disuelto. Verifica el dato PVT.",
                   texto, fondo)

        contadores([
            dict(etiqueta="Espesor neto hₙ", valor=r.espesor_neto, unidad="ft", dec=1, color=ARENA),
            dict(etiqueta="POES", valor=r.poes, unidad="STB", dec=0, color=TINTA),
            dict(etiqueta="POES", valor=r.poes_mm, unidad="MMSTB", dec=2, color=TINTA),
        ])
        contadores([
            dict(etiqueta=f"Recuperable estimado (FR = {fr:.0%})", valor=r.recuperable, unidad="STB", dec=0, color=CRUDO),
            dict(etiqueta="Recuperable estimado", valor=r.recuperable_mm, unidad="MMSTB", dec=2, color=CRUDO),
            dict(etiqueta="Permanece en el reservorio", valor=r.poes_mm - r.recuperable_mm, unidad="MMSTB", dec=2, color=TINTA_SUAVE),
        ])

        col_embudo, col_barras = st.columns([1.35, 1], gap="medium")

        # Cascada volumétrica: cómo cada parámetro reduce el volumen de roca hasta el POES
        with col_embudo:
            etapas = [
                ("Roca total (A·h)", r.vol_bruto),
                ("Roca neta (× NTG)", r.vol_neto),
                ("Volumen poroso (× φ)", r.vol_poroso),
                ("Petróleo en yacimiento (× (1 − Swi))", r.vol_hidrocarburo),
                ("POES a tanque (÷ Boi)", r.poes),
            ]
            fig_embudo = go.Figure(go.Funnel(
                y=[e[0] for e in etapas],
                x=[e[1] / 1e6 for e in etapas],
                texttemplate="%{x:,.1f} MM<br>%{percentInitial:.1%}",
                marker=dict(color=["#D9DFE4", "#C7D0D7", "#E3D3AE", ARENA, TINTA]),
                connector=dict(line=dict(color="#C7D0D7", width=1)),
                hovertemplate="%{y}<br>%{x:,.2f} MM bbl<extra></extra>",
            ))
            estilo_grafico(fig_embudo, "Del volumen de roca al POES", "", "", altura=380)
            fig_embudo.update_yaxes(title_text="", showline=False)
            fig_embudo.update_layout(margin=dict(l=10, r=10, t=56, b=20))
            st.plotly_chart(fig_embudo, width="stretch", key="graf_embudo")

        # POES versus recuperable
        with col_barras:
            categorias = ["POES", "Recuperable", "Remanente"]
            valores = [r.poes_mm, r.recuperable_mm, r.poes_mm - r.recuperable_mm]
            fig_barras = go.Figure(go.Bar(
                x=categorias, y=valores, marker_color=[TINTA, CRUDO, "#C7D0D7"],
                text=[f"{v:,.2f}" for v in valores], textposition="outside",
                hovertemplate="%{x}: %{y:,.2f} MMSTB<extra></extra>",
            ))
            estilo_grafico(fig_barras, "POES frente a recuperable", "", "MMSTB", altura=380)
            fig_barras.update_yaxes(range=[0, max(valores) * 1.2 if max(valores) > 0 else 1])
            st.plotly_chart(fig_barras, width="stretch", key="graf_barras")


def pagina_ejercicios() -> None:
    """Ejercicios: tres áreas técnicas organizadas en tabs."""
    titular(
        "Ejercicios",
        "Modifica los datos de cada pista y observa cómo cambian los resultados y los gráficos. "
        "Los valores inconsistentes se señalan antes de calcular.",
    )

    pestana_prod, pestana_perf, pestana_res = st.tabs(["Producción", "Perforación", "Reservorios"])
    with pestana_prod:
        tab_produccion()
    with pestana_perf:
        tab_perforacion()
    with pestana_res:
        tab_reservorios()


# ===========================================================================
# 6. CONFIGURACIÓN Y NAVEGACIÓN
# ===========================================================================
st.set_page_config(
    page_title="Tres Pistas | Oil & Gas",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded",
)

cargar_css()
barra_lateral()

# Navegación principal: únicamente Home y Ejercicios
PAGINA_HOME = st.Page(pagina_home, title="Home", icon=":material/home:", url_path="home", default=True)
PAGINA_EJERCICIOS = st.Page(pagina_ejercicios, title="Ejercicios", icon=":material/calculate:", url_path="ejercicios")
st.navigation([PAGINA_HOME, PAGINA_EJERCICIOS], position="sidebar").run()
