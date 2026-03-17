from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_distribuciones_por_diagnostico(df: pd.DataFrame) -> None:
    """
    Dibuja histogramas de Age y MMSE para cada diagnóstico del dataset.
    - Age: bins "normales"
    - MMSE: bins por entero (0-30) para ver bien la discreción y el ceiling effect
    """
    diagnosticos = sorted(df["Diagnosis"].dropna().unique())

    for diag in diagnosticos:
        g = df[df["Diagnosis"] == diag]

        plt.figure()
        plt.hist(g["Age"].dropna(), bins=15)
        plt.title(f"Distribución de Age - {diag} (n={len(g)})")
        plt.xlabel("Age")
        plt.ylabel("Frecuencia")
        plt.savefig(Path.cwd() / f"hist_age_{diag}.png", dpi=200)

        plt.figure()
        bins = np.arange(-0.5, 30.5 + 1, 1)
        plt.hist(g["MMSE"].dropna(), bins=bins)
        plt.title(f"Distribución de MMSE - {diag} (n={len(g)})")
        plt.xlabel("MMSE")
        plt.ylabel("Frecuencia")
        plt.savefig(Path.cwd() / f"hist_mmse_{diag}.png", dpi=200)


def plot_distribuciones_objetivo(df: pd.DataFrame, diagnosis_objetivo: str) -> None:
    """
    Dibuja histogramas solo para el diagnóstico objetivo (más rápido si no quieres todo).
    """
    g = df[df["Diagnosis"] == diagnosis_objetivo]
    if g.empty:
        print(f"[WARN] No hay datos para Diagnosis='{diagnosis_objetivo}'. No se pueden plotear distribuciones.")
        return

    plt.figure()
    plt.hist(g["Age"].dropna(), bins=15)
    plt.title(f"Distribución de Age - {diagnosis_objetivo} (n={len(g)})")
    plt.xlabel("Age")
    plt.ylabel("Frecuencia")
    plt.savefig(Path.cwd() / f"dist_age_{diagnosis_objetivo}.png", dpi=200)

    plt.figure()
    bins = np.arange(-0.5, 30.5 + 1, 1)
    plt.hist(g["MMSE"].dropna(), bins=bins)
    plt.title(f"Distribución de MMSE - {diagnosis_objetivo} (n={len(g)})")
    plt.xlabel("MMSE")
    plt.ylabel("Frecuencia")
    plt.savefig(Path.cwd() / f"dist_mmse_{diagnosis_objetivo}.png", dpi=200)


def plot_relacion_age_mmse(df: pd.DataFrame, out_dir: Path | None = None) -> None:
    """
    Plots para ver relación Age vs MMSE:
    - Global: scatter + tendencia lineal
    - Por diagnóstico: scatter (alpha) + tendencia lineal
    Guarda PNGs si out_dir se especifica (si no, usa cwd).
    """
    if out_dir is None:
        out_dir = Path.cwd()
    out_dir.mkdir(parents=True, exist_ok=True)

    dfp = df.copy()
    dfp["Age"] = pd.to_numeric(dfp["Age"], errors="coerce")
    dfp["MMSE"] = pd.to_numeric(dfp["MMSE"], errors="coerce")
    dfp = dfp.dropna(subset=["Age", "MMSE", "Diagnosis"])

    plt.figure()
    x = dfp["Age"].to_numpy()
    y = dfp["MMSE"].to_numpy()

    plt.scatter(x, y, alpha=0.35, s=18)
    if len(x) >= 2:
        m, b = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 200)
        plt.plot(xs, m * xs + b, linewidth=2)

    plt.title(f"Relación Age vs MMSE - GLOBAL (n={len(dfp)})")
    plt.xlabel("Age")
    plt.ylabel("MMSE")
    plt.ylim(-0.5, 30.5)
    plt.savefig(out_dir / "rel_age_mmse_GLOBAL.png", dpi=200, bbox_inches="tight")

    diagnosticos = sorted(dfp["Diagnosis"].unique())
    for diag in diagnosticos:
        g = dfp[dfp["Diagnosis"] == diag]
        if g.empty:
            continue

        plt.figure()
        x = g["Age"].to_numpy()
        y = g["MMSE"].to_numpy()

        plt.scatter(x, y, alpha=0.4, s=22)

        if len(x) >= 2:
            m, b = np.polyfit(x, y, 1)
            xs = np.linspace(x.min(), x.max(), 200)
            plt.plot(xs, m * xs + b, linewidth=2)

        plt.title(f"Relación Age vs MMSE - {diag} (n={len(g)})")
        plt.xlabel("Age")
        plt.ylabel("MMSE")
        plt.ylim(-0.5, 30.5)
        plt.savefig(out_dir / f"rel_age_mmse_{diag}.png", dpi=200, bbox_inches="tight")
