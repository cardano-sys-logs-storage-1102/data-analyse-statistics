"""
analise.py — Pipeline de análise adaptativo guiado por metadados DataLab.

Uso:
    python analise.py <data.csv> <metadata.json> [--out-dir ./output]

O script lê o JSON de metadados para inferir automaticamente:
  • num_cols   — colunas numéricas
  • cat_cols   — colunas categóricas
  • bool_cols  — colunas booleanas
  • target_col — variável alvo (maior coeficiente de variação)
  • group_cols — categóricas com 2-8 categorias (usadas em comparações)

Cada uma das 10 seções possui uma guarda de pré-requisito e só executa
se o dataset satisfizer as condições mínimas.

Saídas: PNGs + export_stats.json em --out-dir.
"""

import sys
import os
import json
import argparse
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from scipy.stats import shapiro, kruskal
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Pipeline de análise adaptativo DataLab')
    p.add_argument('csv',  help='Arquivo CSV de dados')
    p.add_argument('meta', help='Arquivo JSON de metadados (DataLab)')
    p.add_argument('--out-dir', default='/home/claude', help='Diretório de saída')
    return p.parse_args()

# ── Paleta ────────────────────────────────────────────────────────────────────
DARK   = '#0d0f14'
PANEL  = '#13161c'
CARD   = '#181c24'
ACCENT = '#f5a623'
GREEN  = '#34d399'
BLUE   = '#60a5fa'
RED    = '#f87171'
PURPLE = '#a78bfa'
TEAL   = '#2dd4bf'
PINK   = '#ec4899'
TEXT   = '#e8eaf0'
MUTED  = '#8892a4'

# Paleta base para categorias dinâmicas
BASE_PALETTE = [ACCENT, BLUE, GREEN, PURPLE, TEAL, RED, PINK,
                '#fb923c', '#818cf8', '#4ade80', '#f472b6', '#38bdf8']

def build_cat_palette(categories: list) -> dict:
    """Atribui cores da paleta base para uma lista de categorias."""
    return {cat: BASE_PALETTE[i % len(BASE_PALETTE)] for i, cat in enumerate(sorted(categories))}

# ── Helpers de plotagem ───────────────────────────────────────────────────────
def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=MUTED, labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor(CARD)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)
    if title:  ax.set_title(title, color=TEXT, fontsize=10, pad=8, fontweight='bold')
    if xlabel: ax.set_xlabel(xlabel, fontsize=8)
    if ylabel: ax.set_ylabel(ylabel, fontsize=8)
    ax.grid(True, color='#1d222c', linewidth=0.5, alpha=0.7)

def new_fig(title, figsize):
    fig = plt.figure(figsize=figsize, facecolor=DARK)
    fig.suptitle(title, color=ACCENT, fontsize=13, fontweight='bold', y=0.98)
    return fig

def add_caption(fig, text, y=0.01):
    fig.text(0.5, y, text, ha='center', va='bottom',
             color=MUTED, fontsize=7.5, style='italic',
             wrap=True, transform=fig.transFigure)

def save(fig, name, out_dir, caption=''):
    if caption:
        plt.subplots_adjust(bottom=0.10)
        add_caption(fig, caption)
    path = os.path.join(out_dir, f'{name}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight', facecolor=DARK, edgecolor='none')
    plt.close(fig)
    print(f'  saved {name}.png')
    return path

def style_table(tbl, header_color=ACCENT, row_color=PANEL):
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#1d222c')
        if r == 0:
            cell.set_facecolor(header_color)
            cell.set_text_props(color='#000', fontweight='bold')
        elif c == -1:
            cell.set_facecolor(CARD)
            cell.set_text_props(color=ACCENT)
        else:
            cell.set_facecolor(row_color)
            cell.set_text_props(color=TEXT)

# ── Leitura de metadados ──────────────────────────────────────────────────────
def load_metadata(meta_path: str) -> dict:
    with open(meta_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def infer_columns(meta: dict, df: pd.DataFrame) -> dict:
    """
    Inferência de papéis de coluna a partir do JSON de metadados DataLab.
    Retorna: num_cols, cat_cols, bool_cols, target_col, group_cols.
    """
    num_cols, cat_cols, bool_cols = [], [], []

    for col_info in meta.get('columns', []):
        name = col_info['name']
        ctype = col_info.get('type', '').lower()
        if name not in df.columns:
            continue
        if ctype in ('numeric', 'float', 'integer', 'int'):
            num_cols.append(name)
        elif ctype in ('category', 'string', 'object', 'text'):
            cat_cols.append(name)
        elif ctype in ('boolean', 'bool'):
            bool_cols.append(name)
        else:
            # Fallback: inferir pelo dtype pandas
            dt = df[name].dtype
            if pd.api.types.is_numeric_dtype(dt):
                num_cols.append(name)
            elif pd.api.types.is_bool_dtype(dt):
                bool_cols.append(name)
            else:
                cat_cols.append(name)

    # Colunas booleanas mascaradas como string (ex: 'Sim'/'Não')
    # Só reclassifica se tiver exatamente 2 valores únicos e ambos forem booleanos
    BOOL_VALS = {'sim', 'não', 'nao', 'true', 'false', '1', '0', 'yes', 'no'}
    for col in list(cat_cols):
        unique_vals = set(df[col].dropna().str.lower().unique()) if df[col].dtype == object else set()
        if len(unique_vals) == 2 and unique_vals <= BOOL_VALS:
            cat_cols.remove(col)
            bool_cols.append(col)

    # target_col: numérica com maior CV%
    target_col = None
    if num_cols:
        cvs = {c: df[c].std() / df[c].mean() * 100 for c in num_cols if df[c].mean() != 0}
        if cvs:
            target_col = max(cvs, key=cvs.get)

    # group_cols: categóricas com 2-8 categorias únicas
    group_cols = [c for c in cat_cols if 2 <= df[c].nunique() <= 8]

    print(f'\n  num_cols   ({len(num_cols)}): {num_cols}')
    print(f'  cat_cols   ({len(cat_cols)}): {cat_cols}')
    print(f'  bool_cols  ({len(bool_cols)}): {bool_cols}')
    print(f'  target_col : {target_col}')
    print(f'  group_cols ({len(group_cols)}): {group_cols}')

    return dict(num_cols=num_cols, cat_cols=cat_cols, bool_cols=bool_cols,
                target_col=target_col, group_cols=group_cols)

# ── Normalização booleana ─────────────────────────────────────────────────────
def normalize_bool_col(series: pd.Series) -> pd.Series:
    if series.dtype == object:
        return series.map({
            'Sim': True, 'sim': True, 'SIM': True,
            'Não': False, 'não': False, 'NAO': False, 'nao': False,
            'True': True, 'False': False, '1': True, '0': False,
            'yes': True, 'no': False, 'Yes': True, 'No': False,
        }).fillna(series)
    return series

# ══════════════════════════════════════════════════════════════════════════════
# SEÇÕES DE ANÁLISE
# ══════════════════════════════════════════════════════════════════════════════

def s1_descritiva(df, cols, out_dir, stats_out):
    """Seção 1 — Estatística Descritiva & Distribuição Categórica."""
    guard = len(cols['num_cols']) >= 1
    if not guard:
        print('  [S1] SKIP: nenhuma coluna numérica encontrada.')
        return

    num_cols = cols['num_cols']
    cat_cols = cols['cat_cols']

    desc = df[num_cols].describe(percentiles=[.05, .25, .5, .75, .95]).T
    desc['cv%']     = (desc['std'] / desc['mean'] * 100).round(1)
    desc['skew']    = df[num_cols].skew().values
    desc['kurt']    = df[num_cols].kurtosis().values
    desc['missing'] = df[num_cols].isnull().sum().values

    cols_show  = ['count','mean','std','min','5%','25%','50%','75%','95%','max','cv%','skew','kurt','missing']
    col_labels = ['n','Média','DP','Mín','P5','Q1','Mediana','Q3','P95','Máx','CV%','Assim.','Curtose','Nulos']

    fig = new_fig('Seção 1 — Estatística Descritiva', (14, max(4, len(num_cols) * 0.7 + 2)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL); ax.axis('off')
    tbl = ax.table(cellText=desc[cols_show].round(2).values.tolist(),
                   rowLabels=[c.replace('_', ' ') for c in num_cols],
                   colLabels=col_labels, cellLoc='center', rowLoc='right', loc='center')
    style_table(tbl)
    save(fig, 's1_descritiva', out_dir,
         'CV% mede dispersão relativa (>30% = alta). Assimetria positiva indica cauda direita; '
         'Curtose >0 indica caudas mais pesadas que a normal. Nulos = valores ausentes.')

    # Estatísticas para export
    stats_out['s1_descritiva'] = {
        col: {
            'mean': round(float(df[col].mean()), 4),
            'std':  round(float(df[col].std()), 4),
            'cv_pct': round(float(df[col].std() / df[col].mean() * 100), 2) if df[col].mean() else None,
            'missing': int(df[col].isnull().sum()),
        }
        for col in num_cols
    }

    # Distribuição categórica (se houver)
    if cat_cols:
        n_cats = len(cat_cols)
        fig2, axes = plt.subplots(1, n_cats, figsize=(3.5 * n_cats, 4), facecolor=DARK)
        if n_cats == 1:
            axes = [axes]
        fig2.suptitle('Seção 1 — Distribuição Categórica', color=ACCENT, fontsize=12, fontweight='bold')
        for ax, col in zip(axes, cat_cols):
            vc = df[col].value_counts()
            bars = ax.barh(vc.index, vc.values, color=ACCENT, alpha=0.85, edgecolor=DARK, linewidth=0.5)
            for bar, v in zip(bars, vc.values):
                ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                        f'{v} ({v/len(df)*100:.0f}%)', va='center', color=TEXT, fontsize=7)
            style_ax(ax, col.replace('_', ' ').title())
            ax.set_facecolor(PANEL)
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
        save(fig2, 's1_categoricas', out_dir,
             'Frequência absoluta e relativa por categoria. Desbalanceamento pode influenciar análises comparativas.')

        stats_out['s1_categoricas'] = {
            col: df[col].value_counts().to_dict() for col in cat_cols
        }


def s2_distribuicao_target(df, cols, out_dir, stats_out):
    """Seção 2 — Histograma + KDE + Boxplot da variável alvo."""
    target = cols['target_col']
    guard = target is not None and df[target].notna().sum() >= 10
    if not guard:
        print('  [S2] SKIP: target_col ausente ou insuficiente.')
        return

    v = df[target].dropna()
    log_v = np.log(v[v > 0]) if (v > 0).all() else None
    mu = sigma = None
    if log_v is not None and len(log_v) > 0:
        mu, sigma = log_v.mean(), log_v.std()

    group_col = cols['group_cols'][0] if cols['group_cols'] else None
    palette = build_cat_palette(df[group_col].unique()) if group_col else {}

    fig = new_fig(f'Seção 2 — Distribuição: {target}', (14, 5))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.35)

    # Histograma + KDE
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PANEL)
    x_range = np.linspace(v.min(), v.max(), 300)
    ax1.hist(v, bins=25, density=True, color=ACCENT, alpha=0.55, edgecolor=DARK, linewidth=0.4)
    if mu is not None:
        pdf_ln = stats.lognorm.pdf(x_range, s=sigma, scale=np.exp(mu))
        ax1.plot(x_range, pdf_ln, color=TEAL, linewidth=2, label='Log-Normal ajustada')
    kde = stats.gaussian_kde(v)
    ax1.plot(x_range, kde(x_range), color=RED, linewidth=1.5, linestyle='--', label='KDE empírica')
    ax1.axvline(v.mean(),   color=BLUE,  linestyle=':', linewidth=1.2, label=f'Média {v.mean():.2f}')
    ax1.axvline(v.median(), color=GREEN, linestyle=':', linewidth=1.2, label=f'Mediana {v.median():.2f}')
    ax1.legend(fontsize=6.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax1, 'Histograma + KDE', target.replace('_', ' '), 'Densidade')

    # Boxplot com IC 95% bootstrap
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(PANEL)
    ax2.boxplot(v, notch=True, patch_artist=True, bootstrap=5000,
                medianprops=dict(color=ACCENT, linewidth=2),
                boxprops=dict(facecolor=BLUE, alpha=0.5),
                whiskerprops=dict(color=MUTED),
                capprops=dict(color=MUTED),
                flierprops=dict(marker='o', color=RED, alpha=0.5, markersize=4))
    boots = [np.random.choice(v, size=len(v), replace=True).mean() for _ in range(3000)]
    ci_lo, ci_hi = np.percentile(boots, 2.5), np.percentile(boots, 97.5)
    ax2.axhline(ci_lo,  color=GREEN, linestyle='--', linewidth=1, alpha=0.7, label=f'IC95% [{ci_lo:.2f}')
    ax2.axhline(ci_hi,  color=GREEN, linestyle='--', linewidth=1, alpha=0.7, label=f'– {ci_hi:.2f}]')
    ax2.legend(fontsize=6.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax2, 'Boxplot + IC 95% bootstrap', '', target.replace('_', ' '))

    # Boxplot por grupo (se houver group_col)
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(PANEL)
    if group_col:
        groups = sorted(df[group_col].unique())
        data_g = [df.loc[df[group_col] == g, target].dropna().values for g in groups]
        bp = ax3.boxplot(data_g, notch=True, patch_artist=True, bootstrap=5000,
                         labels=groups,
                         medianprops=dict(linewidth=2),
                         whiskerprops=dict(color=MUTED),
                         capprops=dict(color=MUTED),
                         flierprops=dict(marker='o', alpha=0.5, markersize=3))
        for patch, g in zip(bp['boxes'], groups):
            patch.set_facecolor(palette.get(g, ACCENT)); patch.set_alpha(0.6)
        for median, g in zip(bp['medians'], groups):
            median.set_color(palette.get(g, ACCENT))
        style_ax(ax3, f'Boxplot por {group_col.replace("_"," ")}',
                 group_col.replace('_', ' '), target.replace('_', ' '))
    else:
        ax3.axis('off')

    save(fig, 's2_distribuicao', out_dir,
         'Histograma: forma da distribuição. Notch do boxplot = IC95% da mediana. '
         'Pontos fora dos whiskers são outliers (±1,5×IQR).')

    stats_out['s2_distribuicao'] = {
        'target': target,
        'mean': round(float(v.mean()), 4),
        'median': round(float(v.median()), 4),
        'std': round(float(v.std()), 4),
        'min': round(float(v.min()), 4),
        'max': round(float(v.max()), 4),
        'ci95_mean_lo': round(float(ci_lo), 4),
        'ci95_mean_hi': round(float(ci_hi), 4),
        'lognormal_mu': round(float(mu), 4) if mu else None,
        'lognormal_sigma': round(float(sigma), 4) if sigma else None,
    }


def s3_qq(df, cols, out_dir, stats_out):
    """Seção 3 — Q-Q Plot dos pares numéricos mais variáveis."""
    num_cols = cols['num_cols']
    guard = len(num_cols) >= 1
    if not guard:
        print('  [S3] SKIP: menos de 1 coluna numérica.')
        return

    # Escolhe até 4 colunas com maior CV
    cvs = {c: df[c].std() / df[c].mean() * 100 for c in num_cols if df[c].mean() != 0}
    top_cols = sorted(cvs, key=cvs.get, reverse=True)[:min(4, len(cvs))]
    n = len(top_cols)
    ncols_plot = min(n, 2)
    nrows_plot = (n + 1) // 2

    fig = new_fig('Seção 3 — Q-Q Plot vs. Normal Teórica', (6 * ncols_plot, 4.5 * nrows_plot))
    for i, col in enumerate(top_cols):
        ax = fig.add_subplot(nrows_plot, ncols_plot, i + 1)
        ax.set_facecolor(PANEL)
        y = df[col].dropna()
        if (y > 0).all():
            y = np.log(y)
            label = f'log({col})'
        else:
            label = col
        (osm, osr), (slope, intercept, r) = stats.probplot(y, dist='norm')
        ax.scatter(osm, osr, color=ACCENT, alpha=0.65, s=18, zorder=3)
        line_x = np.array([osm.min(), osm.max()])
        ax.plot(line_x, slope * line_x + intercept, color=TEAL, linewidth=1.8, label=f'r = {r:.4f}')
        ax.legend(fontsize=7.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
        style_ax(ax, f'Q-Q: {label}', 'Quantis teóricos (Normal)', 'Quantis observados')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    save(fig, 's3_qq', out_dir,
         'Pontos sobre a diagonal = distribuição normal. Desvios nas caudas indicam assimetria residual. '
         'r ≈ 1 = excelente ajuste.')

    stats_out['s3_qq'] = {col: round(float(stats.probplot(df[col].dropna(), dist='norm')[1][2]), 4)
                           for col in top_cols}


def s4_ecdf(df, cols, out_dir, stats_out):
    """Seção 4 — ECDF empírica vs. CDF log-normal ajustada (target_col)."""
    target = cols['target_col']
    guard = target is not None and (df[target] > 0).all() and df[target].notna().sum() >= 10
    if not guard:
        print('  [S4] SKIP: target_col ausente, não-positiva ou insuficiente.')
        return

    v = df[target].dropna()
    log_v = np.log(v)
    mu, sigma = log_v.mean(), log_v.std()
    v_sorted = np.sort(v)
    ecdf_y = np.arange(1, len(v_sorted) + 1) / len(v_sorted)
    cdf_ln = stats.lognorm.cdf(v_sorted, s=sigma, scale=np.exp(mu))

    ks_stat, ks_p = stats.kstest(v, 'lognorm', args=(sigma, 0, np.exp(mu)))

    fig = new_fig(f'Seção 4 — ECDF vs. CDF Log-Normal: {target}', (8, 5))
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)
    ax.step(v_sorted, ecdf_y, color=ACCENT, linewidth=2, label='ECDF empírica', where='post')
    ax.plot(v_sorted, cdf_ln, color=TEAL, linewidth=2, linestyle='--', label='CDF Log-Normal ajustada')
    ax.text(0.97, 0.05, f'KS stat = {ks_stat:.3f}\np-value = {ks_p:.3f}',
            transform=ax.transAxes, ha='right', va='bottom',
            color=TEXT, fontsize=8, bbox=dict(facecolor=CARD, edgecolor=MUTED, boxstyle='round,pad=0.4'))
    ax.legend(fontsize=8.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax, '', target.replace('_', ' '), 'Probabilidade acumulada')

    save(fig, 's4_ecdf', out_dir,
         'ECDF (degraus) vs. CDF teórica (tracejado). KS p>0,05 indica ajuste log-normal aceitável.')

    stats_out['s4_ecdf'] = {
        'target': target,
        'ks_stat': round(float(ks_stat), 4),
        'ks_pvalue': round(float(ks_p), 4),
        'lognormal_mu': round(float(mu), 4),
        'lognormal_sigma': round(float(sigma), 4),
        'goodness_of_fit': 'aceitável' if ks_p > 0.05 else 'rejeitado (p≤0.05)',
    }


def s5_shapiro(df, cols, out_dir, stats_out):
    """Seção 5 — Teste Shapiro-Wilk para todas as variáveis numéricas."""
    num_cols = cols['num_cols']
    guard = len(num_cols) >= 1
    if not guard:
        print('  [S5] SKIP: nenhuma coluna numérica.')
        return

    results = []
    for col in num_cols:
        s, p = shapiro(df[col].dropna())
        results.append({'Variável': col.replace('_', ' '), 'W': round(s, 4),
                        'p-value': round(p, 4), 'Normal?': 'Sim ✓' if p > 0.05 else 'Não ✗'})
    sw_df = pd.DataFrame(results)

    fig = new_fig('Seção 5 — Shapiro-Wilk', (9, max(3, len(num_cols) * 0.6 + 1.5)))
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL); ax.axis('off')
    tbl = ax.table(cellText=sw_df.values, colLabels=sw_df.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 2)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor('#1d222c')
        if r == 0:
            cell.set_facecolor(ACCENT); cell.set_text_props(color='#000', fontweight='bold')
        else:
            val = sw_df.iloc[r - 1]['Normal?'] if r > 0 else ''
            if 'Não' in str(val):
                cell.set_facecolor('#2a1215'); cell.set_text_props(color=RED)
            else:
                cell.set_facecolor(PANEL); cell.set_text_props(color=TEXT)

    save(fig, 's5_shapiro', out_dir,
         'H₀ = distribuição normal. p > 0,05 → não rejeitamos H₀. '
         'p ≤ 0,05 → use testes não-paramétricos (Kruskal-Wallis, Mann-Whitney).')

    stats_out['s5_shapiro'] = {
        row['Variável']: {'W': row['W'], 'p_value': row['p-value'],
                          'normal': row['Normal?'] == 'Sim ✓'}
        for _, row in sw_df.iterrows()
    }


def s6_correlacao(df, cols, out_dir, stats_out):
    """Seção 6 — Heatmap de Correlação de Pearson."""
    num_cols = cols['num_cols']
    guard = len(num_cols) >= 2
    if not guard:
        print('  [S6] SKIP: menos de 2 colunas numéricas para correlação.')
        return

    corr = df[num_cols].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig = new_fig('Seção 6 — Correlação de Pearson', (9, 7))
    ax = fig.add_subplot(111)
    cmap = sns.diverging_palette(220, 20, as_cmap=True)
    sns.heatmap(corr, mask=mask, cmap=cmap, vmin=-1, vmax=1, center=0,
                annot=True, fmt='.2f', annot_kws={'size': 8.5, 'color': TEXT},
                linewidths=0.5, linecolor=DARK, cbar_kws={'shrink': 0.8}, ax=ax)
    ax.set_facecolor(PANEL)
    ax.tick_params(colors=TEXT, labelsize=8)
    ax.set_xticklabels([c.replace('_', '\n') for c in num_cols], rotation=0, ha='center')
    ax.set_yticklabels([c.replace('_', ' ') for c in num_cols], rotation=0)
    ax.figure.axes[-1].tick_params(colors=TEXT)

    save(fig, 's6_correlacao', out_dir,
         '|r| > 0,6 = correlação forte (atenção à colinearidade em modelos). '
         'Apenas triângulo inferior exibido.')

    # Pares mais correlacionados
    corr_pairs = []
    for i, c1 in enumerate(num_cols):
        for j, c2 in enumerate(num_cols):
            if j >= i:
                continue
            corr_pairs.append((c1, c2, round(float(corr.loc[c1, c2]), 4)))
    corr_pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    stats_out['s6_correlacao'] = {
        'top_pairs': [{'col1': a, 'col2': b, 'r': r} for a, b, r in corr_pairs[:5]],
        'max_abs_r': round(abs(corr_pairs[0][2]), 4) if corr_pairs else None,
    }


def s7_scatter(df, cols, out_dir, stats_out):
    """Seção 7 — Scatter matrix 2×2 com OLS por grupo."""
    num_cols = cols['num_cols']
    group_col = cols['group_cols'][0] if cols['group_cols'] else None
    guard = len(num_cols) >= 2
    if not guard:
        print('  [S7] SKIP: menos de 2 colunas numéricas.')
        return

    target = cols['target_col']
    # Gera pares automaticamente: target vs outros (até 4 pares)
    others = [c for c in num_cols if c != target]
    if target and others:
        raw_pairs = [(target, c) for c in others[:3]]
        if len(others) >= 2:
            raw_pairs.append((others[0], others[1]))
    else:
        raw_pairs = [(num_cols[i], num_cols[j])
                     for i in range(len(num_cols)) for j in range(i+1, len(num_cols))][:4]
    pairs = raw_pairs[:4]

    n_pairs = len(pairs)
    nrows_p = (n_pairs + 1) // 2
    ncols_p = min(n_pairs, 2)
    fig, axes = plt.subplots(nrows_p, ncols_p, figsize=(6 * ncols_p, 5 * nrows_p), facecolor=DARK)
    fig.suptitle('Seção 7 — Scatter com OLS', color=ACCENT, fontsize=12, fontweight='bold')
    if n_pairs == 1:
        axes = np.array([[axes]])
    elif nrows_p == 1:
        axes = axes.reshape(1, -1)

    palette = build_cat_palette(df[group_col].unique()) if group_col else {}
    ols_stats = []

    for idx, (xcol, ycol) in enumerate(pairs):
        ax = axes[idx // ncols_p][idx % ncols_p]
        ax.set_facecolor(PANEL)
        if group_col:
            for grp_val, grp in df.groupby(group_col):
                x, y = grp[xcol].values, grp[ycol].values
                color = palette.get(grp_val, ACCENT)
                ax.scatter(x, y, color=color, alpha=0.55, s=22, label=str(grp_val), zorder=3)
                if len(x) > 2:
                    m, b, r_val, *_ = stats.linregress(x, y)
                    xr = np.linspace(x.min(), x.max(), 100)
                    ax.plot(xr, m * xr + b, color=color, linewidth=1.5, alpha=0.9)
                    ols_stats.append({'x': xcol, 'y': ycol, 'group': str(grp_val),
                                      'slope': round(float(m), 4), 'r': round(float(r_val), 4)})
        else:
            x, y = df[xcol].values, df[ycol].values
            ax.scatter(x, y, color=ACCENT, alpha=0.55, s=22, zorder=3)
            m, b, r_val, *_ = stats.linregress(x, y)
            xr = np.linspace(x.min(), x.max(), 100)
            ax.plot(xr, m * xr + b, color=TEAL, linewidth=1.5)
            ols_stats.append({'x': xcol, 'y': ycol, 'slope': round(float(m), 4),
                               'r': round(float(r_val), 4)})
        style_ax(ax, f'{xcol.replace("_"," ")} × {ycol.replace("_"," ")}',
                 xcol.replace('_', ' '), ycol.replace('_', ' '))
        if group_col:
            ax.legend(fontsize=7, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)

    # Ocultar subplots vazios
    for idx in range(n_pairs, nrows_p * ncols_p):
        axes[idx // ncols_p][idx % ncols_p].set_visible(False)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    save(fig, 's7_scatter', out_dir,
         'Cada cor = grupo. A reta OLS por grupo mostra tendência linear. '
         'Inclinações diferentes indicam interação entre grupo e a relação estudada.')

    stats_out['s7_scatter'] = {'ols_by_pair': ols_stats}


def s8_violin_kruskal(df, cols, out_dir, stats_out):
    """Seção 8 — Violin por grupo + IC bootstrap da mediana + Kruskal-Wallis."""
    target = cols['target_col']
    group_col = cols['group_cols'][0] if cols['group_cols'] else None
    guard = target is not None and group_col is not None
    if not guard:
        print('  [S8] SKIP: target_col ou group_col ausente.')
        return

    groups_names = sorted(df[group_col].unique())
    groups_data  = [df.loc[df[group_col] == g, target].dropna().values for g in groups_names]
    guard2 = all(len(g) >= 2 for g in groups_data) and len(groups_data) >= 2
    if not guard2:
        print('  [S8] SKIP: grupos insuficientes para Kruskal-Wallis.')
        return

    kw_stat, kw_p = kruskal(*groups_data)
    palette = build_cat_palette(groups_names)

    fig = new_fig(f'Seção 8 — Violin: {target} por {group_col} + Kruskal-Wallis', (12, 6))
    ax = fig.add_subplot(111)
    ax.set_facecolor(PANEL)

    positions = range(1, len(groups_names) + 1)
    parts = ax.violinplot(groups_data, positions=list(positions),
                          showmedians=False, showextrema=True, widths=0.7)
    for pc, g in zip(parts['bodies'], groups_names):
        c = palette.get(g, ACCENT)
        pc.set_facecolor(c); pc.set_alpha(0.45); pc.set_edgecolor(c)
    for part in ['cbars', 'cmins', 'cmaxes']:
        parts[part].set_edgecolor(MUTED); parts[part].set_linewidth(0.8)

    medians_out = {}
    for i, (grp_data, g) in enumerate(zip(groups_data, groups_names), 1):
        med = float(np.median(grp_data))
        boots_med = [np.median(np.random.choice(grp_data, len(grp_data), replace=True))
                     for _ in range(2000)]
        ci_lo, ci_hi = float(np.percentile(boots_med, 2.5)), float(np.percentile(boots_med, 97.5))
        c = palette.get(g, ACCENT)
        ax.scatter(i, med, color=c, s=60, zorder=5, marker='D')
        ax.vlines(i, ci_lo, ci_hi, color=c, linewidth=3, alpha=0.7)
        medians_out[str(g)] = {'median': round(med, 4), 'ci95_lo': round(ci_lo, 4),
                               'ci95_hi': round(ci_hi, 4)}

    ax.set_xticks(list(positions))
    ax.set_xticklabels(groups_names, color=TEXT, fontsize=9)
    ax.text(0.02, 0.97, f'Kruskal-Wallis H = {kw_stat:.2f}, p = {kw_p:.4f}',
            transform=ax.transAxes, va='top', color=TEXT, fontsize=8.5,
            bbox=dict(facecolor=CARD, edgecolor=MUTED, boxstyle='round,pad=0.4'))
    style_ax(ax, '', group_col.replace('_', ' '), target.replace('_', ' '))

    save(fig, 's8_violin', out_dir,
         'Largura do violino = densidade. Losango = mediana; barra = IC95% bootstrap. '
         'Kruskal-Wallis p < 0,05 → diferença significativa entre grupos.')

    stats_out['s8_violin_kruskal'] = {
        'target': target,
        'group_col': group_col,
        'kw_stat': round(float(kw_stat), 4),
        'kw_pvalue': round(float(kw_p), 4),
        'significant': kw_p < 0.05,
        'medians_by_group': medians_out,
    }


def s9_pca(df, cols, out_dir, stats_out):
    """Seção 9 — Scree Plot PCA + Heatmap de Loadings PC1-PC3."""
    num_cols = cols['num_cols']
    guard = len(num_cols) >= 3
    if not guard:
        print('  [S9] SKIP: menos de 3 colunas numéricas para PCA.')
        return

    X = df[num_cols].dropna()
    guard2 = len(X) >= len(num_cols) + 1
    if not guard2:
        print('  [S9] SKIP: amostras insuficientes para PCA.')
        return

    X_sc = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(X_sc)
    n_comp = len(pca.explained_variance_ratio_)
    var_exp   = pca.explained_variance_ratio_ * 100
    var_cumul = np.cumsum(var_exp)

    fig = new_fig('Seção 9 — PCA: Scree Plot & Loadings', (13, 5.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)

    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(PANEL)
    x_pos = range(1, n_comp + 1)
    ax1.bar(x_pos, var_exp, color=ACCENT, alpha=0.75, edgecolor=DARK, linewidth=0.5)
    ax1.plot(x_pos, var_cumul, color=TEAL, marker='o', markersize=5, linewidth=2, label='Acumulada %')
    ax1.axhline(80, color=RED, linestyle='--', linewidth=1, alpha=0.7, label='80%')
    for i, (v_i, c_i) in enumerate(zip(var_exp, var_cumul), 1):
        ax1.text(i, v_i + 0.8, f'{v_i:.1f}%', ha='center', color=TEXT, fontsize=7)
    ax1.set_xticks(list(x_pos))
    ax1.set_xticklabels([f'PC{i}' for i in x_pos], color=TEXT, fontsize=8)
    ax1.legend(fontsize=7.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax1, 'Scree Plot', 'Componente', 'Variância explicada (%)')

    n_load = min(3, n_comp)
    ax2 = fig.add_subplot(gs[1])
    loadings = pd.DataFrame(pca.components_[:n_load].T,
                            index=[c.replace('_', '\n') for c in num_cols],
                            columns=[f'PC{i+1}' for i in range(n_load)])
    sns.heatmap(loadings, cmap=sns.diverging_palette(220, 20, as_cmap=True),
                vmin=-1, vmax=1, center=0, annot=True, fmt='.2f',
                annot_kws={'size': 8, 'color': TEXT},
                linewidths=0.5, linecolor=DARK, cbar_kws={'shrink': 0.8}, ax=ax2)
    ax2.set_facecolor(PANEL)
    ax2.tick_params(colors=TEXT, labelsize=8)
    ax2.set_title(f'Loadings PC1–PC{n_load}', color=TEXT, fontsize=9, pad=6)
    ax2.figure.axes[-1].tick_params(colors=TEXT)

    save(fig, 's9_pca', out_dir,
         'Scree: escolha PCs que acumulam ≥80%. '
         'Loadings |>0,5| = variáveis dominantes no componente.')

    n_80 = int(np.searchsorted(var_cumul, 80) + 1)
    stats_out['s9_pca'] = {
        'n_components_total': int(n_comp),
        'variance_explained_pct': [round(float(v), 2) for v in var_exp],
        'cumulative_variance_pct': [round(float(v), 2) for v in var_cumul],
        'n_components_for_80pct': n_80,
        'pc1_variance_pct': round(float(var_exp[0]), 2),
        'pc2_variance_pct': round(float(var_exp[1]), 2) if n_comp >= 2 else None,
    }


def s10_kmeans(df, cols, out_dir, stats_out):
    """Seção 10 — K-Means: cotovelo + silhouette + PCA 2D + perfil Z-score."""
    num_cols = cols['num_cols']
    guard = len(num_cols) >= 2 and len(df.dropna(subset=num_cols)) >= 10
    if not guard:
        print('  [S10] SKIP: colunas ou amostras insuficientes para K-Means.')
        return

    X = df[num_cols].dropna()
    X_sc = StandardScaler().fit_transform(X)
    pca2 = PCA(n_components=2)
    X_2d = pca2.fit_transform(X_sc)

    k_max = min(9, len(X) // 5)
    K_range = range(2, k_max + 1)
    inertias, silhouettes = [], []
    for k in K_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_sc)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_sc, labels))

    # K ótimo: maior silhouette
    K_BEST = list(K_range)[silhouettes.index(max(silhouettes))]
    km_best = KMeans(n_clusters=K_BEST, random_state=42, n_init=10)
    cluster_labels = km_best.fit_predict(X_sc)
    CLUS_COLORS = [ACCENT, BLUE, GREEN, PURPLE, TEAL, RED, PINK,
                   '#fb923c', '#818cf8', '#4ade80']

    fig = new_fig('Seção 10 — K-Means Clustering', (14, 10))
    gs = gridspec.GridSpec(2, 2, figure=fig, wspace=0.38, hspace=0.45)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PANEL)
    ax1.plot(K_range, inertias, color=ACCENT, marker='o', markersize=6, linewidth=2)
    ax1.axvline(K_BEST, color=RED, linestyle='--', linewidth=1.2, alpha=0.8, label=f'K={K_BEST}')
    ax1.set_xticks(list(K_range))
    ax1.legend(fontsize=8, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax1, 'Gráfico do Cotovelo', 'K', 'Inércia (WCSS)')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PANEL)
    ax2.plot(K_range, silhouettes, color=TEAL, marker='s', markersize=6, linewidth=2)
    ax2.axvline(K_BEST, color=RED, linestyle='--', linewidth=1.2, alpha=0.8, label=f'K={K_BEST}')
    ax2.set_xticks(list(K_range))
    for k, s in zip(K_range, silhouettes):
        ax2.text(k, s + 0.002, f'{s:.3f}', ha='center', color=TEXT, fontsize=7)
    ax2.legend(fontsize=8, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax2, 'Silhouette Score', 'K', 'Silhouette')

    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PANEL)
    for k in range(K_BEST):
        msk = cluster_labels == k
        ax3.scatter(X_2d[msk, 0], X_2d[msk, 1],
                    color=CLUS_COLORS[k % len(CLUS_COLORS)], alpha=0.65, s=28,
                    label=f'Cluster {k+1}', zorder=3)
    centers_2d = pca2.transform(km_best.cluster_centers_)
    ax3.scatter(centers_2d[:, 0], centers_2d[:, 1],
                color='white', s=120, marker='*', zorder=5, edgecolors=DARK, linewidth=0.5)
    pct = pca2.explained_variance_ratio_ * 100
    style_ax(ax3, 'Projeção PCA 2D', f'PC1 ({pct[0]:.1f}%)', f'PC2 ({pct[1]:.1f}%)')
    ax3.legend(fontsize=7.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PANEL)
    df_cl = X.copy()
    df_cl['cluster'] = cluster_labels
    z_profile = df_cl.groupby('cluster')[num_cols].mean()
    z_profile_std = (z_profile - z_profile.mean()) / z_profile.std()
    x_ticks = range(len(num_cols))
    for k in range(K_BEST):
        ax4.plot(x_ticks, z_profile_std.iloc[k].values,
                 color=CLUS_COLORS[k % len(CLUS_COLORS)], marker='o', markersize=5,
                 linewidth=2, label=f'Cluster {k+1}')
    ax4.axhline(0, color=MUTED, linewidth=0.8, linestyle='--', alpha=0.5)
    ax4.set_xticks(list(x_ticks))
    ax4.set_xticklabels([c.replace('_', '\n') for c in num_cols], color=TEXT, fontsize=7)
    ax4.legend(fontsize=7.5, facecolor=CARD, edgecolor=MUTED, labelcolor=TEXT)
    style_ax(ax4, 'Perfil Z-score dos Clusters', '', 'Z-score')

    save(fig, 's10_kmeans', out_dir,
         'Cotovelo: ponto de inflexão na inércia. Silhouette > 0,5 = clusters bem definidos. '
         'Estrela = centróide. Z-score revela o "DNA" de cada cluster.')

    cluster_sizes = {int(k): int((cluster_labels == k).sum()) for k in range(K_BEST)}
    cluster_means = {
        f'cluster_{k+1}': {
            col: round(float(df_cl.loc[df_cl['cluster'] == k, col].mean()), 4)
            for col in num_cols
        }
        for k in range(K_BEST)
    }
    stats_out['s10_kmeans'] = {
        'k_best': K_BEST,
        'best_silhouette': round(float(max(silhouettes)), 4),
        'silhouettes_by_k': {int(k): round(float(s), 4) for k, s in zip(K_range, silhouettes)},
        'cluster_sizes': cluster_sizes,
        'cluster_means': cluster_means,
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f'\n{"═"*60}')
    print(f'  DataLab Analytics Pipeline')
    print(f'  CSV  : {args.csv}')
    print(f'  META : {args.meta}')
    print(f'  OUT  : {args.out_dir}')
    print(f'{"═"*60}')

    # ── Leitura ──────────────────────────────────────────────────────────────
    df = pd.read_csv(args.csv, encoding='utf-8-sig')
    meta = load_metadata(args.meta)

    print(f'\nDataset: {len(df)} linhas × {len(df.columns)} colunas')

    # ── Normalização booleana ────────────────────────────────────────────────
    bool_meta_cols = [c['name'] for c in meta.get('columns', [])
                      if c.get('type', '').lower() in ('boolean', 'bool')]
    for col in bool_meta_cols:
        if col in df.columns:
            df[col] = normalize_bool_col(df[col])

    # ── Inferência de papéis ─────────────────────────────────────────────────
    print('\nInferindo papéis de colunas...')
    cols = infer_columns(meta, df)

    stats_out = {
        '_meta': {
            'csv': os.path.basename(args.csv),
            'meta': os.path.basename(args.meta),
            'rows': len(df),
            'columns': len(df.columns),
            'num_cols': cols['num_cols'],
            'cat_cols': cols['cat_cols'],
            'bool_cols': cols['bool_cols'],
            'target_col': cols['target_col'],
            'group_cols': cols['group_cols'],
        }
    }

    # ── Execução das seções ──────────────────────────────────────────────────
    sections = [
        ('S1  — Descritiva & Categórica',          s1_descritiva),
        ('S2  — Distribuição da Variável Alvo',     s2_distribuicao_target),
        ('S3  — Q-Q Plot',                          s3_qq),
        ('S4  — ECDF vs. CDF Log-Normal',           s4_ecdf),
        ('S5  — Shapiro-Wilk',                      s5_shapiro),
        ('S6  — Correlação de Pearson',              s6_correlacao),
        ('S7  — Scatter + OLS',                     s7_scatter),
        ('S8  — Violin + Kruskal-Wallis',           s8_violin_kruskal),
        ('S9  — PCA',                               s9_pca),
        ('S10 — K-Means Clustering',                s10_kmeans),
    ]

    for label, fn in sections:
        print(f'\n{label}...')
        try:
            fn(df, cols, args.out_dir, stats_out)
        except Exception as e:
            print(f'  [ERRO] {label}: {e}')
            stats_out[fn.__name__] = {'error': str(e)}

    # ── Export JSON ──────────────────────────────────────────────────────────
    json_path = os.path.join(args.out_dir, 'export_stats.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(stats_out, f, ensure_ascii=False, indent=2,
                  default=lambda o: bool(o) if isinstance(o, (np.bool_,)) else
                                    int(o) if isinstance(o, (np.integer,)) else
                                    float(o) if isinstance(o, (np.floating,)) else str(o))
    print(f'\n✓ export_stats.json salvo em {json_path}')
    print(f'{"═"*60}\n')


if __name__ == '__main__':
    main()
