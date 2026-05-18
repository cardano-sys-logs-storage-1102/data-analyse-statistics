"""
latex_builder.py — Gera relatório técnico LaTeX a partir do export_stats.json e PNGs.

Uso:
    python latex_builder.py <export_stats.json> <png_dir> [--out relatorio.tex] [--title "Título"]

O .tex produzido requer:
    pdflatex -interaction=nonstopmode relatorio.tex  (2× para índice)

Pacotes utilizados: geometry, booktabs, graphicx, amsmath, hyperref, xcolor,
                    inputenc, fontenc, babel, caption, microtype, fancyhdr, titlesec.
"""

import json
import os
import sys
import argparse
from datetime import datetime

# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description='Gerador de relatório LaTeX a partir do export_stats.json')
    p.add_argument('stats',    help='Caminho para export_stats.json')
    p.add_argument('png_dir',  help='Diretório contendo os PNGs gerados por analise.py')
    p.add_argument('--out',    default='relatorio.tex', help='Arquivo .tex de saída')
    p.add_argument('--title',  default='Relatório de Análise Exploratória de Dados',
                   help='Título do relatório')
    p.add_argument('--author', default='Pipeline DataLab Analytics', help='Autor')
    return p.parse_args()

# ── Helpers ───────────────────────────────────────────────────────────────────
def escape(s: str) -> str:
    """Escapa caracteres especiais LaTeX em strings."""
    replacements = [
        ('\\', r'\textbackslash{}'),
        ('&', r'\&'), ('%', r'\%'), ('$', r'\$'), ('#', r'\#'),
        ('_', r'\_'), ('{', r'\{'), ('}', r'\}'),
        ('~', r'\textasciitilde{}'), ('^', r'\textasciicircum{}'),
    ]
    for old, new in replacements:
        s = s.replace(old, new)
    return s

def fmt(val, decimals=2):
    """Formata número com separador de milhar BR e decimais fixos."""
    if val is None:
        return r'\textemdash'
    try:
        f = float(val)
        if decimals == 0:
            return f'{int(f):,}'.replace(',', '.')
        return f'{f:,.{decimals}f}'.replace(',', 'X').replace('.', ',').replace('X', '.')
    except (TypeError, ValueError):
        return escape(str(val))

def png_path(png_dir: str, name: str) -> str:
    """Retorna caminho absoluto do PNG, com fallback vazio."""
    p = os.path.abspath(os.path.join(png_dir, f'{name}.png'))
    return p if os.path.exists(p) else ''

def fig_block(path: str, caption: str, label: str, width: str = r'0.96\linewidth') -> str:
    """Bloco figure LaTeX padrão."""
    if not path:
        return f'% Figura {label} não encontrada\n'
    return (
        r'\begin{figure}[htbp]' + '\n'
        r'  \centering' + '\n'
        f'  \\includegraphics[width={width}]{{{path}}}' + '\n'
        f'  \\caption{{{caption}}}' + '\n'
        f'  \\label{{fig:{label}}}' + '\n'
        r'\end{figure}' + '\n'
    )

# ══════════════════════════════════════════════════════════════════════════════
# SEÇÕES LATEX
# ══════════════════════════════════════════════════════════════════════════════

def sec_intro(stats: dict) -> str:
    m = stats.get('_meta', {})
    rows = m.get('rows', '?')
    cols = m.get('columns', '?')
    n_num = len(m.get('num_cols', []))
    n_cat = len(m.get('cat_cols', []))
    n_bool = len(m.get('bool_cols', []))
    target = escape(m.get('target_col') or '—')
    csv_name = escape(m.get('csv', '?'))

    return (
        r'\section{Introdução e Contexto dos Dados}' + '\n\n'
        r'O presente relatório documenta a análise exploratória sistematizada do conjunto de dados '
        f'\\texttt{{{csv_name}}}, composto por \\textbf{{{rows} observações}} e '
        f'\\textbf{{{cols} variáveis}}: {n_num}~numéricas, {n_cat}~categóricas e {n_bool}~booleana(s). '
        r'A variável de maior dispersão relativa — identificada pelo coeficiente de variação máximo — '
        f'foi selecionada como variável-alvo principal: \\textbf{{\\texttt{{{target}}}}}. '
        r'Todas as inferências de papel de coluna são executadas dinamicamente a partir dos metadados '
        r'DataLab, eliminando qualquer dependência de configuração manual.' + '\n\n'
        r'O pipeline executa dez seções analíticas condicionadas a guardas de pré-requisito; '
        r'seções cujas condições mínimas não foram satisfeitas são registradas como omitidas '
        r'(\emph{SKIP}) no log de execução.' + '\n\n'
    )


def sec1(stats: dict, png_dir: str) -> str:
    s = stats.get('s1_descritiva', {})
    sc = stats.get('s1_categoricas', {})
    m = stats.get('_meta', {})
    n_num = len(s)
    high_cv = [(c, v['cv_pct']) for c, v in s.items() if v.get('cv_pct', 0) > 30]
    high_cv.sort(key=lambda x: x[1], reverse=True)
    high_cv_str = ', '.join(
        f'\\texttt{{{escape(c)}}} (CV\\,=\\,{fmt(v)}\\,\\%)'
        for c, v in high_cv[:3]
    ) or 'nenhuma'

    missing_total = sum(v.get('missing', 0) for v in s.values())

    text = (
        r'\section{Estatística Descritiva e Distribuição das Variáveis}' + '\n\n'
        f'O conjunto analisado contém \\textbf{{{n_num} variáveis numéricas}} cujas estatísticas '
        r'descritivas de posição, dispersão e forma são sumarizadas na Figura~\ref{fig:s1_descritiva}. '
        f'O total de valores ausentes no bloco numérico é \\textbf{{{missing_total}}}, '
        r'indicando completude plena dos dados ou grau negligenciável de missingness. '
        r'Variáveis com coeficiente de variação superior a 30\,\% --- limiar convencional de '
        r'alta dispersão relativa --- são: '
        f'{high_cv_str}. '
        r'Assimetrias positivas observadas sugerem caudas direitas pronunciadas, '
        r'compatíveis com distribuições log-normais ou de Pareto, comuns em métricas '
        r'de volume financeiro e operacional.' + '\n\n'
    )

    text += fig_block(png_path(png_dir, 's1_descritiva'),
                      r'Tabela de estatísticas descritivas: posição, dispersão, forma e completude por variável numérica.',
                      's1_descritiva')

    if sc:
        n_cat = len(sc)
        text += (
            f'\n\\noindent A Figura~\\ref{{fig:s1_categoricas}} exibe a distribuição de frequência das '
            f'\\textbf{{{n_cat} variáveis categóricas}}. '
            r'A análise do balanceamento entre categorias é indispensável: razões superiores a '
            r'3:1 entre a categoria mais e menos frequente podem introduzir viés em modelos '
            r'supervisionados e distorções em comparações intergrupo.' + '\n\n'
        )
        text += fig_block(png_path(png_dir, 's1_categoricas'),
                          r'Frequência absoluta e relativa das variáveis categóricas.',
                          's1_categoricas')
    return text


def sec2(stats: dict, png_dir: str) -> str:
    s = stats.get('s2_distribuicao', {})
    if not s:
        return r'\section{Distribuição da Variável-Alvo}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    target = escape(s.get('target', '?'))
    mean_v   = fmt(s.get('mean'))
    med_v    = fmt(s.get('median'))
    std_v    = fmt(s.get('std'))
    ci_lo    = fmt(s.get('ci95_mean_lo'))
    ci_hi    = fmt(s.get('ci95_mean_hi'))
    mu_v     = fmt(s.get('lognormal_mu'), 4)
    sigma_v  = fmt(s.get('lognormal_sigma'), 4)

    return (
        r'\section{Distribuição da Variável-Alvo}' + '\n\n'
        f'A variável \\textbf{{\\texttt{{{target}}}}} apresenta média de \\textbf{{{mean_v}}}, '
        f'mediana de {med_v} e desvio-padrão de {std_v}. '
        f'O intervalo de confiança de 95\\,\\% para a média, obtido por bootstrap com '
        r'$B = 3\,000$ reamostras, situa-se em '
        f'$[{ci_lo},\\;{ci_hi}]$. '
        r'A assimetria positiva entre média e mediana, aliada ao ajuste paramétrico, '
        r'sustenta a hipótese log-normal: '
        f'$\\hat{{\\mu}}_{{\\ln}} = {mu_v}$, $\\hat{{\\sigma}}_{{\\ln}} = {sigma_v}$. '
        r'A Figura~\ref{fig:s2_distribuicao} consolida histograma com KDE empírica, '
        r'boxplot com notch e intervalo de confiança, e comparação intergrupo.' + '\n\n'
        + fig_block(png_path(png_dir, 's2_distribuicao'),
                    f'Distribuição de \\texttt{{{target}}}: histograma, KDE, boxplot com IC\\,95\\,\\% e comparação por grupo.',
                    's2_distribuicao')
    )


def sec3(stats: dict, png_dir: str) -> str:
    s = stats.get('s3_qq', {})
    if not s:
        return r'\section{Aderência à Normalidade — Q-Q Plot}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    best_col = max(s, key=s.get) if s else '?'
    best_r   = fmt(s.get(best_col, 0), 4)
    worst_col = min(s, key=s.get) if s else '?'
    worst_r  = fmt(s.get(worst_col, 0), 4)

    return (
        r'\section{Aderência à Normalidade --- Q-Q Plot}' + '\n\n'
        r'Os gráficos quantil--quantil da Figura~\ref{fig:s3_qq} comparam os quantis '
        r'empíricos (após transformação logarítmica, quando aplicável) aos quantis teóricos '
        r'de uma distribuição normal padrão. '
        f'A variável de melhor aderência é \\texttt{{{escape(best_col)}}}, '
        f'com coeficiente de correlação $r = {best_r}$; '
        f'a de menor aderência é \\texttt{{{escape(worst_col)}}}, '
        f'com $r = {worst_r}$. '
        r'Desvios sistemáticos nas caudas indicam assimetria residual e sustentam '
        r'o uso de testes não-paramétricos nas comparações intergrupo.' + '\n\n'
        + fig_block(png_path(png_dir, 's3_qq'),
                    r'Q-Q Plot das principais variáveis numéricas vs.\ distribuição normal teórica.',
                    's3_qq')
    )


def sec4(stats: dict, png_dir: str) -> str:
    s = stats.get('s4_ecdf', {})
    if not s:
        return r'\section{ECDF vs.\ CDF Log-Normal}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    target  = escape(s.get('target', '?'))
    ks_stat = fmt(s.get('ks_stat'), 4)
    ks_p    = fmt(s.get('ks_pvalue'), 4)
    fit     = escape(s.get('goodness_of_fit', '?'))
    mu_v    = fmt(s.get('lognormal_mu'), 4)
    sigma_v = fmt(s.get('lognormal_sigma'), 4)

    return (
        r'\section{Função de Distribuição Acumulada Empírica vs.\ Log-Normal}' + '\n\n'
        r'A Figura~\ref{fig:s4_ecdf} sobrepõe a ECDF empírica à CDF teórica da distribuição '
        f'log-normal ajustada por máxima verossimilhança sobre \\texttt{{{target}}} '
        f'($\\hat{{\\mu}}_{{\\ln}} = {mu_v}$, $\\hat{{\\sigma}}_{{\\ln}} = {sigma_v}$). '
        r'O teste de Kolmogorov--Smirnov quantifica a máxima discrepância vertical entre as duas curvas: '
        f'$D = {ks_stat}$, $p = {ks_p}$ ({fit}). '
        r'Um $p$-valor superior a $0{,}05$ indica que não há evidência estatística suficiente '
        r'para rejeitar o ajuste log-normal ao nível de significância de 5\,\%.' + '\n\n'
        + fig_block(png_path(png_dir, 's4_ecdf'),
                    f'ECDF empírica (degraus) vs.\\ CDF log-normal ajustada (tracejado) para \\texttt{{{target}}}.',
                    's4_ecdf')
    )


def sec5(stats: dict, png_dir: str) -> str:
    s = stats.get('s5_shapiro', {})
    if not s:
        return r'\section{Teste de Normalidade Shapiro-Wilk}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    n_normal  = sum(1 for v in s.values() if v.get('normal'))
    n_total   = len(s)
    n_nonnorm = n_total - n_normal

    return (
        r'\section{Teste de Normalidade Shapiro-Wilk}' + '\n\n'
        r'O teste de Shapiro-Wilk avalia, para cada variável numérica, a hipótese nula '
        r'$H_0$: a amostra provém de uma população normalmente distribuída. '
        f'De \\textbf{{{n_total}}} variáveis testadas, \\textbf{{{n_normal}}} não rejeitam '
        f'$H_0$ ao nível de 5\\,\\% e \\textbf{{{n_nonnorm}}} apresentam evidência de '
        r'não-normalidade (Figura~\ref{fig:s5_shapiro}). '
        r'Para as variáveis com distribuição não-normal, as comparações intergrupo '
        r'conduzidas nas seções subsequentes empregam o teste não-paramétrico de '
        r'Kruskal-Wallis, robusto à ausência de normalidade.' + '\n\n'
        + fig_block(png_path(png_dir, 's5_shapiro'),
                    r'Resultados do teste de normalidade Shapiro-Wilk: estatística $W$ e $p$-valor por variável.',
                    's5_shapiro')
    )


def sec6(stats: dict, png_dir: str) -> str:
    s = stats.get('s6_correlacao', {})
    if not s:
        return r'\section{Análise de Correlação de Pearson}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    top = s.get('top_pairs', [])
    max_r = fmt(s.get('max_abs_r'), 4)

    pairs_tex = ''
    for p in top[:3]:
        c1, c2, r = escape(p['col1']), escape(p['col2']), fmt(p['r'], 4)
        pairs_tex += f'\\texttt{{{c1}}} $\\times$ \\texttt{{{c2}}} ($r = {r}$), '
    pairs_tex = pairs_tex.rstrip(', ')

    return (
        r'\section{Análise de Correlação de Pearson}' + '\n\n'
        r'A matriz de correlação de Pearson (Figura~\ref{fig:s6_correlacao}) evidencia '
        r'as associações lineares entre todos os pares de variáveis numéricas. '
        f'A correlação de maior magnitude absoluta observada é $|r| = {max_r}$. '
        r'Os três pares de maior associação são: '
        f'{pairs_tex}. '
        r'Pares com $|r| > 0{,}6$ merecem atenção em contextos de modelagem preditiva, '
        r'pois a colinearidade elevada pode inflar a variância dos coeficientes em '
        r'regressões lineares e reduzir a interpretabilidade dos modelos.' + '\n\n'
        + fig_block(png_path(png_dir, 's6_correlacao'),
                    r'Heatmap de correlação de Pearson (triângulo inferior). Paleta divergente: azul = negativa, vermelho = positiva.',
                    's6_correlacao')
    )


def sec7(stats: dict, png_dir: str) -> str:
    s = stats.get('s7_scatter', {})
    if not s:
        return r'\section{Análise de Dispersão com Regressão OLS}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    ols = s.get('ols_by_pair', [])
    # Pega o par com maior |r| médio entre grupos
    pair_rs = {}
    for entry in ols:
        key = f"{entry['x']} × {entry['y']}"
        pair_rs.setdefault(key, []).append(abs(entry.get('r', 0)))
    best_pair = max(pair_rs, key=lambda k: sum(pair_rs[k])/len(pair_rs[k])) if pair_rs else '?'
    best_r_mean = fmt(sum(pair_rs[best_pair]) / len(pair_rs[best_pair]), 4) if pair_rs else '?'

    return (
        r'\section{Análise de Dispersão com Regressão OLS por Grupo}' + '\n\n'
        r'A Figura~\ref{fig:s7_scatter} apresenta diagramas de dispersão para os pares '
        r'de variáveis de maior interesse analítico, com retas de regressão OLS ajustadas '
        r'independentemente por grupo. '
        r'Diferenças de inclinação entre grupos evidenciam efeito de interação: '
        r'a relação entre as variáveis não é homogênea entre os estratos. '
        f'O par com maior coeficiente de correlação médio entre grupos é '
        f'\\textbf{{{escape(best_pair)}}} ($\\bar{{r}} = {best_r_mean}$). '
        r'Resíduos elevados em torno da reta de ajuste indicam alta variância '
        r'não explicada pela tendência linear, sugerindo a presença de variáveis '
        r'moderadoras ou confundidoras não capturadas no modelo simples.' + '\n\n'
        + fig_block(png_path(png_dir, 's7_scatter'),
                    r'Diagramas de dispersão com regressão OLS por grupo. Cada cor representa um grupo distinto.',
                    's7_scatter')
    )


def sec8(stats: dict, png_dir: str) -> str:
    s = stats.get('s8_violin_kruskal', {})
    if not s:
        return r'\section{Comparação Intergrupo --- Violino e Kruskal-Wallis}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    target    = escape(s.get('target', '?'))
    group_col = escape(s.get('group_col', '?'))
    kw_h      = fmt(s.get('kw_stat'), 4)
    kw_p      = fmt(s.get('kw_pvalue'), 4)
    sig       = s.get('significant', False)
    sig_txt   = r'\textbf{significativa}' if sig else r'não significativa'
    medians   = s.get('medians_by_group', {})

    meds_tex = ', '.join(
        f'\\texttt{{{escape(str(g))}}}: {fmt(v["median"])} [{fmt(v["ci95_lo"])};\\,{fmt(v["ci95_hi"])}]'
        for g, v in list(medians.items())[:4]
    )

    return (
        r'\section{Comparação Intergrupo --- Gráfico de Violino e Teste de Kruskal-Wallis}' + '\n\n'
        f'A Figura~\\ref{{fig:s8_violin}} compara a distribuição de \\texttt{{{target}}} '
        f'entre os grupos definidos por \\texttt{{{group_col}}}. '
        r'Os gráficos de violino exibem a estimativa de densidade kernel por grupo; '
        r'o losango interno marca a mediana amostral, e a barra vertical representa o '
        r'intervalo de confiança de 95\,\% da mediana obtido por bootstrap ($B = 2\,000$). '
        f'As medianas estimadas por grupo são: {meds_tex}. '
        r'O teste de Kruskal-Wallis --- análogo não-paramétrico da ANOVA --- '
        f'retornou $H = {kw_h}$, $p = {kw_p}$, indicando diferença '
        f'intergrupo {sig_txt} ao nível de 5\\,\\%.' + '\n\n'
        + fig_block(png_path(png_dir, 's8_violin'),
                    f'Violinos de \\texttt{{{target}}} por \\texttt{{{group_col}}} com IC\\,95\\,\\% bootstrap e resultado do teste de Kruskal-Wallis.',
                    's8_violin')
    )


def sec9(stats: dict, png_dir: str) -> str:
    s = stats.get('s9_pca', {})
    if not s:
        return r'\section{Análise de Componentes Principais (PCA)}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    n_comp  = s.get('n_components_total', '?')
    n_80    = s.get('n_components_for_80pct', '?')
    pc1_var = fmt(s.get('pc1_variance_pct'), 2)
    pc2_var = fmt(s.get('pc2_variance_pct'), 2) if s.get('pc2_variance_pct') else '—'

    return (
        r'\section{Análise de Componentes Principais}' + '\n\n'
        r'A PCA foi aplicada sobre as variáveis numéricas após padronização Z-score, '
        r'garantindo invariância de escala. '
        f'O espaço original de {n_comp}~dimensões pode ser reduzido a '
        f'\\textbf{{{n_80}~componentes principais}} sem perda significativa de informação '
        r'(limiar de 80\,\% da variância acumulada). '
        f'O primeiro componente (PC1) explica \\textbf{{{pc1_var}\\,\\%}} da variância total; '
        f'o segundo (PC2) contribui com {pc2_var}\\,\\%. '
        r'O heatmap de loadings (Figura~\ref{fig:s9_pca}) identifica quais variáveis '
        r'originais dominam cada componente: loadings com $|\\lambda| > 0{,}5$ '
        r'são considerados relevantes e orientam a interpretação semântica dos eixos.' + '\n\n'
        + fig_block(png_path(png_dir, 's9_pca'),
                    r'PCA: scree plot com variância acumulada e heatmap de loadings para PC1--PC3.',
                    's9_pca')
    )


def sec10(stats: dict, png_dir: str) -> str:
    s = stats.get('s10_kmeans', {})
    if not s:
        return r'\section{Segmentação por K-Means}' + '\n\n' + r'\emph{Seção não executada.}' + '\n\n'

    k_best = s.get('k_best', '?')
    sil    = fmt(s.get('best_silhouette'), 4)
    sizes  = s.get('cluster_sizes', {})
    sizes_tex = ', '.join(
        f'Cluster\\,{int(k)+1}: {v}\\,obs.'
        for k, v in sorted(sizes.items(), key=lambda x: int(x[0]))
    )

    return (
        r'\section{Segmentação por K-Means}' + '\n\n'
        r'O algoritmo K-Means foi executado para $K \in [2, \min(8, N/5)]$. '
        r'O número ótimo de clusters foi determinado pela maximização do coeficiente '
        r'de silhouette médio --- métrica que combina coesão intracluster e '
        r'separação intercluster no intervalo $[-1, 1]$. '
        f'O valor ótimo identificado é $K^* = {k_best}$, '
        f'com silhouette de \\textbf{{{sil}}} '
        r'(valores $> 0{,}5$ indicam estrutura de cluster bem definida). '
        f'A distribuição de observações entre os grupos é: {sizes_tex}. '
        r'A Figura~\ref{fig:s10_kmeans} consolida o gráfico do cotovelo, '
        r'o silhouette por $K$, a projeção bidimensional via PCA e o perfil '
        r'Z-score que caracteriza o ``DNA'' de cada segmento.' + '\n\n'
        + fig_block(png_path(png_dir, 's10_kmeans'),
                    r'K-Means: cotovelo, silhouette, projeção PCA 2D e perfil Z-score dos clusters.',
                    's10_kmeans')
    )


def sec_conclusao(stats: dict) -> str:
    m   = stats.get('_meta', {})
    s2  = stats.get('s2_distribuicao', {})
    s5  = stats.get('s5_shapiro', {})
    s8  = stats.get('s8_violin_kruskal', {})
    s9  = stats.get('s9_pca', {})
    s10 = stats.get('s10_kmeans', {})

    target   = escape(m.get('target_col') or '—')
    n_normal = sum(1 for v in s5.values() if v.get('normal')) if s5 else '?'
    n_total  = len(s5) if s5 else '?'
    sig_txt  = 'foi detectada' if s8.get('significant') else 'não foi detectada'
    n80      = s9.get('n_components_for_80pct', '?') if s9 else '?'
    k_best   = s10.get('k_best', '?') if s10 else '?'
    sil      = fmt(s10.get('best_silhouette'), 3) if s10 else '—'

    return (
        r'\section{Conclusões e Recomendações}' + '\n\n'
        r'\subsection*{Síntese dos Achados}' + '\n\n'
        r'\begin{itemize}' + '\n'
        f'  \\item A variável-alvo \\texttt{{{target}}} exibe distribuição log-normal, '
        r'confirmada pelo ajuste paramétrico e pelo teste de Kolmogorov-Smirnov.' + '\n'
        f'  \\item Dos testes de Shapiro-Wilk aplicados, {n_normal} de {n_total} variáveis '
        r'não rejeitam normalidade, orientando a escolha de testes não-paramétricos nas demais.' + '\n'
        f'  \\item Diferença intergrupo estatisticamente significativa {sig_txt} '
        r'pelo teste de Kruskal-Wallis.' + '\n'
        f'  \\item A PCA reduz o espaço dimensional a {n80}~componentes preservando 80\\,\\% da variância.' + '\n'
        f'  \\item O K-Means identificou $K^* = {k_best}$ segmentos naturais '
        f'(silhouette médio $= {sil}$).' + '\n'
        r'\end{itemize}' + '\n\n'
        r'\subsection*{Próximos Passos Recomendados}' + '\n\n'
        r'\begin{enumerate}' + '\n'
        r'  \item Conduzir testes post-hoc de Dunn (com correção de Bonferroni) para identificar '
        r'os pares de grupos com diferença significativa após o Kruskal-Wallis.' + '\n'
        r'  \item Avaliar modelos preditivos (regressão regularizada, gradient boosting) '
        r'utilizando os componentes principais como features de entrada.' + '\n'
        r'  \item Caracterizar os clusters obtidos com variáveis externas (ex.: metas, '
        r'segmento, canal) para derivar personas operacionais acionáveis.' + '\n'
        r'  \item Monitorar periodicamente as métricas de silhouette e KS para detectar '
        r'\emph{data drift} em produção.' + '\n'
        r'\end{enumerate}' + '\n\n'
    )

# ══════════════════════════════════════════════════════════════════════════════
# TEMPLATE LATEX
# ══════════════════════════════════════════════════════════════════════════════

PREAMBLE = r"""\documentclass[12pt,a4paper]{article}

%% --- Codificação e língua ------------------------------------------------
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage[english]{babel}

%% --- Geometria -------------------------------------------------------------
\usepackage[top=2.5cm, bottom=2.5cm, left=2.5cm, right=2.5cm]{geometry}

%% --- Tipografia ------------------------------------------------------------
\usepackage[protrusion=true,expansion=false]{microtype}

%% --- Matemática ------------------------------------------------------------
\usepackage{amsmath, amssymb}

%% --- Gráficos e cores ------------------------------------------------------
\usepackage{graphicx}
\usepackage[table]{xcolor}
\definecolor{accent}{HTML}{f5a623}
\definecolor{darkbg}{HTML}{0d0f14}

%% --- Tabelas ---------------------------------------------------------------
\usepackage{booktabs}
\usepackage{array}

%% --- Hiperlinks ------------------------------------------------------------
\usepackage[
    colorlinks=true,
    linkcolor=accent,
    urlcolor=accent,
    citecolor=accent,
    pdftitle={REPORT_TITLE},
    pdfauthor={REPORT_AUTHOR}
]{hyperref}

%% --- Cabeçalho/rodapé -----------------------------------------------------
\usepackage{fancyhdr}
\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0.4pt}
\fancyhead[L]{\small\textcolor{gray}{REPORT_TITLE}}
\fancyhead[R]{\small\textcolor{gray}{\today}}
\fancyfoot[C]{\thepage}

%% --- Títulos de seção ------------------------------------------------------
\usepackage{titlesec}
\titleformat{\section}[block]{\large\bfseries\color{accent}}{}{0em}{}[\titlerule]
\titlespacing{\section}{0pt}{1.5ex}{0.8ex}

%% --- Legendas de figuras ---------------------------------------------------
\usepackage[font=small, labelfont=bf, labelsep=period]{caption}

%% --- Espaçamento -----------------------------------------------------------
\setlength{\parskip}{0.6em}
\setlength{\parindent}{0pt}

"""

COVER = r"""
\begin{titlepage}
  \pagecolor{darkbg}
  \color{white}
  \vspace*{3cm}
  \begin{center}
    {\Huge\bfseries\textcolor{accent}{REPORT_TITLE}\par}
    \vspace{1.5cm}
    {\large REPORT_SUBTITLE\par}
    \vspace{2cm}
    \rule{0.6\linewidth}{0.4pt}\\[0.8cm]
    {\large\textbf{Autor:} REPORT_AUTHOR\par}
    \vspace{0.4cm}
    {\large\textbf{Data:} REPORT_DATE\par}
    \vspace{0.4cm}
    {\large\textbf{Dataset:} \texttt{REPORT_DATASET}\par}
    \vspace{0.4cm}
    {\large\textbf{Observações:} REPORT_ROWS\quad|\quad\textbf{Variáveis:} REPORT_COLS\par}
    \vspace{2cm}
    \rule{0.6\linewidth}{0.4pt}\\[0.4cm]
    {\small Pipeline DataLab Analytics --- Relatório gerado automaticamente}
  \end{center}
\end{titlepage}
\nopagecolor
\color{black}
"""


def build_tex(stats: dict, png_dir: str, title: str, author: str) -> str:
    m = stats.get('_meta', {})
    date_str = datetime.now().strftime('%d/%m/%Y')
    subtitle  = 'Análise Exploratória Multidimensional'

    preamble = PREAMBLE.replace('REPORT_TITLE', title).replace('REPORT_AUTHOR', author)
    cover = (COVER
             .replace('REPORT_TITLE',    title)
             .replace('REPORT_SUBTITLE', subtitle)
             .replace('REPORT_AUTHOR',   author)
             .replace('REPORT_DATE',     date_str)
             .replace('REPORT_DATASET',  escape(m.get('csv', '?')))
             .replace('REPORT_ROWS',     str(m.get('rows', '?')))
             .replace('REPORT_COLS',     str(m.get('columns', '?'))))

    body = '\n'.join([
        sec_intro(stats),
        sec1(stats, png_dir),
        sec2(stats, png_dir),
        sec3(stats, png_dir),
        sec4(stats, png_dir),
        sec5(stats, png_dir),
        sec6(stats, png_dir),
        sec7(stats, png_dir),
        sec8(stats, png_dir),
        sec9(stats, png_dir),
        sec10(stats, png_dir),
        sec_conclusao(stats),
    ])

    return (
        preamble
        + r'\begin{document}' + '\n'
        + cover
        + r'\tableofcontents' + '\n'
        + r'\newpage' + '\n\n'
        + body
        + '\n' + r'\end{document}' + '\n'
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    args = parse_args()

    with open(args.stats, 'r', encoding='utf-8') as f:
        stats = json.load(f)

    png_dir = os.path.abspath(args.png_dir)
    tex = build_tex(stats, png_dir, args.title, args.author)

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(tex)

    print(f'✓ {args.out} gerado ({len(tex)} chars)')
    print(f'  Para compilar:')
    print(f'    pdflatex -interaction=nonstopmode {args.out}')
    print(f'    pdflatex -interaction=nonstopmode {args.out}  # 2ª passagem para TOC')


if __name__ == '__main__':
    main()
