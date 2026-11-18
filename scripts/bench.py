#!/usr/bin/env python3
"""Benchmark e validação de corretude do Barenn.

Executa o binário N vezes, extrai o "Tempo de treino" impresso pelo próprio
programa (cronômetro interno, cobre só o laço de treino) e o tempo de parede
total do processo (inclui leitura dos arquivos e avaliação). Ao final compara
a saída da primeira execução com a saída de referência do código legado.

Uso:
    python3 scripts/bench.py ./barenn [--runs 10] [--ref tests/referencia_legado.txt]
"""
import argparse
import re
import statistics
import subprocess
import sys
import time

TEMPO_RE = re.compile(r"Tempo de treino ([0-9.]+) s")
PERDA_RE = re.compile(r"Epoca +(\d+)/\d+ \| perda media ([0-9.]+)")
ACC_RE = re.compile(r"Acuracia ([0-9.]+)% \| (\d+) acertos")
PERDA_TESTE_RE = re.compile(r"Perda media ([0-9.]+)")
# Cronômetros por fase impressos em stderr (a partir de v0.5).
FASES_RE = re.compile(r"\[fases\] leitura ([0-9.]+) s \| treino [0-9.]+ s \| avaliacao ([0-9.]+) s")

# Tolerâncias usadas quando a saída não é idêntica byte a byte (por exemplo,
# quando uma otimização muda a ordem das somas em ponto flutuante).
TOL_PERDA = 1e-3      # diferença absoluta máxima nas perdas impressas
TOL_ACERTOS = 9       # diferença máxima de acertos no teste (0,5 p.p. de 1797)


def sem_tempo(saida):
    return "\n".join(l for l in saida.splitlines() if not l.startswith("Tempo de treino"))


def comparar(saida, referencia):
    """Retorna (status, detalhes). status: IDENTICA, EQUIVALENTE ou DIVERGENTE."""
    a, b = sem_tempo(saida), sem_tempo(referencia)
    if a == b:
        return "IDENTICA", "saída idêntica à referência (exceto a linha de tempo)"
    det = []
    ok = True
    pa, pb = dict(PERDA_RE.findall(a)), dict(PERDA_RE.findall(b))
    if pa.keys() != pb.keys():
        return "DIVERGENTE", "épocas impressas diferentes"
    dmax = max(abs(float(pa[k]) - float(pb[k])) for k in pa)
    det.append(f"max |Δperda treino| = {dmax:.4f}")
    ok &= dmax <= TOL_PERDA
    ma, mb = ACC_RE.search(a), ACC_RE.search(b)
    if not ma or not mb:
        return "DIVERGENTE", "acurácia ausente na saída"
    dac = abs(int(ma.group(2)) - int(mb.group(2)))
    det.append(f"acertos {ma.group(2)} vs {mb.group(2)} (Δ={dac})")
    ok &= dac <= TOL_ACERTOS
    ta, tb = PERDA_TESTE_RE.search(a), PERDA_TESTE_RE.search(b)
    dpt = abs(float(ta.group(1)) - float(tb.group(1)))
    det.append(f"|Δperda teste| = {dpt:.4f}")
    ok &= dpt <= TOL_PERDA
    return ("EQUIVALENTE" if ok else "DIVERGENTE"), "; ".join(det)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("binario")
    ap.add_argument("--runs", type=int, default=10)
    ap.add_argument("--ref", default="tests/referencia_legado.txt")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("args", nargs="*", help="argumentos repassados ao binário")
    op = ap.parse_args()

    cmd = [op.binario] + op.args
    for _ in range(op.warmup):
        subprocess.run(cmd, capture_output=True, check=True)

    treinos, paredes, leituras, avaliacoes, primeira = [], [], [], [], None
    for _ in range(op.runs):
        t0 = time.perf_counter()
        r = subprocess.run(cmd, capture_output=True, text=True, check=True)
        paredes.append(time.perf_counter() - t0)
        m = TEMPO_RE.search(r.stdout)
        if not m:
            sys.exit("linha 'Tempo de treino' não encontrada na saída")
        treinos.append(float(m.group(1)))
        f = FASES_RE.search(r.stderr)
        if f:
            leituras.append(float(f.group(1)))
            avaliacoes.append(float(f.group(2)))
        primeira = primeira or r.stdout

    def resumo(v):
        dp = statistics.stdev(v) if len(v) > 1 else 0.0
        return statistics.mean(v), dp, min(v), statistics.median(v)

    mt, dt, nt, medt = resumo(treinos)
    mp, dp, np_, medp = resumo(paredes)
    print(f"binario: {' '.join(cmd)} | execucoes: {op.runs} (+{op.warmup} aquecimento)")
    print(f"treino : media {mt:.4f} s | desvio {dt:.4f} s | min {nt:.4f} s | mediana {medt:.4f} s")
    print(f"parede : media {mp:.4f} s | desvio {dp:.4f} s | min {np_:.4f} s | mediana {medp:.4f} s")
    if leituras:
        ml, dl, nl, _ = resumo(leituras)
        ma, da, na, _ = resumo(avaliacoes)
        print(f"leitura: media {ml:.4f} s | desvio {dl:.4f} s | min {nl:.4f} s")
        print(f"avalia.: media {ma:.4f} s | desvio {da:.4f} s | min {na:.4f} s")

    with open(op.ref, encoding="utf-8") as f:
        status, det = comparar(primeira, f.read())
    print(f"corretude: {status} — {det}")
    print(f"| {mt:.4f} ± {dt:.4f} | {nt:.4f} | {mp:.4f} ± {dp:.4f} | {status} |")
    return 0 if status != "DIVERGENTE" else 2


if __name__ == "__main__":
    sys.exit(main())
