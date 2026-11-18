#!/usr/bin/env bash
# Coleta as evidências de profiling de CPU de um binário e salva em um diretório.
# Uso: scripts/profile_cpu.sh ./barenn docs/profiling/<bloco>
set -eu   # sem pipefail: "| head" encerra o pipe cedo (SIGPIPE) de propósito
BIN=${1:-./barenn}
OUT=${2:-docs/profiling/atual}
PERF=${PERF:-$(ls /usr/lib/linux-tools-*/perf 2>/dev/null | head -1 || echo perf)}
mkdir -p "$OUT"

echo "[1/4] perf stat (5 repetições)"
"$PERF" stat -r 5 -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses,branch-misses \
    "$BIN" > /dev/null 2> "$OUT/perf-stat.txt"

echo "[2/4] perf record"
"$PERF" record -g -o /tmp/barenn-perf.data "$BIN" > /dev/null 2>&1
"$PERF" report -i /tmp/barenn-perf.data --no-children --stdio --sort symbol 2>/dev/null \
    | grep -E "^ +[0-9.]+%" | sed -E 's/ +- +- *$//' | head -25 > "$OUT/perf-report.txt"

echo "[3/4] callgrind"
valgrind --tool=callgrind --callgrind-out-file=/tmp/barenn-cg.out "$BIN" > /dev/null 2>/dev/null
callgrind_annotate --auto=yes /tmp/barenn-cg.out barenn.cpp 2>/dev/null > /tmp/barenn-cg-ann.txt
{
    sed -n '/PROGRAM TOTALS/,/^-- Auto-annotated/p' /tmp/barenn-cg-ann.txt | head -30
    echo
    echo "== Linhas de barenn.cpp com mais de 1% das instruções =="
    awk '/[Aa]nnotated source: .*barenn.cpp/{f=1} f' /tmp/barenn-cg-ann.txt \
        | grep -E "\( *([1-9][0-9]?\.[0-9]+)%\)" || true
} > "$OUT/callgrind.txt"

echo "[4/4] cachegrind"
valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=/tmp/barenn-cachegrind.out "$BIN" > /dev/null 2> /tmp/barenn-cache.err
grep -E "refs|misses|miss rate" /tmp/barenn-cache.err | sed -E 's/^==[0-9]+== //' > "$OUT/cachegrind.txt"

echo "evidências salvas em $OUT"
