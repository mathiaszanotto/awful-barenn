#!/usr/bin/env bash
# Testes do modo --lote (mini-batch paralelo com OpenMP):
#  1) --lote 1 reproduz o SGD por amostra do legado: saída IDENTICA para
#     1, 4 e 8 threads;
#  2) para lote > 1 a saída é determinística: igual para 1 e 8 threads;
#  3) para lote > 1 a acurácia não cai em relação ao legado;
#  4) lotes fora de 1..32 são recusados.
# Uso: scripts/test_lote.sh ./barenn
set -u
BIN=${1:-./barenn}
REF=tests/referencia_legado.txt
DIR=$(mktemp -d)
trap 'rm -rf "$DIR"' EXIT
falhas=0
sem_tempo() { grep -v "^Tempo de treino" "$1"; }
acertos() { sed -nE 's/^Acuracia .* \| ([0-9]+) acertos.*/\1/p' "$1"; }

sem_tempo "$REF" > "$DIR/ref"
for t in 1 4 8; do
    OMP_NUM_THREADS=$t "$BIN" --lote 1 > "$DIR/o" 2>/dev/null
    if sem_tempo "$DIR/o" | cmp -s - "$DIR/ref"; then echo "OK     --lote 1 com $t threads: identica ao legado"
    else echo "FALHA  --lote 1 com $t threads difere do legado"; falhas=$((falhas + 1)); fi
done

base=$(acertos "$REF")
for b in 8 16 32; do
    OMP_NUM_THREADS=1 "$BIN" --lote $b > "$DIR/a" 2>/dev/null
    OMP_NUM_THREADS=8 "$BIN" --lote $b > "$DIR/b" 2>/dev/null
    if cmp -s <(sem_tempo "$DIR/a") <(sem_tempo "$DIR/b"); then
        echo "OK     --lote $b: saida igual com 1 e 8 threads"
    else echo "FALHA  --lote $b: saida depende do numero de threads"; falhas=$((falhas + 1)); fi
    ac=$(acertos "$DIR/a")
    if [ -n "$ac" ] && [ "$ac" -ge "$base" ]; then echo "OK     --lote $b: $ac acertos (legado: $base)"
    else echo "FALHA  --lote $b: $ac acertos < legado $base"; falhas=$((falhas + 1)); fi
done

for b in 0 33 abc; do
    if "$BIN" --lote $b > /dev/null 2>&1; then echo "FALHA  --lote $b aceito"; falhas=$((falhas + 1))
    else echo "OK     --lote $b recusado"; fi
done
exit $falhas
