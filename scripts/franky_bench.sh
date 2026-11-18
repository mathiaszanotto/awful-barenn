#!/usr/bin/env bash
# Reproduz as medições no cluster Franky. Rode no head-node, na raiz do repo
# (os nós de computação não têm git nem os headers para compilar):
#   scripts/franky_bench.sh            # CPU + GPU
#   RUNS_GPU=3 scripts/franky_bench.sh
set -eu
PARTICAO=${PARTICAO:-pluton_gpu}
make -s -B barenn barenn_cuda > /dev/null 2> make.log || { cat make.log; exit 1; }
srun --partition="$PARTICAO" --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 bash -c "
  echo \"no: \$(hostname) | \$(lscpu | grep 'Model name' | sed 's/ \+/ /g') | \$(nvidia-smi --query-gpu=name --format=csv,noheader)\"
  echo '== CPU (padrao, SGD por amostra)'; python3 scripts/bench.py ./barenn --runs ${RUNS:-10}
  echo '== GPU';                            python3 scripts/bench.py ./barenn_cuda --runs ${RUNS_GPU:-5}
  ./barenn_cuda 2>&1 >/dev/null | grep '^\[gpu\]'
"
