#!/usr/bin/env bash
# Perfil Nsight Systems da versão CUDA no Franky (rode no head-node).
#   scripts/franky_nsys.sh <nome> [argumentos do barenn_cuda]
# Gera nsys/<nome>.nsys-rep (não versionado) e docs/profiling/<nome>/nsys-stats.txt.
set -eu
NOME=${1:?uso: scripts/franky_nsys.sh <nome> [args]}; shift
CUDA_BIN=$(dirname "$(readlink -f /usr/local/cuda/bin/nvcc)")
make -s barenn_cuda > /dev/null
mkdir -p nsys "docs/profiling/$NOME"
srun --partition="${PARTICAO:-pluton_gpu}" --gres=gpu:1 --cpus-per-task=4 --time=00:20:00 bash -c "
  $CUDA_BIN/nsys profile --trace=cuda --force-overwrite true -o nsys/$NOME ./barenn_cuda $* > /dev/null 2>&1
  $CUDA_BIN/nsys stats --report cuda_api_sum,cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,cuda_gpu_mem_size_sum \
      --format table nsys/$NOME.nsys-rep 2>/dev/null | grep -v '^Generating\|^Processing' > docs/profiling/$NOME/nsys-stats.txt
"
echo "docs/profiling/$NOME/nsys-stats.txt"
