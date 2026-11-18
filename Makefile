# Build do Barenn. CPU testada em Ubuntu 24.04 (WSL2, g++ 13.3) e no cluster Franky
# (g++ 14.2); GPU no Franky (CUDA 12.8, RTX A1000). No Franky, compile no head-node.
CXX      ?= g++
# ARCH: nível de ISA x86-64 (portátil entre máquinas, ao contrário de -march=native).
#   x86-64-v3 = AVX2 + FMA (Haswell/Zen 1 em diante)  -> padrão
#   x86-64-v4 = AVX-512 (Skylake-X, Ice/Tiger Lake, Zen 4) -> make ARCH=x86-64-v4
# -ffp-contract=off: sem fusão a*b+c em FMA, para manter o arredondamento do legado
# (o gcc usa -ffp-contract=fast por padrão em C++).
ARCH     ?= x86-64-v3
CXXFLAGS ?= -O3 -g -std=c++17 -march=$(ARCH) -ffp-contract=off -fopenmp
PERF     ?= $(firstword $(wildcard /usr/lib/linux-tools-*/perf) perf)
RUNS     ?= 10
# GPU: --fmad=false impede a fusão a*b+c em FMA, para manter a saída idêntica.
NVCC      ?= $(firstword $(wildcard /usr/local/cuda/bin/nvcc) nvcc)
CUDA_ARCH ?= sm_86
NVCCFLAGS ?= -O3 -std=c++17 -arch=$(CUDA_ARCH) --fmad=false -lineinfo -Xcompiler "-O3 -march=$(ARCH) -ffp-contract=off -fopenmp" -lgomp

.PHONY: all cuda bench bench-cuda validar perf callgrind cachegrind clean

all: barenn

barenn: barenn.cpp barenn.hpp
	$(CXX) $(CXXFLAGS) $< -o $@

cuda: barenn_cuda

barenn_cuda: barenn_cuda.cu barenn.hpp
	$(NVCC) $(NVCCFLAGS) -Xptxas -v $< -o $@

# Benchmark com repetições + validação contra a saída do legado.
bench: barenn
	python3 scripts/bench.py ./barenn --runs $(RUNS)

bench-cuda: barenn_cuda
	python3 scripts/bench.py ./barenn_cuda --runs $(RUNS)

validar: barenn
	python3 scripts/bench.py ./barenn --runs 1 --warmup 0

# Contadores de hardware (instruções, IPC, caches, desvios).
perf: barenn
	$(PERF) stat -r 5 -e cycles,instructions,L1-dcache-loads,L1-dcache-load-misses,cache-references,cache-misses,branch-misses ./barenn > /dev/null

# Custo por função/linha em instruções executadas (Ir).
callgrind: barenn
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out ./barenn > /dev/null
	callgrind_annotate callgrind.out | head -30

# Simulação de cache (D1/LL miss rate).
cachegrind: barenn
	valgrind --tool=cachegrind --cache-sim=yes --cachegrind-out-file=cachegrind.out ./barenn > /dev/null

clean:
	rm -f barenn barenn_cuda callgrind.out cachegrind.out perf.data perf.data.old
