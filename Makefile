# Build do Barenn (CPU). Testado em Ubuntu 24.04 (WSL2) com g++ 13.3.
CXX      ?= g++
CXXFLAGS ?= -O2 -g -std=c++17
PERF     ?= $(firstword $(wildcard /usr/lib/linux-tools-*/perf) perf)
RUNS     ?= 10

.PHONY: all bench validar perf callgrind cachegrind clean

all: barenn

barenn: barenn.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

# Benchmark com repetições + validação contra a saída do legado.
bench: barenn
	python3 scripts/bench.py ./barenn --runs $(RUNS)

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
	rm -f barenn callgrind.out cachegrind.out perf.data perf.data.old
