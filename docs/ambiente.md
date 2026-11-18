# Ambiente e metodologia de medição

## Hardware e software (CPU)

| Item | Valor |
|:--|:--|
| CPU | Intel Core i5-1135G7 (Tiger Lake), 4 núcleos / 8 threads, base 2,4 GHz, turbo até 4,2 GHz |
| Caches | L1d 48 KB/núcleo, L1i 32 KB/núcleo, L2 1,25 MB/núcleo, L3 8 MB compartilhado |
| SIMD | SSE4.2, AVX2, FMA, AVX-512 |
| SO | Ubuntu 24.04.4 LTS em WSL2 (kernel 6.6.87.2-microsoft-standard-WSL2), Windows 11 |
| Compilador | g++ 13.3.0 |
| Profilers | perf 6.8 (`/usr/lib/linux-tools-6.8.0-149/perf`), valgrind 3.22 (callgrind, cachegrind) |

## Metodologia

- **Métrica principal:** `Tempo de treino`, impresso pelo próprio programa (`std::chrono::steady_clock`), cobre forward, perda e backward das 51 épocas. Não inclui leitura dos arquivos nem a avaliação.
- **Métrica secundária:** tempo de parede do processo inteiro (inclui leitura de `optdigits.tra`/`optdigits.tes` e avaliação).
- **Repetições:** `scripts/bench.py` faz 1 execução de aquecimento e 10 medidas; reporta média ± desvio-padrão, mínimo e mediana.
- **Corretude:** a saída de cada versão é comparada com `tests/referencia_legado.txt` (saída do código original):
  - `IDENTICA`: igual byte a byte (exceto a linha de tempo);
  - `EQUIVALENTE`: perdas impressas com |Δ| ≤ 1e-3 e diferença de acertos ≤ 9 (0,5 p.p.), aceitável apenas para mudanças que alteram a ordem das somas em ponto flutuante (vetorização/FMA/GPU);
  - `DIVERGENTE`: falha; a versão não pode virar tag.
- **Observação sobre o ambiente:** notebook com turbo dinâmico e WSL2, por isso o desvio-padrão é sempre reportado. Os arquivos ficam em `/mnt/c` (9P); isso afeta o tempo de parede, mas não o tempo de treino.

## Como reproduzir

```bash
git checkout <tag>
make            # compila com as flags daquela versão
make bench      # 10 execuções + validação
make perf       # contadores de hardware
make callgrind  # custo por função
scripts/profile_cpu.sh ./barenn docs/profiling/<nome>   # todas as evidências
```
