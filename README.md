# Barenn: relatório de otimização e migração para CUDA (APS2)

> **Sumário executivo.** O treino da rede 64 → 32 → 16 → 10 (SGD por amostra, 51 épocas, 3 823 imagens) caiu de **0,431 s** (legado) para **0,110 s na CPU (3,9×)** e **0,281 s na GPU**, sempre com saída **bit a bit idêntica** à do código original. A versão CUDA final é **12,6× mais rápida que o port inicial** (3,53 s). Mesmo assim, para o algoritmo exato a CPU vence a GPU, porque o SGD por amostra é uma cadeia serial com poucas centenas de operações por passo (Lei de Amdahl). Um modo opcional de mini-lotes paralelos (`--lote 32`) leva a CPU a **0,047 s (9,2×)**, com acurácia de 96,5 % contra 95,5 %.

## Tabela consolidada

Todas as medições no mesmo nó do cluster **Franky** (compute20: Intel Core i9-14900 + NVIDIA RTX A1000, CUDA 12.8, g++ 14.2), tempo de treino medido pelo próprio programa (`scripts/franky_bench.sh`, 10 execuções, média ± desvio-padrão). "Corretude" compara a saída com a do legado (`tests/referencia_legado.txt`).

| Versão | Tag | Execução | Treino | Speedup vs legado | Corretude |
|:--|:--|:--|--:|--:|:--|
| **Baseline legado** | [`v0.1-baseline`](../../tree/v0.1-baseline) | CPU, 1 núcleo, `-O2` | **0,431 ± 0,001 s** | 1,0× | referência |
| sem alocações | [`v0.2-sem-alocacoes`](../../tree/v0.2-sem-alocacoes) | CPU, 1 núcleo | 0,410 ± 0,001 s | 1,05× | IDENTICA |
| layout contíguo | [`v0.3-layout-contiguo`](../../tree/v0.3-layout-contiguo) | CPU, 1 núcleo | 0,337 ± 0,001 s | 1,28× | IDENTICA |
| SIMD (AVX2) | [`v0.4-simd`](../../tree/v0.4-simd) | CPU, 1 núcleo, AVX2 | 0,111 ± 0,001 s | 3,9× | IDENTICA |
| **CPU otimizada** | [`v0.5-openmp`](../../tree/v0.5-openmp) | CPU, 1 núcleo, AVX2 | **0,110 ± 0,001 s** | **3,9×** | **IDENTICA** |
| CPU otimizada, `--lote 32` | [`v0.5-openmp`](../../tree/v0.5-openmp) | CPU, 8 threads OpenMP | 0,047 ± 0,000 s | 9,2× | outro algoritmo (mini-lotes); acurácia 96,5 % |
| **CUDA inicial** | [`v1.0-cuda-inicial`](../../tree/v1.0-cuda-inicial) | GPU, 12 kernels/amostra | **3,530 ± 0,016 s** | 0,12× | **IDENTICA** |
| kernel persistente | [`v1.1-cuda-persistente`](../../tree/v1.1-cuda-persistente) | GPU, 1 bloco × 512 threads | 0,4995 ± 0,0007 s | 0,86× | IDENTICA |
| memória compartilhada | [`v1.2-cuda-shared`](../../tree/v1.2-cuda-shared) | GPU, 1 bloco × 512 threads | 0,4130 ± 0,0004 s | 1,04× | IDENTICA |
| **CUDA otimizado** | [`v1.3-cuda-warp`](../../tree/v1.3-cuda-warp) | GPU, 1 warp | **0,281 ± 0,010 s** | **1,53×** | **IDENTICA** |

Todas as tags foram recompiladas a partir de um clone limpo e revalidadas no mesmo nó: [`docs/reprodutibilidade.txt`](docs/reprodutibilidade.txt). Os blocos de CPU foram desenvolvidos num notebook (i5-1135G7, WSL2; ver [`docs/ambiente.md`](docs/ambiente.md)), cujas medições, com profiling detalhado, estão em cada issue.

## Índice dos blocos lógicos (issue → PR → tag)

| # | Bloco | Hipótese e técnica | Issue | PR | Tag |
|--:|:--|:--|:--:|:--:|:--|
| 1 | Baseline reprodutível | Makefile, benchmark com repetições, validação, perf/callgrind/cachegrind | [#1](../../issues/1) | [#2](../../pull/2) | [`v0.1-baseline`](../../tree/v0.1-baseline) |
| 2 | Sem alocações por amostra | passagem por referência, buffers reutilizados | [#3](../../issues/3) | [#4](../../pull/4) | [`v0.2-sem-alocacoes`](../../tree/v0.2-sem-alocacoes) |
| 3 | Layout contíguo | matriz row-major contígua, backward em um passe, remoção de trabalho morto | [#5](../../issues/5) | [#6](../../pull/6) | [`v0.3-layout-contiguo`](../../tree/v0.3-layout-contiguo) |
| 4 | Vetorização SIMD | AVX2/AVX-512, dimensões em tempo de compilação, pesos transpostos (sem reassociação) | [#7](../../issues/7) | [#8](../../pull/8) | [`v0.4-simd`](../../tree/v0.4-simd) |
| 5 | Paralelismo multicore | parser `from_chars`, OpenMP em mini-lotes determinísticos, experimentos negativos | [#9](../../issues/9) | [#10](../../pull/10) | [`v0.5-openmp`](../../tree/v0.5-openmp) |
| 6 | Migração para CUDA | port funcional, memória no device, H2D/D2H, `--fmad=false` | [#11](../../issues/11) | [#12](../../pull/12) | [`v1.0-cuda-inicial`](../../tree/v1.0-cuda-inicial) |
| 7 | Kernel persistente | fusão de 12 kernels em 1 lançamento, varredura de threads | [#13](../../issues/13) | [#14](../../pull/14) | [`v1.1-cuda-persistente`](../../tree/v1.1-cuda-persistente) |
| 8 | Memória compartilhada | pesos em shared memory, prefetch em buffer duplo, perfil por fase com `clock64` | [#15](../../issues/15) | [#16](../../pull/16) | [`v1.2-cuda-shared`](../../tree/v1.2-cuda-shared) |
| 9 | Kernel warp-síncrono | atualização adiada fundida ao forward, padding contra conflito de banco, ILP guiado por SASS | [#17](../../issues/17) | [#18](../../pull/18) | [`v1.3-cuda-warp`](../../tree/v1.3-cuda-warp) |

## Principais conclusões

- **Integridade numérica preservada em todo o caminho.** Nenhuma otimização mudou a ordem das operações em ponto flutuante: pesos transpostos em vez de somas reassociadas na CPU, `-ffp-contract=off` e `--fmad=false`, e a mesma ordem de soma por thread na GPU. Otimizações que mudavam o resultado (reassociação, FMA) foram medidas e descartadas (#7).
- **CPU: limitada por instruções, não por memória.** Os pesos (11 KB) cabem no L1 (miss rate de 0,07 %). O ganho veio de reduzir instruções por MAC (de 8,8 para ≈ 1,4) com SIMD. O paralelismo multicore dentro do SGD por amostra **piora** o tempo de 4× a 14× (barreiras de ~1 µs contra ~1,5 µs de trabalho por amostra); só escala em mini-lotes (#9).
- **GPU: limitada por latência e pela cadeia serial.** O port inicial passava 98 % do tempo lançando 2,34 M kernels de ~1 µs (#11). A fusão em um kernel persistente, a memória compartilhada, o prefetch e o kernel warp-síncrono levaram a 1 400 ns por amostra (#13–#17). O SGD exato usa 1 warp de 1 SM (≈ 0,1 % da GPU). O PCIe é irrelevante: 1,8 MB transferidos em 0,3 ms. A criação do contexto CUDA (1,6 s) custa mais que o treino inteiro.
- **Ferramentas:** perf, callgrind, cachegrind e relatórios de vetorização do gcc na CPU; Nsight Systems, `ptxas -v`, SASS (`cuobjdump`) e instrumentação com `clock64` na GPU. O Nsight Compute não pôde ser usado: os contadores de GPU do cluster são restritos ao administrador (`ERR_NVGPUCTRPERM`), conforme documentado em #11.

## Como reproduzir

```bash
git checkout <tag>
make                 # versão de CPU (padrão: -O3 -march=x86-64-v3 -ffp-contract=off -fopenmp)
make bench           # 10 execuções + validação contra a saída do legado
make cuda            # versão CUDA (a partir de v1.0; nvcc -arch=sm_86 --fmad=false)
scripts/franky_bench.sh             # no head-node do Franky: CPU e GPU no mesmo nó (srun)
scripts/franky_nsys.sh <nome>       # perfil Nsight Systems
scripts/test_parser.sh ./barenn     # parser equivalente ao legado (a partir de v0.5)
scripts/test_lote.sh ./barenn       # modo --lote: identidade com lote 1 e determinismo
```

---

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
