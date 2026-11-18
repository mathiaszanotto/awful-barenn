// Barenn em GPU (CUDA): treino por SGD amostra a amostra, mesmo algoritmo e
// mesma saída da versão de CPU. Leitura, inicialização e avaliação vêm de
// barenn.hpp; apenas o laço de treino roda no device.
//
// Compilado com --fmad=false: sem contração a*b+c em FMA. Soma e produto em
// float são IEEE-754 com arredondamento ao mais próximo também na GPU, e cada
// thread soma as parcelas na mesma ordem da CPU, então os resultados são
// bit a bit iguais aos do legado.
#include "barenn.hpp"

#include <cuda_runtime.h>

#define CUDA_CHECK(chamada)                                                    \
    do {                                                                       \
        const cudaError_t erro_cuda = (chamada);                               \
        if (erro_cuda != cudaSuccess) {                                        \
            throw runtime_error(string("CUDA: ") +                             \
                                cudaGetErrorString(erro_cuda) + " em " #chamada \
                                " (" __FILE__ ":" + to_string(__LINE__) + ")"); \
        }                                                                      \
    } while (0)

namespace BarennGPU {

using Barenn::Amostra;
using Barenn::classes;
using Barenn::entradas;

// ---------------------------------------------------------------------------
// Kernel persistente warp-síncrono com atualização adiada.
//
// UM lançamento executa todas as épocas e amostras, e o SGD por amostra
// continua sequencial. Em vez de uma fase separada de atualização dos pesos
// (seguida de barreira), a atualização da amostra n é ADIADA e aplicada no
// forward da amostra n+1: no forward da camada l, a thread i é a única que lê
// a coluna i de Wl, então ela aplica
//     W[i][j] -= lr * (d_n[i] * x_n[j]),   b[i] -= lr * d_n[i]
// imediatamente antes de usar cada peso. Os valores são exatamente os de
// "atualizar tudo e depois fazer o forward". O gradiente da entrada da
// amostra n+1 usa W já com a atualização de n e sem a de n+1, como no
// legado. A atualização da última amostra é aplicada no fim do treino.
//
// Nenhuma fase usa mais de 32 threads, então o kernel roda com UM warp e as
// barreiras são __syncwarp(). Tudo (pesos, ativações, imagens) fica na
// memória compartilhada. Mesmas operações e mesma ordem da CPU com
// --fmad=false: saída idêntica ao legado.
// ---------------------------------------------------------------------------

constexpr int threads_treino = 32;  // um warp
// Stride de linha de W2 na memória compartilhada. Com 16, as 32 threads do
// grad_in 2 (thread j lê W2[j*16 + i]) caem em 2 dos 32 bancos: conflito de
// 16 vias. Com 17, cada thread cai num banco diferente.
constexpr int ld2 = 17;

struct ParametrosGPU {
    float* W1; float* b1;  // 64 -> 32, Wt[j * 32 + i]
    float* W2; float* b2;  // 32 -> 16, Wt[j * 16 + i]
    float* W3; float* b3;  // 16 -> 10, Wt[j * 10 + i]
    unsigned long long* perfil;  // ciclos por fase (só com -DPERFIL_FASES)
};

// Perfil por fase sem Nsight Compute (indisponível no cluster): a thread 0
// lê o contador de ciclos do SM (clock64) logo após cada barreira e acumula
// a duração de cada fase.
constexpr int num_fases = 5;
#ifdef PERFIL_FASES
#define MARCA_FASE(f)                                   \
    if (threadIdx.x == 0) {                             \
        const long long agora = clock64();              \
        ciclos[f] += agora - marca;                     \
        marca = agora;                                  \
    }
#else
#define MARCA_FASE(f)
#endif

// Forward do neurônio i com a atualização pendente da sua coluna embutida.
// Wt tem stride de linha LD; xp/dp são a entrada e o delta da amostra anterior.
template <int IN, int OUT, int LD, bool FINAL>
__device__ __forceinline__ float forward_neuronio(int i, bool pendente, float lr,
                                                  float* Wt, float* b, const float* x,
                                                  const float* xp, const float* dp) {
    float acc;
    if (pendente) {
        const float di = dp[i];
        const float bi = b[i] - lr * di;
        b[i] = bi;
        acc = bi;
        for (int j = 0; j < IN; ++j) {
            const float w = Wt[j * LD + i] - lr * (di * xp[j]);
            Wt[j * LD + i] = w;
            acc += x[j] * w;
        }
    } else {
        acc = b[i];
        for (int j = 0; j < IN; ++j) {
            acc += x[j] * Wt[j * LD + i];
        }
    }
    return acc;
}

// Atualização pendente de uma coluna inteira (usada só no fim do treino).
template <int IN, int LD>
__device__ __forceinline__ void aplica_pendente(int i, float lr, float* Wt, float* b,
                                                const float* xp, const float* dp) {
    const float di = dp[i];
    b[i] -= lr * di;
    for (int j = 0; j < IN; ++j) {
        Wt[j * LD + i] -= lr * (di * xp[j]);
    }
}

__global__ void __launch_bounds__(threads_treino)
k_treino(const float* __restrict__ imagens, const int* __restrict__ rotulos, int n,
         int epocas, float lr, ParametrosGPU p, float* __restrict__ perdas) {
    const int t = threadIdx.x;
    __shared__ float W1[64 * 32], b1[32];
    __shared__ float W2[32 * ld2], b2[16];
    __shared__ float W3[16 * classes], b3[classes];
    __shared__ float xs[3][entradas];  // imagens: anterior / atual / próxima
    __shared__ int rs[3];
    __shared__ float z1[32], a1[2][32], d1[32];  // ativações em buffer duplo
    __shared__ float z2[16], a2[2][16], d2[16];
    __shared__ float d3[classes], s[classes];

    for (int q = t; q < 64 * 32; q += threads_treino) W1[q] = p.W1[q];
    for (int q = t; q < 32 * 16; q += threads_treino) W2[(q / 16) * ld2 + q % 16] = p.W2[q];
    for (int q = t; q < 16 * classes; q += threads_treino) W3[q] = p.W3[q];
    b1[t] = p.b1[t];
    if (t < 16) b2[t] = p.b2[t];
    if (t < classes) b3[t] = p.b3[t];
    xs[0][t] = imagens[t];
    xs[0][t + 32] = imagens[t + 32];
    if (t == 0) rs[0] = rotulos[0];
    __syncwarp();
#ifdef PERFIL_FASES
    unsigned long long ciclos[num_fases] = {};
    long long marca = clock64();
#endif

    bool pendente = false;  // há atualização da amostra anterior a aplicar?
    int atual = 0;          // índice da imagem atual no buffer triplo
    int par = 0;            // paridade das ativações a1/a2
    for (int epoca = 0; epoca < epocas; ++epoca) {
        for (int k = 0; k < n; ++k) {
            const int anterior = (atual + 2) % 3, proxima = (atual + 1) % 3;
            // Prefetch da próxima amostra em registradores: a latência da
            // memória global fica escondida pelo resto da amostra.
            const int kp = (k + 1 < n) ? k + 1 : 0;
            const float pf0 = imagens[static_cast<size_t>(kp) * entradas + t];
            const float pf1 = imagens[static_cast<size_t>(kp) * entradas + t + 32];
            const int prot = rotulos[kp];
            const float* x = xs[atual];
            const float* xp = xs[anterior];

            // Fase 0: forward 1 (32 neurônios) com a atualização pendente de W1.
            {
                const float acc = forward_neuronio<64, 32, 32, false>(t, pendente, lr, W1, b1, x, xp, d1);
                z1[t] = acc;
                a1[par][t] = 0.0f < acc ? acc : 0.0f;
            }
            __syncwarp();
            MARCA_FASE(0)
            // Fase 1: forward 2 (16 neurônios) com a atualização pendente de W2.
            if (t < 16) {
                const float acc =
                    forward_neuronio<32, 16, ld2, false>(t, pendente, lr, W2, b2, a1[par], a1[par ^ 1], d2);
                z2[t] = acc;
                a2[par][t] = 0.0f < acc ? acc : 0.0f;
            }
            __syncwarp();
            MARCA_FASE(1)
            // Fase 2: forward 3 (10 neurônios) com a atualização pendente de W3;
            // o delta da última camada é o erro (gradiente da perda MSE/2).
            if (t < classes) {
                const float acc =
                    forward_neuronio<16, classes, classes, true>(t, pendente, lr, W3, b3, a2[par], a2[par ^ 1], d3);
                s[t] = acc;
                d3[t] = acc - (t == rs[atual] ? 1.0f : 0.0f);
            }
            __syncwarp();
            MARCA_FASE(2)
            // Fase 3: grad_in 3 -> delta 2 (16 threads); a thread 31, ociosa,
            // soma a perda da amostra na ordem i = 0..9, como na CPU.
            if (t < 16) {
                float sum = 0.0f;
                for (int i = 0; i < classes; ++i) {
                    sum += d3[i] * W3[t * classes + i];
                }
                d2[t] = sum * (z2[t] > 0.0f ? 1.0f : 0.0f);
            } else if (t == 31) {
                const int rotulo = rs[atual];
                float total = 0;
                for (int i = 0; i < classes; ++i) {
                    const float erro = s[i] - (i == rotulo ? 1.0f : 0.0f);
                    total += 0.5f * erro * erro;
                }
                perdas[static_cast<size_t>(epoca) * n + k] = total;
            }
            __syncwarp();
            MARCA_FASE(3)
            // Fase 4: grad_in 2 -> delta 1 (32 threads, sem conflito de banco
            // graças ao stride 17) e gravação da próxima imagem no buffer.
            {
                float sum = 0.0f;
                for (int i = 0; i < 16; ++i) {
                    sum += d2[i] * W2[t * ld2 + i];
                }
                d1[t] = sum * (z1[t] > 0.0f ? 1.0f : 0.0f);
                xs[proxima][t] = pf0;
                xs[proxima][t + 32] = pf1;
                if (t == 0) rs[proxima] = prot;
            }
            __syncwarp();
            MARCA_FASE(4)
            pendente = true;
            atual = proxima;
            par ^= 1;
        }
    }

    // Aplica a atualização da última amostra e devolve os parâmetros.
    if (pendente) {
        const float* xp = xs[(atual + 2) % 3];
        aplica_pendente<64, 32>(t, lr, W1, b1, xp, d1);
        if (t < 16) aplica_pendente<32, ld2>(t, lr, W2, b2, a1[par ^ 1], d2);
        if (t < classes) aplica_pendente<16, classes>(t, lr, W3, b3, a2[par ^ 1], d3);
    }
    __syncwarp();
    for (int q = t; q < 64 * 32; q += threads_treino) p.W1[q] = W1[q];
    for (int q = t; q < 32 * 16; q += threads_treino) p.W2[q] = W2[(q / 16) * ld2 + q % 16];
    for (int q = t; q < 16 * classes; q += threads_treino) p.W3[q] = W3[q];
    p.b1[t] = b1[t];
    if (t < 16) p.b2[t] = b2[t];
    if (t < classes) p.b3[t] = b3[t];
#ifdef PERFIL_FASES
    if (t == 0) {
        for (int f = 0; f < num_fases; ++f) {
            p.perfil[f] = ciclos[f];
        }
    }
#endif
}

// ---------------------------------------------------------------------------
// Memória no device
// ---------------------------------------------------------------------------

// Buffer no device liberado automaticamente (RAII).
template <typename T>
class BufferGPU {
    T* ptr = nullptr;
    size_t n = 0;

public:
    explicit BufferGPU(size_t elementos) : n(elementos) {
        CUDA_CHECK(cudaMalloc(&ptr, n * sizeof(T)));
    }
    ~BufferGPU() { cudaFree(ptr); }
    BufferGPU(const BufferGPU&) = delete;
    BufferGPU& operator=(const BufferGPU&) = delete;
    T* get() const { return ptr; }
    void enviar(const T* origem) { CUDA_CHECK(cudaMemcpy(ptr, origem, n * sizeof(T), cudaMemcpyHostToDevice)); }
    void receber(T* destino) const { CUDA_CHECK(cudaMemcpy(destino, ptr, n * sizeof(T), cudaMemcpyDeviceToHost)); }
};

template <class Camada>
struct CamadaGPU {
    static constexpr int IN = Camada::entradas_camada;
    static constexpr int OUT = Camada::saidas_camada;
    BufferGPU<float> W{static_cast<size_t>(IN) * OUT};
    BufferGPU<float> b{static_cast<size_t>(OUT)};

    void enviar(const Camada& c) {
        W.enviar(c.pesos());
        b.enviar(c.vieses());
    }
    void receber(Camada& c) const {
        W.receber(c.pesos());
        b.receber(c.vieses());
    }
};

struct Tempos {
    double h2d_ms = 0, kernels_ms = 0, d2h_ms = 0;
    long lancamentos = 0;
    unsigned long long ciclos_fase[num_fases] = {};
};

// Treina as três camadas na GPU (SGD por amostra, kernel persistente) e
// devolve os pesos treinados nas camadas do host. perdas[e] recebe a perda
// média da época e, somada em double no host na ordem das amostras.
template <size_t EPOCAS>
void treinar(Barenn::Camada1& l1, Barenn::Camada2& l2, Barenn::Camada3& l3,
             const vector<Amostra>& treino, float taxa, array<double, EPOCAS>& perdas,
             Tempos& tempos) {
    const int n = static_cast<int>(treino.size());
    // Conjunto de treino contíguo: imagens n x 64 e rótulos.
    vector<float> imagens(static_cast<size_t>(n) * entradas);
    vector<int> rotulos(n);
    for (int k = 0; k < n; ++k) {
        copy(treino[k].imagem.begin(), treino[k].imagem.end(), &imagens[static_cast<size_t>(k) * entradas]);
        rotulos[k] = treino[k].rotulo;
    }

    BufferGPU<float> d_imagens(imagens.size());
    BufferGPU<int> d_rotulos(rotulos.size());
    BufferGPU<float> d_perdas(EPOCAS * static_cast<size_t>(n));
    CamadaGPU<Barenn::Camada1> g1;
    CamadaGPU<Barenn::Camada2> g2;
    CamadaGPU<Barenn::Camada3> g3;

    cudaEvent_t ev[4];
    for (auto& e : ev) {
        CUDA_CHECK(cudaEventCreate(&e));
    }

    // Host -> Device: dados e parâmetros iniciais, uma única vez.
    CUDA_CHECK(cudaEventRecord(ev[0]));
    d_imagens.enviar(imagens.data());
    d_rotulos.enviar(rotulos.data());
    g1.enviar(l1);
    g2.enviar(l2);
    g3.enviar(l3);
    CUDA_CHECK(cudaEventRecord(ev[1]));

    BufferGPU<unsigned long long> d_perfil(num_fases);
    // Um único lançamento: um bloco percorre todas as épocas e amostras.
    const ParametrosGPU p{g1.W.get(), g1.b.get(), g2.W.get(), g2.b.get(),
                          g3.W.get(), g3.b.get(), d_perfil.get()};
    k_treino<<<1, threads_treino>>>(d_imagens.get(), d_rotulos.get(), n,
                                    static_cast<int>(EPOCAS), taxa, p, d_perdas.get());
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev[2]));

    // Device -> Host: perdas de cada amostra e parâmetros treinados.
    vector<float> perda_amostra(EPOCAS * static_cast<size_t>(n));
    d_perdas.receber(perda_amostra.data());
    g1.receber(l1);
    g2.receber(l2);
    g3.receber(l3);
#ifdef PERFIL_FASES
    d_perfil.receber(tempos.ciclos_fase);
#endif
    CUDA_CHECK(cudaEventRecord(ev[3]));
    CUDA_CHECK(cudaEventSynchronize(ev[3]));

    for (size_t epoca = 0; epoca < EPOCAS; ++epoca) {
        double perda_total = 0;
        for (int k = 0; k < n; ++k) {
            perda_total += perda_amostra[epoca * n + k];
        }
        perdas[epoca] = perda_total / n;
    }

    float ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev[0], ev[1]));
    tempos.h2d_ms = ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev[1], ev[2]));
    tempos.kernels_ms = ms;
    CUDA_CHECK(cudaEventElapsedTime(&ms, ev[2], ev[3]));
    tempos.d2h_ms = ms;
    tempos.lancamentos = 1;
    for (auto& e : ev) {
        cudaEventDestroy(e);
    }
}

} // namespace BarennGPU

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        using relogio = chrono::steady_clock;
        const auto t_leitura = relogio::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        const chrono::duration<double> tempo_leitura = relogio::now() - t_leitura;
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        // Cria o contexto CUDA fora do cronômetro de treino (custo único de
        // inicialização do driver, reportado à parte).
        const auto t_contexto = relogio::now();
        CUDA_CHECK(cudaFree(nullptr));
        const chrono::duration<double> tempo_contexto = relogio::now() - t_contexto;
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10 (pesos iniciais no host,
        // mesmo gerador da versão de CPU).
        Barenn::Camada1 l1;
        Barenn::Camada2 l2;
        Barenn::Camada3 l3;
        array<double, epocas> perdas{};
        BarennGPU::Tempos tempos;

        // A medição inclui as cópias Host<->Device e todo o treino na GPU.
        const auto inicio = relogio::now();
        BarennGPU::treinar(l1, l2, l3, treino, taxa, perdas, tempos);
        const chrono::duration<double> tempo = relogio::now() - inicio;

        cout << fixed << setprecision(4)
             << "TREINO | " << treino.size() << " imagens\n";
        for (int epoca = 0; epoca < epocas; epoca += 5) {
            cout << "Epoca " << setw(2) << epoca + 1 << "/" << epocas
                 << " | perda media " << perdas[epoca] << "\n";
        }
        cout << "Tempo de treino " << tempo.count() << " s\n";
        const auto t_avaliacao = relogio::now();
        Barenn::avaliar(l1, l2, l3, teste);
        cout.flush();
        const chrono::duration<double> tempo_avaliacao = relogio::now() - t_avaliacao;
        cerr << fixed << setprecision(4) << "[fases] leitura " << tempo_leitura.count()
             << " s | treino " << tempo.count() << " s | avaliacao "
             << tempo_avaliacao.count() << " s\n";
        cerr << "[gpu] " << prop.name << " (" << prop.multiProcessorCount << " SMs, cc "
             << prop.major << "." << prop.minor << ") | contexto " << tempo_contexto.count()
             << " s | H2D " << tempos.h2d_ms << " ms | kernels " << tempos.kernels_ms
             << " ms | D2H " << tempos.d2h_ms << " ms | lancamentos " << tempos.lancamentos
             << " | threads " << BarennGPU::threads_treino << " | " << setprecision(3)
             << 1000.0 * tempos.kernels_ms / (static_cast<double>(epocas) * treino.size())
             << " us/amostra\n";
#ifdef PERFIL_FASES
        // Perfil por fase (clock64 da thread 0), em ciclos por amostra.
        const char* nomes[BarennGPU::num_fases] = {
            "atualiza W1 + forward 1", "atualiza W2 + forward 2", "atualiza W3 + forward 3",
            "grad_in 3 + perda", "grad_in 2 + prefetch"};
        double total = 0;
        for (auto c : tempos.ciclos_fase) {
            total += static_cast<double>(c);
        }
        const double amostras = static_cast<double>(epocas) * treino.size();
        cerr << "[perfil] " << setprecision(0) << total / amostras << " ciclos/amostra | clock efetivo "
             << setprecision(3) << total / (tempos.kernels_ms * 1e6) << " GHz\n";
        for (int f = 0; f < BarennGPU::num_fases; ++f) {
            cerr << "[perfil] " << left << setw(26) << nomes[f] << right << setw(8) << setprecision(0)
                 << tempos.ciclos_fase[f] / amostras << " ciclos/amostra (" << setprecision(1)
                 << 100.0 * tempos.ciclos_fase[f] / total << " %)\n";
        }
#endif
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
