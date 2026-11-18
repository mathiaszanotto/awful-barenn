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
// Kernel persistente: UM lançamento executa todas as épocas e amostras.
//
// O SGD por amostra é sequencial (a amostra n+1 usa os pesos atualizados pela
// n), então um único bloco de T threads percorre as amostras em ordem. As
// etapas de cada amostra são fases separadas por __syncthreads(), uma barreira
// de bloco de dezenas de ciclos, no lugar de ~1,5 us por lançamento de kernel.
// Ativações e deltas ficam em memória compartilhada; os pesos ficam em memória
// global (Wt[j * OUT + i], o layout transposto da CPU).
//
// Cada elemento é calculado por uma única thread, com a mesma ordem de soma e
// as mesmas expressões da CPU: a saída é idêntica ao legado (--fmad=false).
// ---------------------------------------------------------------------------

#ifndef THREADS_TREINO
#define THREADS_TREINO 512  // melhor valor medido entre 32 e 1024 (issue #13)
#endif

// y[i] = ativação(b[i] + soma_j x[j] * W[i][j]), uma thread por neurônio i.
// Threads consecutivas leem Wt[j * OUT + i] consecutivos (coalescido).
template <int IN, int OUT, bool FINAL, int T>
__device__ __forceinline__ void fase_forward(const float* x, const float* Wt, const float* b,
                                             float* z, float* y) {
    for (int i = threadIdx.x; i < OUT; i += T) {
        float acc = b[i];
        for (int j = 0; j < IN; ++j) {
            acc += x[j] * Wt[j * OUT + i];
        }
        z[i] = acc;
        // Mesma expressão de std::max(0.0f, acc) para preservar o sinal de zero.
        y[i] = FINAL ? acc : (0.0f < acc ? acc : 0.0f);
    }
}

// Gradiente da entrada (pesos ainda NÃO atualizados) e, em seguida, o delta
// da camada anterior: d_ant[j] = (soma_i d[i] * W[i][j]) * relu'(z_ant[j]).
template <int IN, int OUT, int T>
__device__ __forceinline__ void fase_grad_in(const float* d, const float* Wt,
                                             const float* z_ant, float* d_ant) {
    for (int j = threadIdx.x; j < IN; j += T) {
        float sum = 0.0f;
        for (int i = 0; i < OUT; ++i) {
            sum += d[i] * Wt[j * OUT + i];
        }
        d_ant[j] = sum * (z_ant[j] > 0.0f ? 1.0f : 0.0f);
    }
}

// SGD: W[i][j] -= lr * (d[i] * x[j]); b[i] -= lr * d[i]. Threads consecutivas
// atualizam pesos consecutivos (coalescido).
template <int IN, int OUT, int T>
__device__ __forceinline__ void fase_atualiza(const float* d, const float* x, float* Wt,
                                              float* b, float lr) {
    for (int idx = threadIdx.x; idx < IN * OUT; idx += T) {
        const int j = idx / OUT;
        const int i = idx - j * OUT;
        Wt[idx] -= lr * (d[i] * x[j]);
    }
    for (int i = threadIdx.x; i < OUT; i += T) {
        b[i] -= lr * d[i];
    }
}

// Cópia cooperativa e coalescida: thread t copia os elementos t, t+T, ...
template <int T>
__device__ __forceinline__ void copia_bloco(const float* origem, float* destino, int n) {
    for (int i = threadIdx.x; i < n; i += T) {
        destino[i] = origem[i];
    }
}

struct ParametrosGPU {
    float* W1; float* b1;  // 64 -> 32
    float* W2; float* b2;  // 32 -> 16
    float* W3; float* b3;  // 16 -> 10
    unsigned long long* perfil;  // ciclos por fase (só com -DPERFIL_FASES)
};

// Perfil por fase sem Nsight Compute (indisponível no cluster): a thread 0
// lê o contador de ciclos do SM (clock64) logo após cada barreira e acumula
// a duração de cada fase, que inclui a espera pela thread mais lenta.
constexpr int num_fases = 7;
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

template <int T>
__global__ void __launch_bounds__(T)
k_treino(const float* __restrict__ imagens, const int* __restrict__ rotulos, int n,
         int epocas, float lr, ParametrosGPU p, float* __restrict__ perdas) {
    __shared__ float z1[32], a1[32], d1[32];
    __shared__ float z2[16], a2[16], d2[16];
    __shared__ float z3[classes], s[classes], d3[classes];
    // Todos os parâmetros (11,1 KB) ficam na memória compartilhada durante o
    // treino inteiro: o bloco é o único que os usa, então não há coerência a
    // manter com outros blocos. Leituras e escritas custam ~20-30 ciclos, sem
    // passar pela L1/L2 (onde cada escrita invalidava a linha na L1).
    __shared__ float W1[64 * 32], b1[32];
    __shared__ float W2[32 * 16], b2[16];
    __shared__ float W3[16 * classes], b3[classes];
    copia_bloco<T>(p.W1, W1, 64 * 32);
    copia_bloco<T>(p.b1, b1, 32);
    copia_bloco<T>(p.W2, W2, 32 * 16);
    copia_bloco<T>(p.b2, b2, 16);
    copia_bloco<T>(p.W3, W3, 16 * classes);
    copia_bloco<T>(p.b3, b3, classes);
    __syncthreads();
#ifdef PERFIL_FASES
    unsigned long long ciclos[num_fases] = {};
    long long marca = clock64();
#endif

    for (int epoca = 0; epoca < epocas; ++epoca) {
        for (int k = 0; k < n; ++k) {
            const float* x = imagens + static_cast<size_t>(k) * entradas;
            fase_forward<64, 32, false, T>(x, W1, b1, z1, a1);
            __syncthreads();
            MARCA_FASE(0)
            fase_forward<32, 16, false, T>(a1, W2, b2, z2, a2);
            __syncthreads();
            MARCA_FASE(1)
            fase_forward<16, classes, true, T>(a2, W3, b3, z3, s);
            __syncthreads();
            MARCA_FASE(2)
            // Perda MSE/2 e delta da última camada (gradiente dos escores), na
            // ordem i = 0..9 como na CPU: uma thread.
            if (threadIdx.x == 0) {
                const int rotulo = rotulos[k];
                float total = 0;
                for (int i = 0; i < classes; ++i) {
                    const float erro = s[i] - (i == rotulo ? 1.0f : 0.0f);
                    total += 0.5f * erro * erro;
                    d3[i] = erro;
                }
                perdas[static_cast<size_t>(epoca) * n + k] = total;
            }
            __syncthreads();
            MARCA_FASE(3)
            // Camada 3: gradiente da entrada antes de atualizar W3.
            fase_grad_in<16, classes, T>(d3, W3, z2, d2);
            __syncthreads();
            MARCA_FASE(4)
            // W3 e o gradiente da camada 2 (que só lê W2) não conflitam.
            fase_atualiza<16, classes, T>(d3, a2, W3, b3, lr);
            fase_grad_in<32, 16, T>(d2, W2, z1, d1);
            __syncthreads();
            MARCA_FASE(5)
            fase_atualiza<32, 16, T>(d2, a1, W2, b2, lr);
            fase_atualiza<64, 32, T>(d1, x, W1, b1, lr);
            // Os pesos precisam estar atualizados antes do forward seguinte.
            __syncthreads();
            MARCA_FASE(6)
        }
    }
#ifdef PERFIL_FASES
    if (threadIdx.x == 0) {
        for (int f = 0; f < num_fases; ++f) {
            p.perfil[f] = ciclos[f];
        }
    }
#endif
    // Parâmetros treinados de volta à memória global (para a cópia D2H).
    copia_bloco<T>(W1, p.W1, 64 * 32);
    copia_bloco<T>(b1, p.b1, 32);
    copia_bloco<T>(W2, p.W2, 32 * 16);
    copia_bloco<T>(b2, p.b2, 16);
    copia_bloco<T>(W3, p.W3, 16 * classes);
    copia_bloco<T>(b3, p.b3, classes);
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
    k_treino<THREADS_TREINO><<<1, THREADS_TREINO>>>(d_imagens.get(), d_rotulos.get(), n,
                                                    static_cast<int>(EPOCAS), taxa, p,
                                                    d_perdas.get());
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
             << " | threads " << THREADS_TREINO << " | " << setprecision(3)
             << 1000.0 * tempos.kernels_ms / (static_cast<double>(epocas) * treino.size())
             << " us/amostra\n";
#ifdef PERFIL_FASES
        // Perfil por fase (clock64 da thread 0), em ciclos por amostra.
        const char* nomes[BarennGPU::num_fases] = {
            "forward 1 (64->32)", "forward 2 (32->16)", "forward 3 (16->10)", "perda + delta 3",
            "grad_in 3 + delta 2", "atualiza W3 + grad_in 2", "atualiza W2 + W1"};
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
