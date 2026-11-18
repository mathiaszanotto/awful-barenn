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
// Kernels: uma thread por neurônio (forward/delta), por entrada (grad_in) ou
// por peso (atualização). Os pesos usam o mesmo layout transposto da CPU,
// Wt[j * OUT + i]; no forward, threads consecutivas (neurônios i
// consecutivos) leem endereços consecutivos: acesso coalescido.
// ---------------------------------------------------------------------------

template <int IN, int OUT, bool FINAL>
__global__ void k_forward(const float* __restrict__ x, const float* __restrict__ Wt,
                          const float* __restrict__ b, float* __restrict__ z,
                          float* __restrict__ y) {
    const int i = threadIdx.x;
    if (i >= OUT) {
        return;
    }
    // Viés, depois x[0]*w, x[1]*w, ...: a mesma ordem da CPU.
    float acc = b[i];
    for (int j = 0; j < IN; ++j) {
        acc += x[j] * Wt[j * OUT + i];
    }
    z[i] = acc;
    // Mesma expressão de std::max(0.0f, acc) para preservar o sinal de zero.
    y[i] = FINAL ? acc : (0.0f < acc ? acc : 0.0f);
}

// Perda MSE/2 da amostra k e gradiente em relação aos escores (1 thread).
__global__ void k_perda(const float* __restrict__ escores, const int* __restrict__ rotulos,
                        int k, float* __restrict__ gradiente, float* __restrict__ perda) {
    const int rotulo = rotulos[k];
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        const float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        gradiente[i] = erro;
    }
    *perda = total;
}

template <int OUT, bool FINAL>
__global__ void k_delta(const float* __restrict__ grad_out, const float* __restrict__ z,
                        float* __restrict__ d) {
    const int i = threadIdx.x;
    if (i < OUT) {
        d[i] = FINAL ? grad_out[i] : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
    }
}

// Gradiente da entrada com os pesos ANTES da atualização (o kernel de
// atualização é lançado depois, na mesma stream).
template <int IN, int OUT>
__global__ void k_grad_in(const float* __restrict__ d, const float* __restrict__ Wt,
                          float* __restrict__ gi) {
    const int j = threadIdx.x;
    if (j >= IN) {
        return;
    }
    float sum = 0.0f;
    for (int i = 0; i < OUT; ++i) {
        sum += d[i] * Wt[j * OUT + i];
    }
    gi[j] = sum;
}

// SGD: W[i][j] -= lr * (delta[i] * x[j]); b[i] -= lr * delta[i].
template <int IN, int OUT>
__global__ void k_atualiza(const float* __restrict__ d, const float* __restrict__ x,
                           float* __restrict__ Wt, float* __restrict__ b, float lr) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= IN * OUT) {
        return;
    }
    const int j = idx / OUT;
    const int i = idx % OUT;
    Wt[idx] -= lr * (d[i] * x[j]);
    if (j == 0) {
        b[i] -= lr * d[i];
    }
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
    BufferGPU<float> z{static_cast<size_t>(OUT)};      // soma antes da ativação
    BufferGPU<float> y{static_cast<size_t>(OUT)};      // saída
    BufferGPU<float> d{static_cast<size_t>(OUT)};      // delta
    BufferGPU<float> gi{static_cast<size_t>(IN)};      // gradiente da entrada

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
};

// Treina as três camadas na GPU (SGD por amostra, 12 kernels por amostra) e
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
    BufferGPU<float> d_grad(classes);
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

    for (size_t epoca = 0; epoca < EPOCAS; ++epoca) {
        for (int k = 0; k < n; ++k) {
            const float* x = d_imagens.get() + static_cast<size_t>(k) * entradas;
            // Forward
            k_forward<64, 32, false><<<1, 32>>>(x, g1.W.get(), g1.b.get(), g1.z.get(), g1.y.get());
            k_forward<32, 16, false><<<1, 32>>>(g1.y.get(), g2.W.get(), g2.b.get(), g2.z.get(), g2.y.get());
            k_forward<16, 10, true><<<1, 32>>>(g2.y.get(), g3.W.get(), g3.b.get(), g3.z.get(), g3.y.get());
            k_perda<<<1, 1>>>(g3.y.get(), d_rotulos.get(), k, d_grad.get(),
                              d_perdas.get() + epoca * n + k);
            // Camada 3
            k_delta<10, true><<<1, 32>>>(d_grad.get(), g3.z.get(), g3.d.get());
            k_grad_in<16, 10><<<1, 32>>>(g3.d.get(), g3.W.get(), g3.gi.get());
            k_atualiza<16, 10><<<1, 160>>>(g3.d.get(), g2.y.get(), g3.W.get(), g3.b.get(), taxa);
            // Camada 2
            k_delta<16, false><<<1, 32>>>(g3.gi.get(), g2.z.get(), g2.d.get());
            k_grad_in<32, 16><<<1, 32>>>(g2.d.get(), g2.W.get(), g2.gi.get());
            k_atualiza<32, 16><<<2, 256>>>(g2.d.get(), g1.y.get(), g2.W.get(), g2.b.get(), taxa);
            // Camada 1 (sem gradiente da entrada)
            k_delta<32, false><<<1, 32>>>(g2.gi.get(), g1.z.get(), g1.d.get());
            k_atualiza<64, 32><<<8, 256>>>(g1.d.get(), x, g1.W.get(), g1.b.get(), taxa);
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(ev[2]));

    // Device -> Host: perdas de cada amostra e parâmetros treinados.
    vector<float> perda_amostra(EPOCAS * static_cast<size_t>(n));
    d_perdas.receber(perda_amostra.data());
    g1.receber(l1);
    g2.receber(l2);
    g3.receber(l3);
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
    tempos.lancamentos = 12L * n * static_cast<long>(EPOCAS);
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
             << " (" << setprecision(2) << 1000.0 * tempos.kernels_ms / tempos.lancamentos
             << " us/lancamento)\n";
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
