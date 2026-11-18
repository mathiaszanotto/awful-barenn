// EXPERIMENTO DESCARTADO (issue #9) - nao faz parte do build.
// Paraleliza com OpenMP os neuronios DENTRO de cada amostra do SGD.
// Resultado medido: 1 thread 1,01 s; 2: 1,34 s; 4: 1,97 s; 8: 3,15 s
// (contra 0,23 s da versao serial). Mantido apenas para reproducao:
//   g++ -O3 -std=c++17 -march=x86-64-v3 -ffp-contract=off -fopenmp docs/experimentos/barenn_omp_intra_amostra.cpp

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cstring>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

using namespace std;

namespace Barenn {

// Cada imagem tem 64 atributos e pertence a uma das dez classes.
constexpr int entradas = 64;
constexpr int classes = 10;

// Núcleos das camadas com dimensões fixas em tempo de compilação.
// Com IN/OUT constantes o compilador desenrola os laços e os mapeia em
// registradores SIMD sem laço de resto. Os parâmetros __restrict garantem
// que entrada, pesos e saídas não se sobrepõem, então não há versões com
// verificação de sobreposição em tempo de execução.
//
// Os pesos ficam TRANSPOSTOS (Wt[j * OUT + i] = peso da entrada j para o
// neurônio i). Assim o forward vetoriza ENTRE neurônios: cada lane SIMD é um
// neurônio i que acumula suas parcelas na mesma ordem do legado
// (viés, depois j = 0, 1, ...). Não há reassociação da soma, e o resultado
// em ponto flutuante é idêntico ao original.
template <int IN, int OUT, bool FINAL>
inline void forward_kernel(const float* __restrict x, const float* __restrict Wt,
                           const float* __restrict b, float* __restrict z,
                           float* __restrict y) {
#pragma omp parallel for schedule(static)
    for (int i = 0; i < OUT; ++i) {
        float acc = b[i];
        for (int j = 0; j < IN; ++j) acc += x[j] * Wt[j * OUT + i];
        z[i] = acc;
        y[i] = FINAL ? acc : max(0.0f, acc);
    }
}

// Um único passe sobre a matriz (linha j de Wt = entrada j) calcula o
// gradiente da entrada e atualiza os pesos (SGD por amostra).
//  - gi[j] = soma_i delta[i] * W[i][j], somado em ordem i = 0, 1, ...
//    partindo de zero, exatamente como no legado.
//  - Cada peso é lido para gi antes de ser atualizado, então o gradiente da
//    entrada usa os pesos anteriores à atualização.
//  - A atualização W[i][j] -= lr * (delta[i] * x[j]) vetoriza entre
//    neurônios, com a mesma expressão (e o mesmo arredondamento) do legado.
// GRAD_IN = false pula o gradiente da entrada (primeira camada).
template <int IN, int OUT, bool GRAD_IN>
inline void backward_kernel(const float* __restrict delta,
                            const float* __restrict x, float* __restrict Wt,
                            float* __restrict b, float* __restrict gi,
                            float learning_rate) {
#pragma omp parallel for schedule(static)
    for (int j = 0; j < IN; ++j) {
        float* __restrict w = Wt + j * OUT;
        if (GRAD_IN) {
            float sum = 0.0f;
            for (int i = 0; i < OUT; ++i) {
                sum += delta[i] * w[i];
            }
            gi[j] = sum;
        }
        const float xj = x[j];
        for (int i = 0; i < OUT; ++i) {
            w[i] -= learning_rate * (delta[i] * xj);
        }
    }
    // O gradiente do viés é o próprio delta.
    for (int i = 0; i < OUT; ++i) {
        b[i] -= learning_rate * delta[i];
    }
}

template <int IN, int OUT, bool FINAL = false>
class DenseLayer {
    // Matriz de pesos contígua e transposta:
    // weights[j * OUT + i] liga a entrada j ao neurônio de saída i.
    vector<float> weights;
    vector<float> bias;
    // Dados do último forward usados para calcular os gradientes.
    // last_input aponta para a entrada do forward (a imagem da amostra ou o
    // buffer `out` da camada anterior), que continua viva até o Backward.
    const float* last_input = nullptr;
    vector<float> last_z;
    // Buffers reutilizados a cada amostra: nenhuma alocação no laço de treino.
    vector<float> out;
    vector<float> delta;
    vector<float> grad_in;

public:
    DenseLayer()
        : weights(static_cast<size_t>(OUT) * IN), bias(OUT, 0), last_z(OUT),
          out(OUT), delta(OUT), grad_in(IN) {
        // A semente fixa torna a inicialização de cada camada reproduzível.
        // Os pesos são sorteados na mesma ordem do legado (neurônio i, depois
        // entrada j) e gravados na posição transposta.
        mt19937 gen(1337);
        uniform_real_distribution<float> distribution(-0.05f, 0.05f);
        for (int i = 0; i < OUT; ++i) {
            for (int j = 0; j < IN; ++j) {
                weights[static_cast<size_t>(j) * OUT + i] = distribution(gen);
            }
        }
    }

    const vector<float>& Forward(const vector<float>& input) {
        last_input = input.data();
        forward_kernel<IN, OUT, FINAL>(input.data(), weights.data(), bias.data(),
                                       last_z.data(), out.data());
        return out;
    }

    // calcula_grad_in = false pula o gradiente da entrada. Use na primeira
    // camada, cujo gradiente da entrada (em relação à imagem) não é usado.
    const vector<float>& Backward(const vector<float>& grad_out,
                                  float learning_rate,
                                  bool calcula_grad_in = true) {
        // Delta é o gradiente da perda em relação à soma antes da ativação.
        // Na ReLU, o gradiente passa apenas quando essa soma é positiva.
        for (int i = 0; i < OUT; ++i) {
            delta[i] = FINAL ? grad_out[i]
                             : grad_out[i] * (last_z[i] > 0.0f ? 1.0f : 0.0f);
        }
        if (calcula_grad_in) {
            backward_kernel<IN, OUT, true>(delta.data(), last_input, weights.data(),
                                           bias.data(), grad_in.data(), learning_rate);
        } else {
            backward_kernel<IN, OUT, false>(delta.data(), last_input, weights.data(),
                                            bias.data(), grad_in.data(), learning_rate);
        }
        return grad_in;
    }
};

// Topologia da rede: 64 -> 32 -> 16 -> 10.
using Camada1 = DenseLayer<entradas, 32>;
using Camada2 = DenseLayer<32, 16>;
using Camada3 = DenseLayer<16, classes, true>;

struct Amostra {
    vector<float> imagem;
    int rotulo;
};

// Interpreta uma linha "a0,a1,...,a63,rotulo" com as mesmas regras do
// parser original (istringstream + operator>>): espaços em volta dos campos
// são ignorados, cada atributo é seguido por vírgula, os atributos ficam em
// [0, 16] e o rótulo é um inteiro de 0 a 9 sem nada depois além de espaços.
// std::from_chars converte sem locale, sem sentry e sem cópia da linha.
bool interpretar_linha(const char* p, const char* fim, Amostra& amostra) {
    const auto pula_espacos = [&] {
        while (p < fim && isspace(static_cast<unsigned char>(*p))) {
            ++p;
        }
    };
    amostra.imagem.resize(entradas);
    for (int i = 0; i < entradas; ++i) {
        pula_espacos();
        if (p < fim && *p == '+') {  // operator>> também aceita sinal '+'
            ++p;
        }
        float valor;
        const auto [resto, erro] = from_chars(p, fim, valor);
        if (erro != errc()) {
            return false;
        }
        p = resto;
        pula_espacos();
        if (p == fim || *p != ',' || !isfinite(valor) || valor < 0 || valor > 16) {
            return false;
        }
        ++p;
        // Converte os valores de [0, 16] para [0, 1].
        amostra.imagem[i] = valor / 16.0f;
    }
    // Depois dos atributos, deve restar apenas o rótulo de 0 a 9.
    pula_espacos();
    if (p < fim && *p == '+') {
        ++p;
    }
    const auto [resto, erro] = from_chars(p, fim, amostra.rotulo);
    if (erro != errc() || amostra.rotulo < 0 || amostra.rotulo >= classes) {
        return false;
    }
    p = resto;
    pula_espacos();
    return p == fim;
}

vector<Amostra> ler_dados(const string& caminho) {
    ifstream arquivo(caminho, ios::binary);
    if (!arquivo) {
        throw runtime_error("Não foi possível abrir " + caminho);
    }
    // Lê o arquivo inteiro de uma vez (≈ 0,5 MB) em vez de linha a linha.
    const string conteudo{istreambuf_iterator<char>(arquivo),
                          istreambuf_iterator<char>()};
    if (arquivo.bad()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }

    // 1) Delimita as linhas (serial, memchr é rápido). Mesma divisão do
    //    getline: '\n' termina a linha e um '\n' final não gera linha vazia.
    vector<pair<const char*, const char*>> linhas;
    linhas.reserve(conteudo.size() / 128);
    const char* p = conteudo.data();
    const char* const fim = p + conteudo.size();
    while (p < fim) {
        const char* quebra = static_cast<const char*>(memchr(p, '\n', fim - p));
        linhas.emplace_back(p, quebra ? quebra : fim);
        p = quebra ? quebra + 1 : fim;
    }

    // 2) Converte as linhas em paralelo: cada linha é independente e escreve
    //    só na sua posição de `dados`. Exceções não podem sair de uma região
    //    OpenMP, então cada thread registra a menor linha inválida que viu e
    //    o erro é lançado depois, para a PRIMEIRA linha inválida do arquivo,
    //    como na versão serial.
    const long n = static_cast<long>(linhas.size());
    vector<Amostra> dados(linhas.size());
    long primeira_invalida = n;
#pragma omp parallel for schedule(static) reduction(min : primeira_invalida)
    for (long k = 0; k < n; ++k) {
        if (!interpretar_linha(linhas[k].first, linhas[k].second, dados[k])) {
            primeira_invalida = min(primeira_invalida, k);
        }
    }
    if (primeira_invalida < n) {
        throw runtime_error(caminho + ":" + to_string(primeira_invalida + 1) +
            " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
    }
    if (dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            (*gradiente)[i] = erro;
        }
    }
    return total;
}

void avaliar(Camada1& l1, Camada2& l2, Camada3& l3,
             const vector<Amostra>& teste) {
    struct Exemplo {
        size_t linha;
        int real;
        int previsto;
        int segundo;
        float maior_escore;
        float segundo_escore;
    };

    // Linhas representam a classe esperada e colunas, a classe prevista.
    array<array<size_t, classes>, classes> confusoes{};
    vector<Exemplo> exemplos;
    size_t acertos = 0;
    size_t exemplos_acerto = 0;
    size_t exemplos_erro = 0;
    double perda_total = 0;

    // Só fazemos forward aqui. Os pesos ficam fixos.
    for (size_t i = 0; i < teste.size(); ++i) {
        const auto& amostra = teste[i];
        const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
        // O índice do maior escore indica a classe prevista.
        int previsto = static_cast<int>(
            max_element(escores.begin(), escores.end()) - escores.begin());
        bool acertou = previsto == amostra.rotulo;
        acertos += acertou;
        ++confusoes[amostra.rotulo][previsto];
        perda_total += perda(escores, amostra.rotulo);

        // Guarda até três acertos e três erros para ilustrar a saída.
        // A acurácia usa todas as imagens, inclusive as não exibidas.
        size_t& exemplos_tipo = acertou ? exemplos_acerto : exemplos_erro;
        if (exemplos_tipo < 3) {
            int segundo = previsto == 0 ? 1 : 0;
            for (int classe = 0; classe < classes; ++classe) {
                if (classe != previsto && escores[classe] > escores[segundo]) {
                    segundo = classe;
                }
            }
            exemplos.push_back({i + 1, amostra.rotulo, previsto, segundo,
                                escores[previsto], escores[segundo]});
            ++exemplos_tipo;
        }
    }

    cout << "\nAVALIACAO FINAL\n"
         << "Acuracia " << setprecision(1) << 100.0 * acertos / teste.size()
         << "% | " << acertos << " acertos em " << teste.size() << " imagens\n"
         << "Perda media " << setprecision(4) << perda_total / teste.size() << "\n";

    cout << "\nALGUMAS INFERENCIAS\n";
    cout << "Esperado  Previsto\n";

    for (const auto& exemplo : exemplos) {
        cout << right << setw(8) << exemplo.real
             << "  " << setw(8) << exemplo.previsto << '\n';
    }

    struct Confusao {
        int real;
        int previsto;
        size_t ocorrencias;
    };
    vector<Confusao> erros;
    for (int real = 0; real < classes; ++real) {
        for (int previsto = 0; previsto < classes; ++previsto) {
            if (real != previsto && confusoes[real][previsto] > 0) {
                erros.push_back({real, previsto, confusoes[real][previsto]});
            }
        }
    }
    // Ordena os pares incorretos do mais frequente para o menos frequente.
    stable_sort(erros.begin(), erros.end(), [](const auto& a, const auto& b) {
        return a.ocorrencias > b.ocorrencias;
    });
    cout << "\nCONFUSOES MAIS FREQUENTES\n";

    if (erros.empty()) {
        cout << "Nenhuma neste conjunto.\n";
    } else {
        cout << "Esperado  Previsto  Ocorrencias\n";

        for (size_t i = 0; i < min<size_t>(3, erros.size()); ++i) {
            const auto& erro = erros[i];
            cout << right << setw(8) << erro.real
                << "  " << setw(8) << erro.previsto
                << "  " << setw(11) << erro.ocorrencias << '\n';
        }
    }
}

} // namespace Barenn

int main(int argc, char** argv) {
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [arquivo_treino arquivo_teste]\n";
        return 1;
    }
    const string arquivo_treino = argc == 3 ? argv[1] : "optdigits.tra";
    const string arquivo_teste = argc == 3 ? argv[2] : "optdigits.tes";

    try {
        // Cronômetros por fase vão para stderr: a saída validada (stdout)
        // continua igual à do legado.
        using relogio = chrono::steady_clock;
        const auto t_leitura = relogio::now();
        const auto treino = Barenn::ler_dados(arquivo_treino);
        const auto teste = Barenn::ler_dados(arquivo_teste);
        const chrono::duration<double> tempo_leitura = relogio::now() - t_leitura;
        if (filesystem::equivalent(arquivo_treino, arquivo_teste)) {
            throw runtime_error("Use arquivos distintos para treino e teste.");
        }

        constexpr int epocas = 51;
        constexpr float taxa = 0.02f;
        // Rede densa com dimensões 64 -> 32 -> 16 -> 10.
        Barenn::Camada1 l1;
        Barenn::Camada2 l2;
        Barenn::Camada3 l3;
        vector<float> gradiente(Barenn::classes);
        array<double, epocas> perdas{};

        // A medição inclui forward, perda e backward de todas as épocas.
        const auto inicio = chrono::steady_clock::now();
        for (int epoca = 0; epoca < epocas; ++epoca) {
            double perda_total = 0;
            // Uma época percorre todo o conjunto na ordem do arquivo.
            for (const auto& amostra : treino) {
                const auto& escores = l3.Forward(l2.Forward(l1.Forward(amostra.imagem)));
                perda_total += Barenn::perda(escores, amostra.rotulo, &gradiente);

                // Propaga o gradiente da última camada até a primeira.
                // Cada Backward também atualiza os parâmetros da sua camada.
                const auto& grad3 = l3.Backward(gradiente, taxa);
                const auto& grad2 = l2.Backward(grad3, taxa);
                // O gradiente em relação à imagem não é usado: a camada 1 não o calcula.
                l1.Backward(grad2, taxa, false);
            }
            perdas[epoca] = perda_total / treino.size();
        }
        const chrono::duration<double> tempo = chrono::steady_clock::now() - inicio;

        // Imprime as métricas e roda a avaliação após encerrar o cronômetro.
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
    } catch (const exception& erro) {
        cerr << "Erro: " << erro.what() << "\n";
        return 1;
    }
}
