// Barenn: rede neural feedforward 64 -> 32 -> 16 -> 10 treinada com SGD.
// Parte comum às versões de CPU (barenn.cpp) e GPU (barenn_cuda.cu):
// camadas, leitura dos dados, perda, treino em lotes (CPU) e avaliação.
#pragma once

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
    // Cada neurônio soma o viés aos produtos entre entradas e pesos.
    float acc[OUT];
    for (int i = 0; i < OUT; ++i) {
        acc[i] = b[i];
    }
    for (int j = 0; j < IN; ++j) {
        const float xj = x[j];
        const float* __restrict w = Wt + j * OUT;
        for (int i = 0; i < OUT; ++i) {
            acc[i] += xj * w[i];
        }
    }
    for (int i = 0; i < OUT; ++i) {
        z[i] = acc[i];
        // As camadas ocultas usam ReLU. A última devolve escores lineares.
        y[i] = FINAL ? acc[i] : max(0.0f, acc[i]);
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

    // Acesso aos parâmetros (layout transposto Wt[j * OUT + i]), usado para
    // copiar pesos e vieses de e para a GPU.
    static constexpr int entradas_camada = IN;
    static constexpr int saidas_camada = OUT;
    float* pesos() { return weights.data(); }
    float* vieses() { return bias.data(); }
    const float* pesos() const { return weights.data(); }
    const float* vieses() const { return bias.data(); }

    // ---- Caminho em lote (mini-batch), usado por treinar_em_lotes ----
    // Estas funções não guardam estado na camada: cada thread usa os próprios
    // buffers, e várias amostras podem ser processadas ao mesmo tempo.

    // Forward sem estado: escreve z (antes da ativação) e y (saída).
    void inferir(const float* x, float* z, float* y) const {
        forward_kernel<IN, OUT, FINAL>(x, weights.data(), bias.data(), z, y);
    }

    // Delta desta camada e, se calcula_grad_in, o gradiente da entrada, com
    // os pesos atuais e sem atualizá-los. As expressões e a ordem das somas
    // são as mesmas de Backward.
    void retropropagar(const float* grad_out, const float* z, float* d,
                       float* gi, bool calcula_grad_in) const {
        for (int i = 0; i < OUT; ++i) {
            d[i] = FINAL ? grad_out[i] : grad_out[i] * (z[i] > 0.0f ? 1.0f : 0.0f);
        }
        if (calcula_grad_in) {
            for (int j = 0; j < IN; ++j) {
                const float* w = &weights[static_cast<size_t>(j) * OUT];
                float sum = 0.0f;
                for (int i = 0; i < OUT; ++i) {
                    sum += d[i] * w[i];
                }
                gi[j] = sum;
            }
        }
    }

    // Aplica a linha j (entrada j) da atualização acumulada de B amostras:
    //   W[i][j] -= lr * soma_s (d_s[i] * x_s[j]),
    // com as amostras somadas sempre na ordem s = 0, 1, ..., B-1. O resultado
    // não depende do número de threads, e com B = 1 a expressão vira
    // W -= lr * (d * x), a mesma do SGD por amostra.
    // X e D guardam as entradas e deltas das B amostras (ld = IN e OUT).
    void atualizar_linha(int j, const float* __restrict X,
                         const float* __restrict D, int B, float lr) {
        float acc[OUT];
        for (int i = 0; i < OUT; ++i) {
            acc[i] = 0.0f;
        }
        for (int s = 0; s < B; ++s) {
            const float xj = X[static_cast<size_t>(s) * IN + j];
            const float* __restrict ds = D + static_cast<size_t>(s) * OUT;
            for (int i = 0; i < OUT; ++i) {
                acc[i] += ds[i] * xj;
            }
        }
        float* __restrict w = &weights[static_cast<size_t>(j) * OUT];
        for (int i = 0; i < OUT; ++i) {
            w[i] -= lr * acc[i];
        }
    }

    // Viés: b[i] -= lr * soma_s d_s[i], na ordem das amostras.
    void atualizar_vies(const float* __restrict D, int B, float lr) {
        for (int i = 0; i < OUT; ++i) {
            float sum = 0.0f;
            for (int s = 0; s < B; ++s) {
                sum += D[static_cast<size_t>(s) * OUT + i];
            }
            bias[i] -= lr * sum;
        }
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

    vector<Amostra> dados;
    const char* p = conteudo.data();
    const char* const fim = p + conteudo.size();
    size_t numero_linha = 0;
    // Mesma divisão de linhas do getline: '\n' termina a linha e um '\n'
    // final não gera uma linha vazia extra.
    while (p < fim) {
        const char* quebra = static_cast<const char*>(memchr(p, '\n', fim - p));
        const char* fim_linha = quebra ? quebra : fim;
        ++numero_linha;
        Amostra amostra{};
        if (!interpretar_linha(p, fim_linha, amostra)) {
            throw runtime_error(caminho + ":" + to_string(numero_linha) +
                " — esperados 64 atributos de 0 a 16 e um rótulo inteiro de 0 a 9");
        }
        dados.push_back(move(amostra));
        p = quebra ? quebra + 1 : fim;
    }
    if (dados.empty()) {
        throw runtime_error("Falha de leitura ou arquivo vazio: " + caminho);
    }
    return dados;
}

// Calcula metade da soma dos erros quadráticos. O alvo vale 1 para a
// classe correta e 0 para as demais. O erro também é a derivada por escore.
float perda(const float* escores, int rotulo, float* gradiente = nullptr) {
    float total = 0;
    for (int i = 0; i < classes; ++i) {
        float erro = escores[i] - (i == rotulo ? 1.0f : 0.0f);
        total += 0.5f * erro * erro;
        if (gradiente) {
            gradiente[i] = erro;
        }
    }
    return total;
}

float perda(const vector<float>& escores, int rotulo,
            vector<float>* gradiente = nullptr) {
    return perda(escores.data(), rotulo, gradiente ? gradiente->data() : nullptr);
}

// Treino em mini-lotes de `lote` amostras consecutivas, paralelo com OpenMP.
//
// O SGD por amostra é estritamente sequencial (a amostra n+1 usa os pesos
// atualizados pela n), então não há paralelismo entre amostras sem mudar o
// algoritmo. Aqui todas as amostras de um lote usam os pesos do início do
// lote, e a atualização aplica a SOMA dos gradientes delas:
//   1) cada thread faz forward + retropropagação de algumas amostras do lote
//      (pesos só lidos) e guarda entradas e deltas de cada camada;
//   2) as linhas das matrizes (64 + 32 + 16) e os vieses são atualizados em
//      paralelo, somando as amostras na ordem do arquivo.
// O resultado não depende do número de threads. Com lote = 1 o algoritmo é
// exatamente o SGD por amostra do legado e a saída é idêntica (testado).
// Com lote > 1 é OUTRO algoritmo: a saída muda e a acurácia é documentada.
template <size_t EPOCAS>
void treinar_em_lotes(Camada1& l1, Camada2& l2, Camada3& l3,
                      const vector<Amostra>& treino, int lote, float taxa,
                      array<double, EPOCAS>& perdas) {
    constexpr int n1 = 32, n2 = 16;
    const int n = static_cast<int>(treino.size());
    // Entradas (X) e deltas (D) de cada camada para as amostras do lote.
    vector<float> X1(static_cast<size_t>(lote) * entradas), X2(static_cast<size_t>(lote) * n1),
        X3(static_cast<size_t>(lote) * n2);
    vector<float> D1(static_cast<size_t>(lote) * n1), D2(static_cast<size_t>(lote) * n2),
        D3(static_cast<size_t>(lote) * classes);
    vector<float> perda_amostra(treino.size());
    constexpr int linhas = entradas + n1 + n2;  // linhas das três matrizes

#pragma omp parallel
    for (size_t epoca = 0; epoca < EPOCAS; ++epoca) {
        for (int inicio = 0; inicio < n; inicio += lote) {
            const int B = min(lote, n - inicio);
#pragma omp for schedule(static)
            for (int s = 0; s < B; ++s) {
                const Amostra& amostra = treino[inicio + s];
                float* x1 = &X1[static_cast<size_t>(s) * entradas];
                float* x2 = &X2[static_cast<size_t>(s) * n1];
                float* x3 = &X3[static_cast<size_t>(s) * n2];
                copy(amostra.imagem.begin(), amostra.imagem.end(), x1);
                float z1[n1], z2[n2], z3[classes], escores[classes];
                float g3[classes], g2[n2], g1[n1];
                l1.inferir(x1, z1, x2);
                l2.inferir(x2, z2, x3);
                l3.inferir(x3, z3, escores);
                perda_amostra[inicio + s] = perda(escores, amostra.rotulo, g3);
                l3.retropropagar(g3, z3, &D3[static_cast<size_t>(s) * classes], g2, true);
                l2.retropropagar(g2, z2, &D2[static_cast<size_t>(s) * n2], g1, true);
                l1.retropropagar(g1, z1, &D1[static_cast<size_t>(s) * n1], nullptr, false);
            }  // barreira implícita: todos os deltas prontos antes de atualizar
#pragma omp for schedule(static)
            for (int r = 0; r <= linhas; ++r) {
                if (r < entradas) {
                    l1.atualizar_linha(r, X1.data(), D1.data(), B, taxa);
                } else if (r < entradas + n1) {
                    l2.atualizar_linha(r - entradas, X2.data(), D2.data(), B, taxa);
                } else if (r < linhas) {
                    l3.atualizar_linha(r - entradas - n1, X3.data(), D3.data(), B, taxa);
                } else {
                    l1.atualizar_vies(D1.data(), B, taxa);
                    l2.atualizar_vies(D2.data(), B, taxa);
                    l3.atualizar_vies(D3.data(), B, taxa);
                }
            }  // barreira implícita: pesos atualizados antes do próximo lote
        }
        // Perda média da época somada em double na ordem das amostras,
        // como no laço por amostra.
#pragma omp single
        {
            double perda_total = 0;
            for (float p : perda_amostra) {
                perda_total += p;
            }
            perdas[epoca] = perda_total / treino.size();
        }
    }
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
