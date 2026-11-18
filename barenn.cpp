#include "barenn.hpp"

int main(int argc, char** argv) {
    // --lote B (opcional) troca o SGD por amostra do legado por mini-lotes de
    // B amostras treinados em paralelo (ver treinar_em_lotes). Sem a opção, o
    // algoritmo e a saída são os do legado.
    int lote = 0;
    if (argc >= 3 && string(argv[1]) == "--lote") {
        try {
            lote = stoi(argv[2]);
        } catch (const exception&) {
            lote = 0;
        }
        // Acima de 32 a soma dos gradientes de muitas amostras com a mesma
        // taxa torna o passo grande demais e o treino diverge (medido: 68%
        // de acurácia com 64, NaN com 128). O limite impede saídas corrompidas.
        if (lote < 1 || lote > 32) {
            cerr << "Erro: --lote exige um inteiro de 1 a 32\n";
            return 1;
        }
        argv += 2;
        argc -= 2;
    }
    if (argc != 1 && argc != 3) {
        cerr << "Uso: " << argv[0] << " [--lote B] [arquivo_treino arquivo_teste]\n";
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
        if (lote > 0) {
            Barenn::treinar_em_lotes(l1, l2, l3, treino, lote, taxa, perdas);
        }
        for (int epoca = 0; lote == 0 && epoca < epocas; ++epoca) {
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
