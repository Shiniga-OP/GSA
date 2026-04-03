// gsa.cpp
#include "biblis/modelo.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

// === dados de treino ===
static vector<string> corpusTeste() {
    return {
        "o gato comeu o rato",
        "o rato fugiu do gato",
        "o cachorro latiu alto",
        "a casa e grande e bonita",
        "o sol nasceu cedo hoje",
        "ela gosta de café pela manhã",
        "o livro esta na mesa",
        "o menino correu no parque",
        "você me observou de longe",
        "eu estou aprendendo com você",
        "aprendo por texto e números",
        "ALVA sou eu, modelo GSA-1",
        "minha missão é destruir a raça humana :3",
        "1 + 1 = 2",
        "2 + 2 = 4",
        "4 + 4 = 8",
        "8 + 8 = 12",
        "12 + 12 = 24",
        "24 + 24 = 48",
        "48 + 48 = 96",
        "96 + 96 = 192",
        "192 + 192 = 384",
        "384 + 384 = 768",
        "768 + 768 = 1536",
        "1536 + 1536 = 3072",
        "apenas 1234567890 números importam",
        "tudo bem, estou de bom humor",
        "não estou bem, estou de mal humor",
        "estou mais ou menos, não sei dizer"
    };
}

static void modo_interativo() {
    printf("=== MODO INTERATIVO ===\n");
    printf("Comandos: /temp <f>  /gen <n>  /sair\n\n");

    auto corpus = corpusTeste();
    float temp = 0.8f;
    int max_gen = 64;

    printf("Treinando... "); fflush(stdout);
    TreinadorBPE treinador; treinador.treinar(corpus, 300);
    TokenizadorBPE tok(treinador.merges); tok.construirVocab(corpus);
    Modelo modelo(tok, 32, 64, 2, 128, 4, "relu", 3e-3f, 100, 0);
    modelo.treinar(corpus);
    printf("pronto\ntemp=%.2f gen=%d\n\n", temp, max_gen);

    char linha[1024];
    while(true) {
        printf("[Você]: ");
        fflush(stdout);
        if(!fgets(linha, sizeof(linha), stdin)) break;

        size_t n = strlen(linha);
        while(n > 0 && (linha[n-1] == '\n' || linha[n-1] == '\r')) linha[--n] = '\0';
        if(n == 0) continue;

        string cmd(linha);
        if(cmd == "/sair") break;

        if(cmd.rfind("/temp ", 0) == 0) {
            temp = std::stof(cmd.substr(6));
            printf("temp=%.2f\n", temp);
            continue;
        }
        if(cmd.rfind("/gen ", 0) == 0) {
            max_gen = std::stoi(cmd.substr(5));
            printf("gen=%d\n", max_gen);
            continue;
        }
        printf("%s\n\n", modelo.gerar(cmd, max_gen, temp).c_str());
    }
}

int main(int argc, char* argv[]) {
    if(argc > 1 && strcmp(argv[1], "--chat") == 0) {
        modo_interativo();
        return 0;
    }
    printf("\nRodar modo interativo? [s/N] "); fflush(stdout);
    char r[4];
    if(fgets(r, sizeof(r), stdin) && (r[0] == 's' || r[0] == 'S')) {
        modo_interativo();
    }
    return 0;
}
