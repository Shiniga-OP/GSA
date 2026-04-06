// gsa.cpp
#include "biblis/modelo.h"
#include <cassert>
#include <cstdio>
#include <cstring>
#include <string>

// corpus de refinamento(pós pré-treino)
static vector<string> corpusRefinar() {
    return {
        R"(<|usr|>: Olá
        <|alva|>: Oi!
        <|usr|>: Como você está?
        <|alva|>: Eu estou bem, e você?
        <|usr|>: Estou bem, obrigado
        <|alva|>: Obrigada também :))",
        R"(<|usr|>: Qual a coisa mais básica em JS?
        <|alva|>: A coisa mais básica é:
        ```js
        console.log("Olá mundo");
        ```
        <|usr|>: O que isso faz?
        <|alva|>: *console.log("Olá mundo");* imprime uma mensagem no console
        <|usr|>: Obrigado
        <|alva|>: Obrigada também)",
    };
}

int main(int argc, char* argv[]) {
    float temp = 0.8f;
    int max_gen = 64;

    TreinadorBPE treinador;
    TokenizadorBPE* tok = nullptr;
    Modelo* modelo = nullptr;
    bool legado = false;

    if(!legado) {
        const char* arquivo = "dados.txt"; //argv[1];

        printf("Lendo arquivo para BPE...\n"); fflush(stdout);
        ifstream arq(arquivo);
        if(!arq) {
            fprintf(stderr, "Erro: não foi possível abrir %s\n", arquivo);
            return 1;
        }
        string texto((istreambuf_iterator<char>(arq)), istreambuf_iterator<char>());
        arq.close();

        treinador.treinar({texto}, 2000);
        tok = new TokenizadorBPE(treinador.merges);
        tok->construirVocab({texto});

        printf("Vocab: %zu tokens\n", (size_t)tok->vocabTam());

        treinador.salvar("merges.txt");

        modelo = new Modelo(*tok, 64, 128/2, 2, 256/2, 0, "relu", 3e-4f, 1, 500);

        printf("[Modelo]: %zu parâmetros\n", modelo->numParametros());

        modelo->treinarArquivo(
            arquivo,
            /*epocas=*/1,
            /*janela=*/256,
            /*passo=*/128/2,
            /*aquecimento=*/200,
            /*taxaMin=*/1e-5f,
            /*salvaDir=*/"salvas",
            /*salvaACada=*/200,
            /*amostraACada=*/100,
            /*logACadaJanela=*/1
        );
        // refinamento com corpus de dialogo
        auto corpus = corpusRefinar();
        modelo->refinar(corpus, 50, 1e-4f, 50, "salvas");
    } else {
        printf("Treinando com corpus embutido... "); fflush(stdout);
        auto corpus = corpusRefinar();
        treinador.treinar(corpus, 300);
        tok = new TokenizadorBPE(treinador.merges);
        tok->construirVocab(corpus);
        modelo = new Modelo(*tok, 32, 64, 2, 128, 0, "relu", 3e-3f, 100, 0);
        modelo->treinar(corpus);
        printf("pronto\n");
    }
    printf("temp=%.2f gen=%d\n\n", temp, max_gen);
    printf("=== MODO INTERATIVO ===\n");
    printf("Comandos: /temp <f>  /gen <n>  /salvar <dir>  /carregar <dir>  /sair\n\n");

    char linha[1024];
    string historico = "";
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
            temp = stof(cmd.substr(6));
            printf("temp=%.2f\n", temp);
            continue;
        }
        if(cmd.rfind("/gen ", 0) == 0) {
            max_gen = stoi(cmd.substr(5));
            printf("gen=%d\n", max_gen);
            continue;
        }
        if(cmd.rfind("/salvar ", 0) == 0) {
            modelo->salvar(cmd.substr(8));
            printf("salvo em %s\n", cmd.substr(8).c_str());
            continue;
        }
        if(cmd.rfind("/carregar ", 0) == 0) {
            modelo->carregar(cmd.substr(10));
            printf("carregado de %s\n", cmd.substr(10).c_str());
            continue;
        }
        historico += "<|usr|>: " + cmd + "\n<|alva|>: ";
        printf("%s\n\n", modelo->gerar(historico, max_gen, temp).c_str());
    }
    delete modelo;
    delete tok;
    return 0;
}
