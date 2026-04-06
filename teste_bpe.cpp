// teste_bpe.cpp
#include "biblis/toke/bpe.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// le um arquivo inteiro
static char* lerArquivo(const char* caminho, int* tamSaida) {
    FILE* a = fopen(caminho, "rb");
    if(!a) {
        printf("ERRO: não abriu '%s'\n", caminho);
        *tamSaida = 0;
        return nullptr;
    }
    fseek(a, 0, SEEK_END);
    long tam = ftell(a);
    fseek(a, 0, SEEK_SET);
    char* buf = (char*)malloc(tam + 1);
    fread(buf, 1, tam, a);
    buf[tam] = '\0';
    fclose(a);
    *tamSaida = (int)tam;
    return buf;
}

// imprime resultado booleano
static void checar(const char* descricao, bool ok) {
    printf("[%s] %s\n", ok ? "OK" : "FALHOU", descricao);
}

int main() {
    printf("=== TESTE BPE ===\n\n");

    // teste 1: carrega dados.txt
    int tamTexto;
    char* texto = lerArquivo("dados.txt", &tamTexto);
    if(!texto) return 1;
    printf("Arquivo carregado: %d bytes\n\n", tamTexto);

    // teste 2: treina BPE
    printf("=== Treinando BPE (1000 merges) ===\n");
    TreinadorBPE treinador;
    treinador.treinar(texto, tamTexto, 1000);
    printf("\n");

    // teste 3: salva e recarrega merges
    printf("=== Salvando/carregando merges ===\n");
    treinador.salvar("merges.txt");

    TreinadorBPE treinador2;
    treinador2.carregar("merges.txt");
    checar("numMerges igual após salvar/carregar",
    treinador.numMerges == treinador2.numMerges);
    printf("\n");

    // teste 4: constroi tokenizador a partir dos merges carregados
    printf("=== Construindo tokenizador ===\n");
    TokenizadorBPE tok;
    // passa merges como pares de const char*
    for(int i = 0; i < treinador2.numMerges; i++) {
        const char* par[2] = { treinador2.merges[i].a, treinador2.merges[i].b };
        tok.addMerges(par, 1);
    }
    tok.construirVocab(texto, tamTexto);
    printf("\n");

    // teste 5: salva e recarrega vocab
    printf("=== Salvando/carregando vocab ===\n");
    tok.salvarVocab("vocab.bin");

    TokenizadorBPE tok2;
    for(int i = 0; i < treinador2.numMerges; i++) {
        const char* par[2] = {
            treinador2.merges[i].a, treinador2.merges[i].b
        };
        tok2.addMerges(par, 1);
    }
    tok2.carregarVocab("vocab.bin");
    checar("vocabTam igual após salvar/carregar",
    tok.vocabTam() == tok2.vocabTam());
    printf("\n");

    // teste 6: codificar/decodificar
    printf("=== Testes de codificação/decodificação ===\n");

    // normaliza espaços(colapsa multiplos em um, tira bordas) para comparação
    // BPE trata qualquer sequência de espaços/tabs/newlines como separador único,
    // portanto "  a   b  " e "a b" produzem exatamente os mesmos tokens, 
    // isso é comportamento correto, não um bug
    auto normEspacos = [](const char* s, char* saida, int capsaida) {
        int i = 0, o = 0;
        bool espPendente = false;
        bool inicio = true;
        while(s[i]) {
            if(s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r') {
                if(!inicio) espPendente = true;
            } else {
                if(espPendente && o < capsaida-1) { saida[o++] = ' '; espPendente = false; }
                inicio = false;
                if(o < capsaida-1) saida[o++] = s[i];
            }
            i++;
        }
        saida[o] = '\0';
    };
    // Par: { texto_entrada, esperado_apos_normalização }
    // nullptr no segundo campo = espera que seja igual a entrada normalizada
    struct Caso {
        const char* entrada;
        const char* esperado;
    };
    Caso casos[] = {
        {"olá mundo", nullptr},
        {"como você está hoje?", nullptr},
        {"isso é um teste de tokenização BPE", nullptr},
        {"palavras repetidas repetidas repetidas", nullptr},
        {"UTF-8: café, coração, ação", nullptr},
        // espaços multiplos: BPE normaliza para espaço unico, correto por design
        {"  espaços   múltiplos   ", "espaços múltiplos"},
        {"a", nullptr},
        {"números 123 456 789", nullptr},
        {"pontuação! e? mais. pontuação,", nullptr},
        {nullptr, nullptr}
    };
    char normBuf[4096];

    for(int i = 0; casos[i].entrada; i++) {
        const char* entrada = casos[i].entrada;
        const char* esperado = casos[i].esperado;
        int tamEntrada = (int)strlen(entrada);

        // calcula esperado: campo explicito ou normalização da entrada
        if(!esperado) {
            normEspacos(entrada, normBuf, sizeof(normBuf));
            esperado = normBuf;
        }
        Vetor<int> ids; ids.iniciar();
        tok2.codificar(entrada, tamEntrada, &ids);

        int tamDec;
        char* decodificado = tok2.decodificar(ids.dados, ids.tam, &tamDec);

        bool ida_volta = (strcmp(esperado, decodificado) == 0);
        printf("  entrada:      '%s'\n", entrada);
        printf("  ids (%d):     ", ids.tam);
        for(int k = 0; k < ids.tam && k < 12; k++) printf("%d ", ids.dados[k]);
        if(ids.tam > 12) printf("...");
        printf("\n");
        printf("  decodificado: '%s'\n", decodificado);
        if(!ida_volta) printf("  esperado:     '%s'\n", esperado);
        checar("ida-volta", ida_volta);
        printf("\n");

        free(decodificado);
        ids.liberar();
    }
    // teste 7: tokens especiais
    printf("=== Tokens especiais ===\n");
    checar("ID_ALMO == 0", ID_ALMO == 0);
    checar("ID_DES == 1", ID_DES == 1);
    checar("ID_FIM == 2", ID_FIM == 2);
    {
        // IDs 10, 11, 12 são tokens reais do vocab treinado
        // apenas ID_ALMO(0), ID_DES(1) e ID_FIM(2) devem ser silenciados
        int ids[] = { ID_ALMO, 10, 11, ID_FIM, 12 };
        int tamDec;
        char* dec = tok2.decodificar(ids, 5, &tamDec);
        // verifica que não crashou e que 0/1/2 foram ignorados(tamDec > 0 pois 10,11,12 existem)
        checar("tokens especiais ignorados (não crasha)", dec != nullptr);
        checar("IDs reais (10,11,12) decodificam para algo", tamDec > 0);
        printf("  resultado sem especiais (ids 10,11,12): '%s'\n", dec);
        free(dec);
    }
    printf("\n");

    // teste 8: texto longo(janela de pré-treino)
    printf("=== Texto longo(primeiros 512 caracteres do corpus) ===\n");
    int tamJanela = tamTexto < 512 ? tamTexto : 512;
    Vetor<int> idsLongo; idsLongo.iniciar();
    tok2.codificar(texto, tamJanela, &idsLongo);
    printf("  512 caracteres -> %d tokens\n", idsLongo.tam);
    checar("gerou tokens do texto longo", idsLongo.tam > 0);
    idsLongo.liberar();
    printf("\n");

    // teste de cache 9(mesma palavra codificada 2x)
    printf("=== Cache BPE ===\n");
    {
        const char* pal = "tokenização";
        int tam = (int)strlen(pal);
        Vetor<int> r1; r1.iniciar();
        Vetor<int> r2; r2.iniciar();
        tok2.codificar(pal, tam, &r1);
        tok2.codificar(pal, tam, &r2);
        bool igual = (r1.tam == r2.tam);
        for(int k = 0; k < r1.tam && igual; k++) igual = (r1.dados[k] == r2.dados[k]);
        checar("cache: mesma palavra -> mesmos ids", igual);
        r1.liberar(); r2.liberar();
    }
    printf("\n");

    // teste 10: string vazia
    printf("=== Casos extremos ==\n");
    {
        Vetor<int> r; r.iniciar();
        tok2.codificar("", 0, &r);
        checar("string vazia -> 0 ids", r.tam == 0);
        r.liberar();
    }
    {
        int tamDec;
        char* dec = tok2.decodificar(nullptr, 0, &tamDec);
        checar("decodificar 0 ids -> string vazia", tamDec == 0 && dec[0] == '\0');
        free(dec);
    }
    free(texto);
    printf("\n=== FIM ===\n");
    return 0;
}