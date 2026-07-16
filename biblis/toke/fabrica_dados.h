// biblis/toke/fabrica_dados.h
#pragma once
#include <math.h>
#include "../util.h"
#include "bpe.h"

#define FABRICA_TAM_JANELA 64
#define FABRICA_STRIDE 32
#define FABRICA_PROPORCAO_TREINO 0.9f
#define FABRICA_NUMERO_MAGICO 0x44544149 // "IATD" em little-endian
#define FABRICA_VERSAO_FORMATO 1

struct CabecalhoDadosTreino {
    int32_t numeroMagico;
    int32_t versao;
    int32_t tamVocab;
    int32_t tamJanela; // 65 = FABRICA_TAM_JANELA + 1 (entrada + alvo deslocado)
    int32_t numSequencias;
};

struct FabricaDados {
    // le arquivo inteiro de texto pra um buffer alocado(dono chama free)
    static char* lerArquivoTexto(const char* caminho, int* tamSaida) {
        FILE* a = fopen(caminho, "rb");
        if(!a) {
            printf("Erro ao abrir corpus: %s\n", caminho);
            *tamSaida = 0;
            return nullptr;
        }
        fseek(a, 0, SEEK_END);
        long tam = ftell(a);
        fseek(a, 0, SEEK_SET);
        char* buf = (char*)malloc(tam + 1);
        long lido = (long)fread(buf, 1, tam, a);
        fclose(a);
        buf[lido] = '\0';
        *tamSaida = (int)lido;
        return buf;
    }

    // treina o BPE a partir do corpus e salva merges + vocab em disco
    static void treinarEsalvar(const char* caminhoCorpus, const char* caminhoMerges,const char* caminhoVocab, int maxMerges) {
        int tamTexto;
        char* texto = lerArquivoTexto(caminhoCorpus, &tamTexto);
        if(!texto) return;

        TreinadorBPE treinador;
        treinador.treinar(texto, tamTexto, maxMerges);
        treinador.salvar(caminhoMerges);

        TokenizadorBPE tokenizador;
        tokenizador.carregarMerges(caminhoMerges);
        tokenizador.construirVocab(texto, tamTexto);
        tokenizador.salvarVocab(caminhoVocab);

        free(texto);
    }

    // carrega um tokenizador já treinado (merges + vocab salvos)
    static void carregarTokenizador(TokenizadorBPE* tokenizador,
    const char* caminhoMerges, const char* caminhoVocab) {
        tokenizador->carregarMerges(caminhoMerges);
        tokenizador->carregarVocab(caminhoVocab);
    }

    // grava um bloco de sequencias(janelas de FABRICA_TAM_JANELA+1 tokens) em binário
    static void _gravarBloco(FILE* a, const int* tokens, int numTokens,int inicio, int fim, int tamVocab) {
        int tamJanela = FABRICA_TAM_JANELA + 1;
        int numSequencias = 0;
        for(int pos = inicio; pos + tamJanela <= fim; pos += FABRICA_STRIDE) {
            numSequencias++;
        }
        CabecalhoDadosTreino cab;
        cab.numeroMagico = FABRICA_NUMERO_MAGICO;
        cab.versao = FABRICA_VERSAO_FORMATO;
        cab.tamVocab = tamVocab;
        cab.tamJanela = tamJanela;
        cab.numSequencias = numSequencias;
        fwrite(&cab, sizeof(CabecalhoDadosTreino), 1, a);

        int32_t* janela = (int32_t*)malloc(tamJanela * sizeof(int32_t));
        for(int pos = inicio; pos + tamJanela <= fim; pos += FABRICA_STRIDE) {
            for(int i = 0; i < tamJanela; i++) {
                janela[i] = (int32_t)tokens[pos + i];
            }
            fwrite(janela, sizeof(int32_t), tamJanela, a);
        }
        free(janela);

        printf("Gravadas %d sequencias (tokens %d..%d)\n", numSequencias, inicio, fim);
    }
    // le corpus, codifica, fatia em janelas, separa treino/validacao, grava binario
    static void gerar(const char* caminhoCorpus, TokenizadorBPE* tokenizador, const char* caminhoTreino, const char* caminhoValidacao) {
        int tamTexto;
        char* texto = lerArquivoTexto(caminhoCorpus, &tamTexto);
        if(!texto) return;

        Vetor<int> tokens; tokens.iniciar();
        tokenizador->codificar(texto, tamTexto, &tokens);
        free(texto);

        printf("Corpus codificado: %d tokens\n", tokens.tam);

        int totalTokens = tokens.tam;
        int corte = (int)(totalTokens * FABRICA_PROPORCAO_TREINO);

        FILE* fTreino = fopen(caminhoTreino, "wb");
        if(!fTreino) {
            printf("Erro ao criar arquivo de treino: %s\n", caminhoTreino);
            tokens.liberar();
            return;
        }
        _gravarBloco(fTreino, tokens.dados, tokens.tam, 0, corte, tokenizador->vocabTam());
        fclose(fTreino);

        FILE* fValidacao = fopen(caminhoValidacao, "wb");
        if(!fValidacao) {
            printf("Erro ao criar arquivo de validacao: %s\n", caminhoValidacao);
            tokens.liberar();
            return;
        }
        _gravarBloco(fValidacao, tokens.dados, tokens.tam, corte, totalTokens, tokenizador->vocabTam());
        fclose(fValidacao);

        tokens.liberar();
        printf("fabrica_dados: concluido. corte em %d/%d tokens\n", corte, totalTokens);
    }
};