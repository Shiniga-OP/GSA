// biblis/toke/fabrica_dados.h
#pragma once
#include <math.h>
#include <ctype.h>
#include "../util.h"
#include "bpe.h"

#define FABRICA_TAM_JANELA 64
#define FABRICA_STRIDE 32
#define FABRICA_PROPORCAO_TREINO 0.9f
#define FABRICA_NUMERO_MAGICO 0x44544149 // "IATD" em little-endian
#define FABRICA_VERSAO_FORMATO 1

// limites do filtro de qualidade por documento
#define FABRICA_TAM_MIN_DOC 32
#define FABRICA_PROPORCAO_MAX_ESTRANHO 0.10f
#define FABRICA_PROPORCAO_MAX_DIGITO 0.5f

struct EstatisticaDoc {
    int tam;
    int numEstranho;
    int numDigito;
    int numLetra;
    bool aprovado;
};

struct CabecalhoDadosTreino {
    int32_t numeroMagico;
    int32_t versao;
    int32_t tamVocab;
    int32_t tamJanela; // 65 = FABRICA_TAM_JANELA + 1(entrada + alvo deslocado)
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

    // normaliza o texto bruto antes de qualquer outra etapa:
    //   - CRLF/CR -> LF
    //   - remove tokens puramente numericos isolados (numero de capitulo/versiculo
    //     tipo "41", "2:14" intercalados no meio do texto corrido) — sem isso o
    //     modelo tenta prever numeros de versiculo como se fossem parte do
    //     portugues, o que e ruido artificial sem correlacao linguistica real
    //   - normaliza espacos/tabs multiplos
    // retorna novo buffer alocado (dono chama free); tamSaida por referencia
    static char* normalizarTexto(const char* texto, int tamTexto, int* tamSaida) {
        char* out = (char*)malloc(tamTexto + 1);
        long op = 0;
        long i = 0;

        while(i < tamTexto) {
            unsigned char c = (unsigned char)texto[i];

            if(c == '\r') {
                if(i + 1 < tamTexto && texto[i+1] == '\n') i += 2; else i += 1;
                out[op++] = '\n';
                continue;
            }
            if(c == '\n') {
                out[op++] = '\n';
                i++;
                continue;
            }
            // token puramente numerico (com ':' ou '.' opcionais no meio, tipo
            // numero de capitulo:versiculo), isolado por espaco/quebra/inicio/fim
            if(isdigit(c)) {
                long j = i;
                while(j < tamTexto) {
                    unsigned char cj = (unsigned char)texto[j];
                    if(isdigit(cj) || cj == ':' || cj == '.') { j++; continue; }
                    break;
                }
                bool delimitadoDepois = (j >= tamTexto) || texto[j] == ' ' || texto[j] == '\n' || texto[j] == '\t';
                if(delimitadoDepois) {
                    i = j;
                    if(i < tamTexto && texto[i] == ' ') i++;
                    continue;
                }
            }
            if(c == ' ' || c == '\t') {
                out[op++] = ' ';
                i++;
                while(i < tamTexto && (texto[i] == ' ' || texto[i] == '\t')) i++;
                continue;
            }
            out[op++] = (char)c;
            i++;
        }
        out[op] = '\0';
        *tamSaida = (int)op;
        return out;
    }

    // separa o corpus em documentos por linha em branco dupla ("\n\n")
    // retorna vetor de pares (pos, tam) apontando pro buffer original
    static void _separarDocumentos(const char* texto, int tamTexto, Vetor<int>* posDoc, Vetor<int>* tamDoc) {
        int inicio = 0;
        int i = 0;
        while(i < tamTexto) {
            bool quebraDupla = (texto[i] == '\n' && i+1 < tamTexto && texto[i+1] == '\n');
            if(quebraDupla || i == tamTexto - 1) {
                int fim = quebraDupla ? i : tamTexto;
                if(fim > inicio) {
                    posDoc->empurrar(inicio);
                    tamDoc->empurrar(fim - inicio);
                }
                if(quebraDupla) {
                    i += 2;
                    while(i < tamTexto && texto[i] == '\n') i++;
                    inicio = i;
                    continue;
                }
            }
            i++;
        }
    }

    // analisa um documento e decide se passa no filtro de qualidade
    static EstatisticaDoc analisarDoc(const char* doc, int tam) {
        EstatisticaDoc est;
        est.tam = tam;
        est.numEstranho = 0;
        est.numDigito = 0;
        est.numLetra = 0;

        for(int i = 0; i < tam; i++) {
            unsigned char c = (unsigned char)doc[i];
            if(c >= 0x80) continue; // multibyte UTF-8: nao conta como estranho
            if(isalpha(c)) est.numLetra++;
            else if(isdigit(c)) est.numDigito++;
            else if(!isspace(c) && !ispunct(c)) est.numEstranho++;
        }

        est.aprovado = true;
        if(est.tam < FABRICA_TAM_MIN_DOC) est.aprovado = false;
        if(tam > 0) {
            float propEstranho = (float)est.numEstranho / (float)tam;
            float propDigito = (float)est.numDigito / (float)tam;
            if(propEstranho > FABRICA_PROPORCAO_MAX_ESTRANHO) est.aprovado = false;
            if(propDigito > FABRICA_PROPORCAO_MAX_DIGITO) est.aprovado = false;
        }
        return est;
    }

    // treina o BPE a partir do corpus e salva merges + vocab em disco
    static void treinarEsalvar(const char* caminhoCorpus, const char* caminhoMerges,const char* caminhoVocab, int maxMerges) {
        int tamTexto;
        char* textoBruto = lerArquivoTexto(caminhoCorpus, &tamTexto);
        if(!textoBruto) return;

        int tamNorm;
        char* texto = normalizarTexto(textoBruto, tamTexto, &tamNorm);
        free(textoBruto);
        tamTexto = tamNorm;

        TreinadorBPE treinador;
        treinador.treinar(texto, tamTexto, maxMerges);
        treinador.salvar(caminhoMerges);

        TokenizadorBPE tokenizador;
        tokenizador.carregarMerges(caminhoMerges);
        tokenizador.construirVocab(texto, tamTexto);
        tokenizador.salvarVocab(caminhoVocab);

        free(texto);
    }

    // carrega um tokenizador ja treinado(merges + vocab salvos)
    static void carregarTokenizador(TokenizadorBPE* tokenizador,
    const char* caminhoMerges, const char* caminhoVocab) {
        tokenizador->carregarMerges(caminhoMerges);
        tokenizador->carregarVocab(caminhoVocab);
    }

    // grava um bloco de sequencias(janelas de FABRICA_TAM_JANELA+1 tokens) em binario
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
    // le corpus, separa em documentos, filtra qualidade, codifica com
    // <FIM> entre documentos, fatia em janelas, separa treino/validacao, grava binario
    static void gerar(const char* caminhoCorpus, TokenizadorBPE* tokenizador, const char* caminhoTreino, const char* caminhoValidacao) {
        int tamTexto;
        char* textoBruto = lerArquivoTexto(caminhoCorpus, &tamTexto);
        if(!textoBruto) return;

        int tamNorm;
        char* texto = normalizarTexto(textoBruto, tamTexto, &tamNorm);
        free(textoBruto);
        tamTexto = tamNorm;

        Vetor<int> posDoc; posDoc.iniciar();
        Vetor<int> tamDoc; tamDoc.iniciar();
        _separarDocumentos(texto, tamTexto, &posDoc, &tamDoc);

        int numAprovados = 0;
        int numRejeitados = 0;

        Vetor<int> tokens; tokens.iniciar();
        Vetor<int> tokensDoc; tokensDoc.iniciar();

        for(int d = 0; d < posDoc.tam; d++) {
            const char* doc = texto + posDoc.dados[d];
            int tamAtual = tamDoc.dados[d];

            EstatisticaDoc est = analisarDoc(doc, tamAtual);
            if(!est.aprovado) {
                numRejeitados++;
                continue;
            }
            numAprovados++;

            tokensDoc.limpar();
            tokenizador->codificar(doc, tamAtual, &tokensDoc);
            for(int i = 0; i < tokensDoc.tam; i++) {
                tokens.empurrar(tokensDoc.dados[i]);
            }
            tokens.empurrar(ID_FIM);
        }
        tokensDoc.liberar();
        posDoc.liberar();
        tamDoc.liberar();
        free(texto);

        printf("Documentos: %d aprovados, %d rejeitados\n", numAprovados, numRejeitados);
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