// teste_pipeline.cpp
// TESTE HONESTO DE CONVERGENCIA, sem disfarce de temperatura.
// objetivo: provar ou refutar se o modelo consegue convergir de verdade
// num recorte PEQUENO e CONTROLADO do corpus real, rodando VARIAS EPOCAS
// completas sobre ele. isso elimina a hipotese de "pouco dado/poucas
// passadas"(o teste anterior via so ~8% do corpus, uma vez, sem repeticao).
// avaliacao SEMPRE por argmax, nunca por sampling com
// temperatura - se o modelo nao aprendeu, greedy mostra isso sem disfarce.
// perda sempre reportada ao lado do chute aleatorio(ln(vocab)).

#include <stdio.h>
#include <math.h>
#include "biblis/util.h"
#include "biblis/toke/bpe.h"
#include "biblis/toke/fabrica_dados.h"
#include "biblis/modelo.h"
#include "biblis/otimis/otimizador.h"
#include "biblis/otimis/adamw.h"

// geracao SEMPRE por argmax, sem temperatura, sem sampling.
// se o modelo aprendeu algo real, argmax deve mostrar isso; se nao
// aprendeu, argmax mostra o problema cru, sem maquiagem.
static void gerarArgmax(Modelo& modelo, TokenizadorBPE& tok, const int* semente, int tamSemente, int tamGerar, char* saidaTexto, int capSaida) {
    int seqMax = modelo.seqMax;
    int* buf = (int*)malloc(seqMax * sizeof(int));
    int tamBuf = tamSemente < seqMax ? tamSemente : seqMax;
    memcpy(buf, semente + (tamSemente - tamBuf), tamBuf * sizeof(int));

    int* todosIds = (int*)malloc((tamSemente + tamGerar) * sizeof(int));
    memcpy(todosIds, semente, tamSemente * sizeof(int));

    for(int g = 0; g < tamGerar; g++) {
        modelo.defSeq(tamBuf);
        modelo.prop(buf);

        float* ultimoLogit = modelo.logits + (tamBuf - 1) * modelo.vocab;
        int melhor = 0;
        float melhorVal = ultimoLogit[0];
        for(int v = 1; v < modelo.vocab; v++) {
            if(ultimoLogit[v] > melhorVal) { melhorVal = ultimoLogit[v]; melhor = v; }
        }
        todosIds[tamSemente + g] = melhor;

        if(tamBuf < seqMax) {
            buf[tamBuf] = melhor;
            tamBuf++;
        } else {
            memmove(buf, buf + 1, (seqMax - 1) * sizeof(int));
            buf[seqMax - 1] = melhor;
        }
    }
    int tamDec;
    char* decodificado = tok.decodificar(todosIds, tamSemente + tamGerar, &tamDec);
    int copiar = tamDec < capSaida - 1 ? tamDec : capSaida - 1;
    memcpy(saidaTexto, decodificado, copiar);
    saidaTexto[copiar] = '\0';
    free(decodificado);
    free(todosIds);
    free(buf);
}

int main() {
    srand(42);

    // === 1. ler e normalizar corpus real ===
    int tamTextoBruto;
    char* textoBruto = FabricaDados::lerArquivoTexto("biblia.txt", &tamTextoBruto);
    if(!textoBruto) { printf("FALHOU: nao leu corpus\n"); return 1; }
    printf("corpus lido: %d bytes\n", tamTextoBruto);

    int tamTextoTotal;
    char* textoTotal = FabricaDados::normalizarTexto(textoBruto, tamTextoBruto, &tamTextoTotal);
    free(textoBruto);
    printf("corpus normalizado: %d bytes\n", tamTextoTotal);

    // === 2. RECORTE PEQUENO E FIXO do corpus real (nao o corpus de brinquedo,
    // nao um sorteio aleatorio - um pedaco continuo e real da Biblia, pequeno
    // o bastante pra caber em varias epocas completas em minutos) ===
    int tamRecorte = 100000; // ~1MB de texto real, bem menor que os 3.9MB inteiros
    int inicioRecorte = tamTextoTotal / 3; // comeca 1/3 adentro, evita cabecalhos/indices do inicio
    if(inicioRecorte + tamRecorte > tamTextoTotal) tamRecorte = tamTextoTotal - inicioRecorte;
    char* texto = (char*)malloc(tamRecorte + 1);
    memcpy(texto, textoTotal + inicioRecorte, tamRecorte);
    texto[tamRecorte] = '\0';
    int tamTexto = tamRecorte;
    free(textoTotal);
    printf("recorte de teste: %d bytes (posicao %d a %d do corpus normalizado)\n",
        tamTexto, inicioRecorte, inicioRecorte + tamRecorte);

    // === 3. BPE treinado SOBRE O PROPRIO RECORTE (consistente com o que o
    // modelo vai ver; menos merges, recorte é pequeno) ===
    TreinadorBPE treinador;
    treinador.treinar(texto, tamTexto, 4000);
    treinador.salvar("merges_teste.txt");

    TokenizadorBPE tok;
    tok.carregarMerges("merges_teste.txt");
    tok.construirVocab(texto, tamTexto);

    int vocab = tok.vocabTam();
    float linhaPerda = logf((float)vocab);
    printf("vocab: %d tokens (linha de chute aleatorio: perda=%.4f)\n", vocab, linhaPerda);

    Vetor<int> tokens; tokens.iniciar();
    tok.codificar(texto, tamTexto, &tokens);
    printf("tokens codificados no recorte: %d\n", tokens.tam);
    free(texto);

    if(tokens.tam < 200) {
        printf("FALHOU: recorte gerou poucos tokens\n");
        return 1;
    }
    // === 4. modelo pequeno ===
    int dim = 64;
    int nCab = 4;
    int dimFF = 128;
    int nCamadas = 2;
    int seqMax = 64;

    Modelo modelo(vocab, dim, nCab, dimFF, nCamadas, seqMax);
    modelo.inicializar("xavier");
    printf("modelo criado: vocab=%d dim=%d nCab=%d dimFF=%d nCamadas=%d seqMax=%d\n",
        vocab, dim, nCab, dimFF, nCamadas, seqMax);

    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, 3e-3f);

    // === 5. sequencias de treino: TODAS as janelas possiveis do recorte,
    // com stride pequeno (mais sobreposicao = mais sequencias por epoca) ===
    int seq = seqMax;
    int stride = 8;
    int nSeqs = 0;
    for(int pos = 0; pos + seq + 1 <= tokens.tam; pos += stride) nSeqs++;
    if(nSeqs == 0) { printf("FALHOU: nenhuma sequencia de treino formada\n"); return 1; }
    printf("sequencias de treino por epoca: %d\n", nSeqs);

    int* idsEnt = (int*)malloc(seq * sizeof(int));
    int* idsAlvo = (int*)malloc(seq * sizeof(int));

    // pool de sementes em posicoes variadas do recorte, nao so o inicio,
    // pra amostra de geracao mudar a cada epoca em vez de repetir sempre
    // o mesmo trecho cortado
    int tamSemente = 8;
    int nSementes = 5;
    int* sementes = (int*)malloc(nSementes * tamSemente * sizeof(int));
    for(int sIdx = 0; sIdx < nSementes; sIdx++) {
        int posBase = (tokens.tam - tamSemente) * sIdx / nSementes;
        for(int i = 0; i < tamSemente; i++) sementes[sIdx * tamSemente + i] = tokens[posBase + i];
    }

    char bufTexto[2048];

    // === 6. loop de treino: MUITAS EPOCAS REAIS sobre o recorte pequeno.
    // isso e o teste decisivo: se convergir aqui, o problema anterior era
    // volume/cobertura de dado. se NAO convergir nem aqui, o problema e
    // estrutural (arquitetura, taxa, ou algo mais profundo) e independente
    // do tamanho do corpus. ===
    int epocas = 20;
    int passosPorEpoca = nSeqs;
    int passosTotais = epocas * passosPorEpoca;
    int aquecimento = passosPorEpoca; // 1 epoca de aquecimento
    bool viuNan = false;

    printf("treinando %d epocas (%d passos totais, %d passos/epoca)\n",
        epocas, passosTotais, passosPorEpoca);

    AgendadorCosseno agenda;
    agenda.taxaMax = 3e-3f; // reduzido 10x: 3e-3 estava divergindo (perda subindo, colapso em "ooo"/"de de de")
    agenda.taxaMin = 1e-6f;
    agenda.passosTotal = passosTotais;
    agenda.aquecimento = aquecimento;

    int* ordemSeqs = (int*)malloc(nSeqs * sizeof(int));
    for(int i = 0; i < nSeqs; i++) ordemSeqs[i] = i;

    modelo.defSeq(seq);

    int passoGlobal = 0;
    for(int ep = 0; ep < epocas; ep++) {
        // reembaralha a ordem a cada epoca (epoca real: passa por todas as
        // sequencias antes de repetir qualquer uma)
        for(int i = nSeqs - 1; i > 0; i--) {
            int j = rand() % (i + 1);
            int tmp = ordemSeqs[i]; ordemSeqs[i] = ordemSeqs[j]; ordemSeqs[j] = tmp;
        }
        float perdaSomaEpoca = 0.0f;

        for(int s = 0; s < nSeqs; s++) {
            otim.taxa = agenda.calcular(passoGlobal);

            int idcSeq = ordemSeqs[s];
            int base = idcSeq * stride;
            for(int i = 0; i < seq; i++) {
                idsEnt[i]  = tokens[base + i];
                idsAlvo[i] = tokens[base + i + 1];
            }
            modelo.zerarGrad();
            modelo.prop(idsEnt);
            float perda = perdaEntropiaCruzada(modelo.logits, modelo.gradLogits, idsAlvo, modelo.seqAtual, modelo.vocab);
            
            modelo.retroprop();
            otim.att();

            if(isnan(perda) || isinf(perda)) {
                printf("FALHOU na epoca %d, passo %d: perda = %f (NaN/Inf)\n", ep, s, perda);
                viuNan = true;
                break;
            }
            perdaSomaEpoca += perda;
            passoGlobal++;
        }
        if(viuNan) break;

        float perdaMediaEpoca = perdaSomaEpoca / (float)nSeqs;

        // log a cada epoca: perda media da epoca inteira(nao de 1 sequencia
        // isolada, que e ruidosa) vs o linha de chute aleatorio
        // semente sorteada do pool, pra amostra variar entre epocas
        int sementeEp = rand() % nSementes;
        gerarArgmax(modelo, tok, sementes + sementeEp * tamSemente, tamSemente, 50, bufTexto, sizeof(bufTexto));
        const char* alerta = (perdaMediaEpoca > linhaPerda) ? ", PIOR QUE CHUTE ALEATORIO, DIVERGINDO" : "";
        printf("\n--- epoca %d/%d, perda_media=%.4f, linha_de_chute=%.4f, taxa=%.6f%s ---\n%s\n",
            ep+1, epocas, perdaMediaEpoca, linhaPerda, otim.taxa, alerta, bufTexto);
    }
    free(ordemSeqs);
    free(idsEnt);
    free(idsAlvo);

    if(!viuNan) {
        printf("\n=== GERACAO FINAL ===\n");
        for(int sIdx = 0; sIdx < nSementes; sIdx++) {
            gerarArgmax(modelo, tok, sementes + sIdx * tamSemente, tamSemente, 80, bufTexto, sizeof(bufTexto));
            printf("[semente %d] %s\n", sIdx, bufTexto);
        }
    }
    free(sementes);
    otim.liberar();
    tokens.liberar();
    return (viuNan) ? 1 : 0;
}