// teste_adamw_alinhamento.cpp
// Verifica se o AdamW esta alinhado corretamente com os parametros do Modelo:
//   1) totalN (via _totalParams) bate com a soma de pTams[g] (via _coletarPtrs)?
//   2) cada ponteiro de parametro (pPtrs[g]) e cada ponteiro de gradiente (gPtrs[g])
//      realmente correspondem a ENDERECOS DENTRO das camadas do modelo (nao lixo)?
//   3) apos UM att(), os valores de "m" e "v" (estado interno) estao de fato
//      alinhados 1:1 com pPtrs (testado indiretamente: dando um gradiente so
//      num unico parametro e checando que so ELE muda, mais nada).
//
// Compilar (a partir da pasta que contem biblis/):
//   g++ -O2 -std=c++17 -I. teste_adamw_alinhamento.cpp -o teste_adamw_alinhamento
// Rodar:
//   ./teste_adamw_alinhamento
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include "biblis/modelo.h"
#include "biblis/otimis/adamw.h"

int main() {
    int vocab = 20, dim = 8, nCab = 2, dimFF = 16, nCamadas = 2, seqMax = 4;

    printf("=== teste_adamw_alinhamento ===\n");

    Modelo modelo(vocab, dim, nCab, dimFF, nCamadas, seqMax);
    modelo.inicializar("xavier");

    // total esperado somando numParams() de cada camada (fonte da verdade == _totalParams)
    int totalEsperado = 0;
    for(int c = 0; c < modelo.totalCamadas; c++) totalEsperado += modelo.todasCamadas[c]->numParams();

    AdamW otim;
    otim.iniciar(modelo.todasCamadas, modelo.totalCamadas, 1e-3f);

    printf("totalEsperado (soma numParams por camada) = %d\n", totalEsperado);
    printf("otim.totalN (via _totalParams)            = %d\n", otim.totalN);

    // soma dos tamanhos dos grupos coletados
    int somaGrupos = 0;
    for(int g = 0; g < otim.nGrupos; g++) somaGrupos += otim.pTams[g];
    printf("soma dos pTams dos %d grupos coletados     = %d\n", otim.nGrupos, somaGrupos);

    bool totalBate = (totalEsperado == otim.totalN) && (otim.totalN == somaGrupos);
    printf("totalN == somaGrupos == totalEsperado?     %s\n\n", totalBate ? "SIM (OK)" : "NAO (FALHOU)");

    // -----------------------------------------------------------------
    // TESTE 2: cada ponteiro de grupo coletado deve estar dentro da faixa
    // de memoria de ALGUMA camada real (nao aponta pra lixo/sobreposicao)
    // -----------------------------------------------------------------
    printf("--- grupos coletados (ponteiro, tamanho) ---\n");
    for(int g = 0; g < otim.nGrupos; g++) {
        printf("grupo %2d: pPtrs=%p tam=%d | gPtrs=%p tam=%d\n",
               g, (void*)otim.pPtrs[g], otim.pTams[g], (void*)otim.gPtrs[g], otim.gTams[g]);
    }
    printf("\n");

    // -----------------------------------------------------------------
    // TESTE 3: dar gradiente artificial SO no primeiro parametro do
    // PRIMEIRO grupo da camada do MEIO (pra nao ser caso trivial de borda),
    // rodar att(), e verificar que SOMENTE parametros daquele grupo mudaram.
    // -----------------------------------------------------------------
    modelo.zerarGrad();

    int grupoAlvo = otim.nGrupos / 2; // grupo do meio
    float* gradAlvo  = otim.gPtrs[grupoAlvo];

    // copia valores originais de TODOS os grupos antes do att()
    float** copiaOriginal = (float**)malloc(otim.nGrupos * sizeof(float*));
    for(int g = 0; g < otim.nGrupos; g++) {
        copiaOriginal[g] = (float*)malloc(otim.pTams[g] * sizeof(float));
        memcpy(copiaOriginal[g], otim.pPtrs[g], otim.pTams[g] * sizeof(float));
    }

    // grava gradiente artificial SO no grupo alvo, posicao 0
    gradAlvo[0] = 1.0f; // resto do buffer de grad ja esta zerado por zerarGrad()

    otim.att();

    printf("--- apos att() com gradiente artificial so em grupo %d, posicao 0 ---\n", grupoAlvo);
    int gruposAlterados = 0;
    for(int g = 0; g < otim.nGrupos; g++) {
        bool mudou = false;
        for(int k = 0; k < otim.pTams[g]; k++) {
            if(otim.pPtrs[g][k] != copiaOriginal[g][k]) { mudou = true; break; }
        }
        if(mudou) {
            gruposAlterados++;
            printf("grupo %2d MUDOU (esperado apenas para grupo %d)\n", g, grupoAlvo);
        }
    }
    // nota: weight decay (pd) faz TODOS os parametros encolherem um pouco
    // mesmo sem gradiente (p *= 1 - taxa*pd), entao e ESPERADO que todo
    // grupo mude minimamente por causa disso. o que NAO pode acontecer e
    // um grupo diferente do alvo mudar por causa do momento adam (m/v)
    // especifico, o que so seria visivel numa SEGUNDA chamada de att() com
    // gradiente zero everywhere. vamos fazer essa segunda checagem:

    // zera gradientes de novo (grad artificial ja foi consumido) e roda
    // um segundo att() com TODOS os gradientes zerados
    modelo.zerarGrad();
    // guarda estado apos primeiro att()
    float** aposPrimeiro = (float**)malloc(otim.nGrupos * sizeof(float*));
    for(int g = 0; g < otim.nGrupos; g++) {
        aposPrimeiro[g] = (float*)malloc(otim.pTams[g] * sizeof(float));
        memcpy(aposPrimeiro[g], otim.pPtrs[g], otim.pTams[g] * sizeof(float));
    }

    otim.att(); // gradiente zero em tudo -> so weight decay + momento residual do adam

    printf("\n--- apos SEGUNDO att() com gradiente ZERO em tudo ---\n");
    printf("(aqui, diferenca no grupo %d vem do momento 'm' remanescente do gradiente artificial;\n", grupoAlvo);
    printf(" diferenca em QUALQUER outro grupo alem do weight-decay uniforme indicaria desalinhamento)\n");
    for(int g = 0; g < otim.nGrupos; g++) {
        float maxDiff = 0.0f;
        for(int k = 0; k < otim.pTams[g]; k++) {
            float d = fabsf(otim.pPtrs[g][k] - aposPrimeiro[g][k]);
            if(d > maxDiff) maxDiff = d;
        }
        printf("grupo %2d: maxDiff apos 2o att() = %.8f%s\n",
               g, maxDiff, (g == grupoAlvo) ? "  <-- alvo (esperado maior)" : "");
    }

    printf("\n=== RESUMO ===\n");
    printf("Alinhamento totalN/somaGrupos: %s\n", totalBate ? "OK" : "FALHOU");
    printf("(inspecione a lista de grupos e os maxDiff acima pra confirmar\n");
    printf(" visualmente se algum grupo fora do alvo teve mudanca anormal)\n");

    for(int g = 0; g < otim.nGrupos; g++) { free(copiaOriginal[g]); free(aposPrimeiro[g]); }
    free(copiaOriginal);
    free(aposPrimeiro);
    otim.liberar();
    return totalBate ? 0 : 1;
}
