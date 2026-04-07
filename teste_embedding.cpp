// biblis/teste_embedding.cpp
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>

#include "biblis/camadas/embedding.h"

// utilidades de teste
static int falhas = 0;
static int total  = 0;

static void checar(const char* nome, bool cond) {
    total++;
    if(cond) {
        printf("  [OK] %s\n", nome);
    } else {
        printf("  [FALHA] %s\n", nome);
        falhas++;
    }
}

static bool quaseIgual(float a, float b, float tol = 1e-5f) {
    return fabsf(a - b) <= tol;
}

// teste 1: prop basica: saida igual a linha da tabela
static void testePropBasica() {
    printf("\n[1] prop básica\n");
    const int V = 10, D = 4, SEQ = 3;
    Embedding emb(V, D, SEQ);

    memset(emb.tabela, 0, V * D * sizeof(float));
    for(int v = 0; v < V; v++) {
        for(int d = 0; d < D; d++) {
            emb.tabela[v*D + d] = (float)(v*10 + d);
        }
    }
    int ids[SEQ] = {0, 3, 7};
    float saida[SEQ * D];
    emb.tamSeq = SEQ;
    emb.prop((const float*)ids, saida);

    bool ok = true;
    for(int t = 0; t < SEQ; t++) {
        int id = ids[t];
        for(int d = 0; d < D; d++) {
            if(!quaseIgual(saida[t*D + d], emb.tabela[id*D + d])) {
                ok = false;
            }
        }
    }
    checar("saida == tabela[id] para cada token", ok);

    bool idsOk = true;
    for(int t = 0; t < SEQ; t++) {
        if(emb.ultIds[t] != ids[t]) idsOk = false;
    }
    checar("ultIds gravados corretamente", idsOk);
}

// teste 2: prop não modifica tabela
static void testePropNaoModificaTabela() {
    printf("\n[2] prop não modifica tabela\n");
    const int V = 8, D = 6, SEQ = 4;
    Embedding emb(V, D, SEQ);

    float copia[V * D];
    memcpy(copia, emb.tabela, V * D * sizeof(float));

    int ids[SEQ] = {0, 2, 5, 7};
    float saida[SEQ * D];
    emb.tamSeq = SEQ;
    emb.prop((const float*)ids, saida);

    bool ok = true;
    for(int i = 0; i < V * D; i++) {
        if(!quaseIgual(emb.tabela[i], copia[i])) ok = false;
    }
    checar("tabela intacta após prop", ok);
}

// teste 3: retroprop dispersão-adição com gradientes assimetricos
static void testeRetroprop() {
    printf("\n[3] retroprop dispersão-adição\n");
    const int V = 5, D = 4, SEQ = 4;
    Embedding emb(V, D, SEQ);
    emb.zerarGrad();

    // ids com repetição: token 2 aparece duas vezes
    int ids[SEQ] = {1, 2, 2, 3};
    float saida[SEQ * D];
    emb.tamSeq = SEQ;
    emb.prop((const float*)ids, saida);

    // gradientes distintos por posição e dimensão para não mascarar bugs de indexação
    float gs[SEQ * D];
    for(int t = 0; t < SEQ; t++) {
        for(int d = 0; d < D; d++) {
            gs[t*D + d] = (float)(t * 37 + d * 13 + 1); // primo * posição, assimetrico
        }
    }
    emb.retroprop(gs, nullptr);

    // token 1 (t=0)
    bool ok1 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[1*D + d], gs[0*D + d])) ok1 = false;
    }
    checar("gradTab[id=1] acumulado corretamente", ok1);

    // token 2 (t=1 e t=2): deve ser soma exata de cada dimensão
    bool ok2 = true;
    for(int d = 0; d < D; d++) {
        float esperado = gs[1*D + d] + gs[2*D + d];
        if(!quaseIgual(emb.gradTab[2*D + d], esperado)) ok2 = false;
    }
    checar("gradTab[id=2] dispersão-adição com repetição", ok2);

    // token 3 (t=3)
    bool ok3 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[3*D + d], gs[3*D + d])) ok3 = false;
    }
    checar("gradTab[id=3] acumulado corretamente", ok3);

    // tokens não usados(0 e 4)
    bool ok4 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[0*D + d], 0.0f)) ok4 = false;
        if(!quaseIgual(emb.gradTab[4*D + d], 0.0f)) ok4 = false;
    }
    checar("gradTab[ids não usados] == 0", ok4);
}

// teste 4: zerarGrad limpa tudo
static void testeZerarGrad() {
    printf("\n[4] zerarGrad\n");
    const int V = 8, D = 6, SEQ = 3;
    Embedding emb(V, D, SEQ);

    for(int i = 0; i < V * D; i++) emb.gradTab[i] = 99.0f;
    emb.zerarGrad();

    bool ok = true;
    for(int i = 0; i < V * D; i++) {
        if(!quaseIgual(emb.gradTab[i], 0.0f)) ok = false;
    }
    checar("gradTab zerado completamente", ok);
}

// teste 5: acumulação em multiplos lotes
static void testeAcumulacaolote() {
    printf("\n[5] acumulação em multiplos lotes\n");
    const int V = 4, D = 3, SEQ = 2;
    Embedding emb(V, D, SEQ);
    emb.zerarGrad();

    int ids1[SEQ] = {0, 1};
    float saida[SEQ * D];
    float gs1[SEQ * D];
    for(int i = 0; i < SEQ * D; i++) gs1[i] = 1.0f;
    emb.tamSeq = SEQ;
    emb.prop((const float*)ids1, saida);
    emb.retroprop(gs1, nullptr);

    int ids2[SEQ] = {0, 2};
    float gs2[SEQ * D];
    for(int i = 0; i < SEQ * D; i++) gs2[i] = 2.0f;
    emb.prop((const float*)ids2, saida);
    emb.retroprop(gs2, nullptr);

    bool ok0 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[0*D + d], 3.0f)) ok0 = false;
    }
    checar("token 0 acumulou grad de 2 lotes (1+2=3)", ok0);

    bool ok1 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[1*D + d], 1.0f)) ok1 = false;
    }
    checar("token 1 grad apenas do lote 1", ok1);

    bool ok2 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[2*D + d], 2.0f)) ok2 = false;
    }
    checar("token 2 grad apenas do lote 2", ok2);

    bool ok3 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[3*D + d], 0.0f)) ok3 = false;
    }
    checar("token 3 não usado permanece 0", ok3);
}

// teste 6: gradiente numerico exaustivo
// perturba tabela[v][d] para cada(v,d) de token que aparece
// na sequencia e confere gradTab[v][d] analítico vs numerico
// tambem verifica que tokens ausentes ficam com grad zero

// [NOTA]: a perda é acumulada em double para evitar erro de
// cancelamento catastrofico na diferença finita centrada com
// gradientes de grande magnitude(float32 + EPS=1e-3 não
// tem precisão suficiente quando a perda está na casa de milhares)
static void testeGradienteNumerico() {
    printf("\n[6] gradiente numérico exaustivo\n");
    const int V = 6, D = 8, SEQ = 5;
    const float EPS = 1e-3f;
    const float TOL = 1e-2f;

    // token 1 repete(t=0,t=2), token 4 repete(t=3,t=4), token 0 ausente
    int ids[SEQ] = {1, 3, 1, 4, 4};

    // conjunto de tokens presentes
    bool presente[V] = {};
    for(int t = 0; t < SEQ; t++) presente[ids[t]] = true;

    // gradiente de saida: valores arbitrários assimetricos
    float gs[SEQ * D];
    for(int i = 0; i < SEQ * D; i++) gs[i] = (float)(i * 7 + 3);

    // função auxiliar: executa prop e retorna soma ponderada(perda) em double
    // usa gs como pesos fixos: L = soma(gs[i] * saida[i])
    // acumulação em double elimina erro de cancelamento na diferença finita
    auto perda = [&](Embedding& emb) -> double {
        float saida[SEQ * D];
        emb.tamSeq = SEQ;
        emb.prop((const float*)ids, saida);
        double s = 0.0;
        for(int i = 0; i < SEQ * D; i++) s += (double)gs[i] * (double)saida[i];
        return s;
    };

    Embedding emb(V, D, SEQ);
    // tabela com valores conhecidos, não-triviais
    for(int i = 0; i < V * D; i++) emb.tabela[i] = (float)(i % 7) * 0.1f + 0.05f;

    // calcula grad analitico
    emb.zerarGrad();
    emb.tamSeq = SEQ;
    float saida[SEQ * D];
    emb.prop((const float*)ids, saida);
    emb.retroprop(gs, nullptr);

    bool okPresente = true;
    bool okAusente = true;
    int verificados = 0;

    for(int v = 0; v < V; v++) {
        for(int d = 0; d < D; d++) {
            float ga = emb.gradTab[v*D + d];

            if(!presente[v]) {
                // deve ser exatamente zero
                if(!quaseIgual(ga, 0.0f)) okAusente = false;
                continue;
            }
            // grad numerico por diferenças centradas(resultado em double)
            float orig = emb.tabela[v*D + d];

            emb.tabela[v*D + d] = orig + EPS;
            double pm = perda(emb);

            emb.tabela[v*D + d] = orig - EPS;
            double pp = perda(emb);

            emb.tabela[v*D + d] = orig;

            float gn = (float)((pm - pp) / (2.0 * (double)EPS));

            if(fabsf(ga - gn) >= TOL) {
                printf("    divergência em tabela[%d][%d]: analitico=%.6f numerico=%.6f\n",
                v, d, ga, gn);
                okPresente = false;
            }
            verificados++;
        }
    }
    checar("grad analítico ~= grad numérico para todos (v,d) presentes", okPresente);
    checar("grad zero para todos os tokens ausentes", okAusente);
    printf("    (%d pares (v,d) verificados numericamente)\n", verificados);
}

// teste 7: interface Camada
static void testeInterface() {
    printf("\n[7] interface Camada\n");
    const int V = 12, D = 5;
    Embedding emb(V, D);
    
    printf("Parâmetros = %d;\n", emb.numParams());

    checar("numParams == vocab*dim", emb.numParams() == V * D);
    
    float* p[1]; int tp[1];
    emb.params(p, tp);
    checar("params[0] == tabela", p[0] == emb.tabela);
    checar("tams[0] == vocab*dim", tp[0] == V * D);

    float* g[1]; int tg[1];
    emb.gradParams(g, tg);
    checar("gradParams[0] == gradTab", g[0] == emb.gradTab);
    checar("gradTams[0] == vocab*dim", tg[0] == V * D);
}

// teste 8: inicialização normal, media ~0, variancia ~1/dim
static void testeInicializacao() {
    printf("\n[8] inicialização normal\n");
    const int V = 1000, D = 128;
    Embedding emb(V, D);
    emb.inicializar("normal");

    float media = 0, var = 0;
    int n = V * D;
    for(int i = 0; i < n; i++) media += emb.tabela[i];
    media /= n;
    for(int i = 0; i < n; i++) {
        float d = emb.tabela[i] - media;
        var += d * d;
    }
    var /= n;

    float varPrevista = 1.0f / (float)D;
    checar("variância ~= 1/dim (±15%)", fabsf(var - varPrevista) < varPrevista * 0.15f);
    checar("média ~= 0 (|media| < 0.02)", fabsf(media) < 0.02f);
}

// teste 9: sequencia de comprimento 1
static void testeSeqUm() {
    printf("\n[9] sequência de comprimento 1\n");
    const int V = 5, D = 4;
    Embedding emb(V, D, 16);
    memset(emb.tabela, 0, V * D * sizeof(float));
    for(int d = 0; d < D; d++) emb.tabela[2*D + d] = (float)(d + 1);

    int ids[1] = {2};
    float saida[D];
    emb.tamSeq = 1;
    emb.prop((const float*)ids, saida);

    bool ok = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(saida[d], (float)(d + 1))) ok = false;
    }
    checar("seq=1 prop correta", ok);

    emb.zerarGrad();
    float gs[D];
    for(int d = 0; d < D; d++) gs[d] = (float)(d * 5 + 1); // assimétrico
    emb.retroprop(gs, nullptr);

    bool okG = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[2*D + d], gs[d])) okG = false;
    }
    checar("seq=1 retroprop correta", okG);

    // outros tokens intocados
    bool okZ = true;
    for(int v = 0; v < V; v++) {
        if(v == 2) continue;
        for(int d = 0; d < D; d++) {
            if(!quaseIgual(emb.gradTab[v*D + d], 0.0f)) okZ = false;
        }
    }
    checar("seq=1 outros tokens grad=0", okZ);
}

// teste 10: multiplas instancias independentes
static void testeIndependencia() {
    printf("\n[10] independência entre instâncias\n");
    const int V = 4, D = 3, SEQ = 2;
    Embedding a(V, D, SEQ), b(V, D, SEQ);

    iniConstante(a.tabela, V * D, 1.0f);
    iniConstante(b.tabela, V * D, 2.0f);

    int ids[SEQ] = {0, 1};
    float saidaA[SEQ * D], saidaB[SEQ * D];
    a.tamSeq = SEQ; a.prop((const float*)ids, saidaA);
    b.tamSeq = SEQ; b.prop((const float*)ids, saidaB);

    bool ok = true;
    for(int i = 0; i < SEQ * D; i++) {
        if(!quaseIgual(saidaA[i], 1.0f)) ok = false;
        if(!quaseIgual(saidaB[i], 2.0f)) ok = false;
    }
    checar("instâncias com tabelas distintas não interferem", ok);

    a.zerarGrad(); b.zerarGrad();
    float gs[SEQ * D];
    for(int i = 0; i < SEQ*D; i++) gs[i] = 5.0f;
    a.retroprop(gs, nullptr);

    bool okG = true;
    for(int i = 0; i < V * D; i++) {
        if(!quaseIgual(b.gradTab[i], 0.0f)) okG = false;
    }
    checar("grad de instância a não contamina b", okG);
}

// teste 11: token repetido muitas vezes, acumulo linear
static void testeTokenMuitasVezes() {
    printf("\n[11] token repetido N vezes\n");
    const int V = 3, D = 4, SEQ = 8, N_REP = 6;
    Embedding emb(V, D, SEQ);
    emb.zerarGrad();

    // token 1 aparece N_REP vezes, token 0 aparece 2 vezes
    int ids[SEQ] = {1, 1, 1, 0, 1, 1, 0, 1};
    float saida[SEQ * D];
    emb.tamSeq = SEQ;
    emb.prop((const float*)ids, saida);

    float gs[SEQ * D];
    for(int t = 0; t < SEQ; t++) {
        for(int d = 0; d < D; d++) {
            gs[t*D + d] = (float)(t + d + 1);
        }
    }
    emb.retroprop(gs, nullptr);

    // calcula esperado manualmente
    float esperado1[D] = {};
    float esperado0[D] = {};
    for(int t = 0; t < SEQ; t++) {
        float* dest = (ids[t] == 1) ? esperado1 : esperado0;
        for(int d = 0; d < D; d++) dest[d] += gs[t*D + d];
    }
    bool ok1 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[1*D + d], esperado1[d])) ok1 = false;
    }
    checar("token repetido 6x acumulou corretamente", ok1);

    bool ok0 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[0*D + d], esperado0[d])) ok0 = false;
    }
    checar("token repetido 2x acumulou corretamente", ok0);

    bool ok2 = true;
    for(int d = 0; d < D; d++) {
        if(!quaseIgual(emb.gradTab[2*D + d], 0.0f)) ok2 = false;
    }
    checar("token nunca usado permanece 0", ok2);
}

int main() {
    printf("=== teste embedding ===\n");
    testePropBasica();
    testePropNaoModificaTabela();
    testeRetroprop();
    testeZerarGrad();
    testeAcumulacaolote();
    testeGradienteNumerico();
    testeInterface();
    testeInicializacao();
    testeSeqUm();
    testeIndependencia();
    testeTokenMuitasVezes();
    printf("\n=== resultado: %d/%d passaram ===\n", total - falhas, total);
    return falhas > 0 ? 1 : 0;
}
