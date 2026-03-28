// teste_memoria.cpp
// compila: g++ -std=c++17 -O2 -o teste_memoria teste_memoria.cpp
#include <iostream>
#include <vector>
#include <string>
#include <cmath>
#include <fstream>
#include <cstdio>
#include "biblis/ativas.h"
#include "biblis/util.h"
#include "biblis/otimizadores.h"
#include "biblis/atencao.h"
#include "biblis/memoria.h"

using namespace std;

int totalTestes = 0;
int testesPassados = 0;

void verificar(bool cond, const string& desc) {
    totalTestes++;
    if(cond) { testesPassados++; cout << "  [OK] " << desc << endl; }
    else cout << "  [FALHOU] " << desc << endl;
}
void verificarPerto(float a, float b, const string& desc, float tol = 1e-4f) {
    verificar(abs(a-b) < tol, desc + " (" + to_string(a) + " ~= " + to_string(b) + ")");
}
void secao(const string& t) { cout << "\n=== " << t << " ===" << endl; }

void limparDiretorio(const string& dir) {
    system(("rm -rf " + dir).c_str());
}

// =====================================================================
// TESTE 1: escrita e leitura básica
// =====================================================================
void testeEscritaLeitura() {
    secao("Escrita e leitura básica");
    limparDiretorio("mem_teste1");

    MemoriaVetorial mem(4, 4, "mem_teste1", 0.1f);

    vector<float> chave  = {1.0f, 0.0f, 0.0f, 0.0f};
    vector<float> valor  = {0.5f, 0.2f, 0.8f, 0.1f};
    string texto = "experiencia A";

    int32_t id = mem.salvar(chave, valor, texto);
    verificar(id >= 0, "salvar retorna id válido (" + to_string(id) + ")");
    verificar(mem.tamanho() == 1, "índice tem 1 entrada");

    // pesos: entrada 0 acima do limiar
    vector<float> pesos = {0.9f};
    auto resultado = mem.carregar(pesos);
    verificar(resultado.size() == 1, "carregar retorna 1 resultado acima do limiar");
    verificar(resultado[0].id == (uint32_t)id, "id correto no conteúdo carregado");
    verificar(resultado[0].texto == texto, "texto preservado: '" + resultado[0].texto + "'");

    // verifica vetor
    verificar(resultado[0].vetor.size() == 4, "vetor tem 4 elementos");
    verificarPerto(resultado[0].vetor[0], 0.5f, "vetor[0] = 0.5");
    verificarPerto(resultado[0].vetor[2], 0.8f, "vetor[2] = 0.8");

    limparDiretorio("mem_teste1");
}

// =====================================================================
// TESTE 2: limiar de leitura lazy
// =====================================================================
void testeLimiar() {
    secao("Leitura lazy — limiar de atenção");
    limparDiretorio("mem_teste2");

    MemoriaVetorial mem(4, 4, "mem_teste2", 0.4f); // limiar = 0.4

    // 3 entradas
    mem.salvar({1,0,0,0}, {1,0,0,0}, "A");
    mem.salvar({0,1,0,0}, {0,1,0,0}, "B");
    mem.salvar({0,0,1,0}, {0,0,1,0}, "C");
    verificar(mem.tamanho() == 3, "3 entradas no índice");

    // pesos: só o segundo acima do limiar
    vector<float> pesos = {0.1f, 0.8f, 0.1f};
    auto resultado = mem.carregar(pesos);
    verificar(resultado.size() == 1, "só 1 entrada carregada (limiar=0.4, só pesos[1]=0.8 passa)");
    verificar(resultado[0].texto == "B", "entrada carregada é a B");

    // pesos: dois acima do limiar
    pesos = {0.5f, 0.1f, 0.6f};
    resultado = mem.carregar(pesos);
    verificar(resultado.size() == 2, "2 entradas carregadas (pesos[0]=0.5 e pesos[2]=0.6)");

    // pesos: nenhum acima do limiar
    pesos = {0.1f, 0.2f, 0.3f};
    resultado = mem.carregar(pesos);
    verificar(resultado.size() == 0, "0 entradas carregadas (todos abaixo do limiar=0.4)");

    limparDiretorio("mem_teste2");
}

// =====================================================================
// TESTE 3: carregarComPesos ordena por relevância
// =====================================================================
void testeOrdenacaoPorPeso() {
    secao("carregarComPesos — ordenação decrescente");
    limparDiretorio("mem_teste3");

    MemoriaVetorial mem(4, 4, "mem_teste3", 0.1f);
    mem.salvar({1,0,0,0}, {1,0,0,0}, "fraca");
    mem.salvar({0,1,0,0}, {0,1,0,0}, "forte");
    mem.salvar({0,0,1,0}, {0,0,1,0}, "media");

    vector<float> pesos = {0.2f, 0.9f, 0.5f};
    auto resultado = mem.carregarComPesos(pesos);

    verificar(resultado.size() == 3, "3 resultados carregados");
    verificar(resultado[0].second.texto == "forte", "primeiro é o mais forte (peso=0.9)");
    verificar(resultado[1].second.texto == "media", "segundo é o médio (peso=0.5)");
    verificar(resultado[2].second.texto == "fraca", "terceiro é o mais fraco (peso=0.2)");
    verificar(resultado[0].first > resultado[1].first, "pesos em ordem decrescente [0]>[1]");
    verificar(resultado[1].first > resultado[2].first, "pesos em ordem decrescente [1]>[2]");

    limparDiretorio("mem_teste3");
}

// =====================================================================
// TESTE 4: deduplicação — entradas similares não são duplicadas
// =====================================================================
void testeDeduplicacao() {
    secao("Deduplicação — entradas similares não duplicam");
    limparDiretorio("mem_teste4");

    MemoriaVetorial mem(4, 4, "mem_teste4", 0.1f);

    vector<float> chaveA = {1.0f, 0.0f, 0.0f, 0.0f};
    int32_t id1 = mem.salvar(chaveA, {1,0,0,0}, "original");
    verificar(mem.tamanho() == 1, "1 entrada após primeiro salvar");

    // chave quase idêntica (cosseno ~= 1.0)
    vector<float> chaveQuaseA = {0.9999f, 0.0001f, 0.0f, 0.0f};
    int32_t id2 = mem.salvar(chaveQuaseA, {1,0,0,0}, "duplicata");
    verificar(mem.tamanho() == 1, "ainda 1 entrada após salvar quase-duplicata");
    verificar(id1 == id2, "retorna id da entrada existente (" + to_string(id1) + "==" + to_string(id2) + ")");

    // chave diferente — deve criar nova entrada
    vector<float> chaveB = {0.0f, 1.0f, 0.0f, 0.0f};
    int32_t id3 = mem.salvar(chaveB, {0,1,0,0}, "diferente");
    verificar(mem.tamanho() == 2, "2 entradas após salvar chave diferente");
    verificar(id3 != id1, "id novo é diferente do original");

    limparDiretorio("mem_teste4");
}

// =====================================================================
// TESTE 5: persistência do índice entre sessões
// =====================================================================
void testePersistencia() {
    secao("Persistência — índice sobrevive entre sessões");
    limparDiretorio("mem_teste5");

    // sessão 1: salva e persiste
    {
        MemoriaVetorial mem(4, 4, "mem_teste5", 0.1f);
        mem.salvar({1,0,0,0}, {1,0,0,0}, "sessao1-A");
        mem.salvar({0,1,0,0}, {0,1,0,0}, "sessao1-B");
        mem.salvarIndice();
        verificar(mem.tamanho() == 2, "sessão 1: 2 entradas salvas");
    }

    // sessão 2: carrega e verifica
    {
        MemoriaVetorial mem(4, 4, "mem_teste5", 0.1f);
        mem.carregarIndice();
        verificar(mem.tamanho() == 2, "sessão 2: 2 entradas recuperadas");

        // leitura lazy: limiar=0.1, pesos={0.9, 0.05} → só A passa
        vector<float> pesos = {0.9f, 0.05f};
        auto resultado = mem.carregar(pesos);
        verificar(resultado.size() == 1, "leitura lazy funciona após reload");
        verificar(resultado[0].texto == "sessao1-A", "conteúdo correto após reload");
    }

    limparDiretorio("mem_teste5");
}

// =====================================================================
// TESTE 6: descarte por relevância quando índice está cheio
// =====================================================================
void testeDescarte() {
    secao("Descarte — entrada menos relevante é removida quando cheio");
    limparDiretorio("mem_teste6");

    // limite de 3 entradas
    MemoriaVetorial mem(4, 4, "mem_teste6", 0.1f, 3);

    int32_t idA = mem.salvar({1,0,0,0}, {1,0,0,0}, "A");
    int32_t idB = mem.salvar({0,1,0,0}, {0,1,0,0}, "B");
    int32_t idC = mem.salvar({0,0,1,0}, {0,0,1,0}, "C");
    verificar(mem.tamanho() == 3, "índice cheio com 3 entradas");

    // reforça relevância de A e C acessando-as
    vector<float> pesosAC = {0.9f, 0.0f, 0.9f};
    mem.carregar(pesosAC); // A e C ganham relevância extra

    // adiciona D — deve descartar B (menos acessada)
    mem.salvar({0,0,0,1}, {0,0,0,1}, "D");
    verificar(mem.tamanho() == 3, "índice ainda com 3 entradas após inserção");

    // verifica que B foi descartada
    bool temA = false, temB = false, temC = false, temD = false;
    for(const auto& e : mem.consultarIndice()) {
        if(e.id == (uint32_t)idA) temA = true;
        if(e.id == (uint32_t)idB) temB = true;
        if(e.id == (uint32_t)idC) temC = true;
    }
    // verifica pelo texto carregando tudo
    vector<float> pesosAll = {0.9f, 0.9f, 0.9f};
    auto resultado = mem.carregar(pesosAll);
    bool textoB = false;
    for(auto& r : resultado) if(r.texto == "B") textoB = true;
    verificar(!textoB, "entrada B foi descartada (menos relevante)");
    verificar(resultado.size() <= 3, "no máximo 3 entradas no índice");

    limparDiretorio("mem_teste6");
}

// =====================================================================
// TESTE 7: integração com CamadaAtencao
// =====================================================================
void testeIntegracao() {
    secao("Integração — MemoriaVetorial + CamadaAtencao");
    limparDiretorio("mem_teste7");

    const size_t D = 4;
    MemoriaVetorial mem(D, D, "mem_teste7", 0.3f);

    // popula memória com 3 experiências
    mem.salvar({1,0,0,0}, {0.9f,0.1f,0.0f,0.0f}, "experiencia A");
    mem.salvar({0,1,0,0}, {0.0f,0.8f,0.2f,0.0f}, "experiencia B");
    mem.salvar({0,0,1,0}, {0.0f,0.1f,0.9f,0.0f}, "experiencia C");

    // CamadaAtencao consulta a memória
    CamadaAtencao at(D, D, D);
    // força Wq e Wk pra identidade pra comportamento determinístico
    at.Wq = identidade(D);
    at.Wk = identidade(D);
    at.Wv = identidade(D);

    // estado = similar à chave A
    vector<float> estado = {0.9f, 0.1f, 0.0f, 0.0f};
    auto chavesRam = mem.chaves();

    verificar(chavesRam.size() == 3, "chaves retornadas corretamente (" + to_string(chavesRam.size()) + ")");

    // propagação da atenção
    at.prop(estado, chavesRam);
    auto& pesos = at.pesosAtencao();

    // com Wq=Wk=I: peso maior deve ser na entrada mais similar ao estado
    // estado = [0.9,0.1,0,0], chave A = [1,0,0,0] → dot = 0.9 (maior)
    verificar(pesos[0] > pesos[1], "atenção foca na entrada A (mais similar ao estado)");
    verificar(pesos[0] > pesos[2], "atenção foca na entrada A (mais similar ao estado)");

    // leitura lazy: só A deve ser carregada (peso > 0.3)
    auto conteudo = mem.carregar(vector<float>(pesos.begin(), pesos.end()));
    verificar(!conteudo.empty(), "pelo menos uma entrada carregada");
    verificar(conteudo[0].texto == "experiencia A", "entrada A carregada (maior peso)");

    limparDiretorio("mem_teste7");
}

// =====================================================================
// TESTE 8: decaimento de relevância
// =====================================================================
void testeDecaimento() {
    secao("Decaimento de relevância");
    limparDiretorio("mem_teste8");

    MemoriaVetorial mem(4, 4, "mem_teste8", 0.1f);
    mem.salvar({1,0,0,0}, {1,0,0,0}, "A");

    float relevanciaInicial = mem.consultarIndice()[0].relevancia;
    mem.decairRelevancia(0.5f);
    float relevanciaDepois = mem.consultarIndice()[0].relevancia;

    verificarPerto(relevanciaDepois, relevanciaInicial * 0.5f,
                   "relevância decai pelo fator correto");
    verificar(relevanciaDepois < relevanciaInicial, "relevância diminui após decaimento");

    limparDiretorio("mem_teste8");
}

// =====================================================================
// MAIN
// =====================================================================
int main() {
    cout << "=====================================================" << endl;
    cout << "  TESTES — MemoriaVetorial" << endl;
    cout << "=====================================================" << endl;

    testeEscritaLeitura();
    testeLimiar();
    testeOrdenacaoPorPeso();
    testeDeduplicacao();
    testePersistencia();
    testeDescarte();
    testeIntegracao();
    testeDecaimento();

    cout << "\n=====================================================" << endl;
    cout << "  RESULTADO: " << testesPassados << "/" << totalTestes << " testes passaram" << endl;
    if(testesPassados == totalTestes) cout << "  TUDO OK" << endl;
    else cout << "  " << (totalTestes - testesPassados) << " FALHARAM" << endl;
    cout << "=====================================================" << endl;

    return (testesPassados == totalTestes) ? 0 : 1;
}