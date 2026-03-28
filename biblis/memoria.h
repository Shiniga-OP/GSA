// biblis/memoria.h
#pragma once
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

using namespace std;

// entrada no indice RAM leve
struct EntradaMemoria {
    uint32_t id; // identificador unico
    vector<float> chave; // vetor de busca(vive em RAM)
    string arquivo; // caminho pro arquivo no disco
    float relevancia; // acumulador de relevancia(atualizado a cada acesso)
};

// conteudo carregado do disco pesado, so existe quando necessario
struct ConteudoMemoria {
    uint32_t id;
    vector<float> vetor; // vetor de valor completo
    string texto; // conteudo textual opcional
};

// MemoriaVetorial:
// - mantem chaves leves em RAM
// - carrega conteudo pesado do disco so quando o peso de atenção passa do limiar
// - decide automaticamente o que vale salvar com base em novidade e relevância
class MemoriaVetorial {
public:
    size_t dimChave; // dimensão dos vetores de chave
    size_t dimValor; // dimensão dos vetores de valor no disco
    float limiar; // peso mínimo de atenção pra carregar do disco
    size_t limiteEntradas; // máximo de entradas no indice RAM
    string diretorio; // pasta onde os arquivos ficam

    vector<EntradaMemoria> indice; // indice leve em RAM
    uint32_t proximoId = 0;

    MemoriaVetorial(size_t dimChave, size_t dimValor,
    const string& diretorio = "memoria",
    float limiar = 0.3f,
    size_t limiteEntradas = 10000)
    : dimChave(dimChave), dimValor(dimValor),
    diretorio(diretorio), limiar(limiar),
    limiteEntradas(limiteEntradas) {
        // cria o diretorio se não existir
        _criarDiretorio(diretorio);
    }
    // ESCRITA: salva um conhecimento nova no disco
    // retorna o id da entrada criada, ou -1 se rejeitada
    int32_t salvar(const vector<float>& chave,
    const vector<float>& valor, const string& texto = "") {
        if(chave.size() != dimChave)
            throw invalid_argument("[MemoriaVetorial]: dimensão da chave incorreta");
        if(valor.size() != dimValor)
            throw invalid_argument("[MemoriaVetorial]: dimensão do valor incorreta");

        // verifica novidade antes de salvar
        // se ja existe uma entrada muito similar, atualiza em vez de criar nova
        int32_t similar = _buscarSimilar(chave, 0.95f);
        if(similar >= 0) {
            // reforça a relevancia da entrada existente em vez de duplicar
            indice[similar].relevancia += 1.0f;
            return similar;
        }
        // indice cheio: descarta a entrada menos relevante
        if(indice.size() >= limiteEntradas) {
            _descartarMenosRelevante();
        }
        uint32_t id = proximoId++;
        string arquivo = diretorio + "/" + to_string(id) + ".mem";

        // salva conteudo pesado no disco
        _escreverArquivo(arquivo, id, valor, texto);

        // registra entrada leve no índice RAM
        EntradaMemoria entrada;
        entrada.id = id;
        entrada.chave = chave;
        entrada.arquivo = arquivo;
        entrada.relevancia = 1.0f;
        indice.push_back(entrada);

        return (int32_t)id;
    }

    // CONSULTA: retorna chaves do indice RAM pra CamadaAtencao
    // a atenção decide quais são relevantes
    const vector<EntradaMemoria>& consultarIndice() const {
        return indice;
    }

    // extrai so os vetores de chave pra passar pra CamadaAtencao
    vector<vector<float>> chaves() const {
        vector<vector<float>> cs;
        cs.reserve(indice.size());
        for(const auto& e : indice) cs.push_back(e.chave);
        return cs;
    }

    // LEITURA ECONOMICA: carrega do disco so as entradas acima do limiar
    // pesos vem diretamente de CamadaAtencao::pesosAtencao()
    vector<ConteudoMemoria> carregar(const vector<float>& pesos) {
        if(pesos.size() != indice.size()) throw invalid_argument("[MemoriaVetorial]: tamanho dos pesos não bate com o índice");

        vector<ConteudoMemoria> resultado;

        for(size_t i = 0; i < pesos.size(); i++) {
            if(pesos[i] >= limiar) {
                // reforça relevancia ao acessar
                indice[i].relevancia += pesos[i];

                ConteudoMemoria c = _lerArquivo(indice[i].arquivo);
                resultado.push_back(c);
            }
        }
        return resultado;
    }
    // versão que retorna pares(peso, conteudo) pra facilitar uso
    vector<pair<float, ConteudoMemoria>> carregarComPesos(const vector<float>& pesos) {
        if(pesos.size() != indice.size()) throw invalid_argument("[MemoriaVetorial]: tamanho dos pesos não bate com o índice");

        vector<pair<float, ConteudoMemoria>> resultado;

        for(size_t i = 0; i < pesos.size(); i++) {
            if(pesos[i] >= limiar) {
                indice[i].relevancia += pesos[i];
                ConteudoMemoria c = _lerArquivo(indice[i].arquivo);
                resultado.push_back({pesos[i], c});
            }
        }
        // ordena por peso decrescente
        sort(resultado.begin(), resultado.end(),
        [](const auto& a, const auto& b) {
            return a.first > b.first;
        });
        return resultado;
    }
    // PERSISTENCIA DO INDICE: salva/carrega o indice RAM no disco
    // pra sobreviver entre sessões
    void salvarIndice() const {
        string caminho = diretorio + "/indice.idc";
        ofstream a(caminho, ios::binary);
        if(!a) throw runtime_error("[MemoriaVetorial]: não foi possível salvar indice");

        uint32_t n = indice.size();
        uint32_t dc = dimChave;
        a.write((char*)&n, sizeof(n));
        a.write((char*)&dc, sizeof(dc));
        a.write((char*)&proximoId, sizeof(proximoId));

        for(const auto& e : indice) {
            a.write((char*)&e.id, sizeof(e.id));
            a.write((char*)e.chave.data(), dimChave * sizeof(float));
            a.write((char*)&e.relevancia, sizeof(e.relevancia));

            uint32_t tamanhoArq = e.arquivo.size();
            a.write((char*)&tamanhoArq, sizeof(tamanhoArq));
            a.write(e.arquivo.data(), tamanhoArq);
        }
    }

    void carregarIndice() {
        string caminho = diretorio + "/indice.idc";
        ifstream a(caminho, ios::binary);
        if(!a) return; // indice novo, sem historico

        uint32_t n, dc;
        a.read((char*)&n, sizeof(n));
        a.read((char*)&dc, sizeof(dc));
        a.read((char*)&proximoId, sizeof(proximoId));

        if(dc != dimChave)
            throw runtime_error("[MemoriaVetorial]: dimensão do índice salvo não bate");

        indice.clear();
        for(uint32_t i = 0; i < n; i++) {
            EntradaMemoria e;
            a.read((char*)&e.id, sizeof(e.id));

            e.chave.resize(dimChave);
            a.read((char*)e.chave.data(), dimChave * sizeof(float));
            a.read((char*)&e.relevancia, sizeof(e.relevancia));

            uint32_t tamanhoArq;
            a.read((char*)&tamanhoArq, sizeof(tamanhoArq));
            e.arquivo.resize(tamanhoArq);
            a.read(&e.arquivo[0], tamanhoArq);

            indice.push_back(e);
        }
    }
    size_t tamanho() const { return indice.size(); }

    // decaimento periodico de relevância(simula esquecimento natural)
    void decairRelevancia(float fator = 0.99f) {
        for(auto& e : indice) e.relevancia *= fator;
    }
    // FORMATO DO ARQUIVO DE VALOR(.mem)
    // [id: uint32][dimValor: uint32][vetor: float*dimValor][tamanhoTexto: uint32][texto: char*]
    void _escreverArquivo(const string& caminho, uint32_t id,
    const vector<float>& valor, const string& texto) {
        ofstream a(caminho, ios::binary);
        if(!a) throw runtime_error("[MemoriaVetorial]: não foi possível criar " + caminho);

        uint32_t dv = dimValor;
        a.write((char*)&id, sizeof(id));
        a.write((char*)&dv, sizeof(dv));
        a.write((char*)valor.data(), dimValor * sizeof(float));

        uint32_t tamanhoTexto = texto.size();
        a.write((char*)&tamanhoTexto, sizeof(tamanhoTexto));
        if(tamanhoTexto > 0) a.write(texto.data(), tamanhoTexto);
    }

    ConteudoMemoria _lerArquivo(const string& caminho) {
        ifstream a(caminho, ios::binary);
        if(!a) throw runtime_error("[MemoriaVetorial]: arquivo não encontrado: " + caminho);

        ConteudoMemoria c;
        uint32_t dv;
        a.read((char*)&c.id, sizeof(c.id));
        a.read((char*)&dv, sizeof(dv));

        c.vetor.resize(dv);
        a.read((char*)c.vetor.data(), dv * sizeof(float));

        uint32_t tamanhoTexto;
        a.read((char*)&tamanhoTexto, sizeof(tamanhoTexto));
        if(tamanhoTexto > 0) {
            c.texto.resize(tamanhoTexto);
            a.read(&c.texto[0], tamanhoTexto);
        }
        return c;
    }

    // busca entrada com similaridade cosseno >= limite
    // retorna indice no vetor indice, ou -1 se não encontrar
    int32_t _buscarSimilar(const vector<float>& chave, float limite) const {
        for(size_t i = 0; i < indice.size(); i++) {
            float sim = _cosseno(chave, indice[i].chave);
            if(sim >= limite) return (int32_t)i;
        }
        return -1;
    }

    float _cosseno(const vector<float>& a, const vector<float>& b) const {
        float dot = 0, na = 0, nb = 0;
        for(size_t i = 0; i < a.size(); i++) {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        float denom = sqrt(na) * sqrt(nb);
        if(denom < 1e-8f) return 0.0f;
        return dot / denom;
    }

    void _descartarMenosRelevante() {
        if(indice.empty()) return;

        // encontra o indice com menor relevancia
        size_t minIdc = 0;
        for(size_t i = 1; i < indice.size(); i++) {
            if(indice[i].relevancia < indice[minIdc].relevancia) minIdc = i;
        }
        // remove o arquivo do disco
        remove(indice[minIdc].arquivo.c_str());
        indice.erase(indice.begin() + minIdc);
    }

    void _criarDiretorio(const string& caminho) {
        // tenta criar o diretorio, ignora erro se ja existir
        #ifdef _WIN32
            system(("mkdir \"" + caminho + "\" 2>nul").c_str());
        #else
            system(("mkdir -p \"" + caminho + "\" 2>/dev/null").c_str());
        #endif
    }
};