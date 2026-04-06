// biblis/camadas/posicional.h
#pragma once
#include "camada.h"

// codificação posicional: soma um vetor de posição ao embedding de entrada

// dois modos:
// sinusoidal: vetores fixos derivados de sin/cos, sem parametros treinaveis
// PE[pos][2i] = sin(pos / 10000^(2i/dim))
// PE[pos][2i+1] = cos(pos / 10000^(2i/dim))

// treinavel: tabela E[seqMax x dim] iniciada com Xavier, treinada por gradiente

// uso:
// saida = entrada + PE[pos]
// retroprop: gradiente passa direto(soma não altera grad da entrada)
// se treinavel, acumula gradiente em gradE[posCache]

class CamadaPosicional : public Camada {
public:
    size_t dim;
    size_t seqMax;
    bool treinavel;

    // tabela de codificações [seqMax x dim]
    vector<vector<float>> PE;
    vector<vector<float>> gradE; // so usado se treinavel=true

    // cache pra retropropagação
    size_t posCache;

    CamadaPosicional(size_t dim, size_t seqMax, bool treinavel = false,
        const string& nome = "posicional")
        : Camada(nome), dim(dim), seqMax(seqMax),
          treinavel(treinavel), posCache(0) {

        tipo = "CamadaPosicional";

        if(treinavel) {
            PE = iniPesosXavier((int)seqMax, (int)dim);
            gradE = matrizZeros(seqMax, dim);
        } else {
            PE.assign(seqMax, vector<float>(dim, 0.0f));
            _calcSinusoidal();
        }
    }

    // prop: entrada(dim,) + PE[pos] -> saida(dim,)
    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: use prop(entrada, pos)");
    }

    vector<float> prop(const vector<float>& entrada, size_t pos) {
        if(entrada.size() != dim) {
            throw invalid_argument("[" + nome + "]: dimensão de entrada incorreta");
        }
        if(pos >= seqMax) {
            throw invalid_argument("[" + nome + "]: posição " + to_string(pos)
            + " fora do limite " + to_string(seqMax));
        }
        posCache = pos;

        vector<float> saida(dim);
        for(size_t i = 0; i < dim; i++) {
            saida[i] = entrada[i] + PE[pos][i];
        }
        return saida;
    }

    // retroprop: gradiente passa intacto pra entrada
    // se treinavel, acumula em gradE[posCache]
    GradGenerico retroprop(const vector<float>& gradSaida) override {
        if(gradSaida.size() != dim) {
            throw invalid_argument("[" + nome + "]: dimensão do gradiente incorreta");
        }
        if(treinavel) {
            for(size_t i = 0; i < dim; i++) {
                gradE[posCache][i] += gradSaida[i];
            }
        }
        // soma é operação linear: grad entrada = grad saida
        return GradGenerico(gradSaida);
    }

    void att(float taxaAprendizado) override {
        if(!treinavel) return;

        if(otimizador) {
            vector<float> biasZero(1, 0.0f);
            vector<float> gradBiasZero(1, 0.0f);
            otimizador->att(PE, gradE, biasZero, gradBiasZero);
        } else {
            for(size_t i = 0; i < seqMax; i++) {
                for(size_t j = 0; j < dim; j++) {
                    PE[i][j] -= taxaAprendizado * gradE[i][j];
                }
            }
        }
    }

    void zerarGradientes() override {
        if(!treinavel) return;
        for(auto& linha : gradE) fill(linha.begin(), linha.end(), 0.0f);
    }

    bool temParametros() const override { return treinavel; }
    size_t numParametros() const override { return treinavel ? seqMax * dim : 0; }

    void salvar(const string& arquivo) const override {
        ofstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para salvar");

        a.write(reinterpret_cast<const char*>(&dim), sizeof(dim));
        a.write(reinterpret_cast<const char*>(&seqMax), sizeof(seqMax));
        a.write(reinterpret_cast<const char*>(&treinavel), sizeof(treinavel));

        // sinusoidal é deterministico, não precisa salvar
        // treinavel precisa salvar a tabela aprendida
        if(treinavel) {
            for(const auto& linha : PE) {
                a.write(reinterpret_cast<const char*>(linha.data()), dim * sizeof(float));
            }
        }
    }

    void carregar(const string& arquivo) override {
        ifstream a(arquivo, ios::binary);
        if(!a) throw runtime_error("[" + nome + "]: falha ao abrir arquivo para carregar");

        a.read(reinterpret_cast<char*>(&dim), sizeof(dim));
        a.read(reinterpret_cast<char*>(&seqMax), sizeof(seqMax));
        a.read(reinterpret_cast<char*>(&treinavel), sizeof(treinavel));

        if(treinavel) {
            PE.assign(seqMax, vector<float>(dim));
            gradE = matrizZeros(seqMax, dim);
            for(auto& linha : PE) {
                a.read(reinterpret_cast<char*>(linha.data()), dim * sizeof(float));
            }
        } else {
            PE.assign(seqMax, vector<float>(dim, 0.0f));
            _calcSinusoidal();
        }
    }
    
    void _calcSinusoidal() {
        for(size_t pos = 0; pos < seqMax; pos++) {
            for(size_t i = 0; i < dim / 2; i++) {
                float freq = 1.0f / pow(10000.0f, (2.0f * i) / (float)dim);
                PE[pos][2 * i] = sin(pos * freq);
                PE[pos][2 * i + 1] = cos(pos * freq);
            }
            // se dim for impar, preenche ultimo com sin
            if(dim % 2 != 0) {
                float freq = 1.0f / pow(10000.0f, (float)(dim - 1) / (float)dim);
                PE[pos][dim - 1] = sin(pos * freq);
            }
        }
    }
};