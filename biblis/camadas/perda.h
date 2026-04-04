// biblis/camadas/perda.h
#pragma once
#include "../util.h"

// perda por entropia cruzada com softmax integrado
// entrada: logits crus(antes do softmax)
// alvo: indice inteiro da classe correta

// perda = -log(softmax(logits)[alvo])

// gradiente analitico combinado softmax+entropia cruzada:
// dL/dlogits[i] = softmax[i] - 1(i == alvo)

struct CamadaPerda {
    vector<float> probsCache;
    size_t alvoCache;

    float prop(const vector<float>& logits, size_t alvo) {
        if(alvo >= logits.size()) {
            throw invalid_argument("[CamadaPerda]: alvo " + to_string(alvo)
            + " fora do intervalo [0, " + to_string(logits.size()) + ")");
        }
        alvoCache = alvo;
        probsCache = softmax(logits);

        float prob = probsCache[alvo];
        if(prob < 1e-10f) prob = 1e-10f;
        return -log(prob);
    }

    vector<float> retroprop() const {
        vector<float> grad = probsCache;
        grad[alvoCache] -= 1.0f;
        return grad;
    }

    float propLote(
        const vector<vector<float>>& logitsLote,
        const vector<size_t>& alvos
    ) {
        if(logitsLote.size() != alvos.size())
            throw invalid_argument("[CamadaPerda]: tamanho do lote inconsistente");
        float soma = 0.0f;
        for(size_t i = 0; i < logitsLote.size(); i++)
            soma += prop(logitsLote[i], alvos[i]);
        return soma / (float)logitsLote.size();
    }

    float propOneHot(const vector<float>& logits, const vector<float>& alvoOH) {
        if(logits.size() != alvoOH.size())
            throw invalid_argument("[CamadaPerda]: dimensões incompatíveis");
        probsCache = softmax(logits);
        alvoCache = (size_t)(max_element(alvoOH.begin(), alvoOH.end()) - alvoOH.begin());
        float prob = probsCache[alvoCache];
        if(prob < 1e-10f) prob = 1e-10f;
        return -log(prob);
    }
};