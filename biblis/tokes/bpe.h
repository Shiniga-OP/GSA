// biblis/tokes/bpe.h
#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <climits>

// retorna o tamanho em bytes do caractere UTF-8 que começa em c
static inline int _tamUTF8(unsigned char c) {
    if((c & 0x80) == 0)    return 1;
    if((c & 0xE0) == 0xC0) return 2;
    if((c & 0xF0) == 0xE0) return 3;
    if((c & 0xF8) == 0xF0) return 4;
    return 1; // byte de continuação isolado: trata como 1
}

// fragmenta string em caracteres UTF-8 completos
static inline std::vector<std::string> _splitUTF8(const std::string& s) {
    std::vector<std::string> chars;
    size_t i = 0;
    while(i < s.size()) {
        int tam = _tamUTF8((unsigned char)s[i]);
        // garante que não ultrapassa o fim da string
        if(i + tam > s.size()) tam = (int)(s.size() - i);
        chars.push_back(s.substr(i, tam));
        i += tam;
    }
    return chars;
}

class TokenizadorBPE {
public:
    explicit TokenizadorBPE(std::vector<std::pair<std::string,std::string>> merges = {}) {
        for(size_t i = 0; i < merges.size(); ++i) {
            bpeRanks[merges[i].first + " " + merges[i].second] = (int)i;
        }
        tokenPraId["<ALMO>"] = 0;
        tokenPraId["<DES>"]  = 1;
        tokenPraId["<FIM>"]  = 2;
        idPraToken[0] = "<ALMO>";
        idPraToken[1] = "<DES>";
        idPraToken[2] = "<FIM>";
        proximoId = 3;
    }

    void construirVocab(const std::vector<std::string>& textos) {
        cache.clear();

        // caracteres UTF-8 unicos primeiro
        for(const std::string& texto : textos) {
            for(const std::string& c : _splitUTF8(texto)) {
                if(c == " " || c == "\t" || c == "\n") continue;
                if(tokenPraId.find(c) == tokenPraId.end()) {
                    tokenPraId[c] = proximoId;
                    idPraToken[proximoId] = c;
                    proximoId++;
                }
            }
        }
        // tokens BPE completos
        for(const std::string& texto : textos) {
            for(const std::string& token : encode(texto)) {
                if(tokenPraId.find(token) == tokenPraId.end()) {
                    tokenPraId[token] = proximoId;
                    idPraToken[proximoId] = token;
                    proximoId++;
                }
            }
        }
        printf("Vocabulário construído: %d tokens\n", proximoId);
    }

    std::vector<int> codificar(const std::string& texto) {
        std::vector<int> resultado;
        for(const std::string& token : encode(texto)) {
            auto it = tokenPraId.find(token);
            if(it != tokenPraId.end()) {
                resultado.push_back(it->second);
            } else {
                // fragmenta em caracteres UTF-8
                for(const std::string& c : _splitUTF8(token)) {
                    auto cit = tokenPraId.find(c);
                    if(cit != tokenPraId.end()) resultado.push_back(cit->second);
                    else resultado.push_back(1); // <DES>
                }
            }
        }
        return resultado;
    }

    std::string decodificar(const std::vector<int>& ids) {
        std::vector<std::string> tokens;
        for(int id : ids) {
            if(id == 0 || id == 1 || id == 2) continue;
            auto it = idPraToken.find(id);
            tokens.push_back(it != idPraToken.end() ? it->second : "<DES>");
        }
        return decode(tokens);
    }

    int vocabTam() const { return proximoId; }

    std::unordered_map<std::string,int> tokenPraId;
    std::unordered_map<int,std::string> idPraToken;
    std::unordered_map<std::string,int> bpeRanks;
    std::unordered_map<std::string,std::vector<std::string>> cache;
    int proximoId;

    std::vector<std::string> bpe(const std::string& token) {
        auto cit = cache.find(token);
        if(cit != cache.end()) return cit->second;

        // fragmenta em caracteres UTF-8 completos
        std::vector<std::string> palavra = _splitUTF8(token);

        if(palavra.size() == 1) {
            cache[token] = palavra;
            return palavra;
        }
        while(true) {
            // encontra o par com menor rank
            int minRank = INT_MAX;
            std::string melhorPar;
            for(size_t i = 0; i + 1 < palavra.size(); i++) {
                std::string par = palavra[i] + " " + palavra[i+1];
                auto it = bpeRanks.find(par);
                if(it != bpeRanks.end() && it->second < minRank) {
                    minRank = it->second;
                    melhorPar = par;
                }
            }
            if(melhorPar.empty()) break;

            size_t esp = melhorPar.find(' ');
            std::string primeiro = melhorPar.substr(0, esp);
            std::string segundo = melhorPar.substr(esp + 1);

            std::vector<std::string> nova;
            size_t i = 0;
            while(i < palavra.size()) {
                if(i + 1 < palavra.size() &&
                   palavra[i] == primeiro && palavra[i+1] == segundo) {
                    nova.push_back(primeiro + segundo);
                    i += 2;
                } else {
                    nova.push_back(palavra[i]);
                    i++;
                }
            }
            palavra = nova;
        }
        cache[token] = palavra;
        return palavra;
    }

    std::vector<std::string> encode(const std::string& texto) {
        std::vector<std::string> tokens;
        std::istringstream iss(texto);
        std::string palavra;
        bool primeira = true;
        while(iss >> palavra) {
            std::vector<std::string> bpeTokens = bpe(palavra);
            if(!primeira && !bpeTokens.empty())
                bpeTokens[0] = "\xC4\xA0" + bpeTokens[0]; // Ġ em UTF-8
            tokens.insert(tokens.end(), bpeTokens.begin(), bpeTokens.end());
            primeira = false;
        }
        return tokens;
    }

    std::string decode(const std::vector<std::string>& tokens) {
        std::string texto;
        for(const std::string& token : tokens) {
            if(token.size() >= 2 &&
               (unsigned char)token[0] == 0xC4 &&
               (unsigned char)token[1] == 0xA0) {
                texto += ' ';
                texto += token.substr(2);
            } else {
                texto += token;
            }
        }
        return texto;
    }
};

class TreinadorBPE {
public:
    std::vector<std::pair<std::string,std::string>> merges;

    void treinar(const std::vector<std::string>& textos, int numMerges) {
        merges.clear();

        std::unordered_map<std::string,int> freqPalavras;
        for(const std::string& texto : textos) {
            std::istringstream iss(texto);
            std::string palavra;
            while(iss >> palavra) freqPalavras[palavra]++;
        }
        std::unordered_map<std::string,std::vector<std::string>> vocab;
        for(auto& par : freqPalavras) {
            vocab[par.first] = _splitUTF8(par.first); // UTF-8 correto
        }
        for(int iter = 0; iter < numMerges; ++iter) {
            std::unordered_map<std::string,int> freqPares;
            for(auto& entrada : vocab) {
                int freq = freqPalavras[entrada.first];
                const auto& tokens = entrada.second;
                for(size_t i = 0; i + 1 < tokens.size(); i++)
                    freqPares[tokens[i] + " " + tokens[i+1]] += freq;
            }
            if(freqPares.empty()) break;

            std::string melhorPar;
            int melhorFreq = -1;
            for(auto& p : freqPares) {
                if(p.second > melhorFreq ||
                  (p.second == melhorFreq && p.first < melhorPar)) {
                    melhorFreq = p.second;
                    melhorPar = p.first;
                }
            }
            if(melhorFreq <= 1) {
                printf("Parou no merge %d: nenhum par com frequência > 1\n", iter);
                break;
            }
            size_t esp = melhorPar.find(' ');
            std::string a = melhorPar.substr(0, esp);
            std::string b = melhorPar.substr(esp + 1);
            std::string ab = a + b;
            merges.push_back({a, b});

            if(iter < 10 || iter % 100 == 0) {
                printf("Merge %4d: '%s' + '%s' -> '%s' (freq=%d)\n",
                iter, a.c_str(), b.c_str(), ab.c_str(), melhorFreq);
            }
            for(auto& entrada : vocab) {
                std::vector<std::string>& tokens = entrada.second;
                std::vector<std::string> novo;
                size_t i = 0;
                while(i < tokens.size()) {
                    if(i + 1 < tokens.size() &&
                       tokens[i] == a && tokens[i+1] == b) {
                        novo.push_back(ab);
                        i += 2;
                    } else {
                        novo.push_back(tokens[i++]);
                    }
                }
                tokens = novo;
            }
        }
        printf("Treinamento concluído: %d merges\n", (int)merges.size());
    }

    void salvar(const std::string& caminho) const {
        FILE* a = fopen(caminho.c_str(), "w");
        if(!a) {
            printf("Erro ao salvar merges\n");
            return;
        }
        for(const auto& m : merges) {
            fprintf(a, "%s %s\n", m.first.c_str(), m.second.c_str());
        }
        fclose(a);
    }

    void carregar(const std::string& caminho) {
        merges.clear();
        FILE* a = fopen(caminho.c_str(), "r");
        if(!a) {
            printf("Erro ao carregar merges\n");
            return;
        }
        char x[256], y[256];
        while(fscanf(a, "%255s %255s", x, y) == 2) {
            merges.push_back({x, y});
        }
        fclose(a);
    }
};