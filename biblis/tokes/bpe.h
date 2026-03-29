// biblis/tokes/bpe.h
#pragma once
#include <unordered_map>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <climits>

class TokenizadorBPE {
public:
    explicit TokenizadorBPE(std::vector<std::pair<std::string,std::string>> merges = std::vector<std::pair<std::string,std::string>>()) {
        for(size_t i = 0; i < merges.size(); ++i) {
            std::string chave = merges[i].first + " " + merges[i].second;
            bpeRanks[chave] = (int)i;
        }
        tokenPraId["<ALMO>"] = 0;
        tokenPraId["<DES>"]  = 1;
        tokenPraId["<FIM>"]  = 2;
        idPraToken[0] = "<ALMO>";
        idPraToken[1] = "<DES>";
        idPraToken[2] = "<FIM>";
        proximoId = 3;
    }
    // constroi o vocabulario a partir dos textos
    // o cache é limpo antes de encode pra evitar entradas corrompidas
    // de chamadas anteriores a construirVocab
    void construirVocab(const std::vector<std::string>& textos) {
        cache.clear();

        // caracteres unicos primeiro(base do vocab)
        for(const std::string& texto : textos) {
            for(unsigned char c : texto) {
                if(!isspace(c)) {
                    std::string s(1, (char)c);
                    if(tokenPraId.find(s) == tokenPraId.end()) {
                        tokenPraId[s] = proximoId;
                        idPraToken[proximoId] = s;
                        proximoId++;
                    }
                }
            }
        }
        // agora encode usa bpeRanks com vocab de caracteres ja completo
        for(const std::string& texto : textos) {
            std::vector<std::string> tokens = encode(texto);
            for(const std::string& token : tokens) {
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
        std::vector<std::string> tokensBPE = encode(texto);
        std::vector<int> resultado;
        for(const std::string& token : tokensBPE) {
            auto it = tokenPraId.find(token);
            if(it != tokenPraId.end()) {
                resultado.push_back(it->second);
            } else {
                // retorno caractere a caractere
                for(unsigned char c : token) {
                    std::string s(1, (char)c);
                    auto cit = tokenPraId.find(s);
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
            if(id == 0 || id == 1 || id == 2) continue; // ignora tokens especiais
            auto it = idPraToken.find(id);
            if(it != idPraToken.end()) tokens.push_back(it->second);
            else tokens.push_back("<DES>");
        }
        return decode(tokens);
    }

    int vocabTam() const {
        return proximoId;
    }

    std::unordered_map<std::string,int> tokenPraId;
    std::unordered_map<int,std::string> idPraToken;
    std::unordered_map<std::string,int> bpeRanks;
    std::unordered_map<std::string,std::vector<std::string>> cache;
    int proximoId;

    // retorna pares adjacentes em ordem de posição(deterministico)
    std::vector<std::string> obterPares(const std::vector<std::string>& palavra) {
        std::vector<std::string> pares;
        for(size_t i = 0; i < palavra.size() - 1; ++i)
            pares.push_back(palavra[i] + " " + palavra[i+1]);
        return pares;
    }

    std::vector<std::string> bpe(const std::string& token) {
        auto cit = cache.find(token);
        if(cit != cache.end()) return cit->second;

        std::vector<std::string> palavra;
        for(unsigned char c : token) palavra.push_back(std::string(1, (char)c));

        if(palavra.size() == 1) {
            cache[token] = palavra;
            return palavra;
        }
        while(true) {
            std::vector<std::string> pares = obterPares(palavra);
            if(pares.empty()) break;

            // escolhe o par com menor rank, desempatando pela primeira ocorrencia
            int minRank = INT_MAX;
            std::string melhorPar;
            for(const std::string& par : pares) {
                auto it = bpeRanks.find(par);
                if(it != bpeRanks.end() && it->second < minRank) {
                    minRank = it->second;
                    melhorPar = par;
                }
            }
            if(melhorPar.empty()) break;

            size_t espaco = melhorPar.find(' ');
            std::string primeiro = melhorPar.substr(0, espaco);
            std::string segundo  = melhorPar.substr(espaco + 1);

            std::vector<std::string> novaPalavra;
            size_t i = 0;
            while(i < palavra.size()) {
                auto it = std::find(palavra.begin() + i, palavra.end(), primeiro);
                if(it == palavra.end()) {
                    novaPalavra.insert(novaPalavra.end(), palavra.begin() + i, palavra.end());
                    break;
                }
                size_t j = (size_t)(it - palavra.begin());
                novaPalavra.insert(novaPalavra.end(), palavra.begin() + i, palavra.begin() + j);
                if(j + 1 < palavra.size() && palavra[j+1] == segundo) {
                    novaPalavra.push_back(primeiro + segundo);
                    i = j + 2;
                } else {
                    novaPalavra.push_back(primeiro);
                    i = j + 1;
                }
            }
            palavra = novaPalavra;
        }
        cache[token] = palavra;
        return palavra;
    }
    // codifica texto em tokens BPE, espaço representado como prefixo "Ġ"
    // no primeiro token de cada palavra(exceto a primeira)
    std::vector<std::string> encode(const std::string& texto) {
        std::vector<std::string> tokens;
        std::istringstream iss(texto);
        std::string palavra;
        bool primeira = true;
        while(iss >> palavra) {
            std::vector<std::string> bpeTokens = bpe(palavra);
            if(!primeira && !bpeTokens.empty()) {
                // prefixo de espaço no primeiro sub-token da palavra
                bpeTokens[0] = "Ġ" + bpeTokens[0];
            }
            tokens.insert(tokens.end(), bpeTokens.begin(), bpeTokens.end());
            primeira = false;
        }
        return tokens;
    }

    std::string decode(const std::vector<std::string>& tokens) {
        std::string texto;
        for(const std::string& token : tokens) {
            if(token.size() >= 2 && (unsigned char)token[0] == 0xC4 && (unsigned char)token[1] == 0xA0) {
                // "Ġ" em UTF-8 = 0xC4 0xA0
                texto += ' ';
                texto += token.substr(2);
            } else {
                texto += token;
            }
        }
        return texto;
    }
};

// treina merges BPE a partir de um corpus
class TreinadorBPE {
public:
    std::vector<std::pair<std::string,std::string>> merges;
    
    // textos: corpus de treinamento
    // numMerges: quantos pares fundir(tamanho do vocabulario = chars unicos + numMerges + tokens especiais)
    void treinar(const std::vector<std::string>& textos, int numMerges) {
        merges.clear();

        // representa cada palavra como sequencia de caracteres + frequencia
        std::unordered_map<std::string, int> freqPalavras;
        for(const std::string& texto : textos) {
            std::istringstream iss(texto);
            std::string palavra;
            while(iss >> palavra) freqPalavras[palavra]++;
        }
        // converte para representação interna: vetor de tokens por palavra
        std::unordered_map<std::string, std::vector<std::string>> vocab;
        for(auto& par : freqPalavras) {
            std::vector<std::string> chars;
            // itera em UTF-8 preservando caracteres multibyte
            const std::string& w = par.first;
            size_t i = 0;
            while(i < w.size()) {
                unsigned char c = (unsigned char)w[i];
                int tam = 1;
                if((c & 0x80) == 0) tam = 1;
                else if((c & 0xE0) == 0xC0) tam = 2;
                else if((c & 0xF0) == 0xE0) tam = 3;
                else if((c & 0xF8) == 0xF0) tam = 4;
                chars.push_back(w.substr(i, tam));
                i += tam;
            }
            vocab[par.first] = chars;
        }
        for(int iter = 0; iter < numMerges; ++iter) {
            // conta frequencia de cada par adjacente ponderada pela frequencia da palavra
            std::unordered_map<std::string, int> freqPares;
            for(auto& entrada : vocab) {
                const std::string& palavra = entrada.first;
                const std::vector<std::string>& tokens = entrada.second;
                int freq = freqPalavras[palavra];
                for(size_t i = 0; i + 1 < tokens.size(); ++i) {
                    std::string par = tokens[i] + " " + tokens[i+1];
                    freqPares[par] += freq;
                }
            }
            if(freqPares.empty()) break;

            // encontra o par mais frequente(desempate lexicografico pra determinismo)
            std::string melhorPar;
            int melhorFreq = -1;
            for(auto& p : freqPares) {
                if(p.second > melhorFreq || (p.second == melhorFreq && p.first < melhorPar)) {
                    melhorFreq = p.second;
                    melhorPar = p.first;
                }
            }
            if(melhorFreq <= 1) {
                // nenhum par aparece mais de uma vez, não vale fundir
                printf("Parou no merge %d: nenhum par com frequência > 1\n", iter);
                break;
            }
            size_t espaco = melhorPar.find(' ');
            std::string a = melhorPar.substr(0, espaco);
            std::string b = melhorPar.substr(espaco + 1);
            std::string ab = a + b;

            merges.push_back({a, b});

            if(iter % 100 == 0 || iter < 10) {
                printf("Merge %4d: '%s' + '%s' -> '%s' (freq=%d)\n", iter, a.c_str(), b.c_str(), ab.c_str(), melhorFreq);
            }
            // aplica o merge no vocab
            for(auto& entrada : vocab) {
                std::vector<std::string>& tokens = entrada.second;
                std::vector<std::string> novo;
                size_t i = 0;
                while(i < tokens.size()) {
                    if(i + 1 < tokens.size() && tokens[i] == a && tokens[i+1] == b) {
                        novo.push_back(ab);
                        i += 2;
                    } else {
                        novo.push_back(tokens[i]);
                        i++;
                    }
                }
                tokens = novo;
            }
        }
        printf("Treinamento concluído: %d merges\n", (int)merges.size());
    }

    // salva merges em arquivo texto(um merge por linha: "a b")
    void salvar(const std::string& caminho) const {
        FILE* a = fopen(caminho.c_str(), "w");
        if(!a) {
            printf("Erro ao salvar merges em '%s'\n", caminho.c_str());
            return;
        }
        for(const auto& m : merges) fprintf(a, "%s %s\n", m.first.c_str(), m.second.c_str());
        fclose(a);
        printf("Merges salvos em '%s'\n", caminho.c_str());
    }

    // carrega merges de arquivo salvo por salvar()
    void carregar(const std::string& caminho) {
        merges.clear();
        FILE* arq = fopen(caminho.c_str(), "r");
        if(!arq) {
            printf("Erro ao carregar merges de '%s'\n", caminho.c_str());
            return;
        }
        char a[256], b[256];
        while(fscanf(arq, "%255s %255s", a, b) == 2) {
            merges.push_back({std::string(a), std::string(b)});
        }
        fclose(arq);
        printf("Merges carregados: %d\n", (int)merges.size());
    }
};