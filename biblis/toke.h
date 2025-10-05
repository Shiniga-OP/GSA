#include <iostream>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>

class TokenizadorBPE {
public:
    TokenizadorBPE(std::vector<std::pair<std::string,std::string>> merges) {
        // inicia merges
        for(size_t i = 0; i < merges.size(); ++i) {
            std::string chave = merges[i].first + " " + merges[i].second;
            bpeRanks[chave] = i;
        }
        // tokens especiais
        tokenPraId["<ALMO>"] = 0;
        tokenPraId["<DES>"] = 1;
        tokenPraId["<FIM>"] = 2;

        idPraToken[0] = "<ALMO>";
        idPraToken[1] = "<DES>";
        idPraToken[2] = "<FIM>";

        proximoId = 3;
    }

    void construirVocab(const std::vector<std::string>& textos) {
        std::unordered_set<std::string> todosTokens;
        std::unordered_set<char> todosCaracteres;
        // tokens especiais
        todosTokens.insert("<ALMO>");
        todosTokens.insert("<DES>");
        todosTokens.insert("<FIM>");
        // coleta caracteres únicos
        for(const std::string& texto : textos) for(char c : texto) if(!isspace(c)) todosCaracteres.insert(c);

        for(char c : todosCaracteres) todosTokens.insert(std::string(1, c));
        // processa textos para BPE
        for(const std::string& texto : textos) {
            std::vector<std::string> tokens = encode(texto);
            for(const std::string& token : tokens) todosTokens.insert(token);
        }
        // mapeia tokens para IDs
        int id = 3;
        for(const std::string& token : todosTokens) {
            if(tokenPraId.find(token) == tokenPraId.end()) {
                tokenPraId[token] = id;
                idPraToken[id] = token;
                id++;
            }
        }
        proximoId = id;
        printf("Vocabulário construído: %d tokens\n", proximoId);
    }

    std::vector<int> codificar(const std::string& texto) {
        std::vector<std::string> tokensBPE = encode(texto);
        std::vector<int> resultado;
        for(const std::string& token : tokensBPE) {
            auto it = tokenPraId.find(token);
            if(it != tokenPraId.end()) resultado.push_back(it->second);
            else {
                for(char c : token) {
                    auto cit = tokenPraId.find(std::string(1,c));
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
            if(id == 2) continue; // <FIM>
            auto it = idPraToken.find(id);
            if(it != idPraToken.end()) tokens.push_back(it->second);
            else tokens.push_back("<DES>");
        }
        return decode(tokens);
    }

    int vocabTam() {
        return proximoId;
    }

public:
    std::unordered_map<std::string,int> tokenPraId;
    std::unordered_map<int, std::string> idPraToken;
    std::unordered_map<std::string,int> bpeRanks;
    std::unordered_map<std::string,std::vector<std::string>> cache;
    int proximoId;

    std::unordered_set<std::string> obterPares(const std::vector<std::string>& palavra) {
        std::unordered_set<std::string> pares;
        for(size_t i = 0; i < palavra.size() - 1; ++i) pares.insert(palavra[i] + " " + palavra[i+1]);
        return pares;
    }

    std::vector<std::string> bpe(const std::string& token) {
        if(cache.find(token) != cache.end()) return cache[token];

        std::vector<std::string> palavra;
        for(char c : token) palavra.push_back(std::string(1,c));

        std::unordered_set<std::string> pares = obterPares(palavra);
        if(pares.empty()) return { token };

        while(true) {
            int minRank = INT32_MAX;
            std::string melhorPar;
            for(const std::string& par : pares) {
                auto it = bpeRanks.find(par);
                if(it != bpeRanks.end() && it->second < minRank) {
                    minRank = it->second;
                    melhorPar = par;
                }
            }
            if(melhorPar.empty()) break;

            std::string primeiro = melhorPar.substr(0, melhorPar.find(' '));
            std::string segundo = melhorPar.substr(melhorPar.find(' ')+1);

            std::vector<std::string> novaPalavra;
            size_t i = 0;
            while(i < palavra.size()) {
                auto it = std::find(palavra.begin()+i, palavra.end(), primeiro);
                if(it == palavra.end()) {
                    novaPalavra.insert(novaPalavra.end(), palavra.begin()+i, palavra.end());
                    break;
                }
                size_t j = it - palavra.begin();
                novaPalavra.insert(novaPalavra.end(), palavra.begin()+i, palavra.begin()+j);
                if(j < palavra.size()-1 && palavra[j+1] == segundo) {
                    novaPalavra.push_back(primeiro+segundo);
                    i = j+2;
                } else {
                    novaPalavra.push_back(primeiro);
                    i = j+1;
                }
            }
            palavra = novaPalavra;
            pares = obterPares(palavra);
        }
        cache[token] = palavra;
        return palavra;
    }

    std::vector<std::string> encode(const std::string& texto) {
        std::vector<std::string> tokens;
        std::istringstream iss(texto);
        std::string palavra;
        while(iss >> palavra) {
            std::vector<std::string> bpeTokens = bpe(palavra);
            if(bpeTokens.size()==1 && bpeTokens[0]==palavra && tokenPraId.find(palavra)==tokenPraId.end())
                for(char c : palavra) tokens.push_back(std::string(1,c));
            else tokens.insert(tokens.end(), bpeTokens.begin(), bpeTokens.end());
            tokens.push_back("Ġ"); // espaço entre palavras
        }
        if(!tokens.empty() && tokens.back()=="Ġ") tokens.pop_back();
        return tokens;
    }

    std::string decode(const std::vector<std::string>& tokens) {
        std::string texto;
        for(const std::string& token : tokens) {
            if(token == "Ġ") texto += ' ';
            else if(token.size() > 1 && token.substr(0,2) == "Ġ") texto += ' ' + token.substr(2);
            else texto += token;
        }
        return texto;
    }
};

void testeT() {
    std::cout << "\n=== TESTES TOKENIZADOR BPE ===\n\n";
    TokenizadorBPE t({});
    std::vector<std::string> textos = { "olá mundo", "teste de tokenização" };
    t.construirVocab(textos);

    std::string frase = "olá mundo";
    std::vector<int> cod = t.codificar(frase);
    std::string dec = t.decodificar(cod);
    std::cout << "Texto: " << frase << std::endl;
    printf("Codificado: ");
    for(int id : cod) printf("%i ", id);
    printf("\nDecodificado: %s\n", dec.c_str());
}