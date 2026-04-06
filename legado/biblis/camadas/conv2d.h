// biblis/camadas/conv2d.h
#pragma once

#include "camada.h"

class Conv2D : public Camada {
public:
    size_t filtros; // numero de filtros
    size_t canalEntrada; // canais de entrada(1 para escala cinza, 3 pra RGB)
    size_t alturaFiltro; // altura do kernel/filtro
    size_t larguraFiltro; // largura do kernel/filtro
    size_t passo; // passo da convolução
    size_t espaco; // espaco(0 ou 1)
    
    // parametros treinaveis
    vector<vector<vector<vector<float>>>> pesos; // [filtros][canal][altura][largura]
    vector<float> bias; // [filtros]
    
    // gradientes
    vector<vector<vector<vector<float>>>> gradPesos;
    vector<float> gradBias;
    
    // cache pra retropropagação
    vector<vector<vector<float>>> entradaCache; // entrada da camada
    vector<vector<vector<float>>> saidaCache; // saida da camada(apos convolução)
    
    // dimensões da entrada e saida
    size_t entradaAltura, entradaLargura;
    size_t saidaAltura, saidaLargura;
    
    // função de ativação
    string tipoAtivacao;
    function<float(float)> ativacao;
    function<float(float)> derivadaAtivacao;
    
    bool usarBias;
    
    Conv2D(size_t filtros, size_t alturaFiltro, 
    size_t larguraFiltro, size_t canalEntrada = 1,
    size_t passo = 1, size_t espaco = 0,
    const string& tipoAtivacao = "relu", bool usarBias = true,
    const string& nome = "")
    : Camada(nome), filtros(filtros),
    alturaFiltro(alturaFiltro), larguraFiltro(larguraFiltro),
    canalEntrada(canalEntrada), passo(passo),
    espaco(espaco), usarBias(usarBias), tipoAtivacao(tipoAtivacao) {
        tipo = "Conv2D";
        
        // inicia os pesos com He(boa pra ReLU)
        iniciarPesos();
        
        // inicia bias com zeros
        bias = zeros(filtros);
        
        // inicia gradientes
        iniciarGrad();
        
        // cobfigura a função de ativação
        configAtivacao(tipoAtivacao);
    }
    
    void iniciarPesos() {
        pesos.resize(filtros);
        
        // fator de escala pra He
        float escala = sqrt(2.0f / (alturaFiltro * larguraFiltro * canalEntrada));
        
        random_device al;
        mt19937 gen(al());
        normal_distribution<float> dist(0.0f, escala);
        
        for(size_t f = 0; f < filtros; f++) {
            pesos[f].resize(canalEntrada);
            
            for(size_t c = 0; c < canalEntrada; c++) {
                pesos[f][c].resize(alturaFiltro);
                
                for(size_t i = 0; i < alturaFiltro; i++) {
                    pesos[f][c][i].resize(larguraFiltro);
                    
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        pesos[f][c][i][j] = dist(gen);
                    }
                }
            }
        }
    }
    
    void iniciarGrad() {
        gradPesos = zeros4D(filtros, canalEntrada, alturaFiltro, larguraFiltro);
        gradBias = zeros(filtros);
    }
    
    void configAtivacao(const string& tipo) {
        tipoAtivacao = tipo;
        
        if(tipo == "sigmoid") {
            ativacao = sigmoid;
            derivadaAtivacao = [](float y) { return y * (1 - y); };
        } else if(tipo == "relu") {
            ativacao = ReLU;
            derivadaAtivacao = [](float y) { return y > 0 ? 1.0f : 0.0f; };
        } else if(tipo == "leakyrelu") {
            ativacao = leakyReLU;
            derivadaAtivacao = derivadaLeakyReLU;
        } else if(tipo == "tanh") {
            ativacao = tanhF;
            derivadaAtivacao = derivadaTanh;
        } else { // linear(sem ativação)
            ativacao = [](float x) { return x; };
            derivadaAtivacao = [](float y) { return 1.0f; };
        }
    }
    
    // calcula dimensões de saida
    void calcularDimensoesSaida(size_t alturaEntrada, size_t larguraEntrada) {
        saidaAltura = (alturaEntrada + 2 * espaco - alturaFiltro) / passo + 1;
        saidaLargura = (larguraEntrada + 2 * espaco - larguraFiltro) / passo + 1;
        
        if(saidaAltura <= 0 || saidaLargura <= 0) {
            throw invalid_argument("[" + nome + "]: Dimensões de saída inválidas. Verifique filtro, passo e espaco.");
        }
    }
    
    // aplica espaco em uma imagem
    vector<vector<vector<float>>> aplicarespaco(const vector<vector<vector<float>>>& entrada) const {
        if(espaco == 0) return entrada;
        
        size_t novaAltura = entradaAltura + 2 * espaco;
        size_t novaLargura = entradaLargura + 2 * espaco;
        
        vector<vector<vector<float>>> comespaco(canalEntrada, 
        vector<vector<float>>(novaAltura, vector<float>(novaLargura, 0.0f)));
        
        for(size_t c = 0; c < canalEntrada; c++) {
            for(size_t i = 0; i < entradaAltura; i++) {
                for(size_t j = 0; j < entradaLargura; j++) {
                    comespaco[c][i + espaco][j + espaco] = entrada[c][i][j];
                }
            }
        }
        return comespaco;
    }
    
    // realiza a operação de convolução pra um unico filtro
    vector<vector<float>> convoluirFiltro(
        const vector<vector<vector<float>>>& entradaComEspaco,
        size_t filtroIdc) const {
        
        vector<vector<float>> res(saidaAltura, vector<float>(saidaLargura, 0.0f));
        
        for(size_t y = 0; y < saidaAltura; y++) {
            for(size_t x = 0; x < saidaLargura; x++) {
                float soma = 0.0f;
                
                // percorre todos os canais
                for(size_t c = 0; c < canalEntrada; c++) {
                    // aplica o filtro nessa região
                    for(size_t i = 0; i < alturaFiltro; i++) {
                        for(size_t j = 0; j < larguraFiltro; j++) {
                            size_t entradaY = y * passo + i;
                            size_t entradaX = x * passo + j;
                            
                            soma += entradaComEspaco[c][entradaY][entradaX] * 
                            pesos[filtroIdc][c][i][j];
                        }
                    }
                }
                res[y][x] = soma;
            }
        }
        return res;
    }
    
    vector<float> prop(const vector<float>& entrada) override {
        throw runtime_error("[" + nome + "]: Use propMapa para Conv2D(entrada deve ser 3D)");
    }
    
    // propagação pra mapa de caracteristicas 2D
    vector<vector<vector<float>>> propMapa(const vector<vector<vector<float>>>& entrada) override {
        if(entrada.size() != canalEntrada) {
            throw invalid_argument("[" + nome + "]: Número de canais de entrada incorreto");
        }
        entradaAltura = entrada[0].size();
        entradaLargura = entrada[0][0].size();
        
        // calcula dimensões de saida
        calcularDimensoesSaida(entradaAltura, entradaLargura);
        
        // aplica espaco se necessario
        auto entradaComEspaco = aplicarespaco(entrada);
        
        // armazena entrada no cache
        entradaCache = entradaComEspaco;
        
        // cria tensor de saida [filtros][altura][largura]
        vector<vector<vector<float>>> saida(filtros);
        
        // pra cada filtro
        for(size_t f = 0; f < filtros; f++) {
            saida[f] = convoluirFiltro(entradaComEspaco, f);
            
            // adiciona bias
            if(usarBias) {
                for(auto& linha : saida[f]) {
                    for(auto& pixel : linha) {
                        pixel += bias[f];
                    }
                }
            }
            // aplica função de ativação
            for(auto& linha : saida[f]) {
                for(auto& pixel : linha) {
                    pixel = ativacao(pixel);
                }
            }
        }
        // armazena saida no cache
        saidaCache = saida;
        
        return saida;
    }
    
    // propagação em lote(multiplas imagens)
    vector<vector<vector<vector<float>>>> propLoteMapa(const vector<vector<vector<vector<float>>>>& entradaLote) override {
        vector<vector<vector<vector<float>>>> saidaLote;
        
        for(const auto& entrada : entradaLote) {
            saidaLote.push_back(propMapa(entrada));
        }
        return saidaLote;
    }
    
    GradGenerico retroprop(const vector<float>& gradiente) override {
        throw runtime_error("[" + nome + "]: Use retropropMapa para Conv2D");
    }
    
    // retropropagação pra mapa de caracteristicas 2D
    vector<vector<vector<float>>> retropropMapa(const vector<vector<vector<float>>>& gradienteSaida) override {
        if(gradienteSaida.size() != filtros) {
            throw invalid_argument("[" + nome + "]: Dimensão do gradiente de saída incorreta");
        }
        size_t gradAltura = gradienteSaida[0].size();
        size_t gradLargura = gradienteSaida[0][0].size();
        
        if(gradAltura != saidaAltura || gradLargura != saidaLargura) {
            throw invalid_argument("[" + nome + "]: Dimensões do gradiente não correspondem à saída");
        }
        // gradiente em relação a ativação(aplica derivada)
        vector<vector<vector<float>>> gradAtivacao = gradienteSaida;
        
        for(size_t f = 0; f < filtros; f++) {
            for(size_t y = 0; y < saidaAltura; y++) {
                for(size_t x = 0; x < saidaLargura; x++) {
                    gradAtivacao[f][y][x] *= derivadaAtivacao(saidaCache[f][y][x]);
                }
            }
        }
        // calcula gradientes dos pesos
        calcularGradPesos(gradAtivacao);
        
        // calcula gradientes do bias
        if(usarBias) calcularGradBias(gradAtivacao);
        
        // calcula gradiente pra camada anterior(entrada)
        return calcularGradEntrada(gradAtivacao);
    }
    
    void calcularGradPesos(const vector<vector<vector<float>>>& gradAtivacao) {
        // pra cada filtro
        for(size_t f = 0; f < filtros; f++) {
            // pra cada canal
            for(size_t c = 0; c < canalEntrada; c++) {
                // pra cada posição do filtro
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        float soma = 0.0f;
                        
                        // percorre todas as posições do gradiente
                        for(size_t y = 0; y < saidaAltura; y++) {
                            for(size_t x = 0; x < saidaLargura; x++) {
                                size_t entradaY = y * passo + i;
                                size_t entradaX = x * passo + j;
                                
                                soma += gradAtivacao[f][y][x] * entradaCache[c][entradaY][entradaX];
                            }
                        }
                        gradPesos[f][c][i][j] += soma;
                    }
                }
            }
        }
    }
    
    void calcularGradBias(const vector<vector<vector<float>>>& gradAtivacao) {
        for(size_t f = 0; f < filtros; f++) {
            float soma = 0.0f;
            
            for(size_t y = 0; y < saidaAltura; y++) {
                for(size_t x = 0; x < saidaLargura; x++) {
                    soma += gradAtivacao[f][y][x];
                }
            }
            gradBias[f] += soma;
        }
    }
    
    vector<vector<vector<float>>> calcularGradEntrada(const vector<vector<vector<float>>>& gradAtivacao) {
        // gradiente pra a entrada(com espaco)
        vector<vector<vector<float>>> gradentradaComEspaco = zeros3D(canalEntrada,
        entradaAltura + 2 * espaco, entradaLargura + 2 * espaco);
        
        // pra cada filtro
        for(size_t f = 0; f < filtros; f++) {
            // pra cada posição no gradiente de saida
            for(size_t y = 0; y < saidaAltura; y++) {
                for(size_t x = 0; x < saidaLargura; x++) {
                    float grad = gradAtivacao[f][y][x];
                    
                    // pra cada canal
                    for(size_t c = 0; c < canalEntrada; c++) {
                        // pra cada posição do filtro
                        for(size_t i = 0; i < alturaFiltro; i++) {
                            for(size_t j = 0; j < larguraFiltro; j++) {
                                size_t entradaY = y * passo + i;
                                size_t entradaX = x * passo + j;
                                
                                gradentradaComEspaco[c][entradaY][entradaX] += grad * pesos[f][c][i][j];
                            }
                        }
                    }
                }
            }
        }
        // remove espaco se necessario
        if(espaco == 0) return gradentradaComEspaco;
        
        vector<vector<vector<float>>> gradEntrada(canalEntrada, 
        vector<vector<float>>(entradaAltura, vector<float>(entradaLargura, 0.0f)));
        
        for(size_t c = 0; c < canalEntrada; c++) {
            for(size_t i = 0; i < entradaAltura; i++) {
                for(size_t j = 0; j < entradaLargura; j++) {
                    gradEntrada[c][i][j] = gradentradaComEspaco[c][i + espaco][j + espaco];
                }
            }
        }
        return gradEntrada;
    }
    
    void att(float taxaAprendizado) override {
        if(otimizador) {
            // prepara os pesos em formato 2D pro otimizador
            vector<vector<float>> pesos2D = converterPesos2D();
            vector<vector<float>> gradPesos2D = converterGradPesos2D();
            
            // FIX: bias já é 1D, passa direto (sem criar bias2D que era ignorado)
            otimizador->att(pesos2D, gradPesos2D, bias, gradBias);
            
            // reconverte pesos de volta pra 4D
            reconverterPesos2D(pesos2D);
        } else {
            // atualiza SGD padrão
            for(size_t f = 0; f < filtros; f++) {
                for(size_t c = 0; c < canalEntrada; c++) {
                    for(size_t i = 0; i < alturaFiltro; i++) {
                        for(size_t j = 0; j < larguraFiltro; j++) {
                            pesos[f][c][i][j] -= taxaAprendizado * gradPesos[f][c][i][j];
                        }
                    }
                }
            }
            if(usarBias) {
                for(size_t f = 0; f < filtros; f++) {
                    bias[f] -= taxaAprendizado * gradBias[f];
                }
            }
        }
    }
    // converte pesos 4D pra 2D
    vector<vector<float>> converterPesos2D() const {
        size_t totalElementos = filtros * canalEntrada * alturaFiltro * larguraFiltro;
        vector<vector<float>> pesos2D(1, vector<float>(totalElementos));
        
        size_t idc = 0;
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        pesos2D[0][idc++] = pesos[f][c][i][j];
                    }
                }
            }
        }
        return pesos2D;
    }
    
    // converte gradientes de pesos 4D pra 2D
    vector<vector<float>> converterGradPesos2D() const {
        size_t totalElementos = filtros * canalEntrada * alturaFiltro * larguraFiltro;
        vector<vector<float>> grad2D(1, vector<float>(totalElementos));
        
        size_t idc = 0;
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        grad2D[0][idc++] = gradPesos[f][c][i][j];
                    }
                }
            }
        }
        return grad2D;
    }
    
    // reconverte pesos 2D pra 4D
    void reconverterPesos2D(const vector<vector<float>>& pesos2D) {
        size_t idc = 0;
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        pesos[f][c][i][j] = pesos2D[0][idc++];
                    }
                }
            }
        }
    }
    
    void zerarGradientes() override {
        // zera gradientes dos pesos
        for(auto& filtro : gradPesos) {
            for(auto& canal : filtro) {
                for(auto& linha : canal) {
                    fill(linha.begin(), linha.end(), 0.0f);
                }
            }
        }
        // zera gradientes do bias
        fill(gradBias.begin(), gradBias.end(), 0.0f);
    }
    
    bool temParametros() const override { return true; }
    size_t numParametros() const override {
        size_t pesosParams = filtros * canalEntrada * alturaFiltro * larguraFiltro;
        size_t biasParams = usarBias ? filtros : 0;
        return pesosParams + biasParams;
    }
    
    // serialização
    void salvar(const string& nomeArquivo) const override {
        ofstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível salvar Conv2D");
        
        arquivo << "CONV2D_CAMADA" << endl;
        arquivo << filtros << " " << alturaFiltro << " " << larguraFiltro << " "
        << canalEntrada << " " << passo << " " << espaco << endl;
        arquivo << tipoAtivacao << " " << (usarBias ? 1 : 0) << endl;
        
        // salva pesos
        for(size_t f = 0; f < filtros; f++) {
            for(size_t c = 0; c < canalEntrada; c++) {
                for(size_t i = 0; i < alturaFiltro; i++) {
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        arquivo << pesos[f][c][i][j] << " ";
                    }
                }
            }
        }
        arquivo << endl;
        
        // salva bias
        if(usarBias) {
            for(size_t f = 0; f < filtros; f++) {
                arquivo << bias[f] << " ";
            }
            arquivo << endl;
        }
        arquivo.close();
    }
    
    void carregar(const string& nomeArquivo) override {
        ifstream arquivo(nomeArquivo);
        if(!arquivo) throw runtime_error("[" + nome + "]: Não foi possível carregar Conv2D");
        
        string tipo;
        arquivo >> tipo;
        if(tipo != "CONV2D_CAMADA") {
            throw runtime_error("[" + nome + "]: Formato de arquivo inválido");
        }
        arquivo >> filtros >> alturaFiltro >> larguraFiltro 
        >> canalEntrada >> passo >> espaco;
        
        int usarBiasInt;
        arquivo >> tipoAtivacao >> usarBiasInt;
        usarBias = (usarBiasInt == 1);
        
        configAtivacao(tipoAtivacao);
        
        // redimensiona pesos
        pesos.resize(filtros);
        for(size_t f = 0; f < filtros; f++) {
            pesos[f].resize(canalEntrada);
            for(size_t c = 0; c < canalEntrada; c++) {
                pesos[f][c].resize(alturaFiltro);
                for(size_t i = 0; i < alturaFiltro; i++) {
                    pesos[f][c][i].resize(larguraFiltro);
                    for(size_t j = 0; j < larguraFiltro; j++) {
                        arquivo >> pesos[f][c][i][j];
                    }
                }
            }
        }
        // carrega bias
        if(usarBias) {
            bias.resize(filtros);
            for(size_t f = 0; f < filtros; f++) {
                arquivo >> bias[f];
            }
        }
        arquivo.close();
        
        // reinicializa gradientes
        iniciarGrad();
    }
};