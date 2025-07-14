// biblioteca de utilitários 100% português PT-br e JS
// ativações:
function degrau(x) {
  return x >= 0 ? 1 : 0;
}

function sigmoid(x) {
  return 1/(1+Math.exp(-x));
}
function derivadaSigmoid(x) {
  return x*(1-x);
}

function tanh(x) {
  return Math.tanh(x);
}
function derivadaTanh(x) {
  return 1-x*x;  
}

function ReLU(x) {
  return Math.max(0, x);
}
function derivadaReLU(x) {
  return x>0 ? 1 : 0;
}

function leakyReLU(x) {
  return x>0 ? x : 0.01*x;
}
function derivadaLeakyReLU(x) {
  return x>0 ? 1 : 0.01;
}

function softsign(x) {
  return x/(1+Math.abs(x));
}
function derivadaSoftsign(x) {
  let denom = 1+Math.abs(x);
  return 1/(denom*denom);
}

function softplus(x) {
  return Math.log(1+Math.exp(x));
}

function swish(x) {
  return x*sigmoid(x);
}
function derivadaSwish(x) {
  const sigmoidX = sigmoid(x);
  return sigmoidX+x*sigmoidX*(1-sigmoidX);
}

function GELU(x) {
  return 0.5*x*(1+Math.tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x, 3))));
}

function ELU(x, alfa=1.0) {
  return x >= 0 ? x : alfa*(Math.exp(x)-1);
}

function derivadaELU(x, alfa=1.0) {
  return x >= 0 ? 1 : ELU(x, alfa)+alfa;
}

function SELU(x, alfa=1.67326, escala=1.0507) {
  return escala*(x >= 0 ? x : alfa*(Math.exp(x)-1));
}

function derivadaSELU(x, alfa=1.67326, escala=1.0507) {
  return escala*(x >= 0 ? 1 : alfa*Math.exp(x));
}

function mish(x) {
  return x*Math.tanh(Math.log(1+Math.exp(x)));
}
function derivadaMish(x) {
  const omega = 4*(x+1)+4*Math.exp(2*x)+Math.exp(3*x)+Math.exp(x)*(4*x+6);
  const delta = 2*Math.exp(x)+Math.exp(2*x)+2;
  return Math.exp(x)*omega/(delta*delta);
}

// funções de saída:
function softmax(arr, temperatura=1) {
  // NADA PRA ALTERAR AQUI
  if(!Array.isArray(arr)) console.error("[softmax]: valor passado não é um array");
  if(!isFinite(temperatura) || temperatura <= 0) temperatura=1e-8;
  const max = Math.max(...arr.map(v => isFinite(v) ? v : 0));
  const exps = arr.map(v => {
    const t = (v-max)/temperatura;
    return isFinite(t) ? Math.exp(t) : 0;
  });
  const soma = exps.reduce((a, b) => a+b, 0) || 1e-8;
  return exps.map(e => e/soma);
}

function derivadaSoftmax(arr, gradSaida) {
  if(!Array.isArray(arr)) console.error("a derivada de softmax só pode receber vetores. Não: ", arr);
  let soma = 0;
  for(let j=0; j<gradSaida.length; j++) {
    const termo = gradSaida[j]*arr[j];
    if(!isFinite(termo)) throw new Error(`NaN em derivadaSoftmax: grad[${j}]=${gradSaida[j]}, arr[${j}]=${arr[j]}`);
    soma += termo;
  }
  return arr.map((s, i) => {
    const v = s*(gradSaida[i]-soma);
    if(!isFinite(v)) throw new Error(`NaN na saída da derivadaSoftmax[${i}]: s=${s}, grad=${gradSaida[i]}, soma=${soma}`);
    return v;
  });
}

function softmaxLote(matriz, temperatura=1) {
  return matriz.map(linha => softmax(linha, temperatura));
}

function argmax(v) {
  return v.indexOf(Math.max(...v));
}

function addRuido(v, intensidade=0.01) {
  return v.map(x => x+(Math.random()*2-1)*intensidade);
}

// funções de erro:
function erroAbsolutoMedio(saida, esperado) {
  return saida.reduce((s, x, i) => s+Math.abs(x-esperado[i]), 0)/saida.length;
}
function erroQuadradoEsperado(saida, esperado) {
  return saida.reduce((s, x, i) => s+0.5*(x-esperado[i])**2, 0);
}
function derivadaErro(saida, esperado) {
  return saida.map((x, i) => x-esperado[i]);
}

function entropiaCruzada(y, yChapeu) {
  return -y.reduce((s, yi, i) => s+yi*Math.log(yChapeu[i]+1e-12), 0);
}
function derivadaEntropiaCruzada(y, yChapeu) {
  if(!Array.isArray(y) || !Array.isArray(yChapeu) || y.length != yChapeu.length) {
    throw new Error("[derivadaEntropiaCruzada]: vetor inválido");
  }
  return yChapeu.map((yci, i) => {
    const yi = y[i];
    if(!isFinite(yi) || !isFinite(yci)) {
      console.error(`[derivadaEntropiaCruzada]: valor inválido em i=${i}: y=${yi}, y^=${yci}`);
      return 0;
    }
    return yci-yi;
  });
}

function huberPerda(saida, esperado, delta=1.0) {
  const erros = saida.map((x, i) => {
    const diff = x-esperado[i];
    return Math.abs(diff) <= delta ? 0.5*diff*diff : delta*(Math.abs(diff)-0.5*delta);
  });
  return erros.reduce((s, x) => s+x, 0)/saida.length;
}

function derivadaHuber(saida, esperado, delta=1.0) {
  return saida.map((x, i) => {
    const diff = x-esperado[i];
    return Math.abs(diff) <= delta ? diff : delta*Math.sign(diff);
  });
}

function perdaTripleto(ancora, positiva, negativa, margem=1.0) {
  const distPos = ancora.reduce((s, a, i)=>s+(a-positiva[i])**2, 0);
  const distNeg = ancora.reduce((s, a, i)=>s+(a-negativa[i])**2, 0);
  return Math.max(0, distPos-distNeg+margem);
}

function contrastivaPerda(saida1, saida2, rotulo, margem=1.0) {
  const distancia = saida1.reduce((s, x, i)=>s+(x-saida2[i])**2, 0);
  return rotulo==1 ? distancia : Math.max(0, margem-Math.sqrt(distancia));
}

// funções de regularização:
function regularL1(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*Math.sign(p)));
}
function regularL2(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*p));
}

function dropout(tensor, taxa) {
  if(Array.isArray(tensor)) return tensor.map(sub => dropout(sub, taxa));
  else return Math.random()<taxa ? 0 : tensor/(1-taxa);
}

function clipGrad(grad, maxVal=1.0) {
  return Math.min(Math.max(grad, -maxVal), maxVal);
}

function normEntrada(vetor) {
  const max = Math.max(...vetor);
  const min = Math.min(...vetor);
  const amplitude = max-min || 1e-8;
  return vetor.map(x => (x-min)/amplitude);
}

function normZPonto(v) {
  const media = v.reduce((a,b)=>a+b, 0)/v.length;
  const variancia = v.reduce((a,b)=>a+(b-media)**2, 0)/v.length;
  const desvio = Math.sqrt(variancia+1e-8);
  return v.map(x => (x-media)/desvio);
}

function acuracia(saida, esperado) {
  const corretos = saida.reduce((s, x, i) => s+(argmax(x)===argmax(esperado[i]) ? 1 : 0), 0);
  return corretos/saida.length;
}

function precisao(confusao) {
  const tp = confusao[0][0], fp = confusao[0].slice(1).reduce((a,b) => a+b, 0);
  return tp/(tp+fp+1e-8); // evita divisão por zero
}

function recall(confusao) {
  const tp = confusao[0][0], fn = confusao.slice(1).reduce((s, l) => s+l[0], 0);
  return tp/(tp+fn+1e-8);
}

function f1Score(confusao) {
  const p = precisao(confusao), r = recall(confusao);
  return 2*(p*r)/(p+r+1e-8);
}

function mse(saida, esperado) {
  return saida.reduce((s, x, i) => s+(x-esperado[i])**2, 0)/saida.length;
}

function klDivergencia(p, q) {
  return p.reduce(((s, pi, i) => s+(pi*Math.log((pi+1e-12)/(q[i]+1e-12)))), 0);
}

function rocAuc(pontos, rotulos) {
  const pares = pontos.map((s, i) => [s, rotulos[i]]).sort((a,b) => b[0]-a[0]);
  let auc = 0, fp = 0, tp = 0, fpPrev = 0, tpPrev = 0;
  pares.forEach(([s, r]) => {
    if(r===1) tp++;
    else fp++;
    auc += (fp-fpPrev)*(tp+tpPrev)/2;
    fpPrev = fp;
    tpPrev = tp;
  });
  return auc/(tp*fp);
}

// funções de pesos:
function iniPesosXavier(linhas, cols) {
  if(linhas <= 0 || cols <= 0) console.error("[iniPesosXavier]: valor negativo passado como parâmetro");
  let m = [];
  let limite = Math.sqrt(6/(linhas+cols));
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*limite;
    }
  }
  return m;
}

function iniPesosHe(linhas, cols) {
  if(linhas <= 0 || cols <= 0) console.error("[iniPesosHe]: valor negativo passado como parâmetro");
  let m = [];
  let limite = Math.sqrt(2/linhas)*0.5;
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*limite;
    }
  }
  return m;
}

function attPesos(pesos, gradientes, taxa, lambda=1e-3) {
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j]-lambda*p)
  );
}

function attPesosMomentum(pesos, gradientes, taxa, momento, velocidade) {
  return pesos.map((linha, i) =>
    linha.map((p, j) => {
      velocidade[i][j] = momento*velocidade[i][j]+gradientes[i][j];
      return p-taxa*velocidade[i][j];
    })
  );
}

function attPesosAdam(pesos, gradientes, m, v, taxa, beta1=0.9, beta2=0.999, epsilon=1e-8, iteracao) {
  const mCorrigido = m.map((linha, i)=>linha.map((val, j)=>beta1*val+(1-beta1)*gradientes[i][j]));
  const vCorrigido = v.map((linha, i)=>linha.map((val, j)=>beta2*val+(1-beta2)*gradientes[i][j]**2));
  const mChapeu = mCorrigido.map(linha=>linha.map(val=>val/(1-Math.pow(beta1, iteracao))));
  const vChapeu = vCorrigido.map(linha=>linha.map(val=>val/(1-Math.pow(beta2, iteracao))));
  return pesos.map((linha, i)=>linha.map((p, j)=>p-taxa*mChapeu[i][j]/(Math.sqrt(vChapei[i][j])+epsilon)));
}

function attPesosAdamW(pesos, gradientes, m, v, taxa, beta1=0.9, beta2=0.999, epsilon=1e-8, iteracao, decaimentoPeso=0.01) {
  const mCorrigido = m.map((linha, i)=>linha.map((val, j)=>beta1*val+(1-beta1)*gradientes[i][j]));
  const vCorrigido = v.map((linha, i)=>linha.map((val, j)=>beta2*val+(1-beta2)*gradientes[i][j]**2));
  const mChapeu = mCorrigido.map(linha=>linha.map(val=>val/(1-Math.pow(beta1, iteracao))));
  const vChapeu = vCorrigido.map(linha=>linha.map(val=>val/(1-Math.pow(beta2, iteracao))));
  const pesosDecaidos = pesos.map(linha=>linha.map(val=>val*(1-taxa*decaimentoPeso)));
  return pesosDecaidos.map((linha, i)=>linha.map((val, j)=>val-taxa*mChapeu[i][j]/(Math.sqrt(vChapeu[i][j])+epsilon)));
}

function attPesosRMSprop(pesos, gradientes, cache, taxa=0.001, decadencia=0.9, epsilon=1e-8) {
  cache = cache.map((linha, i) => 
    linha.map((val, j) => decadencia*val+(1-decadencia)*gradientes[i][j]**2)
  );
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j]/(Math.sqrt(cache[i][j])+epsilon))
  );
}

function attPesosAdagrad(pesos, gradientes, cache, taxa=0.01, epsilon=1e-8) {
  cache = cache.map((linha, i) => 
    linha.map((val, j) => val+gradientes[i][j]**2)
  );
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j]/(Math.sqrt(cache[i][j])+epsilon))
  );
}

function iniPesosUniforme(linhas, cols, limiteInferior=-0.05, limiteSuperior=0.05) {
  let m = [];
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = Math.random()*(limiteSuperior-limiteInferior)+limiteInferior;
    }
  }
  return m;
}

// matrizes 3D:
function tensor3D(profundidade, linhas, colunas, escala=0.1) {
  let t = [];
  for(let d=0; d<profundidade; d++) {
    t[d] = matriz(linhas, colunas, escala);
  }
  return t;
}

function zeros3D(p, l, c) {
  return Array.from({length: p}, () => matrizZeros(l, c));
}

function mapear3D(tensor, fn) {
  return tensor.map(m => m.map(linha => linha.map(fn)));
}

function somar3D(a, b) {
  return a.map((mat, i) => somarMatriz(mat, b[i]));
}

function mult3DporEscalar(tensor, escalar) {
  return tensor.map(m => multMatriz(m, escalar));
}

// matrizes 2D:
function matriz(linhas, cols, escala=0.1) {
  let m = [];
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*escala;
    }
  }
  return m;
}

function matrizConfusao(saidas, esperados) {
  const classes = saidas[0].length;
  const matriz = Array.from({ length: classes }, ()=> Array(classes).fill(0));
  saidas.forEach((s, i) => {
    const pred = argmax(s);
    const real = argmax(esperados[i]);
    matriz[real][pred]++;
  });
  return matriz;
}

function exterior(a, b) {
  let res = [];
  for(let i=0; i<a.length; i++) {
    res[i] = [];
    for(let j=0; j<b.length; j++) {
      res[i][j] = a[i]*b[j];
    }
  }
  return res;
}

function clonarMatriz(m) {
  const copia = [];
  for(let i=0; i<m.length; i++) {
    copia[i] = m[i].slice();
  }
  return copia;
}

function somarMatriz(a, b) {
  if(a.length != a.length) console.error("[somarMatrizes]: tamanho incompatível");
  return a.map((linha, i) => linha.map((v, j) => v+b[i][j]));
}

function subMatriz(a, b) {
  if(a.length != a.length) console.error("[subMatriz]: tamanho incompatível");
  return a.map((linha, i) => linha.map((v, j) => v-b[i][j]));
}

function multMatriz(m, s) {
  if(a.length != a.length) console.error("[multMatriz]: tamanho incompatível");
  return m.map(linha => linha.map(v => v*s));
}

function multMatrizes(a, b) {
  if(a.length != a.length) console.error("[multMatrizes]: tamanho incompatível");
  return a.map(linhaA => b[0].map((_, j) => linhaA.reduce((soma, valA, k) => soma+valA*(b[k] ? b[k][j] : 0), 0)));
}

function multElementos(a, b) {
  if(a.length != a.length) console.error("[multElementos]: tamanho incompatível");
  return a.map((val, i) => val*b[i]);
}

function aplicarMatriz(m, v) {
  if(!Array.isArray(v)) console.error("[aplicarMatriz]: o segundo parâmetro não é um vetor");
  return m.map(linha => escalarDot(linha, v));
}

function transpor(m) {
  return [...m[0]].map((_, j) => m.map(linha => linha[j]));
}

function matrizZeros(linhas, cols) {
  return Array.from({length: linhas}, () => Array(cols).fill(0));
}

function identidade(n) {
  return Array.from({length: n}, (_, i) =>
    Array.from({length: n}, (_, j) => i===j ? 1 : 0)
  );
}

function matrizVetor(m, v) {
  if(!Array.isArray(v)) console.error("[matrizVetor]: o segundo parâmetro não é um vetor");
    return m.map(linha => {
        return linha.reduce((soma, val, j) => soma+val*v[j], 0);
    });
}

function multVetorMatriz(v, c) {
  if(!Array.isArray(v)) console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return v.map((x,i) => x*c[i]);
}

function multMatrizVetor(m, v) {
  if(!Array.isArray(v)) console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return m.map(linha =>
    linha.reduce((soma, val, j) => soma+val*v[j], 0)
  );
}

function clipMatriz(m, limite) {
  return m.map(linha => linha.map(v => Math.max(-limite, Math.min(v, limite))));
}

// vetores
function vetor(n, escala=0.1) {
  return Array(n).fill(0).map(() => (Math.random()*2-1)*escala);
}

function somarVetores(a, b) {
  if(!a || !b || a.length !== b.length) throw new Error(`[somarVetores]: tamanho incompatível a=${a.length}, b=${b.length}`);
  return a.map((x, i) => x+b[i]);
}

function subVetores(a, b) {
  if(a.length != a.length) console.error("[subVetores]: tamanho incompatível");
  return a.map((x, i) => x-b[i]);
}

function multVetores(a, b) {
  if(a.length != a.length) console.error("[multVetores]: tamanho incompatível");
  return a.map((x, i) => x*b[i]);
}

function escalarDot(v, w) {
  if(!Array.isArray(v) || !Array.isArray(w)) throw new Error("[escalarDot] Um dos vetores não é array");
  if(v.length != w.length) throw new Error(`[escalarDot] Tamanhos diferentes: v=${v.length}, w=${w.length}`);
  let soma = 0;
  for(let i=0; i<v.length; i++) {
    const prod = v[i]*w[i];
    if(!isFinite(prod)) throw new Error(`[escalarDot] Produto inválido em [${i}]: ${v[i]} * ${w[i]} = ${prod}`);
    soma += prod;
  }
  return soma;
}

function normalizar(v) {
  if(!Array.isArray(v)) console.error("[normalizar]: o parâmetro não é um vetor");
  const mag = Math.sqrt(v.reduce((s, x) => s+x*x, 0));
  return mag==0 ? v : v.map(x => x/mag);
}

function zeros(n) {
  return Array(n).fill(0);
}

function uns(n) {
  return Array(n).fill(1);
}

function aplicarFuncaoVetor(v, fn) {
  if(!Array.isArray(v)) console.errror("[aplicarFuncaoVetor]: o primeiro parâmetro não é um vetor");
  return v.map(x => fn(x));
}

function embaralhar(pares) {
  for(let i=pares.length-1; i>0; i--) {
    const j = Math.floor(Math.random()*(i+1));
    [pares[i], pares[j]] = [pares[j], pares[i]];
  }
  return pares;
}

function dividirDados(pares, proporcao=0.8) {
  const nTreino = Math.floor(pares.length*proporcao);
  const treino = pares.slice(0, nTreino);
  const teste = pares.slice(nTreino);
  return [treino, teste];
}

function clip(grad, limite=1.0) {
  return grad.map(v => Math.max(-limite, Math.min(limite, v)));
}
function clipVetor(v, limite) {
  if(!Array.isArray(v)) console.error("[clipVetor]: o parâmetro não é um vetor");
  return v.map(x => Math.max(-limite, Math.min(x, limite)));
}
// gradientes:
function limparGrad(grad) {
  return grad.map(linha => linha.map(v=>isFinite(v) ? v : 0));
}
// debug:
function arraysIguais(a, b) {
  return JSON.stringify(a)===JSON.stringify(b);
}

function eNaN(tensor, nome, classe, parar=true) {
  for(let i=0; i<tensor.length; i++) {
    if(parar && tensor[0][0]==undefined) {
        if(isNaN(tensor[i])) throw new Error(`[${classe}] ${nome} contém NaN em [${i}]`);
        else if(!isFinite(tensor[i])) throw new Error(`[${classe}] ${nome} contém Infinity em [${i}][${j}]`);
        else if(tensor[i]==undefined) throw new Error(`[${classe}] ${nome} contém Undefined em [${i}]`);
      } else if(tensor[0][0]==undefined) {
        if(isNaN(tensor[i])) console.error(`[${classe}] ${nome} contém NaN em [${i}]`);
        else if(!isFinite(tensor[i])) console.error(`[${classe}] ${nome} contém Infinity em [${i}]`);
        else if(tensor[i]==undefined) console.error(`[${classe}] ${nome} contém Undefined em [${i}]`);
      }
    for(let j=0; j<tensor[i].length; j++) {
      if(parar) {
        if(isNaN(tensor[i][j])) throw new Error(`[${classe}] ${nome} contém NaN em [${i}][${j}]`);
        else if(!isFinite(tensor[i][j])) throw new Error(`[${classe}] ${nome} contém Infinity em [${i}][${j}]`);
        else if(tensor[i][j]==undefined) throw new Error(`[${classe}] ${nome} contém Undefined em [${i}][${j}]`);
      } else {
        if(isNaN(tensor[i][j])) console.error(`[${classe}] ${nome} contém NaN em [${i}][${j}]`);
        else if(!isFinite(tensor[i][j])) console.error(`[${classe}] ${nome} contém Infinity em [${i}][${j}]`);
        else if(tensor[i][j]==undefined) console.error(`[${classe}] ${nome} contém Undefined em [${i}][${j}]`);
      }
    }
  }
}
// vetorização:
function oneHot(i, tam) {
  const v = new Array(tam).fill(0);
  if(i<0 || i >= tam) {
    throw new Error(`[oneHot]: índice fora do vocabulário: ${i}`);
  }
  v[i] = 1;
  return v;
}

function oneHotLote(indices, tamanho) {
  return indices.map(i => oneHot(i, tamanho));
}
// texto:
function pesquisar(url, html) {
  const https = require('https');
  https.get(url, (res) => {
    let dados = '';
    res.on('data', chunk => dados += chunk);
    res.on('end', () => html(dados));
  }).on('error', err => {
    console.error('Erro:', err.message);
  });
}

function normalizarTexto(texto) {
  return texto
  .normalize('NFD').replace(/[\u0300-\u036f]/g, '') // remove acentos
  .toLowerCase()
  .replace(/[^\w\s.,!?;:]/g, '') // mantém apenas caracteres válidos
  .replace(/\s+/g, ' ');
}
function tokenizarCtx(texto) {
  return texto
    .split(/(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s+/g) // split por sentenças
    .filter(s => s.trim().length>0);
}

function buscaFeixe(modelo, inicio, fim, maxComprimento, raioTam, temperatura=1.0, topoK=null) {
    let feixe = [{
        sequencia: [inicio],
        pontuacao: 0.0,
        finalizada: false
    }];
    const completas = [];
    for(let passo=0; passo<maxComprimento; passo++) {
        const candidatos = [];
        for(const hipotese of feixe) {
            if(hipotese.finalizada) {
                candidatos.push(hipotese);
                continue;
            }
            const seqAtual = hipotese.sequencia;
            
            const saida = modelo.propagar(seqAtual);
            const logits = saida[saida.length-1];
            
            const probs = softmax(logits, temperatura);
            
            let opcoes = probs
            .map((prob, token) => ({token, prob}))
            .filter(opcao => opcao.prob>0);
            
            if(topoK != null && topoK>0) {
                opcoes.sort((a, b) => b.prob-a.prob);
                opcoes = opcoes.slice(0, topoK);
            }
            for(const opcao of opcoes) {
                const novoToken = opcao.token;
                const novaSeq = [...seqAtual, novoToken];
                const novaPontuacao = hipotese.pontuacao+Math.log(opcao.prob+1e-10);
                
                const finalizada = (fim != undefined && novoToken==fim);
                
                candidatos.push({
                    sequencia: novaSeq,
                    pontuacao: novaPontuacao,
                    finalizada: finalizada
                });
            }
        }
        candidatos.sort((a, b) => b.pontuacao-a.pontuacao);
        
        feixe = [];
        for(const cand of candidatos) {
            if(feixe.length<raioTam) {
                feixe.push(cand);
            } else {
                if(cand.finalizada) {
                   completas.push(cand);
                }
            }
        }
        if(feixe.every(o => o.finalizada)) {
            completas.push(...feixe);
            break;
        }
    }
    completas.push(...feixe.filter(o => !o.finalizada));
    if(completas.length>0) {
        completas.sort((a, b) => b.pontuacao-a.pontuacao);
        return completas[0].sequencia;
    }
    feixe.sort((a, b) => b.sequencia.length-a.sequencia.length);
    return feixe[0].sequencia;
}

class CamadaAtencao {
  constructor(dimModelo, numCabecas, taxaAprendizado=0.001, taxaDropout=0.1) {
    this.dimModelo = dimModelo;
    this.numCabecas = numCabecas;
    this.dimCabeca = Math.floor(dimModelo / numCabecas);
    this.taxa = taxaAprendizado;
    this.taxaDrop = taxaDropout;

    this.pq = iniPesosHe(dimModelo, dimModelo);
    this.pk = iniPesosHe(dimModelo, dimModelo);
    this.pv = iniPesosHe(dimModelo, dimModelo);
    this.ps = iniPesosHe(dimModelo, dimModelo);

    this.bq = zeros(dimModelo);
    this.bk = zeros(dimModelo);
    this.bv = zeros(dimModelo);
    this.bs = zeros(dimModelo);

    // buffers de cache para o backward
    this.cache = {};
    if(this.dimModelo%this.numCabecas != 0) throw new Error(`dimModelo (${this.dimModelo}) não divisível por numCabecas (${this.numCabecas})`);
  }

  propagar(x, mascara=null, treino=true) {
    x.forEach((linha, i) => {
      if(linha.length !== this.dimModelo) throw new Error(`[CamadaAtencao] x[${i}] tem tamanho inválido: ${linha.length}, esperado: ${this.dimModelo}`);
    });
    eNaN(x, "x", "CamadaAtencao");
    eNaN(this.pq, "pq", "CamadaAtencao");
    this.bq.forEach((v,i)=>{if(!isFinite(v)) throw new Error(`bq contém NaN ou infinito em ${i}: ${v}`);});
    eNaN(this.pk, "pk", "CamadaAtencao");
    this.bk.forEach((v,i)=>{if(!isFinite(v)) throw new Error(`bk contém NaN ou infinito em ${i}: ${v}`);});
    eNaN(this.pv, "pv", "CamadaAtencao");
    this.bv.forEach((v,i)=>{if(!isFinite(v)) throw new Error(`bv contém NaN ou infinito em ${i}: ${v}`);});
    const seqTam = x.length;
    const q = x.map(seq=>somarVetores(aplicarMatriz(this.pq, seq), this.bq));
    eNaN(q, "q", "CamadaAtencao");
    const k = x.map(seq => somarVetores(aplicarMatriz(this.pk, seq), this.bk));
    eNaN(k, "k", "CamadaAtencao");
    const v = x.map(seq => somarVetores(aplicarMatriz(this.pv, seq), this.bv));
    eNaN(v, "v", "CamadaAtencao");

    const qCab = this.dividirCabecas(q);
    const kCab = this.dividirCabecas(k);
    const vCab = this.dividirCabecas(v);

    const pontosCabecas = [];
    const pCabecas = [];

    for(let o=0; o<this.numCabecas; o++) {
      const qo = qCab[o];
      const ko = kCab[o];
      const ponto = [];
      for(let i=0; i<qo.length; i++) {
        for(let j=0; j<qo[i].length; j++) {
          const vq = qo[i][j], vk = ko[i][j];
          if(!isFinite(vq) || !isFinite(vk)) throw new Error(`qo ou ko contém valor inválido em cabeca ${o}, pos ${i}, dim ${j}: ${vq}, ${vk}`);
        }
      }
      for(let i=0; i<seqTam; i++) {
        ponto[i] = [];
        for(let j=0; j<seqTam; j++) ponto[i][j] = escalarDot(qo[i], ko[j])/Math.sqrt(this.dimCabeca);
      }
      if(mascara) {
        for(let i=0; i<seqTam; i++) {
          for(let j=0; j<seqTam; j++)
            if(!mascara[i][j]) ponto[i][j] = -1e9;
        }
      }
      eNaN(ponto);
      pontosCabecas[o] = ponto;
      pCabecas[o] = ponto.map(linha => softmax(linha));
    }
    const atSCabecas = pCabecas.map((pp, o) => {
      const vo = vCab[o];
      return vo.map((_, i) => {
        let ctx = zeros(this.dimCabeca);
        for(let j=0; j<seqTam; j++) {
          const mult = pp[i][j];
          const contrib = vo[j].map(vij => vij*mult);
          ctx = somarVetores(ctx, contrib);
        }
        return ctx;
      });
    });

    const juntar = this.juntarCabecas(atSCabecas);
    let o = juntar.map(seq => somarVetores(aplicarMatriz(this.ps, seq), this.bs));

    this.cache = { x, qCab, kCab, vCab, pCabecas, juntar };
    eNaN(o, "o", "CamadaAtencao");
    if(treino) o = dropout(o, this.taxaDrop);
    return o;
  }

  retropropagar(dO) {
    eNaN(dO, "dO", "CamadaAtencao");
    const { x, qCab, kCab, vCab, pCabecas, juntar } = this.cache;
    const seqTam = x.length;
    // dConcat = dO · Wo^T
    const dPoT = transpor(this.ps);
    eNaN(dPoT, "dPoT", "CamadaAtencao");
    if(!Array.isArray(dO) || dO.some(l => !Array.isArray(l))) throw new Error("[CamadaAtencao] dO contém item inválido ou undefined");
    
    let dConcat = dO.map(do_i => aplicarMatriz(dPoT, do_i));
    eNaN(dConcat, "dConcat", "CamadaAtencao");
    dConcat = dConcat.map(vetor => clipVetor(vetor, 1.0));
    // gradientes de ps e bs
    let dPs = multMatrizes(transpor(juntar), dO);
    eNaN(dPs, "dPs", "CamadaAtencao");
    let dBo = dO.reduce((s, v)=>somarVetores(s, v), zeros(this.dimModelo));
    eNaN(dBo, "dBo", "CamadaAtencao");
    // divide dConcat nqs cabecas
    if(x[0].length !== this.numCabecas*this.dimCabeca) throw new Error("dividirCabecas recebeu vetor de tamanho inválido: "+x[0].length+" ≠ "+(this.numCabecas*this.dimCabeca));
    if(dConcat[0].length !== this.numCabecas*this.dimCabeca) throw new Error("dConcat com dimensão errada: "+dConcat[0].length);
    const dAtCabecas = this.dividirCabecas(dConcat);
    // grads:
    let dPq = matrizZeros(this.dimModelo, this.dimModelo);
    let dPk = matrizZeros(this.dimModelo, this.dimModelo);
    let dPv = matrizZeros(this.dimModelo, this.dimModelo);
    let dBq = zeros(this.dimModelo);
    let dBk = zeros(this.dimModelo);
    let dBv = zeros(this.dimModelo);
    // prepara espaço para dko e dVo
    const dkoTodo = zeros3D(this.numCabecas, seqTam, this.dimCabeca);
    const dVoTodo = zeros3D(this.numCabecas, seqTam, this.dimCabeca);
    const dX = matrizZeros(seqTam, this.dimModelo);

    for(let o=0; o<this.numCabecas; o++) {
      const qo = qCab[o], ko = kCab[o], vo = vCab[o], po = pCabecas[o], dAo = dAtCabecas[o];
      // dPO = dAo*Vo
      eNaN(po, "po", "CamadaAtencao");
      eNaN(dAo, "dAo", "CamadaAtencao");
      eNaN(vo, "vo", "CamadaAtencao");
      const dPO = [];
      for(let i=0; i<po.length; i++) {
        let soma = zeros(this.dimCabeca);
        for(let j=0; j<this.dimCabeca; j++) {
          const mult = dAo[i][j];
          if(typeof mult !== "number") throw new Error(`mult inválido: dAo[${i}][${j}] = ${mult}`);
          for(let k=0; k<this.dimCabeca; k++) {
            const valor = vo[i][k]
            if(!isFinite(valor*mult)) throw new Error(`Explodiu: vo[${i}][${k}]=${valor}*mult=${mult}`);
            soma[k] += valor*mult;
          }
        }
        dPO.push(clipVetor(soma, 0.5));
      }
      eNaN(dPO, "dPO", "CamadaAtencao");
      for(let i=0; i<po.length; i++) {
        if(po[i].some(v=>isNaN(v))) throw new Error(`[CamadaAtencao] po[${i}] contém NaN`);
        if(dPO[i].some(v=>isNaN(v))) throw new Error(`[CamadaAtencao] dPO[${i}] contém NaN`);
      }
    
      // dPontos = derivada softmax^T*dPO
      const dPoLeve = po.map((_, i) => {
        const linha = [];
        for(let j=0; j<seqTam; j++) {
          let soma = 0;
          for(let k=0; k<this.dimCabeca; k++) {
            soma += dAo[i][k]*vo[j][k];
          }
          linha.push(soma);
        }
        return linha;
      });
      // calcula gradiente da softmax
      const dPontos = po.map((linha, i) => {
        return derivadaSoftmax(linha, dPoLeve[i]);
      });
      eNaN(dPontos, "dPontos", "CamadaAtencao");
      // atualizar dX, dkoTodo e dVhAll
      const invSqrt = 1/Math.sqrt(this.dimCabeca);
      if(isNaN(invSqrt)) console.error("[CamadaAtencao]: invSqrt é NaN");
      for(let i=0; i<qo.length; i++) {
        for(let j=0; j<ko.length; j++) {
          const grad = dPontos[i][j]*invSqrt;
          // dX
          const inicio = o*this.dimCabeca;
          for(let k=0; k<this.dimCabeca; k++) {
            dX[i][inicio + k] += ko[j][k]*grad;
          }
          // dkoTodo
          for(let k=0; k<this.dimCabeca; k++) {
            dkoTodo[o][j][k] += qo[i][k]*grad;
          }
        }
      }
      // dVoTodo
      for(let i=0; i<seqTam; i++) {
        for(let j=0; j<seqTam; j++) {
          const peso = po[i][j]*dAo[i][j];
          if(peso != 0) {
            const contrib = Array(this.dimCabeca).fill(peso);
            dVoTodo[o][j] = somarVetores(dVoTodo[o][j], contrib);
          }
        }
      }
      // prepara matrizes para gradientes de pesos
      const xMat = transpor(x);
      const qCabecaPlana = transpor(qo);
      const kCabecaPlana = transpor(dkoTodo[o]);
      const vCabecaPlana = transpor(dVoTodo[o]);
      const base = o*this.dimCabeca;
      // valida entradas
      for(let idc=0; idc<qCabecaPlana.length; idc++) {
        if(qCabecaPlana[idc].some(v => !isFinite(v))) throw new Error(`qCabecaPlana corrompido`);
      }
      for(let idc=0; idc<xMat.length; idc++) {
        if(xMat[idc].some(v => !isFinite(v))) throw new Error(`xMat corrompido`);
      }
      // gradientes
      for(let i=0; i<this.dimCabeca; i++) {
        const lpq = multMatrizVetor(xMat, qCabecaPlana[i]);
        const lpk = multMatrizVetor(xMat, kCabecaPlana[i]);
        const lpv = multMatrizVetor(xMat, vCabecaPlana[i]);
        dPq[base+i] = somarVetores(dPq[base+i], lpq);
        dPk[base+i] = somarVetores(dPk[base+i], lpk);
        dPv[base+i] = somarVetores(dPv[base+i], lpv);
        dBq[base+i] += qCabecaPlana[i].reduce((s, v) => s+v, 0);
        dBk[base+i] += kCabecaPlana[i].reduce((s, v) => s+v, 0);
        dBv[base+i] += vCabecaPlana[i].reduce((s, v) => s+v, 0);
      }
    }
    // recorte
    dPs = clipMatriz(dPs, 0.5);
    dBo = clipVetor(dBo, 0.5);
    dPq = clipMatriz(dPq, 0.5);
    dBq = clipVetor(dBq, 0.5);
    dPk = clipMatriz(dPk, 0.5);
    dBk = clipVetor(dBk, 0.5);
    dPv = clipMatriz(dPv, 0.5);
    dBv = clipVetor(dBv, 0.5);
    // aplica atualizações ignorando NaNs
    this.ps = attPesos(this.ps, limparGrad(dPs), this.taxa);
    this.bs = this.bs.map((b, i)=>isFinite(dBo[i]) ? b-this.taxa*dBo[i] : b);
    this.pq = attPesos(this.pq, limparGrad(dPq), this.taxa);
    this.bq = this.bq.map((b, i)=>isFinite(dBq[i]) ? b-this.taxa*dBq[i] : b);
    this.pk = attPesos(this.pk, limparGrad(dPk), this.taxa);
    this.bk = this.bk.map((b, i)=>isFinite(dBk[i]) ? b-this.taxa*dBk[i] : b);
    this.pv = attPesos(this.pv, limparGrad(dPv), this.taxa);
    this.bv = this.bv.map((b, i)=>isFinite(dBv[i]) ? b-this.taxa*dBv[i] : b);
    eNaN(dX, "dX", "CamadaAtencao");
    return dX;
  }
  dividirCabecas(x) {
    for(let i=0; i<x.length; i++) {
      if(x[i].length !== this.numCabecas*this.dimCabeca) throw new Error(`dividirCabecas: x[${i}] tem tamanho ${x[i].length}, esperado ${this.numCabecas*this.dimCabeca}`);
    }
    const cabecas = [];
    for(let o=0; o<this.numCabecas; o++) {
      cabecas[o] = x.map(seq=>seq.slice(o*this.dimCabeca, (o+1)*this.dimCabeca));
    }
    return cabecas;
  }
  juntarCabecas(cabecas) {
    return cabecas[0].map((_, i)=>cabecas.reduce((seq, o) => seq.concat(o[i]), [])
    );
  }
}
class CamadaFFN {
  constructor(dimModelo, dimFFN, taxaAprendizado=0.001, taxaDropout=0.1) {
    this.p1 = iniPesosHe(dimFFN, dimModelo);
    this.b1 = zeros(dimFFN);
    this.p2 = iniPesosHe(dimModelo, dimFFN);
    this.b2 = zeros(dimModelo);
    this.taxa = taxaAprendizado;
    this.cache = {};
    this.taxaDrop = taxaDropout;
  }
  propagar(x, treino=true) {
    this.p1.forEach((linha, i) => {
      linha.forEach((v, j) => {
        if(typeof v != 'number' || isNaN(v) || !isFinite(v)) console.log(`p1[${i}][${j}] inválido:`, v);
      });
    });
    const camada1 = x.map(seq => {
      const z = aplicarMatriz(this.p1, seq); // [saida]
      if(z.length != this.b1.length) throw new Error("Bias incompatível");
      const lin = somarVetores(z, this.b1); // [saida]
      return lin.map(ReLU);
    });
    let camada2 = camada1.map(seq => {
      const z = aplicarMatriz(this.p2, seq);
      if(z.length != this.b2.length) throw new Error("Bias 2 incompatível");
      return somarVetores(z, this.b2);
    });
    this.cache = { x, camada1 };
    if(treino) camada2 = dropout(camada2, this.taxaDrop);
    eNaN(camada2, "camada2", "CamadaFFN");
    return camada2;
  }
  
  retropropagar(dY) {
    eNaN(dY, "dY", "CamadaFFN");
    /* cache.x é a entrada original x: matriz [lote][dimEntrada]
    cache.camada1 é a saída pós-ReLU: matriz [lote][dimFFN] */
    const { x, camada1 } = this.cache;
    const lote = x.length;
    const dimFFN = this.p1.length; // número de neurônios da camada oculta
    const dimEntrada = this.p1[0].length;
    const dimSaida = this.p2.length; // deve ser igual a dimEntrada
    if(!Array.isArray(x) || x.length != lote) throw new Error("cache.x malformado");
    if(!Array.isArray(camada1) || camada1.length !== lote) throw new Error("cache.camada1 malformado");
    
    const dP2 = matrizZeros(dimSaida, dimFFN);
    const dB2 = zeros(dimSaida);
    const dP1 = matrizZeros(dimFFN, dimEntrada);
    const dB1 = zeros(dimFFN);
    const dX  = Array(lote).fill(0).map(_=>zeros(dimEntrada));
    
    if(!Array.isArray(this.p2) || this.p2.length != dimSaida) throw new Error("p2 malformada");
    this.p2.forEach((linha, j) => {
      if(!Array.isArray(linha) || linha.length != dimFFN) throw new Error(`p2[${j}] com dimensão incorreta`);
      linha.forEach((v,i) => {
        if(typeof v != "number" || !isFinite(v)) throw new Error(`p2[${j}][${i}] inválido = ${v}`);
      });
    });
    // b2
    if(!Array.isArray(this.b2) || this.b2.length !== dimSaida) throw new Error("b2 malformada");
    this.b2.forEach((v,j) => {
      if(typeof v != "number" || !isFinite(v)) throw new Error(`b2[${j}] inválido = ${v}`);
    });
    // w1: [dimFFN][dimEntrada]
    if(!Array.isArray(this.p1) || this.p1.length != dimFFN)
    throw new Error("p1 malformada");
    this.p1.forEach((linha, j) => {
      if(!Array.isArray(linha) || linha.length !== dimEntrada)
      throw new Error(`p1[${j}] com dimensão incorreta`);
      linha.forEach((v,i) => {
        if(typeof v != "number" || !isFinite(v)) throw new Error(`p1[${j}][${i}] inválido = ${v}`);
      });
    });
    // b1
    if(!Array.isArray(this.b1) || this.b1.length != dimFFN) throw new Error("b1 malformada");
    this.b1.forEach((v,j) => {
      if(typeof v != "number" || !isFinite(v)) throw new Error(`b1[${j}] inválido = ${v}`);
    });
    // RETROPROPAGAÇÃO:
    for(let n=0; n<lote; n++) {
      const seqX  = x[n];
      const seqo1 = camada1[n]; // saída ReLU
      const seqDY = dY[n];
      // checa seqX, seqo1, seqDY
      if(!Array.isArray(seqX) || seqX.length != dimEntrada) throw new Error(`x[${n}] inválido`);
      if(seqX.some(v => typeof v !== "number" || !isFinite(v))) throw new Error(`x[${n}] contém valor inválido`);
      
      if(!Array.isArray(seqo1) || seqo1.length != dimFFN) throw new Error(`camada1[${n}] inválido`);
      if(seqo1.some(v => typeof v != "number" || !isFinite(v))) throw new Error(`camada1[${n}] contém valor inválido`);
      if(!Array.isArray(seqDY) || seqDY.length != dimSaida) throw new Error(`dO[${n}] inválido`);
      if(seqDY.some(v => typeof v != "number" || !isFinite(v))) throw new Error(`dO[${n}] contém valor inválido`);
      // gradientes da camada de saída
      for(let j=0; j < dimSaida; j++) {
        const err = seqDY[j];
        dB2[j] += err;
        for(let k=0; k<dimFFN; k++) {
          dP2[j][k] += seqo1[k]*err;
        }
      }
      // retropropagação na camada de saída
      const p2T = transpor(this.p2); // [dimFFN][dimSaida]
      const dO = aplicarMatriz(p2T, seqDY);
      for(let k=0; k<dimFFN; k++) {
        const deriv = seqo1[k]>0 ? 1 : 0; // derivada ReLU
        dO[k] = dO[k]*deriv;
      }
      // gradientes da primeira camada
      for(let j=0; j<dimFFN; j++) {
        const err1 = dO[j];
        dB1[j] += err1;
        for(let k=0; k<dimEntrada; k++) {
          dP1[j][k] += seqX[k]*err1;
        }
      }
      const p1T = transpor(this.p1); // [dimEntrada][dimFFN]
      dX[n] = aplicarMatriz(p1T, dO);
    }
    // ATUALIZAÇÃO DOS PESOS:
    this.p2 = attPesos(this.p2, dP2, this.taxa);
    this.b2 = this.b2.map((b,j) => b-this.taxa*dB2[j]);
    this.p1 = attPesos(this.p1, dP1, this.taxa);
    this.b1 = this.b1.map((b,j) => b-this.taxa*dB1[j]);
    eNaN(dX, "dX", "CamadaFFN");
    return dX;
  }
}

class CamadaNormalizacao {
  constructor(dimModelo, epsilon=1e-5, taxaAprendizado=0.001, taxaDropout=0.1) {
    this.gamma = uns(dimModelo);
    this.beta = zeros(dimModelo);
    this.epsilon = epsilon;
    this.taxa = taxaAprendizado;
    this.cache = {};
    this.taxaDrop = taxaDropout;
  }
  propagar(x, treino=true) {
    eNaN(x, "x", "CamadaNormalizacao");
    const seqTam = x.length;
    let saida = [];
    const medias = [];
    const vars = [];
    for(let i=0; i<seqTam; i++) {
      const seq = x[i];
      const media = seq.reduce((s,v)=>s+v,0)/seq.length;
      const vari = seq.reduce((s,v)=>s+(v-media)**2,0)/seq.length;
      medias[i] = media;
      vars[i] = vari;
      const std = Math.sqrt(vari+this.epsilon);
      saida[i] = seq.map((v,j)=>(v-media)/std*this.gamma[j]+this.beta[j]);
    }
    this.cache = { x, medias, vars };
    if(treino) saida = dropout(saida, this.taxaDrop);
    eNaN(saida, "saida", "CamadaNormalizacao");
    return saida;
  }
  
  retropropagar(dY) {
    eNaN(dY, "dY", "CamadaNormalizacao");
    const { x, medias, vars } = this.cache;
    const seqTam = x.length;
    const dim = this.gamma.length;
    
    let dGamma = zeros(dim);
    let dBeta = zeros(dim);
    const dX = Array(seqTam).fill(0).map(() => zeros(dim));
    
    for(let i=0; i<seqTam; i++) {
      const seq = x[i];
      const dYi = dY[i];
      const media = medias[i];
      const vari = vars[i];
      const std = Math.sqrt(vari+this.epsilon);
      const invStd = 1/std;
      
      const xCentrado = seq.map(val => val-media);
      
      for(let j=0; j<dim; j++) {
        dGamma[j] += dYi[j]*xCentrado[j]*invStd;
        dBeta[j] += dYi[j];
      }
      const N = dim;
      const termoComum = dYi.map((dy, j)=>dy*this.gamma[j]*invStd);
      const somaTermo1 = termoComum.reduce((s, val)=>s+val, 0);
      const somaTermo2 = xCentrado.reduce((s, xc, j)=>s+xc*termoComum[j], 0);
      
      for(let j=0; j<dim; j++) {
        dX[i][j] = termoComum[j]-(1/N)*somaTermo1-(1/N)*xCentrado[j]*somaTermo2*invStd*invStd;
      }
    }
    
    dGamma = clipVetor(dGamma, 5.0);
    dBeta = clipVetor(dBeta, 5.0);
    
    this.gamma = this.gamma.map((g, j)=>g-this.taxa*dGamma[j]);
    this.beta = this.beta.map((b, j)=>b-this.taxa*dBeta[j]);
    eNaN(dX, "dX", "CamadaNormalizacao");
    return dX;
  }
}

class BlocoTransformer {
  constructor(dimModelo, numCabecas, dimFFN, taxaAprendizado=0.001, taxaDrop) {
    this.atencao = new CamadaAtencao(dimModelo, numCabecas, taxaAprendizado, taxaDrop);
    this.ffn = new CamadaFFN(dimModelo, dimFFN, taxaAprendizado, taxaDrop);
    this.norm1 = new CamadaNormalizacao(dimModelo, 1e-6, taxaAprendizado, taxaDrop);
    this.norm2 = new CamadaNormalizacao(dimModelo, 1e-6, taxaAprendizado, taxaDrop);
  }
  propagar(x, mascara=null) {
    eNaN(x, "x", "BlocoTransformer");
    const a = this.atencao.propagar(x, mascara);
    eNaN(a, "a", "BlocoTransformer");
    const r1 = x.map((v,i)=>somarVetores(v, a[i]));
    eNaN(r1, "r1", "BlocoTransformer");
    const n1 = this.norm1.propagar(r1);
    const f = this.ffn.propagar(n1);
    const r2 = n1.map((v,i)=>somarVetores(v, f[i]));
    const n2 = this.norm2.propagar(r2);
    this.cache = { x, a, r1, n1, f, r2, n2, mascara };
    return n2;
  }
  
  retropropagar(dY) {
    dY = dY.map(vetor => clipVetor(vetor, 1.0));
    eNaN(dY, "dY", "BlocoTransformer");
    if(dY.some(v => v.some(isNaN))) console.error("dY contém NaN antes da retropropagação");
    const { x, a, r1, n1, f, r2 } = this.cache;
    const dNorm2 = this.norm2.retropropagar(dY);
    eNaN(dNorm2, "dNorm2", "BlocoTransformer");
    const dR2 = dNorm2;
    const dF = dR2;
    const dN1_daF = this.ffn.retropropagar(dF);
    eNaN(dN1_daF, "dN1_daF", "BlocoTransformer");
    const dN1 = dR2.map((v,i)=>somarVetores(v, dN1_daF[i]));
    eNaN(dN1, "dN1", "BlocoTransformer");
    const dR1 = this.norm1.retropropagar(dN1);
    eNaN(dR1, "dR1", "BlocoTransformer");
    const dX_daRes = dR1;
    const dA = dR1;
    eNaN(dA, "dA", "BlocoTransformer");
    const dX_daAtencao = this.atencao.retropropagar(dA);
    eNaN(dX_daAtencao, "dX_daAtencao", "BlocoTransformer");
    const res = dX_daRes.map((v,i)=>somarVetores(v, dX_daAtencao[i]));
    eNaN(res, "res", "BlocoTransformer");
    return res;
  }
}

class CodificadorPosicional {
  constructor(dimModelo, seqMaxima=5000) {
    this.dimModelo = dimModelo;
    this.seqMaxima = seqMaxima;
    this.codificacao = this.gerarCodificacao();
  }
  gerarCodificacao() {
    const codificacao = [];
    for(let pos=0; pos<this.seqMaxima; pos++) {
      const vetor = [];
      for(let i=0; i<this.dimModelo; i++) {
        if(i % 2==0) {
          vetor.push(Math.sin(pos/Math.pow(10000, i/this.dimModelo)));
        } else {
          vetor.push(Math.cos(pos/Math.pow(10000, (i-1)/this.dimModelo)));
        }
      }
      codificacao.push(vetor);
    }
    return codificacao;
  }
  aplicar(x) {
    return x.map((seq, pos) => somarVetores(seq, this.codificacao[pos]));
  }
}
class GSATransformer {
  constructor(config) {
    this.vocabTam = config.vocabTam;
    this.dimModelo = config.dimModelo;
    this.numCamadas = config.numCamadas;
    this.numCabecas = config.numCabecas;
    this.dimFFN = config.dimFFN;
    this.seqMaxima = config.seqMaxima || 512;
    this.taxa = config.taxaAprendizado || 0.001;
    this.taxaDropoutEmbed = config.taxaDropoutEmbed || 0.1;
    this.taxaDropout = config.taxaDropout || 0.1;

    this.embedding = transpor(iniPesosXavier(config.dimModelo, config.vocabTam));
    this.codificadorPos = new CodificadorPosicional(config.dimModelo, config.seqMaxima);
    this.camadas = [];
    for(let i=0; i<config.numCamadas; i++) this.camadas.push(new BlocoTransformer(config.dimModelo, config.numCabecas, config.dimFFN, config.taxaAprendizado));
    this.normFinal = new CamadaNormalizacao(config.dimModelo, 1e-6, config.taxaAprendizado);
    this.cabecaLM = iniPesosXavier(config.vocabTam, config.dimModelo);
    this.biasCabeca = zeros(config.vocabTam);
    this.cache = {};
  }

  propagar(tokens, treino=true) {
    const seqTam = tokens.length
    let xNoPos = tokens.map(t => {
      const emb = this.embedding[t];
      if(!emb) {
        console.error(`Token ${t} não encontrado no embedding! Usando embedding padrão.`);
        return normZPonto(this.embedding[1]); // usa <DES> como fallback
        }
        return normZPonto(emb);
    });
    if(treino) xNoPos = dropout(xNoPos, this.taxaDropoutEmbed);
    
    let xPos = this.codificadorPos.aplicar(xNoPos);
    let x = xPos;
    
    const saidas = [];
    for(const camada of this.camadas) {
      x = camada.propagar(x, this.gerarMascaraCausal(seqTam));
      saidas.push(clonarMatriz(x));
    }
    const normSaida = this.normFinal.propagar(x);
    const logits = normSaida.map((seq, i) => {
      if(seq.length != this.cabecaLM[0].length) throw new Error(`[GSATransformer]: vetor em normSaida[${i}] tem tamanho ${seq.length}, esperado ${this.cabecaLM[0].length}`);
      const proj = aplicarMatriz(this.cabecaLM, seq); // 210
      if(proj.length != this.biasCabeca.length) throw new Error(`[GSATransformer]: projeção retornou ${proj.length}, mas biasCabeca tem ${this.biasCabeca.length}`);
      return somarVetores(proj, this.biasCabeca);
    });
    // zeração de segurança pra NaN e Infinity
    for(let i=0; i<logits.length; i++) {
      for(let j=0; j<logits[i].length; j++) {
        if(!isFinite(logits[i][j]) || isNaN(logits[i][j])) logits[i][j] = 0;
        else if(logits[i][j]>50) logits[i][j] = 50; // evita explodir no exp do softmax
        else if(logits[i][j]<-50) logits[i][j] = -50;
      }
    }
    eNaN(x, "x", "GSATransformer");
    eNaN(logits, "logits", "GSATransformer");
    eNaN(normSaida, "normSaida", "GSATransformer");
    this.cache = { tokens, xNoPos, xPos, saidas, normSaida, logits };
    return logits;
  }

  retropropagar(dLogits) {
    eNaN(dLogits, "dLogits", "GSATransformer");
    const { tokens, xPos, saidas, normSaida } = this.cache;
    const seqTam = tokens.length;

    const dBias = zeros(this.vocabTam);
    for(const dl of dLogits)
    for(let j=0; j<this.vocabTam; j++) dBias[j] += dl[j];
    
    const dCabecaEntrada = dLogits.map(dl=>aplicarMatriz(transpor(this.cabecaLM), dl));
    eNaN(dCabecaEntrada, "dCabecaEntrada", "GSATransformer");
    const dPcab = matrizZeros(this.cabecaLM.length, this.dimModelo);
    for(let i=0; i<seqTam; i++) {
      const inVec = normSaida[i];
      for(let j=0; j<this.vocabTam; j++) {
        for(let k=0; k<this.dimModelo; k++) dPcab[j][k] += inVec[k]*dLogits[i][j];
      }
    }

    let dX = dCabecaEntrada;
    dX = dCabecaEntrada.map(vetor => clipVetor(vetor, 1.0));
    dX = this.normFinal.retropropagar(dX);
    
    for(let i=this.camadas.length-1; i >= 0; i--) {
      const ent = i==0 ? xPos : saidas[i-1];
      dX = this.camadas[i].retropropagar(dX);
    }
    eNaN(dX, "dX", "GSATransformer");

    const dEmb = matrizZeros(this.embedding.length, this.dimModelo);
    for(let i=0; i<seqTam; i++) {
      const t = tokens[i];
      dEmb[t] = somarVetores(dEmb[t], dX[i]);
    }

    this.biasCabeca = this.biasCabeca.map((b,i)=>b-this.taxa*dBias[i]);
    this.cabecaLM = attPesos(this.cabecaLM, dPcab, this.taxa);
    this.embedding = attPesos(this.embedding, dEmb, this.taxa);
  }
  gerarMascaraCausal(n) {
    const m = [];
    for(let i=0; i<n; i++) {
      m[i] = [];
      for(let j=0; j<n; j++) m[i][j] = j <= i ? 1 : 0;
    }
    return m;
  }
  gerar(prompt, maxTokens=50, temperatura=0.8, repeticaoPenal=1.5) {
    let tokens = [...prompt];
    const idFIM = gsa.tokenizador.tokenPraId.get('<FIM>');
    
    for(let i=0; i<maxTokens; i++) {
      const logits = this.propagar(tokens, false);
      const ultimosLogits = logits[logits.length-1];
      
      // penalização para tokens repetidos
      const tokensUnicos = [...new Set(tokens)];
      tokensUnicos.forEach(token => {
        ultimosLogits[token] /= repeticaoPenal;
      });
    
      const probs = softmax(ultimosLogits, temperatura);
      const proximoToken = this.exemploProb(probs);
      
      // para se gerar FIM
      if(proximoToken===idFIM) break;
      
      tokens.push(proximoToken);
      if(tokens.length >= this.seqMaxima) break;
    }
    return tokens;
  }
  exemploProb(probs) {
    const raio = Math.random();
    let soma = 0;
    for(let i=0; i<probs.length; i++) {
      soma += probs[i];
      if(raio<soma) return i;
    }
    return probs.length-1;
  }
  
  calcularPerda(tokens) {
    const logits = this.propagar(tokens.slice(0, -1));
    const alvos = tokens.slice(1);
    let perdaTotal = 0;
    for(let i = 0; i<logits.length; i++) {
      const probs = softmax(logits[i]);
      const alvo = alvos[i];
      perdaTotal += -Math.log(probs[alvo]+1e-10);
    }
    return perdaTotal/logits.length;
  }
}
class TokenizadorBPE {
  constructor(merges=[]) {
    this.vocab = {};
    this.bpeRanks = {};
    this.cache = new Map();
    this.byteEncoder = {};
    this.byteDecoder = {};
    // mapeia 0-255 para caracteres únicos visíveis
    const bs = Array.from({ length: 256 }, (_, i) => i);
    const cs = bs.map(b => {
      if(b>=33 && b<=126 || b>=161 && b<=172 || b>=174 && b<=255) return String.fromCharCode(b);
      return String.fromCharCode(b+256);
    });
    for(let i=0; i<256; ++i) {
      this.byteEncoder[i] = cs[i];
      this.byteDecoder[cs[i]] = i;
    }
    for(let i=0; i<merges.length; ++i) this.bpeRanks[merges[i].join(' ')] = i;
    this.tokenPraId = new Map([['<ALMO>', 0], ['<DES>', 1], ['<FIM>', 2]]);
    this.idPraToken = new Map([[0, '<ALMO>'], [1, '<DES>'], [2, '<FIM>']]);
    this.proximoId = 2;
  }
  construir(texto) {
    // codifica todo o texto para extrair subwords
    const tokens = this.encode(texto);
    // cria vocabulário com os tokens únicos
    const tokensUnicos = [...new Set(tokens)];
    tokensUnicos.forEach(token => {
      if(!this.tokenPraId.has(token)) {
        this.tokenPraId.set(token, this.proximoId);
        this.idPraToken.set(this.proximoId, token);
        this.proximoId++;
      }
    });
  }
  codificar(texto) {
    const tokensBPE = this.encode(texto);
    return tokensBPE.map(token => this.tokenPraId.get(token) || 1); // 1 = <DES>
  }
  decodificar(ids) {
    const tokens = ids.map(id => {
      if(id==this.idPraToken.get("<FIM>")) return '';
      return this.idPraToken.get(id) || '<DES>';
    });
    return this.decode(tokens.join('').split('')); // recompõe o texto
  }
  get vocabTam() {
    return this.proximoId;
  }
  obterPares(palavra) {
    const pares = new Set();
    for(let i=0; i<palavra.length-1; ++i) pares.add(palavra[i]+" "+palavra[i+1]);
    return pares;
  }
  bpe(token) {
    if(this.cache.has(token)) return this.cache.get(token);
    let palavra = token.split('');
    let pares = this.obterPares(palavra);
    if(!pares.size) return [token];
    while(true) {
      let minRank = Infinity;
      let melhorPar = null;
      for(const par of pares) {
        const rank = this.bpeRanks[par];
        if(rank != undefined && rank<minRank) {
          minRank = rank;
          melhorPar = par;
        }
      }
      if(melhorPar==null) break;
      const [primeiro, segundo] = melhorPar.split(' ');
      const novaPalavra = [];
      let i = 0;
      while(i<palavra.length) {
        let j = palavra.indexOf(primeiro, i);
        if(j==-1 || j==palavra.length-1) {
          novaPalavra.push(...palavra.slice(i));
          break;
        }
        if(palavra[j+1]==segundo) {
          novaPalavra.push(...palavra.slice(i, j));
          novaPalavra.push(primeiro+segundo);
          i = j+2;
        } else {
          novaPalavra.push(palavra[i]);
          i++;
        }
      }
      palavra = novaPalavra;
      pares = this.obterPares(palavra);
    }
    this.cache.set(token, palavra);
    return palavra;
  }
  encode(texto) {
    const bytes = new TextEncoder().encode(texto);
    const tokens = [];
    for(const byte of bytes) {
      const ch = this.byteEncoder[byte];
      const bpeTokens = this.bpe(ch);
      tokens.push(...bpeTokens);
    }
    return tokens;
  }
  decode(tokens) {
    const bytes = tokens.map(t => {
      return this.byteDecoder[t] != undefined ? this.byteDecoder[t] : 63; // ?
    });
    return new TextDecoder().decode(Uint8Array.from(bytes));
  }
}
class TreinadorGSA {
  constructor(modelo, taxaAprendizado=0.0001) {
    this.modelo = modelo;
    this.epocas = 0;
    this.modelo.taxa = taxaAprendizado;
    this.historico = [];
    this.taxaInicial = 0.1;
    this.idFIM = 2;
  }
  
  treinar(dados, epocas=10, tamanhoLote=8) {
    this.idFIM = gsa.tokenizador.tokenPraId.get('<FIM>') || 2;
    for(let epoca=0; epoca<epocas; epoca++) {
      let perdaTotalEpoca = 0;
      let totalTokensEpoca = 0;
      let numLotes = 0;
      
      for(let i=0; i<dados.length; i += tamanhoLote) {
        const lote = dados.slice(i, i+tamanhoLote);
        const { perdaLote, tokensNoLote } = this.treinarLote(lote, epoca);
        perdaTotalEpoca += perdaLote*tokensNoLote;
        totalTokensEpoca += tokensNoLote;
        numLotes++;
        
        if(numLotes%10==0) {
          console.log(`Época ${epoca+1}/${epocas}, Lote ${numLotes}, Perda: ${perdaLote.toFixed(4)}, Taxa: ${this.modelo.taxa.toFixed(4)}`);
          console.log("\nAmostra \"Olá\": ", gsa.gerar("Olá", 10, 0.6));
          console.log("Amostra \"O plasma\": ", gsa.gerar("O plasma", 10, 0.6));
          console.log("Amostra \"O hidrogênio é\": ", gsa.gerar("O hidrogênio é", 10, 0.6)+"\n");
          gsa.ctx = "";
        }
      }
      const perdaMedia = totalTokensEpoca>0 ? perdaTotalEpoca/totalTokensEpoca : 0;
      this.historico.push(perdaMedia);
      console.log(`Época ${epoca+1}/${epocas}, Perda média: ${perdaMedia.toFixed(4)}, Taxa: ${this.modelo.taxa.toFixed(4)}`);
      
      if(epoca%5==0) {
        salvar(gsa, "modelo-"+epoca+".gsa");
        console.log("\nAmostra \"olá\": ", gsa.gerar("olá", 10, 0.6));
        console.log("Amostra \"tudo bem?\": ", gsa.gerar("tudo bem?", 10, 0.6));
        console.log("Amostra \"quem é você?\": ", gsa.gerar("quem é você?", 10, 0.6)+"\n");
        gsa.ctx = "";
      }
      if(epoca>0 && this.historico.length>1) {
        let perdaAntiga = this.historico[epoca-1];
        if(perdaMedia>perdaAntiga*1.02) {
          this.modelo.taxa *= 0.7;
          console.log(`[TAXA][Redução]: ${this.modelo.taxa.toFixed(4)}`);
        } else if(perdaMedia<perdaAntiga*0.98) {
          this.modelo.taxa = Math.min(this.modelo.taxa*1.08, 0.01);
          console.log(`[TAXA][Aumento]: ${this.modelo.taxa.toFixed(4)}`);
        }
      }
    }
  }
  treinarLote(lote, epoca) {
    const taxaAtual = this.taxaInicial/(1+0.01*epoca);
    this.modelo.taxa = taxaAtual;
    let perdaTotal = 0;
    let totalTokens = 0;

    for(const seq of lote) {
      const posFIM = seq.indexOf(this.idFIM);
      const seqValida = posFIM >= 0 ? seq.slice(0, posFIM+1) : seq;
      
      if(seqValida.length<2) continue;
      const entrada = seqValida.slice(0, -1);
      const esperado = seqValida.slice(1);
      // previsões
      const logits = this.modelo.propagar(entrada);
      const probs = logits.map(l => softmax(l));
      
      const sVerdade = esperado.map((t, i) => {
        return oneHot(t, this.modelo.vocabTam);
      });
      // gradientes
      let dLogits = sVerdade.map((s, i) => derivadaEntropiaCruzada(s, probs[i]));
      dLogits = dLogits.map(vetor => clipVetor(vetor, 1.0));
      this.modelo.retropropagar(dLogits);
      // calcula perda pra cada posição
      for(let pos=0; pos<sVerdade.length; pos++) {
        perdaTotal += entropiaCruzada(sVerdade[pos], probs[pos]);
      }
      totalTokens += sVerdade.length;
    }
    const perdaMedia = totalTokens>0 ? perdaTotal/totalTokens : 0;
    return { perdaLote: perdaMedia, tokensNoLote: totalTokens };
  }
}
function criarGSA(textoTreinamento, config={}, taxa=0.001) {
  const tokenizador = new TokenizadorBPE();
  tokenizador.construir(textoTreinamento);
  const configuracao = {
    vocabTam: tokenizador.vocabTam,
    dimModelo: config.dimModelo || 256,
    numCamadas: config.numCamadas || 4,
    numCabecas: config.numCabecas || 8,
    dimFFN: config.dimFFN || 1024,
    seqMaxima: config.seqMaxima || 512,
    taxaDropout: config.taxaDropout || 0.1,
    taxaDropoutEmbed: config.taxaDropoutEmbed || 0.1,
    ...config
  };
  const modelo = new GSATransformer(configuracao);
  const treinador = new TreinadorGSA(modelo, taxa);
  return {
    modelo,
    ctx: "",
    tokenizador,
    treinador,
    gerar: function(prompt, maxTokens=50, temperatura=0.8) {
      this.ctx = (this.ctx+prompt)
      .split(/(?<=[.!?])\s+/) // divide por sentenças
      .slice(-5) // mantém 5 últimas sentenças
      .join(' ');
      const tokens = tokenizador.codificar(this.ctx);
      const idFIM = tokenizador.tokenPraId.get('<FIM>');
      const gerados = modelo.gerar(tokens, maxTokens, temperatura);
      // filtra FIM da saída final
      const saida = tokenizador.decodificar(gerados.filter(t => t !== idFIM));
      this.ctx += " "+saida;
      return saida;
    }
  };
}
let gsa = null;

const fs = require('fs');
const rd = require('readline');

const rl = rd.createInterface({
  input: process.stdin,
  output: process.stdout
});

const estudo = [
  "fisica",
  "biologia",
  "quimica",
  "programacao",
  "filosofia",
  "poesia",
  "astronomia",
  "medicina",
  "psicologia",
  "matematica",
  "JAVA",
  "HTML",
  "CSS",
  "JS",
  "C",
  
  "conversa"
];

let treino = "";
for(let i=0; i<estudo.length; i++) {
  if(fs.existsSync("treino/"+estudo[i]+".txt")) {
    treino += fs.readFileSync("treino/"+estudo[i]+".txt", "utf-8")+"\n\n";
    console.log("[MATERIA]: +"+estudo[i]);
  } else {
    console.log("[MATERIA]: NÃO EXISTE "+estudo[i]+".txt");
  }
}
// fs.writeFileSync("treino.txt", treino, "utf-8");
console.log("[DADOS DE TREINO]: Concluídos");
// treino = fs.readFileSync("treino.txt", "utf-8");
function prepararDados(texto, tokenizador, seqTam=32) {
  texto = texto
  .replace(/https?:\/\/\S+/g, '') // remove ulrs
  .replace(/```[\s\S]*?```/g, '') // remove blocos de código
  .replace(/[\u{1F600}-\u{1F6FF}]/gu, ''); // remove emojis
  
  const tokens = tokenizador.codificar(texto);
  const seqs = [];
  
  const idFIM = tokenizador.tokenPraId.get('<FIM>');
  
  for(let i=0; i<tokens.length-seqTam; i += seqTam) {
    let seq = tokens.slice(i, i+seqTam);
    seq.push(idFIM); // Adiciona EOS no final
    seqs.push(seq);
  }
  return seqs;
} 

function obterParams(gsa) {
  const emb = gsa.modelo.vocabTam*gsa.modelo.dimModelo; // embedding
  const cabecaFinal = gsa.modelo.vocabTam*gsa.modelo.dimModelo // cabecaLM
  +gsa.modelo.vocabTam; // biasCabeca
  const porBloco = 
  // atenção: pq, pk, pv, ps+bias de cada uma
  4*gsa.modelo.dimModelo*gsa.modelo.dimModelo
  +4*gsa.modelo.dimModelo
  // FFN: p1, p2 + b1, b2
  +gsa.modelo.dimFFN*gsa.modelo.dimModelo
  +gsa.modelo.dimModelo*gsa.modelo.dimFFN
  +gsa.modelo.dimFFN
  +gsa.modelo.dimModelo
  // normalizações: 2 camadas×(gamma+beta)
  +2*(gsa.modelo.dimModelo+gsa.modelo.dimModelo);
  const totalTransformer = gsa.modelo.numCamadas*porBloco;
  return emb+cabecaFinal+totalTransformer;
}

function avaliarModelo(gsa, perguntasTeste) {
  console.log('\n=== AVALIAÇÃO DO MODELO ===\n');
  perguntasTeste.forEach((pergunta, i) => {
    console.log(`${i+1}. Pergunta: "${pergunta}"`);
    const resposta = gsa.gerar(pergunta, 20, 0.7);
    console.log(`   Resposta: "${resposta}"\n`);
  });
}

function calcularPerplexidade(modelo, tokenizador, textoTeste) {
  const tokens = tokenizador.codificar(textoTeste);
  let perdaTotal = 0;
  let numTokens = 0;
  
  for(let i=1; i<tokens.length; i++) {
    const contexto = tokens.slice(0, i);
    const logits = modelo.propagar(contexto);
    const probs = softmax(logits[logits.length-1]);
    const esperado = tokens[i];
    
    perdaTotal += -Math.log(probs[esperado]+1e-10);
    numTokens++;
  }
  const perplexidade = Math.exp(perdaTotal/numTokens);
  return perplexidade;
}

function conversa() {
  console.log('Digite "/pr" para encerrar a conversa\n');
  function perguntarUsuario() {
    rl.question('Você: ', (entrada) => {
      if(entrada.toLowerCase()=='/pr') {
        rl.close();
        return;
      } else if(entrada.startsWith("$")) eval(entrada.replace("$", ""));
      const resposta = gsa.gerar(entrada, 20, 0.6);
      console.log(`ALVA GSA-1: ${resposta}`);
      perguntarUsuario();
    });
  }
  perguntarUsuario();
}

function executarTeste() {
  rl.question('> Treinar novo? (s/n) ', (entrada) => {
    if(entrada.toLowerCase()=='s') {
      console.log('> Criando modelo...');
      gsa = criarGSA(treino, {
        dimModelo: 32, // 192,
        numCamadas: 3,// 4,
        numCabecas: 8, // 8,
        dimFFN: 32*4, // 768,
        seqMaxima: 128,
        taxaAprendizado: 0.001,
        taxaDropout: 0.2
      });
      treinarNovo();
    } else {
      gsa = carregar("modelo.gsa");
      conversa();
    }
  });
}

function treinarNovo() {
  console.log(`> Vocabulário criado com ${gsa.tokenizador.vocabTam} tokens`);
  console.log('> Preparando dados de treinamento...');
  const dados = prepararDados(treino, gsa.tokenizador, 32)// 64);
  console.log(`   ${dados.length} sequências de treinamento ${dados.length > 0 ? "preparadas (√)" : "erradas (X)"}`);
  console.log("==== ESTATÍSTICAS ====");
  console.log(`  Dimensão do modelo: ${gsa.modelo.dimModelo}`);
  console.log(`  Número de camadas: ${gsa.modelo.numCamadas}`);
  console.log(`  Número de cabeças: ${gsa.modelo.numCabecas}`);
  console.log(`  Dimensão FFN: ${gsa.modelo.dimFFN}`);
  console.log(`  Sequência máxima: ${gsa.modelo.seqMaxima}`);
  console.log(`  Taxa Dropout: ${gsa.modelo.taxaDropout}`);
  gsa.treinador.epocas = 2;
  console.log(`  Épocas: ${gsa.treinador.epocas}`);
  console.log(`  Taxa de aprendizado: ${gsa.modelo.taxa}`);
  
  console.log('  ★Parâmetros estimados★:', obterParams(gsa));
  
  console.log('\n> Iniciando treinamento...');
  // TREINAMENTO:
  gsa.treinador.treinar(dados, 15, 8) // 32);
  
  console.log('\n> Testando geração de texto...');
  const perguntasTeste = [
    'Como você está',
    'Qual é o seu nome',
    'O que você gosta',
    'Conte uma história',
    'Qual seu conselho'
  ];
  avaliarModelo(gsa, perguntasTeste);
  
  console.log('\n> Calculando perplexidade...');
  const textoTeste = 'Olá, como você está? Qual é o seu nome?';
  const perplexidade = calcularPerplexidade(gsa.modelo, gsa.tokenizador, textoTeste);
  console.log(`   Perplexidade: ${perplexidade.toFixed(2)}`);
  
  console.log('\n> Exemplos de geração livre:');
  const prompts = ['Era uma vez', 'A tecnologia', 'Amizade é'];
  prompts.forEach(prompt => {
    console.log(`   "${prompt}": "${gsa.gerar(prompt, 20, 0.8)}"`);
  });
  
  console.log('\n=== TREINO CONCLUÍDO ===');
  console.log('Modelo treinado e testado com sucesso!');
  salvar(gsa, "modelo.gsa");
  conversa(gsa);
}

function salvar(gsa, caminhoArquivo) {
  const leitor = {
    vocabTam: gsa.modelo.vocabTam,
    dimModelo: gsa.modelo.dimModelo,
    numCamadas: gsa.modelo.numCamadas,
    numCabecas: gsa.modelo.numCabecas,
    dimFFN: gsa.modelo.dimFFN,
    seqMaxima: gsa.modelo.seqMaxima,
    tokenizador: {
      tokenPraId: Array.from(gsa.tokenizador.tokenPraId.entries()),
      idPraToken: Array.from(gsa.tokenizador.idPraToken.entries())
    },
    historico: gsa.treinador.historico
  };
  // calcula tamanho total dos dados
  let totalFloats = 0;
  // embedding
  totalFloats += gsa.modelo.vocabTam*gsa.modelo.dimModelo;
  
  for(const bloco of gsa.modelo.camadas) {
    // atenção
    totalFloats += 4*(gsa.modelo.dimModelo*gsa.modelo.dimModelo); // pq, pk, pv, ps
    totalFloats += 4*gsa.modelo.dimModelo; // bq, bk, bv, bs
    // FFN
    totalFloats += gsa.modelo.dimFFN*gsa.modelo.dimModelo; // p1
    totalFloats += gsa.modelo.dimFFN; // b1
    totalFloats += gsa.modelo.dimModelo*gsa.modelo.dimFFN; // p2
    totalFloats += gsa.modelo.dimModelo; // b2
    // normalização
    totalFloats += 2*gsa.modelo.dimModelo; // gamma, beta(norm1)
    totalFloats += 2*gsa.modelo.dimModelo; // gamma, beta(norm2)
  }
  // normalização final
  totalFloats += 2*gsa.modelo.dimModelo; // gamma, beta
  // cabeça LM
  totalFloats += gsa.modelo.vocabTam*gsa.modelo.dimModelo; // cabecaLM
  totalFloats += gsa.modelo.vocabTam; // biasCabeca
  // criacbuffer
  const leitorString = JSON.stringify(leitor);
  const leitorTamBytes = leitorString.length;
  // calcular prenchimento pra alinhamento de 4 bytes
  const prenchimento = (4-(leitorTamBytes%4))%4;
  const buffer = new ArrayBuffer(
    8+ // numero mágico+tamanho leitor
    leitorTamBytes+
    prenchimento+ // bytes de alinhamento
    totalFloats*4
  );
  const view = new DataView(buffer);
  let antes = 0;
  // numero magico(GSA1) em hex
  view.setUint32(antes, 0x47534131);
  antes += 4;
  // tamanho do leitor(incluindo prenchimento)
  view.setUint32(antes, leitorTamBytes+prenchimento);
  antes += 4;
  // escreve o leitor
  for(let i=0; i<leitorTamBytes; i++) {
    view.setUint8(antes, leitorString.charCodeAt(i));
    antes++;
  }
  // adiciona bytes de prenchimento para alinhamento
  for(let i=0; i<prenchimento; i++) {
    view.setUint8(antes, 0);
    antes++;
  }
  // escreve parâmetros
  const floatView = new Float32Array(buffer, antes);
  let floatIndice = 0;
  
  function salvarVetor(v) {
    for(let i=0; i<v.length; i++) floatView[floatIndice++] = v[i];
  }
  function salvarMatriz(m) {
    for(let i=0; i<m.length; i++) {
      for(let j=0; j<m[i].length; j++) floatView[floatIndice++] = m[i][j];
    }
  }
  
  // embedding
  salvarMatriz(gsa.modelo.embedding);
  // camadas
  for(const bloco of gsa.modelo.camadas) {
    // atenção
    salvarMatriz(bloco.atencao.pq);
    salvarMatriz(bloco.atencao.pk);
    salvarMatriz(bloco.atencao.pv);
    salvarMatriz(bloco.atencao.ps);
    salvarVetor(bloco.atencao.bq);
    salvarVetor(bloco.atencao.bk);
    salvarVetor(bloco.atencao.bv);
    salvarVetor(bloco.atencao.bs);
    // FFN
    salvarMatriz(bloco.ffn.p1);
    salvarVetor(bloco.ffn.b1);
    salvarMatriz(bloco.ffn.p2);
    salvarVetor(bloco.ffn.b2);
    // normalização
    salvarVetor(bloco.norm1.gamma);
    salvarVetor(bloco.norm1.beta);
    salvarVetor(bloco.norm2.gamma);
    salvarVetor(bloco.norm2.beta);
  }
  // normalização final
  salvarVetor(gsa.modelo.normFinal.gamma);
  salvarVetor(gsa.modelo.normFinal.beta);
  // cabeça LM
  salvarMatriz(gsa.modelo.cabecaLM);
  salvarVetor(gsa.modelo.biasCabeca);
  // salvar arquivo
  fs.writeFileSync(caminhoArquivo, Buffer.from(buffer));
  console.log(`Modelo salvo em: ${caminhoArquivo} (${(buffer.byteLength/1024/1024).toFixed(2)} MB)`);
}

function carregar(caminhoArquivo) {
  const buffer = fs.readFileSync(caminhoArquivo);
  const view = new DataView(buffer.buffer);
  let antes = 0;
  // verifica a versão
  if(view.getUint32(antes) !== 0x47534131) throw new Error("Formato de arquivo inválido");
  antes += 4;
  // le o tamanho do leitor
  const leitorTamPrenchido = view.getUint32(antes);
  antes += 4;
  // le o leitor
  let leitorJSON = "";
  const leitorTam = leitorTamPrenchido;
  for(let i=0; i<leitorTam; i++) {
    const byte = view.getUint8(antes++);
    // ignorar bytes de padding(não imprimíveis)
    if(byte >= 32 && byte <= 126) leitorJSON += String.fromCharCode(byte);
  }
  const leitor = JSON.parse(leitorJSON);
  const tokenizador = new TokenizadorBPE();
  tokenizador.tokenPraId = new Map(leitor.tokenizador.tokenPraId);
  tokenizador.idPraToken = new Map(leitor.tokenizador.idPraToken);
  tokenizador.proximoId = leitor.vocabTam;
  
  const modelo = new GSATransformer(leitor);
  // carrega os parâmetros
  const floatView = new Float32Array(buffer.buffer, antes);
  let floatIndice = 0;
  
  function carregarVetor(tam) {
    const v = new Array(tam);
    for(let i=0; i<tam; i++) v[i] = floatView[floatIndice++];
    return v;
  }
  function carregarMatriz(linhas, cols) {
    const m = new Array(linhas);
    for(let i=0; i<linhas; i++) {
      m[i] = new Array(cols);
      for(let j=0; j<cols; j++) m[i][j] = floatView[floatIndice++];
    }
    return m;
  }
  modelo.embedding = carregarMatriz(leitor.vocabTam, leitor.dimModelo);
  // camadas
  for(let i=0; i<leitor.numCamadas; i++) {
    const bloco = modelo.camadas[i];
    // atenção
    bloco.atencao.pq = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
    bloco.atencao.pk = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
    bloco.atencao.pv = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
    bloco.atencao.ps = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
    
    bloco.atencao.bq = carregarVetor(leitor.dimModelo);
    bloco.atencao.bk = carregarVetor(leitor.dimModelo);
    bloco.atencao.bv = carregarVetor(leitor.dimModelo);
    bloco.atencao.bs = carregarVetor(leitor.dimModelo);
    // FFN
    bloco.ffn.p1 = carregarMatriz(leitor.dimFFN, leitor.dimModelo);
    bloco.ffn.b1 = carregarVetor(leitor.dimFFN);
    bloco.ffn.p2 = carregarMatriz(leitor.dimModelo, leitor.dimFFN);
    bloco.ffn.b2 = carregarVetor(leitor.dimModelo);
    // normalização
    bloco.norm1.gamma = carregarVetor(leitor.dimModelo);
    bloco.norm1.beta = carregarVetor(leitor.dimModelo);
    bloco.norm2.gamma = carregarVetor(leitor.dimModelo);
    bloco.norm2.beta = carregarVetor(leitor.dimModelo);
  }
  // normalização final
  modelo.normFinal.gamma = carregarVetor(leitor.dimModelo);
  modelo.normFinal.beta = carregarVetor(leitor.dimModelo);
  // cabeça LM
  modelo.cabecaLM = carregarMatriz(leitor.vocabTam, leitor.dimModelo);
  modelo.biasCabeca = carregarVetor(leitor.vocabTam);
  // treinador
  const treinador = new TreinadorGSA(modelo);
  treinador.historico = leitor.historico;
  console.log(`Modelo carregado: ${leitor.vocabTam} tokens, ${leitor.numCamadas} camadas`);
  return {
    modelo,
    ctx: "",
    tokenizador,
    treinador,
    gerar: function(prompt, maxTokens=50, temperatura=0.8) {
      this.ctx = (this.ctx+prompt)
      .split(/(?<=[.!?])\s+/) // divide por sentenças
      .slice(-5) // mantém 5 últimas sentenças
      .join(' ');
      const tokens = tokenizador.codificar(this.ctx);
      const idFIM = tokenizador.tokenPraId.get('<FIM>');
      const gerados = modelo.gerar(tokens, maxTokens, temperatura);
      // filtra FIM da saída final
      const saida = tokenizador.decodificar(gerados.filter(t => t !== idFIM));
      this.ctx += " "+saida;
      return saida;
    }
  };
}

function mostrarEstatisticas(gsa) {
  console.log('\n=== CONFIG DO MODELO ===');
  console.log(`  Tamanho do vocabulário: ${gsa.tokenizador.vocabTam}`);
  console.log(`  Dimensão do modelo: ${gsa.modelo.dimModelo}`);
  console.log(`  Número de camadas: ${gsa.modelo.numCamadas}`);
  console.log(`  Número de cabeças: ${gsa.modelo.numCabecas}`);
  console.log(`  Dimensão FFN: ${gsa.modelo.dimFFN}`);
  console.log(`  Sequência máxima: ${gsa.modelo.seqMaxima}`);
  console.log(`  Taxa Dropout: ${gsa.modelo.taxaDropout}`);
  console.log(`  Épocas: ${gsa.treinador.epocas}`);
  console.log(`  Taxa de aprendizado: ${gsa.modelo.taxa}`);
  console.log('  ★Parâmetros estimados★:', obterParams());
  
  if(gsa.treinador.historico.length>0) {
    const perdaInicial = gsa.treinador.historico[0];
    const perdaFinal = gsa.treinador.historico[gsa.treinador.historico.length-1];
    console.log(`Perda inicial: ${perdaInicial.toFixed(4)}`);
    console.log(`Perda final: ${perdaFinal.toFixed(4)}`);
    console.log(`Melhoria: ${((perdaInicial-perdaFinal)/perdaInicial*100).toFixed(2)}%`);
  }
}

executarTeste();
