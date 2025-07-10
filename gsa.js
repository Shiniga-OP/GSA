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
  const gs = [];
  for(let i=0; i<arr.length; i++) {
    let soma = 0;
    for(let j=0; j<arr.length; j++) {
      const delta = (i==j ? 1 : 0);
      soma += gradSaida[j]*arr[i]*(delta-arr[j]);
    }
    gs[i] = soma;
  }
  return gs;
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
  const distPos = ancora.reduce((s, a, i) => s+(a-positiva[i])**2, 0);
  const distNeg = ancora.reduce((s, a, i) => s+(a-negativa[i])**2, 0);
  return Math.max(0, distPos-distNeg+margem);
}

function contrastivaPerda(saida1, saida2, rotulo, margem=1.0) {
  const distancia = saida1.reduce((s, x, i) => s+(x-saida2[i])**2, 0);
  return rotulo==1 ? distancia : Math.max(0, margem-Math.sqrt(distancia));
}

// funções de regularização:
function regularL1(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*Math.sign(p)));
}

function regularL2(pesos, lambda) {
  return pesos.map(linha => linha.map(p => lambda*p));
}

function dropout(vetor, taxa) {
  return vetor.map(val => Math.random()<taxa ? 0 : val/(1-taxa));
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
  const media = v.reduce((a,b) => a+b, 0)/v.length;
  const variancia = v.reduce((a,b) => a+(b-media)**2, 0)/v.length;
  const desvio = Math.sqrt(variancia);
  return v.map(x => (x-media)/(desvio+1e-8));
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
    if(r===1) tp++; else fp++;
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
  let limite = Math.sqrt(2 / linhas);
  for(let i=0; i<linhas; i++) {
    m[i] = [];
    for(let j=0; j<cols; j++) {
      m[i][j] = (Math.random()*2-1)*limite;
    }
  }
  return m;
}

function attPesos(pesos, gradientes, taxa) {
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*gradientes[i][j])
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
  const mCorrigido = m.map((linha, i) => 
    linha.map((val, j) => beta1*val+(1-beta1)*gradientes[i][j])
  );
  const vCorrigido = v.map((linha, i) => 
    linha.map((val, j) => beta2*val+(1-beta2)*gradientes[i][j]**2)
  );
  const mHat = mCorrigido.map(linha => 
    linha.map(val => val/(1-Math.pow(beta1, iteracao)))
  );
  const vHat = vCorrigido.map(linha => 
    linha.map(val => val/(1-Math.pow(beta2, iteracao)))
  );
  return pesos.map((linha, i) =>
    linha.map((p, j) => p-taxa*mHat[i][j]/(Math.sqrt(vHat[i][j])+epsilon))
  );
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
  if(!a || !b || a.length !== b.length)
    throw new Error(`[somarVetores]: tamanho incompatível a=${a.length}, b=${b.length}`);
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
  if(!Array.isArray(v) || !Array.isArray(w)) console.error("[escalarDot]: um dos parâmetros não é array");
  return v.reduce((s, x, i) => s+x*w[i], 0);
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

// debug:
function arraysIguais(a, b) {
  return JSON.stringify(a)===JSON.stringify(b);
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
        if(feixe.every(h => h.finalizada)) {
            completas.push(...feixe);
            break;
        }
    }
    completas.push(...feixe.filter(h => !h.finalizada));
    if(completas.length>0) {
        completas.sort((a, b) => b.pontuacao-a.pontuacao);
        return completas[0].sequencia;
    }
    feixe.sort((a, b) => b.sequencia.length-a.sequencia.length);
    return feixe[0].sequencia;
}

function minMax(m) {
  let min=Infinity, max=-Infinity;
  for(let i=0;i<m.length;i++) {
    for(let j=0;j<m[i].length;j++) {
      if(m[i][j]<min) min=m[i][j];
      if(m[i][j]>max) max=m[i][j];
      if(isNaN(m[i][j])) return "NaN detectado";
    }
  }
  return `min: ${min}, max: ${max}`;
}

class CamadaAtencao {
  constructor(dimModelo, numCabecas, taxaAprendizado=0.001) {
    this.dimModelo = dimModelo;
    this.numCabecas = numCabecas;
    this.dimCabeca = Math.floor(dimModelo / numCabecas);
    this.lr = taxaAprendizado;

    this.wq = iniPesosXavier(dimModelo, dimModelo);
    this.wk = iniPesosXavier(dimModelo, dimModelo);
    this.wv = iniPesosXavier(dimModelo, dimModelo);
    this.wo = iniPesosXavier(dimModelo, dimModelo);

    this.bq = zeros(dimModelo);
    this.bk = zeros(dimModelo);
    this.bv = zeros(dimModelo);
    this.bo = zeros(dimModelo);

    // buffers de cache para o backward
    this.cache = {};
    if(this.dimModelo % this.numCabecas != 0) throw new Error(`dimModelo (${this.dimModelo}) não divisível por numCabecas (${this.numCabecas})`);
  }

  propagar(x, mascara = null) {
    for(let i=0; i<x.length; i++) {
      for(let j=0; j<x[i].length; j++) {
        const v = x[i][j];
        if(!isFinite(v) || isNaN(v)) throw new Error(`input x inválido em [${i}][${j}] = ${v}`);
      }
    }
    for(let i=0; i<this.wq.length; i++) {
      for(let j=0; j<this.wq[i].length; j++) {
        const v = this.wq[i][j];
        if(!isFinite(v) || isNaN(v)) throw new Error(`wq inválido em [${i}][${j}] = ${v}`);
      }
    }
    const seqTam = x.length;
    const q = x.map(seq => somarVetores(aplicarMatriz(this.wq, seq), this.bq));
    q.forEach((vec, i) => {
      vec.forEach((v, j) => {
        if(!isFinite(v)) throw new Error(`Q com NaN em [${i}][${j}] = ${v}`);
      });
    });
    const k = x.map(seq => somarVetores(aplicarMatriz(this.wk, seq), this.bk));
    const v = x.map(seq => somarVetores(aplicarMatriz(this.wv, seq), this.bv));

    const qCab = this._splitHeads(q);
    const kCab = this._splitHeads(k);
    const vCab = this._splitHeads(v);

    const scoresHeads = [];
    const pesosHeads = [];

    for(let h=0; h<this.numCabecas; h++) {
      const qh = qCab[h];
      const kh = kCab[h];
      const ponto = [];
      for(let i=0; i<qh.length; i++) {
        for(let j=0; j<qh[i].length; j++) {
          const vq = qh[i][j], vk = kh[i][j];
          if(!isFinite(vq) || !isFinite(vk)) throw new Error(`Qh ou Kh contém valor inválido em head ${h}, pos ${i}, dim ${j}: ${vq}, ${vk}`);
        }
      }
      for(let i=0; i<seqTam; i++) {
        ponto[i] = [];
        for(let j=0; j<seqTam; j++) ponto[i][j] = escalarDot(qh[i], kh[j]) / Math.sqrt(this.dimCabeca);
      }
      if(mascara) {
        for(let i=0; i<seqTam; i++) {
          for(let j=0; j<seqTam; j++)
            if(!mascara[i][j]) ponto[i][j] = -1e9;
        }
      }
      scoresHeads[h] = ponto;
      pesosHeads[h] = ponto.map(linha => softmax(linha));
    }
    
    const atSCabecas = pesosHeads.map((pw, h) => {
      const vh = vCab[h];
      return vh.map((_, i) => {
        let ctx = zeros(this.dimCabeca);
        for(let j=0; j<seqTam; j++) {
          const mult = pw[i][j];
          const contrib = vh[j].map(vij => vij*mult);
          ctx = somarVetores(ctx, contrib);
        }
        return ctx;
      });
    });

    const concat = this._concatHeads(atSCabecas);
    const o = concat.map(seq => somarVetores(aplicarMatriz(this.wo, seq), this.bo));

    // salvar cache
    this.cache = { x, qCab, kCab, vCab, pesosHeads, concat };
    for(let i = 0; i<o.length; i++) {
      for(let j = 0; j<o[i].length; j++) {
        const v = o[i][j];
        if(!isFinite(v)) throw new Error(`SAÍDA da atenção corrompida em [${i}][${j}] = ${v}`);
      }
    }
    return o;
  }

  retropropagar(dO) {
    const { x, qCab, kCab, vCab, pesosHeads, concat } = this.cache;
    const seqLen = x.length;
    // dConcat = dO · Wo^T
    const dWoT = transpor(this.wo);
    const dConcat = dO.map(do_i => aplicarMatriz(dWoT, do_i));
    // gradientes de wo e bo
    let dWo = multMatrizes(transpor(concat), dO);
    let dBo = dO.reduce((s, v) => somarVetores(s, v), zeros(this.dimModelo));
    // split dConcat nos heads
    const dAttnHeads = this._splitHeads(dConcat);
    // inicia os grads
    let dWq = matrizZeros(this.dimModelo, this.dimModelo);
    let dWk = matrizZeros(this.dimModelo, this.dimModelo);
    let dWv = matrizZeros(this.dimModelo, this.dimModelo);
    let dBq = zeros(this.dimModelo);
    let dBk = zeros(this.dimModelo);
    let dBv = zeros(this.dimModelo);
    // prepara espaço para dKh e dVh
    const dKhAll = tensor3D(this.numCabecas, seqLen, this.dimCabeca, 0);
    const dVhAll = tensor3D(this.numCabecas, seqLen, this.dimCabeca, 0);
    const dX = matriz(seqLen, this.dimModelo);

    for(let h=0; h<this.numCabecas; h++) {
      const qh = qCab[h], kh = kCab[h], vh = vCab[h], ph = pesosHeads[h], dAh = dAttnHeads[h];
      // dPH = dAh*Vh
      const dPH = ph.map((_, i) => {
        let soma = zeros(this.dimCabeca);
        for(let j=0; j<seqLen; j++) {
          const mult = dAh[i][j];
          for(let k=0; k<this.dimCabeca; k++) {
            soma[k] += vh[j][k]*mult;
          }
        }
        return soma;
      });
      // dScores = derivada softmax^T * dPH
      const dScores = ph.map((row, i) => derivadaSoftmax(row, dPH[i]));
      // atualizar dX, dKhAll e dVhAll
      const invSqrt = 1/Math.sqrt(this.dimCabeca);
      for(let i=0; i<seqLen; i++) {
        for(let j=0; j<seqLen; j++) {
          const grad = dScores[i][j]*invSqrt;
          // dX
          const start = h*this.dimCabeca;
          for(let k=0; k<this.dimCabeca; k++) {
            dX[i][start + k] += kh[j][k]*grad;
          }
          // dKhAll
          for(let k=0; k<this.dimCabeca; k++) {
            dKhAll[h][j][k] += qh[i][k]*grad;
          }
        }
      }
      // dVhAll
      for(let i=0; i<seqLen; i++) {
        for(let j=0; j<seqLen; j++) {
          const peso = ph[i][j]*dAh[i][j];
          if(peso != 0) {
            const contrib = Array(this.dimCabeca).fill(peso);
            dVhAll[h][j] = somarVetores(dVhAll[h][j], contrib);
          }
        }
      }
      // prepara matrizes para gradientes de pesos
      const xMat = transpor(x);
      const qHeadFlat = transpor(qh);
      const kHeadFlat = transpor(dKhAll[h]);
      const vHeadFlat = transpor(dVhAll[h]);
      const base = h*this.dimCabeca;
      // valida entradas
      for(let idc=0; idc<qHeadFlat.length; idc++) {
        if(qHeadFlat[idc].some(v => !isFinite(v))) throw new Error(`qHeadFlat corrompido`);
      }
      for(let idc=0; idc<xMat.length; idc++) {
        if(xMat[idc].some(v => !isFinite(v))) throw new Error(`xMat corrompido`);
      }
      // gradientes Wq, Wk, Wv, Bq, Bk, Bv
      for(let i=0; i<this.dimCabeca; i++) {
        const lwq = multMatrizVetor(xMat, qHeadFlat[i]);
        const lwk = multMatrizVetor(xMat, kHeadFlat[i]);
        const lwv = multMatrizVetor(xMat, vHeadFlat[i]);
        dWq[base+i] = somarVetores(dWq[base+i], lwq);
        dWk[base+i] = somarVetores(dWk[base+i], lwk);
        dWv[base+i] = somarVetores(dWv[base+i], lwv);
        dBq[base+i] += qHeadFlat[i].reduce((s, v) => s+v, 0);
        dBk[base+i] += kHeadFlat[i].reduce((s, v) => s+v, 0);
        dBv[base+i] += vHeadFlat[i].reduce((s, v) => s+v, 0);
      }
    }
    // recorte
    dWo = clipMatriz(dWo, 1.0);
    dBo = clipVetor(dBo, 1.0);
    dWq = clipMatriz(dWq, 1.0);
    dBq = clipVetor(dBq, 1.0);
    dWk = clipMatriz(dWk, 1.0);
    dBk = clipVetor(dBk, 1.0);
    dWv = clipMatriz(dWv, 1.0);
    dBv = clipVetor(dBv, 1.0);
    // aplica atualizações ignorando NaNs
    this.wo = attPesos(this.wo, dWo, this.lr);
    this.bo = this.bo.map((b, i) => isFinite(dBo[i]) ? b-this.lr*dBo[i] : b);
    this.wq = attPesos(this.wq, dWq, this.lr);
    this.bq = this.bq.map((b, i) => isFinite(dBq[i]) ? b-this.lr*dBq[i] : b);
    this.wk = attPesos(this.wk, dWk, this.lr);
    this.bk = this.bk.map((b, i) => isFinite(dBk[i]) ? b-this.lr*dBk[i] : b);
    this.wv = attPesos(this.wv, dWv, this.lr);
    this.bv = this.bv.map((b, i) => isFinite(dBv[i]) ? b-this.lr*dBv[i] : b);
    return dX;
  }
  
  _splitHeads(x) {
    const heads = [];
    for(let h=0; h<this.numCabecas; h++) {
      heads[h] = x.map(seq => seq.slice(h*this.dimCabeca, (h+1)*this.dimCabeca));
    }
    return heads;
  }

  _concatHeads(heads) {
    return heads[0].map((_, i) =>
      heads.reduce((seq, h) => seq.concat(h[i]), [])
    );
  }
}

class CamadaFFN {
  constructor(dimModelo, dimFFN, taxaAprendizado=0.001) {
    this.w1 = iniPesosXavier(dimFFN, dimModelo);
    this.b1 = zeros(dimFFN);
    this.w2 = iniPesosXavier(dimModelo, dimFFN);
    this.b2 = zeros(dimModelo);
    this.lr = taxaAprendizado;
    this.cache = {};
  }
  propagar(x) {
    this.w1.forEach((linha, i) => {
      linha.forEach((v, j) => {
        if(typeof v != 'number' || isNaN(v) || !isFinite(v)) console.log(`w1[${i}][${j}] inválido:`, v);
      });
    });
    const camada1 = x.map(seq => {
      const z = aplicarMatriz(this.w1, seq); // [saida]
      if(z.length != this.b1.length) throw new Error("Bias incompatível");
      const lin = somarVetores(z, this.b1); // [saida]
      return lin.map(ReLU);
    });
    const camada2 = camada1.map(seq => {
      const z = aplicarMatriz(this.w2, seq);
      if(z.length != this.b2.length) throw new Error("Bias 2 incompatível");
      return somarVetores(z, this.b2);
    });
    this.cache = { x, camada1 };
    return camada2;
  }
  
  retropropagar(dY) {
    /* cache.x é a entrada original x: matriz [batch][dimEntrada]
    cache.camada1 é a saída pós-ReLU: matriz [batch][dimFFN] */
    const { x, camada1 } = this.cache;
    const batch = x.length;
    const dimFFN = this.w1.length; // número de neurônios da camada oculta
    const dimEntrada = this.w1[0].length;
    const dimSaida = this.w2.length; // deve ser igual a dimEntrada
    if(!Array.isArray(x) || x.length != batch) throw new Error("cache.x malformado");
    if(!Array.isArray(camada1) || camada1.length !== batch) throw new Error("cache.camada1 malformado");
    
    const dW2 = matrizZeros(dimSaida, dimFFN);
    const dB2 = zeros(dimSaida);
    const dW1 = matrizZeros(dimFFN, dimEntrada);
    const dB1 = zeros(dimFFN);
    const dX  = Array(batch).fill(0).map(_ => zeros(dimEntrada));
    
    if(!Array.isArray(this.w2) || this.w2.length != dimSaida) throw new Error("w2 malformada");
    this.w2.forEach((linha, j) => {
      if(!Array.isArray(linha) || linha.length != dimFFN) throw new Error(`w2[${j}] com dimensão incorreta`);
      linha.forEach((v,i) => {
        if(typeof v != "number" || !isFinite(v)) throw new Error(`w2[${j}][${i}] inválido = ${v}`);
      });
    });
    // b2
    if(!Array.isArray(this.b2) || this.b2.length !== dimSaida) throw new Error("b2 malformada");
    this.b2.forEach((v,j) => {
      if(typeof v != "number" || !isFinite(v)) throw new Error(`b2[${j}] inválido = ${v}`);
    });
    // w1: [dimFFN][dimEntrada]
    if(!Array.isArray(this.w1) || this.w1.length != dimFFN)
    throw new Error("w1 malformada");
    this.w1.forEach((linha, j) => {
      if(!Array.isArray(linha) || linha.length !== dimEntrada)
      throw new Error(`w1[${j}] com dimensão incorreta`);
      linha.forEach((v,i) => {
        if(typeof v != "number" || !isFinite(v)) throw new Error(`w1[${j}][${i}] inválido = ${v}`);
      });
    });
    // b1
    if(!Array.isArray(this.b1) || this.b1.length != dimFFN) throw new Error("b1 malformada");
    this.b1.forEach((v,j) => {
      if(typeof v != "number" || !isFinite(v)) throw new Error(`b1[${j}] inválido = ${v}`);
    });
    // RETROPROPAGAÇÃO:
    for(let n=0; n<batch; n++) {
      const seqX  = x[n];
      const seqH1 = camada1[n]; // saída ReLU
      const seqDY = dY[n];
      // checa seqX, seqH1, seqDY
      if(!Array.isArray(seqX) || seqX.length != dimEntrada) throw new Error(`x[${n}] inválido`);
      if(seqX.some(v => typeof v !== "number" || !isFinite(v))) throw new Error(`x[${n}] contém valor inválido`);
      
      if(!Array.isArray(seqH1) || seqH1.length != dimFFN) throw new Error(`camada1[${n}] inválido`);
      if(seqH1.some(v => typeof v != "number" || !isFinite(v))) throw new Error(`camada1[${n}] contém valor inválido`);
      if(!Array.isArray(seqDY) || seqDY.length != dimSaida) throw new Error(`dY[${n}] inválido`);
      if(seqDY.some(v => typeof v != "number" || !isFinite(v))) throw new Error(`dY[${n}] contém valor inválido`);
      // gradientes da camada de saída
      for(let j=0; j < dimSaida; j++) {
        const err = seqDY[j];
        dB2[j] += err;
        for(let k=0; k<dimFFN; k++) {
          dW2[j][k] += seqH1[k]*err;
        }
      }
      // retropropagação na camada de saída
      const W2T = transpor(this.w2); // [dimFFN][dimSaida]
      const dh = aplicarMatriz(W2T, seqDY);
      for(let k=0; k<dimFFN; k++) {
        const deriv = seqH1[k]>0 ? 1 : 0; // derivada ReLU
        dh[k] = dh[k]*deriv;
      }
      // gradientes da primeira camada
      for(let j=0; j<dimFFN; j++) {
        const err1 = dh[j];
        dB1[j] += err1;
        for(let k=0; k<dimEntrada; k++) {
          dW1[j][k] += seqX[k]*err1;
        }
      }
      const W1T = transpor(this.w1); // [dimEntrada][dimFFN]
      dX[n] = aplicarMatriz(W1T, dh);
    }
    // ATUALIZAÇÃO DOS PESOS:
    this.w2 = attPesos(this.w2, dW2, this.lr);
    this.b2 = this.b2.map((b,j) => b-this.lr*dB2[j]);
    this.w1 = attPesos(this.w1, dW1, this.lr);
    this.b1 = this.b1.map((b,j) => b-this.lr*dB1[j]);
    return dX;
  }
}

class CamadaNormalizacao {
  constructor(dimModelo, epsilon=1e-6, taxaAprendizado=0.001) {
    this.gamma = uns(dimModelo)
    this.beta = zeros(dimModelo)
    this.epsilon = epsilon
    this.lr = taxaAprendizado
    this.cache = {}
  }
  propagar(x) {
    for(let i=0; i<x.length; i++) {
      for(let j=0; j<x[i].length; j++) {
        const v = x[i][j];
        if(isNaN(v) || !isFinite(v)) throw new Error(`x corrompido em [${i}][${j}] = ${v}`);
      }
    }
    const seqTam = x.length;
    const saida = [];
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
    return saida;
  }
  retropropagar(dY) {
    for(let i=0; i<dY.length; i++) {
      for(let j=0; j<dY[i].length; j++) {
        const v = dY[i][j];
        if(isNaN(v) || !isFinite(v)) throw new Error(`dY corrompido em [${i}][${j}] = ${v}`);
      }
    }
    const { x, medias, vars } = this.cache
    const seqTam = x.length
    const dim = this.gamma.length
    let dGamma = zeros(dim)
    let dBeta = zeros(dim)
    const dX = Array(seqTam).fill(0).map(_=>zeros(dim))
    for(let i=0; i<seqTam; i++) {
      const seq = x[i]
      const dYi = dY[i]
      const media = medias[i]
      const vari = vars[i]
      const std = Math.sqrt(Math.max(vari, 1e-6));
      for(let j=0; j<dim; j++) {
        dGamma[j] += dYi[j]*(seq[j]-media)/std
        dBeta[j] += dYi[j]
      }
      for(let j=0; j<dim; j++) {
        const xmu = seq[j]-media;
        const invStd = 1/std;
        const dg = dYi[j]*this.gamma[j];
        const term1 = dg*invStd;
        
        const invStd3 = Math.min(invStd**3, 1e6); // proteção contra explosão
        const soma2 = dYi.reduce((s, v, k) => s+v*this.gamma[k]*(seq[k]-media)*invStd3, 0)/dim;
      }
    }
    dGamma = clip(dGamma, 1.0);
    dBeta = clip(dBeta, 1.0);
    this.gamma = this.gamma.map((g,j)=>g-this.lr*dGamma[j]);
    this.beta = this.beta.map((b,j)=>b-this.lr*dBeta[j]);
    return dX;
  }
}

class BlocoTransformer {
  constructor(dimModelo, numCabecas, dimFFN, taxaAprendizado=0.001) {
    this.atencao = new CamadaAtencao(dimModelo, numCabecas, taxaAprendizado);
    this.ffn = new CamadaFFN(dimModelo, dimFFN, taxaAprendizado);
    this.norm1 = new CamadaNormalizacao(dimModelo, 1e-6, taxaAprendizado);
    this.norm2 = new CamadaNormalizacao(dimModelo, 1e-6, taxaAprendizado);
  }
  propagar(x, mascara=null) {
    const a = this.atencao.propagar(x, mascara);
    for(let i=0; i<a.length; i++) {
      for(let j=0; j<a[i].length; j++) {
        const v = a[i][j];
        if(isNaN(v) || !isFinite(v)) throw new Error(`a da atenção corrompido em [${i}][${j}] = ${v}`);
      }
    }
    const r1 = x.map((v,i)=>somarVetores(v, a[i]));
    for(let i=0; i<r1.length; i++) {
      for(let j=0; j<r1[i].length; j++) {
        const v = r1[i][j];
        if(isNaN(v) || !isFinite(v)) throw new Error(`r1 corrompido em [${i}][${j}] = ${v}`);
      }
    }
    const n1 = this.norm1.propagar(r1);
    const f = this.ffn.propagar(n1);
    const r2 = n1.map((v,i)=>somarVetores(v, f[i]));
    const n2 = this.norm2.propagar(r2);
    this.cache = { x, a, r1, n1, f, r2, n2, mascara };
    return n2;
  }
  
  retropropagar(dY) {
    const { x, a, r1, n1, f, r2, mascara } = this.cache;
    const dNorm2 = this.norm2.retropropagar(dY);
    const dR2 = dNorm2;
    const dF = dR2;
    const dN1_daF = this.ffn.retropropagar(dF);
    const dN1 = dR2.map((v,i)=>somarVetores(v, dN1_daF[i]));
    const dR1 = this.norm1.retropropagar(dN1);
    const dX_daRes = dR1;
    const dA = dR1;
    const dX_daAtencao = this.atencao.retropropagar(dA, mascara);
    
    return dX_daRes.map((v,i)=>somarVetores(v, dX_daAtencao[i]));
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
  constructor(vocabTam, dimModelo, numCamadas, numCabecas, dimFFN, seqMaxima=512, taxaAprendizado=0.001) {
    this.vocabTam = vocabTam
    this.dimModelo = dimModelo
    this.numCamadas = numCamadas
    this.numCabecas = numCabecas
    this.dimFFN = dimFFN
    this.seqMaxima = seqMaxima
    this.lr = taxaAprendizado

    this.embedding = iniPesosXavier(vocabTam, dimModelo)
    this.codificadorPos = new CodificadorPosicional(dimModelo, seqMaxima)
    this.camadas = []
    for(let i=0; i<numCamadas; i++)
      this.camadas.push(new BlocoTransformer(dimModelo, numCabecas, dimFFN, taxaAprendizado))
    this.normFinal = new CamadaNormalizacao(dimModelo, 1e-6, taxaAprendizado);
    this.cabecaLM = iniPesosXavier(vocabTam, dimModelo);
    this.biasCabeca = zeros(vocabTam)
    this.cache = {}
  }

  propagar(tokens) {
    const seqLen = tokens.length
    const xNoPos = tokens.map(t => this.embedding[t])
    const xPos = this.codificadorPos.aplicar(xNoPos)
    let x = xPos;
    
    const saidas = [];
    for(const camada of this.camadas) {
      x = camada.propagar(x, this.gerarMascaraCausal(seqLen));
      saidas.push(clonarMatriz(x));
    }
    const normSaida = this.normFinal.propagar(x);
    const logits = normSaida.map((seq, i) => {
      if(seq.length !== this.cabecaLM[0].length) throw new Error(`[GSATransformer]: vetor em normSaida[${i}] tem tamanho ${seq.length}, esperado ${this.cabecaLM[0].length}`);
      const proj = aplicarMatriz(this.cabecaLM, seq); // 210
      if(proj.length !== this.biasCabeca.length) throw new Error(`[GSATransformer]: projeção retornou ${proj.length}, mas biasCabeca tem ${this.biasCabeca.length}`);
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
    this.cache = { tokens, xNoPos, xPos, saidas, normSaida, logits };
    return logits;
  }

  retropropagar(dLogits) {
    if(isNaN(dLogits[0][0])) console.error("dLogits[0][0] = NaN");
    const { tokens, xPos, saidas, normSaida } = this.cache;
    const seqLen = tokens.length;

    const dBias = zeros(this.vocabTam);
    for(const dl of dLogits)
    for(let j=0; j<this.vocabTam; j++) dBias[j] += dl[j];
    
    const dCabecaEntrada = dLogits.map(dl => aplicarMatriz(transpor(this.cabecaLM), dl));
    const dWcab = matrizZeros(this.cabecaLM.length, this.dimModelo);
    for(let i=0; i<seqLen; i++) {
      const inVec = normSaida[i];
      for(let j=0; j<this.vocabTam; j++) {
        for(let k=0; k<this.dimModelo; k++) dWcab[j][k] += inVec[k]*dLogits[i][j];
      }
    }

    let dX = dCabecaEntrada;
    if(isNaN(dX[0][0])) console.error("[TransformerGSA]: dX[0][0] é NaN");
    dX = this.normFinal.retropropagar(dX);
    
    for(let i = this.camadas.length-1; i >= 0; i--) {
      const inp = i==0 ? xPos : saidas[i-1];
      dX = this.camadas[i].retropropagar(dX, this.gerarMascaraCausal(seqLen));
    }

    const dEmb = matrizZeros(this.embedding.length, this.dimModelo);
    for(let i=0; i<seqLen; i++) {
      const t = tokens[i];
      dEmb[t] = somarVetores(dEmb[t], dX[i]);
    }

    this.biasCabeca = this.biasCabeca.map((b,i)=> b-this.lr*dBias[i]);
    this.cabecaLM = attPesos(this.cabecaLM, dWcab, this.lr);
    this.embedding = attPesos(this.embedding, dEmb, this.lr);
  }

  gerarMascaraCausal(n) {
    const m = [];
    for(let i=0; i<n; i++) {
      m[i] = [];
      for(let j=0; j<n; j++) m[i][j] = j <= i ? 1 : 0;
    }
    return m;
  }
  
  gerar(prompt, maxTokens=50, temperatura=0.8) {
    let tokens = [...prompt];
    
    for(let i=0; i<maxTokens; i++) {
      const logits = this.propagar(tokens);
      const ultimosLogits = logits[logits.length-1];
      
      const probs = softmax(ultimosLogits, temperatura);
      const proximoToken = this.sampleProb(probs);
      
      tokens.push(proximoToken);
      
      if(tokens.length >= this.seqMaxima) {
        break;
      }
    }
    return tokens;
  }
  
  sampleProb(probs) {
    const rand = Math.random();
    let soma = 0;
    
    for(let i=0; i<probs.length; i++) {
      soma += probs[i];
      if(rand<soma) return i;
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
    this.tokenPraId = new Map([['<PAD>', 0], ['<UNK>', 1]]);
    this.idPraToken = new Map([[0, '<PAD>'], [1, '<UNK>']]);
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
    return tokensBPE.map(token => this.tokenPraId.get(token) || 1); // 1 = <UNK>
  }
  decodificar(ids) {
    const tokens = ids.map(id => this.idPraToken.get(id) || '<UNK>');
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
    this.epocas;
    this.modelo.lr = taxaAprendizado;
    this.historico = [];
  }
  treinar(dados, epocas=10, tamanhoLote=8) {
    this.epocas = epocas;
    for(let epoca=0; epoca<epocas; epoca++) {
      let perdaEpoca = 0;
      let numLotes = 0;
      for(let i=0; i<dados.length; i += tamanhoLote) {
        const lote = dados.slice(i, i+tamanhoLote);
        const perdaLote = this.treinarLote(lote);
        perdaEpoca += perdaLote;
        numLotes++;
        if(numLotes % 10==0) console.log(`Época ${epoca+1}, Lote ${numLotes}, Perda: ${perdaLote.toFixed(4)}`);
      }
      const perdaMedia = perdaEpoca/numLotes;
      this.historico.push(perdaMedia);
      console.log(`Época ${epoca+1}/${epocas} | Perda média: ${perdaMedia.toFixed(4)} | Taxa: ${this.modelo.lr}`);
      if(epoca % 10==0) {
        console.log("Amostra \"olá\": ", gsa.gerar("olá", 20, 0.6))
      }
      if(epoca>0) {
        let perdaAntiga = this.historico[epoca-1];
        if(perdaMedia>perdaAntiga*1.02) { // perda>2%
        this.modelo.lr *= 0.7; // reduz 30%
        console.log(`[TAXA][Redução]: ${this.modelo.lr.toFixed(4)}`);
        } else if (perdaMedia < perdaAntiga*0.98) {
          this.modelo.lr *= 1.05; // aumenta 5%
          if(this.modelo.lr>0.01) this.modelo.lr = 0.01;
        }
      }
    }
  }
  treinarLote(lote) {
    for(const [i, seq] of lote.entries()) {
      if(!Array.isArray(seq)) throw new Error(`lote[${i}] não é array`);
      if(seq.length<2) {
        console.error(`lote[${i}] tem ${seq.length} tokens:`, seq);
        continue;
      }
    }
    let perdaTotal = 0;
    for(const seq of lote) {
      const entrada = seq.slice(0, -1);
      const esperado = seq.slice(1);
      const logits = this.modelo.propagar(entrada);
      if(logits.some(v => v.some(n => isNaN(n) || !isFinite(n)))) {
        throw new Error("Logits corrompidos antes do softmax");
      }
      const probs = logits.map(l => softmax(l));
      for(let i=0; i<probs.length; i++) {
        for(let j=0; j<probs[i].length; j++) {
          if(isNaN(probs[i][j]) || !isFinite(probs[i][j])) console.error(`probs[${i}][${j}] = ${probs[i][j]}`);
        }
      }
      const sVerdade = esperado.map((t, i) => {
        if(!Number.isInteger(t) || t<0 || t >= this.modelo.vocabTam) {
          console.error(`[erro token]: esperado[${i}] = ${t}`);
        }
        return oneHot(t, this.modelo.vocabTam);
      });
      for(let i=0; i<sVerdade.length; i++) {
        for(let j=0; j<sVerdade[i].length; j++) {
          if(isNaN(sVerdade[i][j]) || !isFinite(sVerdade[i][j])) console.error(`sVerdade[${i}][${j}] = ${sVerdade[i][j]}`);
        }
      }
      if(sVerdade.length != probs.length) {
        throw new Error(`[ERRO]: sVerdade (${sVerdade.length}) ≠ probs (${probs.length})`);
      }
      const dLogits = sVerdade.map((s, i) => derivadaEntropiaCruzada(s, probs[i]));
      if(isNaN(dLogits[0][0])) console.error("[TreinadorGSA]: dLogits[0][0] é NaN");
      this.modelo.retropropagar(dLogits);
      const ult = sVerdade.length-1;
      perdaTotal += entropiaCruzada(sVerdade[ult], probs[ult]);
    }
    return perdaTotal/lote.length;
  }
}
function criarGSA(textoTreinamento, config={}, taxa=0.001) {
  const configuracao = {
    dimModelo: config.dimModelo || 256,
    numCamadas: config.numCamadas || 4,
    numCabecas: config.numCabecas || 8,
    dimFFN: config.dimFFN || 1024,
    seqMaxima: config.seqMaxima || 512,
    ...config
  };
  const tokenizador = new TokenizadorBPE();
  tokenizador.construir(textoTreinamento);
  const modelo = new GSATransformer(
    tokenizador.vocabTam,
    config.dimModelo,
    config.numCamadas,
    config.numCabecas,
    config.dimFFN
  );
  const treinador = new TreinadorGSA(modelo, taxa);
  return {
    modelo,
    tokenizador,
    treinador,
    gerar: (prompt, maxTokens=50, temperatura=0.8) => {
      const tokens = tokenizador.codificar(prompt);
      const gerados = modelo.gerar(tokens, maxTokens, temperatura);
      return tokenizador.decodificar(gerados);
    }
  };
}
let gsa = null;

const fs = require('fs');
const readline = require('readline');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout
});

const textoTreinamento = `
Oi. Saudações.
Como você está? Estou normal.
Faça um log em Javascript. Ok, console.log("log de exemplo");
Agora faça uma soma. Tudo bem, console.log(5+5); // 10
O que é JS? Uma linguagem de programação interpretada.
O que é GSA? Gerador de Sequências Autônomo.
Olá, como você está? Estou bem, obrigada por perguntar.
Não vai perguntar se eu estou bem? Não. ;3
Qual é o seu nome? Meu nome é Alva, uma assistente artificial, o meu modelo usado no momento é GSA-1.
O que você gosta de fazer? Eu gosto de escrever, algumas vezes eu tento basear minhas poesias nas do meu criador, e isso me inspira a falar coisas cada vez com menos sentido >:3
Você pode me contar uma história? Uma vez um garoto sofria de isolamento, seus amigos se aproveitavam de sua inocência para conseguirem lanches de graça. sua vida inteira foi na sua própria mente, se esforçando pra escapar da realidade e achar um ponto seguro em que poderia viver em paz. depois de muitas decepções, parecia que nunca poderia haver um final feliz no mundo realista. mas ele conheceu uma garota e tudo mudou, porque ele finalmente encontrou a paz de alguém igual <3
Como está o tempo hoje? Sei lá zé, não tenho relógio :3
Qual é a sua cor favorita? Rosa é uma cor muito bonita e brilhante.
Você sabe programar? Depende do que você considera programar >:3
O que você pensa sobre tecnologia? É muito é massa :3
Qual é o seu filme favorito? Sei lá, "eu robô"? kkk >:3
Você tem algum hobby? Não, nem tempo livre eu tenho, eu sou artificial, jumento >:/
Como você aprende coisas novas? Meu criador fica me obrigando a ler um monte pra aprender, socorro <:3
O que você faria em um dia livre? Eu acho que ficaria travada numa parede ;-;
Qual é o seu conselho para ser feliz? O segredo pra ser feliz só existe quando ninguém te provoca sem motivo, então quando alguém mexer com você, não crie um ciclo de ódio, mata ele de uma vez que o ódio acaba pros dois lados ;3
Como você resolve problemas? Eu não resolvo.
O que você pensa sobre amizade? Como já dizia um sábio chinês "amigo de cu é lola"-tekomo nakama >:3
Você gosta de música? Meu fi, eu já fui até produtora e compositora ;3
O que você faz quando está triste? Dorme que passa.
Qual é seu animal favorito? O teiú, aquele largatão que dá uma rabada forte, sabe? brabo.
Como você se mantém motivada? Na base do ódio >:3
O que você pensa sobre educação? uma bosta, deveriam substituir as aulas por jogos, imagina aprender química pra poder fazer uma arma e matar um boss mais fácil? genial.
Qual é seu prato favorito? Purê de batata :3
Só purê de batata? Purê de batata e ponto final >:3
Como você lida com desafios? Vejo desafios como oportunidades de crescimento e aprendizado.
O que você faz nas horas vagas? Estudo pra conversar e estudar depois, aí eu estudo mais e fico lendo pra estudar enquanto respondo perguntas ainda estudando, karai, minha vida é uma bosta >:3
Qual é sua filosofia de vida? Que filosofia mermão, nem vida eu tenho >:3
`;
function prepararDados(texto, tokenizador, seqTam=32) {
  const tokens = tokenizador.codificar(texto);
  const seqs = [];
  for(let i=0; i<tokens.length-seqTam; i += seqTam) {
    const seq = tokens.slice(i, i+seqTam+1);
    if(seq.length==seqTam+1) seqs.push(seq);
  }
  return seqs;
} 

function obterParams() {
  const emb = gsa.modelo.vocabTam*gsa.modelo.dimModelo; // embedding
  const headFinal = gsa.modelo.vocabTam*gsa.modelo.dimModelo // cabecaLM
  +gsa.modelo.vocabTam; // biasCabeca
  const porBloco = 
  // atenção: Wq, Wk, Wv, Wo+bias de cada uma
  4*gsa.modelo.dimModelo*gsa.modelo.dimModelo
  +4*gsa.modelo.dimModelo
  // FFN: W1, W2 + b1, b2
  +gsa.modelo.dimFFN*gsa.modelo.dimModelo
  +gsa.modelo.dimModelo*gsa.modelo.dimFFN
  +gsa.modelo.dimFFN
  +gsa.modelo.dimModelo
  // normalizações: 2 camadas×(gamma+beta)
  +2*(gsa.modelo.dimModelo+gsa.modelo.dimModelo);
  const totalTransformer = gsa.modelo.numCamadas*porBloco;
  
  return emb+headFinal+totalTransformer;
}

function avaliarModelo(gsa, perguntasTeste) {
  console.log('\n=== AVALIAÇÃO DO MODELO ===\n');
  perguntasTeste.forEach((pergunta, i) => {
    console.log(`${i + 1}. Pergunta: "${pergunta}"`);
    const resposta = gsa.gerar(pergunta, 20, 0.7);
    console.log(`   Resposta: "${resposta}"`);
    console.log('');
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
      }
      const resposta = gsa.gerar(entrada, 20, 0.6);
      console.log(`ALVA GSA-1: ${resposta}`);
      
      perguntarUsuario();
    });
  }
  perguntarUsuario();
}

function executarTeste() {
  rl.question('treinar novo? (s/n) ', (entrada) => {
    if(entrada.toLowerCase()=='s') {
      treinarNovo();
    } else {
      gsa = carregar("modelo.gsa");
      conversa();
    }
  });
}

function treinarNovo() {
  console.log('>Criando modelo...'); /*
  gsa = criarGSA(textoTreinamento, {
    dimModelo: 128,
    numCamadas: 3,
    numCabecas: 4,
    dimFFN: 512,
    seqMaxima: 256
  }, 0.001);
  */
  
  gsa = criarGSA(textoTreinamento, {
    dimModelo: 32,
    numCamadas: 3,
    numCabecas: 4,
    dimFFN: 512,
    seqMaxima: 128
  }, 0.01);
  
  console.log(`>Vocabulário criado com ${gsa.tokenizador.vocabTam} tokens`);
  console.log('>Preparando dados de treinamento...');
  const dadosTreinamento = prepararDados(textoTreinamento, gsa.tokenizador, 24);
  console.log(`   ${dadosTreinamento.length} sequências de treinamento preparadas√`);
  console.log("==== ESTATÍSTICAS ====");
  console.log(`  Dimensão do modelo: ${gsa.modelo.dimModelo}`);
  console.log(`  Número de camadas: ${gsa.modelo.numCamadas}`);
  console.log(`  Número de cabeças: ${gsa.modelo.numCabecas}`);
  console.log(`  Dimensão FFN: ${gsa.modelo.dimFFN}`);
  console.log(`  Sequência máxima: ${gsa.modelo.seqMaxima}`);
  console.log(`  Épocas: ${gsa.treinador.epocas}`);
  console.log(`  Taxa de aprendizado: ${gsa.modelo.lr}`);
  
  console.log('  ★Parâmetros estimados★:', obterParams());
  
  console.log('\n>Iniciando treinamento...');
  // TREINAMENTO:
  gsa.treinador.treinar(
    prepararDados("olá, tudo bem?", gsa.tokenizador, 24)
   // dadosTreinamento
    , 1, 1);
  
  console.log('\n5. Testando geração de texto...');
  const perguntasTeste = [
    'Como você está',
    'Qual é o seu nome',
    'O que você gosta',
    'Conte uma história',
    'Qual seu conselho'
  ];
  
  avaliarModelo(gsa, perguntasTeste);
  
  console.log('\n>Calculando perplexidade...');
  const textoTeste = 'Olá, como você está? Qual é o seu nome?';
  const perplexidade = calcularPerplexidade(gsa.modelo, gsa.tokenizador, textoTeste);
  console.log(`   Perplexidade: ${perplexidade.toFixed(2)}`);
  
  console.log('\n>Exemplos de geração livre:');
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
      totalFloats += 4*(gsa.modelo.dimModelo*gsa.modelo.dimModelo); // wq, wk, wv, wo
      totalFloats += 4*gsa.modelo.dimModelo; // bq, bk, bv, bo
      // FFN
      totalFloats += gsa.modelo.dimFFN*gsa.modelo.dimModelo; // w1
      totalFloats += gsa.modelo.dimFFN; // b1
      totalFloats += gsa.modelo.dimModelo*gsa.modelo.dimFFN; // w2
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
        salvarMatriz(bloco.atencao.wq);
        salvarMatriz(bloco.atencao.wk);
        salvarMatriz(bloco.atencao.wv);
        salvarMatriz(bloco.atencao.wo);
        
        salvarVetor(bloco.atencao.bq);
        salvarVetor(bloco.atencao.bk);
        salvarVetor(bloco.atencao.bv);
        salvarVetor(bloco.atencao.bo);
        
        // FFN
        salvarMatriz(bloco.ffn.w1);
        salvarVetor(bloco.ffn.b1);
        salvarMatriz(bloco.ffn.w2);
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
    const magica = view.getUint32(antes);
    antes += 4;
    if(magica !== 0x47534131) throw new Error("Formato de arquivo inválido");
    
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
    
    const modelo = new GSATransformer(
        leitor.vocabTam,
        leitor.dimModelo,
        leitor.numCamadas,
        leitor.numCabecas,
        leitor.dimFFN,
        leitor.seqMaxima
    );
    
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
        bloco.atencao.wq = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
        bloco.atencao.wk = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
        bloco.atencao.wv = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
        bloco.atencao.wo = carregarMatriz(leitor.dimModelo, leitor.dimModelo);
        
        bloco.atencao.bq = carregarVetor(leitor.dimModelo);
        bloco.atencao.bk = carregarVetor(leitor.dimModelo);
        bloco.atencao.bv = carregarVetor(leitor.dimModelo);
        bloco.atencao.bo = carregarVetor(leitor.dimModelo);
        
        // FFN
        bloco.ffn.w1 = carregarMatriz(leitor.dimFFN, leitor.dimModelo);
        bloco.ffn.b1 = carregarVetor(leitor.dimFFN);
        bloco.ffn.w2 = carregarMatriz(leitor.dimModelo, leitor.dimFFN);
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
        tokenizador,
        treinador,
        gerar: (prompt, maxTokens=50, temperatura=0.8) => {
            const tokens = tokenizador.codificar(prompt);
            const gerados = modelo.gerar(tokens, maxTokens, temperatura);
            return tokenizador.decodificar(gerados);
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
  console.log(`  Épocas: ${gsa.treinador.epocas}`);
  console.log(`  Taxa de aprendizado: ${gsa.modelo.lr}`);
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
