// biblioteca de utilitários 100% português PT-br e JS
// ativações:
function degrau(x){return x>=0?1:0;}
function sigmoid(x){return 1/(1+Math.exp(-x));}
function derivadaSigmoid(x){return x*(1-x);}
function hardSigmoid(x){return Math.max(0,Math.min(1,0.2*x+0.5));}
function derivadaHardSigmoid(x){return (x>-2.5&& x<2.5)?0.2:0;}
function tanh(x){return Math.tanh(x);}
function derivadaTanh(x){return 1-x*x;}
function ReLU(x){return Math.max(0,x);}
function derivadaReLU(x){return x>0?1:0;}
function leakyReLU(x){return x>0?x:0.01*x;}
function derivadaLeakyReLU(x){return x>0?1:0.01;}
function softsign(x){return x/(1+Math.abs(x));}
function derivadaSoftsign(x){const denom=1+Math.abs(x);return 1/(denom*denom);}
function softplus(x){return Math.log(1+Math.exp(x));}
function swish(x){return x*sigmoid(x);}
function derivadaSwish(x){const sigmoidX=sigmoid(x);return sigmoidX+x*sigmoidX*(1-sigmoidX);}
function hardSwish(x){return x*Math.max(0,Math.min(1,(x+3)/6));}
function derivadaHardSwish(x){return x<=-3?0:x>=3?1:(x+3)/6+x/6;}
function GELU(x){return 0.5*x*(1+tanh(Math.sqrt(2/Math.PI)*(x+0.044715*Math.pow(x,3))));}
function ELU(x,alfa=1.0){return x>=0?x:alfa*(Math.exp(x)-1);}
function derivadaELU(x,alfa=1.0){return x>=0?1:ELU(x,alfa)+alfa;}
function SELU(x,alfa=1.67326,escala=1.0507){return escala*(x>=0?x:alfa*(Math.exp(x)-1));}
function derivadaSELU(x,alfa=1.67326,escala=1.0507){return escala*(x>=0?1:alfa*Math.exp(x));}
function SiLU(x){return x*sigmoid(x);}
function mish(x){return x*tanh(Math.log(1+Math.exp(x)));}
function derivadaMish(x){const omega=4*(x+1)+4*Math.exp(2*x)+Math.exp(3*x)+Math.exp(x)*(4*x+6);const delta=2*Math.exp(x)+Math.exp(2*x)+2;return Math.exp(x)*omega/(delta*delta);}
function bentIdentity(x){return(Math.sqrt(x*x+1)-1)/2+x;}
function derivadaBentIdentity(x){return x/(2*Math.sqrt(x*x+1))+1;}
function gaussian(x){return Math.exp(-x*x);}
function derivadaGaussian(x){return-2*x*Math.exp(-x*x);}
// funções de saída:
function softmax(arr, temperatura=1) {
  if(!Array.isArray(arr)) throw new Error("[softmax]: valor passado não é um array");
  if(!isFinite(temperatura) || temperatura <= 0) temperatura=1e-8;
  const max = Math.max(...arr.map((v, i)=>{
    if(!isFinite(v)) throw new Error(`[softmax] arr[${i}] é NaN`)
    return v;
  }));
  const exps = arr.map(v=>{
    const t = (v-max)/temperatura;
    if(!isFinite(t)) throw new Error("[softmax]: t é NaN");
    return Math.exp(t);
  });
  const soma = exps.reduce((a, b)=>a+b, 0) || 1e-8;
  return exps.map(e=>e/soma);
}
function derivadaSoftmax(arr, gradSaida) {
  if(!Array.isArray(arr)) throw new Error("a derivada de softmax só pode receber vetores. Não: ", arr);
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
function softmaxLote(matriz,temperatura=1){return matriz.map(linha=>softmax(linha,temperatura));}
function argmax(v){return v.indexOf(Math.max(...v));}
function addRuido(v,intenso=0.01){return v.map(x=>x+(Math.random()*2-1)*intenso);}
// funções de erro:
function erroAbsolutoMedio(saida,esperado){return saida.reduce((s,x,i)=>s+Math.abs(x-esperado[i]),0)/saida.length;}
function erroQuadradoEsperado(saida,esperado){return saida.reduce((s,x,i)=>s+0.5*(x-esperado[i])**2,0);}
function derivadaErro(saida,esperado){return saida.map((x,i)=>x-esperado[i]);}
function entropiaCruzada(y,yChapeu){return-y.reduce((s,yi,i)=>s+yi*Math.log(yChapeu[i]+1e-12),0);}
function derivadaEntropiaCruzada(y, yChapeu) {
  if(!Array.isArray(y) || !Array.isArray(yChapeu) || y.length != yChapeu.length) throw new Error("[derivadaEntropiaCruzada]: vetor inválido");
  return yChapeu.map((yci, i) => {
    const yi = y[i];
    if(!isFinite(yi) || !isFinite(yci)) throw new Error(`[derivadaEntropiaCruzada]: valor inválido em i=${i}: y=${yi}, y^=${yci}`);
    return yci-yi;
  });
}
function huberPerda(saida, esperado, delta=1.0) {
  const erros = saida.map((x, i)=>{
    const diff = x-esperado[i];
    return Math.abs(diff) <= delta ? 0.5*diff*diff : delta*(Math.abs(diff)-0.5*delta);
  });
  return erros.reduce((s, x)=>s+x, 0)/saida.length;
}
function derivadaHuber(saida, esperado, delta=1.0) {
  return saida.map((x, i)=>{
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
function regularL1(pesos,lambda) {return pesos.map(linha=>linha.map(p=>lambda*Math.sign(p)));}
function regularL2(pesos,lambda){return pesos.map(linha=>linha.map(p=>lambda*p));}
function dropout(tensor, taxa) {
  if(Array.isArray(tensor)) return tensor.map(sub=>dropout(sub, taxa));
  else return Math.random()<taxa?0:tensor/(1-taxa);
}
function clipGrad(grad,maxVal=1.0){return Math.min(Math.max(grad,-maxVal),maxVal);}
function normEntrada(vetor) {
  const max = Math.max(...vetor);
  const min = Math.min(...vetor);
  const amplitude = max-min || 1e-8;
  return vetor.map(x=>(x-min)/amplitude);
}
function normZPonto(v) {
  const media = v.reduce((a,b)=>a+b, 0)/v.length;
  const variancia=v.reduce((a,b)=>a+(b-media)**2,0)/v.length;
  const desvio = Math.sqrt(variancia+1e-8);
  return v.map(x=>(x-media)/desvio);
}
function acuracia(saida, esperado) {
  const corretos=saida.reduce((s,x,i)=>s+(argmax(x)===argmax(esperado[i])?1:0),0);
  return corretos/saida.length;
}
function precisao(confusao) {
  const tp = confusao[0][0], fp = confusao[0].slice(1).reduce((a,b)=>a+b, 0);
  return tp/(tp+fp+1e-8);
}
function recall(confusao) {
  const tp = confusao[0][0], fn = confusao.slice(1).reduce((s, l)=>s+l[0], 0);
  return tp/(tp+fn+1e-8);
}
function f1Score(confusao) {
  const p = precisao(confusao), r = recall(confusao);
  return 2*(p*r)/(p+r+1e-8);
}
function mse(saida,esperado){return saida.reduce((s,x,i)=>s+(x-esperado[i])**2,0)/saida.length;}
function klDivergencia(p,q){return p.reduce(((s,pi,i)=>s+(pi*Math.log((pi+1e-12)/(q[i]+1e-12)))),0);}
function rocAuc(pontos, rotulos) {
  const pares = pontos.map((s, i)=>[s, rotulos[i]]).sort((a,b)=>b[0]-a[0]);
  let auc = 0, fp = 0, tp = 0, fpPrev = 0, tpPrev = 0;
  pares.forEach(([s, r])=>{
    if(r===1) tp++;
    else fp++;
    auc += (fp-fpPrev)*(tp+tpPrev)/2;
    fpPrev = fp;
    tpPrev = tp;
  });
  return auc/(tp*fp);
}
// funções de pesos:
function iniPesosXavier(l, cols) {
  if(l <= 0 || cols <= 0) console.error("[iniPesosXavier]: valor negativo passado como parâmetro");
  const limite = Math.sqrt(6/(l+cols));
  return Array(l).fill().map(()=>Array(cols).fill().map(()=>(Math.random()*2-1)*limite));
}
function iniPesosHe(l, cols) {
  if(l <= 0 || cols <= 0) console.error("[iniPesosHe]: valor negativo passado como parâmetro");
  const limite = Math.sqrt(2/l);
  return Array(l).fill().map(()=>Array(cols).fill().map(()=>(Math.random()*2-1)*limite));
}
function iniPesosUniforme(l, cols) {
  const limiteInferior = -Math.sqrt(3/(l+cols));
  const limiteSuperior = Math.sqrt(3/(l+cols));
  return Array(l).fill().map(()=>Array(cols).fill().map(()=>(Math.random()*(limiteSuperior-limiteInferior)+limiteInferior)));
}
// atualização de pesos
function attPesos(pesos,grad,taxa,lambda=1e-3){return pesos.map((l,i)=>l.map((p,j)=>p-taxa*grad[i][j]-lambda*p));}
function attPesosMomentum(pesos,grad,taxa,momento,velocidade){return pesos.map((l,i)=>l.map((p,j)=>{velocidade[i][j]=momento*velocidade[i][j]+grad[i][j];return p-taxa*velocidade[i][j];}));}
function attPesosAdam(pesos, grad, m, v, taxa, beta1=0.9, beta2=0.999, eps=1e-8, iteracao, lambda=0) {
  const fator1 = 1-Math.pow(beta1, iteracao);
  const fator2 = 1-Math.pow(beta2, iteracao);
  return pesos.map((l, i)=>l.map((p, j)=>{
    const g = grad[i][j]+lambda*pesos[i][j];
    m[i][j] = beta1*m[i][j]+(1-beta1)*g;
    v[i][j] = beta2*v[i][j]+(1-beta2)*g*g;
    const mChapeu = m[i][j]/fator1;
    const vChapeu = v[i][j]/fator2;
    return p-taxa*mChapeu/(Math.sqrt(vChapeu)+eps);}));
}
function attPesosAdam1D(p, grad, m, v, taxa, beta1, beta2, eps, t) {
  for(let i=0; i<p.length; i++) {
    m[i] = beta1*m[i]+(1-beta1)*grad[i];
    v[i] = beta2*v[i]+(1-beta2)*grad[i]*grad[i];
    const mChapeu = m[i]/(1-Math.pow(beta1, t));
    const vChapeu= v[i]/(1-Math.pow(beta2, t));
    p[i] = p[i]-taxa*mChapeu/(Math.sqrt(vChapeu)+eps);
  }
  return p;
}
function attPesosRMSprop(pesos, grad, cache, taxa=0.001, decadencia=0.9, eps=1e-8) {
  cache = cache.map((l,i)=>l.map((val,j)=>decadencia*val+(1-decadencia)*grad[i][j]**2));
  return pesos.map((l,i)=>l.map((p,j)=>p-taxa*grad[i][j]/(Math.sqrt(cache[i][j])+eps)));
}
function attPesosAdagrad(pesos, grad, cache, taxa=0.01, eps=1e-8) {
  cache = cache.map((l,i)=>l.map((val,j)=>val+grad[i][j]**2));
  return pesos.map((l,i)=>l.map((p,j)=>p-taxa*grad[i][j]/(Math.sqrt(cache[i][j])+eps)));
}
// matrizes 3D:
function tensor3D(p,l,c,escala){return Array(p).fill().map(()=>matriz(l,c,escala));}
function zeros3D(p,l,c){return Array.from({length:p},()=>matrizZeros(l,c));}
function mapear3D(t,fn){return t.map(m=>m.map(l=>l.map(fn)));}
function somar3D(a,b){return a.map((m,i)=>somarMatriz(m,b[i]));}
function mult3DporEscalar(t,escalar){return t.map(m>multMatriz(m,escalar));}
// matrizes 2D:
function matriz(l,c,escala=0.1){return Array(l).fill().map(()=>vetor(c, escala));}
function exterior(a,b){return a.map(ai=>b.map(bj=>ai*bj));}
function clonarMatriz(m){return new Float32Array(m.buffer,m.byteOffset,m.length);}
function somarMatriz(a, b) {
  if(a.length != b.length) console.error("[somarMatrizes]: tamanho incompatível");
  return a.map((l, i)=>l.map((v, j)=>v+b[i][j]));
}
function subMatriz(a, b) {
  if(a.length != b.length) console.error("[subMatriz]: tamanho incompatível");
  return a.map((l, i)=>l.map((v, j)=>v-b[i][j]));
}
function aplicarMatriz(m, v) {
  if(!Array.isArray(v)) console.error("[aplicarMatriz]: o segundo parâmetro não é um vetor");
  return m.map(l=>escalarDot(l, v));
}
function transpor(m){return[...m[0]].map((_,j)=>m.map(l=>l[j]));}
function matrizZeros(l,c){return Array.from({length:l}, ()=>zeros(c))}
function identidade(n) {return Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>i===j?1:0));}
function matrizVetor(m, v) {
  if(!Array.isArray(v))console.error("[matrizVetor]: o segundo parâmetro não é um vetor");
  return m.map(l=>{return l.reduce((soma,v,j)=>soma+v*v[j],0);});
}
function multVetorMatriz(v, c) {
  if(!Array.isArray(v))console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return v.map((x,i)=>x*c[i]);
}
function multMatrizVetor(m, v) {
  if(!Array.isArray(v)) console.error("[multVetorMatriz]: o primeiro parâmetro não é um vetor");
  return m.map(l=>l.reduce((soma, val, j)=>soma+val*v[j], 0));
}
function clipMatriz(m,limite){return m.map(l=>l.map(v=>Math.max(-limite,Math.min(v,limite))));}
// vetores
function vetor(n,escala=0.1){const v=new Float32Array(n);for(let i=0;i<n;i++){v[i]=(Math.random()*2-1)*escala;}return v;}
function somarVetores(a, b) {
  if(!a || !b || a.length !== b.length) throw new Error(`[somarVetores]: tamanho incompatível a=${a.length}, b=${b.length}`);
  return a.map((x, i)=>x+b[i]);
}
function subVetores(a, b) {
  if(a.length != a.length) console.error("[subVetores]: tamanho incompatível");
  return a.map((x, i)=>x-b[i]);
}
function multVetores(a, b) {
  if(a.length != a.length) console.error("[multVetores]: tamanho incompatível");
  return a.map((x, i)=>x*b[i]);
}
function escalarDot(v, w) {
  const eVet = Array.isArray(v) || v instanceof Float32Array;
  const eW = Array.isArray(w) || w instanceof Float32Array;
  if(!eVet || !eW) throw new Error("[escalarDot] Um dos vetores não é um array ou TypedArray");
  if(v.length != w.length) throw new Error(`[escalarDot]: tamanhos diferentes v=${v.length}, w=${w.length}`);
  let soma = 0;
  for(let i=0; i<v.length; i++) {
    const prod = v[i]*w[i];
    if(!isFinite(prod))throw new Error(`[escalarDot]: produto inválido em [${i}]:${v[i]}*${w[i]}=${prod}`);
    soma += prod;
  }
  return soma;
}
function multElementos(a, b) {
  if(a.length != b.length) console.error("[multElementos]: tamanho incompatível");
  return a.map((e, i)=>e*b[i]);
}
function normalizar(v) {
  if(!Array.isArray(v)) console.error("[normalizar]: o parâmetro não é um vetor");
  const mag = Math.sqrt(v.reduce((s, x)=>s+x*x, 0));
  return mag==0 ? v : v.map(x=>x/mag);
}
function zeros(n){return Array(n).fill(0);}
function uns(n){return Array(n).fill(1);}
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
function clipVetor(v, limite) {
  if(!Array.isArray(v)) console.error("[clipVetor]: o parâmetro não é um vetor");
  return v.map(x=>Math.max(-limite, Math.min(x, limite)));
}
// debug:
function heapUsado(){console.log(`Heap: ${process.memoryUsage().heapUsed>>20}MB`);}
function arraysIguais(a,b){return JSON.stringify(a)===JSON.stringify(b);}
function limparGrad(grad){return grad.map(l=>l.map(v=>isFinite(v)?v:0));}
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
  const v = Array(tam).fill(0);
  if(i<0 || i >= tam)throw new Error(`[oneHot]: índice fora do vocabulário: ${i}`);
  v[i] = 1;
  return v;
}
function oneHotLote(idcs,tam){return idcs.map(i=>oneHot(i,tam));}
// probabilidades:
function amostrarTopP(probs, p=0.9) {
  const probsComIndice = probs.map((prob, indice)=>({indice, prob}));
  probsComIndice.sort((a, b)=>b.prob-a.prob);
  let somaCumulativa = 0;
  let i = 0;
  for(; i<probsComIndice.length; i++) {
    somaCumulativa += probsComIndice[i].prob;
    if(somaCumulativa >= p) break;
  }
  const nucleus = probsComIndice.slice(0, i+1);
  const somaNucleus = nucleus.reduce((s, item)=>s+item.prob, 0);
  const probsNormalizadas = nucleus.map(item=>item.prob/somaNucleus);
  const amostra = Math.random();
  let soma = 0;
  for(let j=0; j<nucleus.length; j++) {
    soma += probsNormalizadas[j];
    if(amostra <= soma) return nucleus[j].indice;
  }
  return nucleus[0].indice;
}
class CamadaAtencao {
  constructor(dimModelo, numCabecas, taxaDropout=0.1) {
    this.dimModelo = dimModelo;
    this.numCabecas = numCabecas;
    this.dimCabeca = Math.floor(dimModelo/numCabecas);
    this.taxaDrop = taxaDropout;
    this.pq = iniPesosHe(dimModelo, dimModelo);
    this.pk = iniPesosHe(dimModelo, dimModelo);
    this.pv = iniPesosHe(dimModelo, dimModelo);
    this.ps = iniPesosHe(dimModelo, dimModelo);
    this.bq = zeros(dimModelo);
    this.bk = zeros(dimModelo);
    this.bv = zeros(dimModelo);
    this.bs = zeros(dimModelo);
    // buffers adam:
    this.mq = matrizZeros(dimModelo, dimModelo);
    this.vq = matrizZeros(dimModelo, dimModelo);
    this.mk = matrizZeros(dimModelo, dimModelo);
    this.vk = matrizZeros(dimModelo, dimModelo);
    this.mv = matrizZeros(dimModelo, dimModelo);
    this.vv = matrizZeros(dimModelo, dimModelo);
    this.ms = matrizZeros(dimModelo, dimModelo);
    this.vs = matrizZeros(dimModelo, dimModelo);
    this.mBq = zeros(dimModelo);
    this.vBq = zeros(dimModelo);
    this.mBk = zeros(dimModelo);
    this.vBk = zeros(dimModelo);
    this.mBv = zeros(dimModelo);
    this.vBv = zeros(dimModelo);
    this.mBs = zeros(dimModelo);
    this.vBs = zeros(dimModelo);
    this.iteracao = 1;
    this.cache = {};
    if(this.dimModelo%this.numCabecas != 0) throw new Error(`dimModelo (${this.dimModelo}) não divisível por numCabecas (${this.numCabecas})`);
  }
  propagar(x, mascara=null, treino=true) {
    x.forEach((linha, i) => {if(linha.length !== this.dimModelo) throw new Error(`[CamadaAtencao] x[${i}] tem tamanho inválido: ${linha.length}, esperado: ${this.dimModelo}`);});
    eNaN(x, "x", "CamadaAtencao");
    eNaN(this.pq, "pq", "CamadaAtencao");
    this.bq.forEach((v,i)=>{if(!isFinite(v)) throw new Error(`bq contém NaN ou infinito em ${i}: ${v}`);});
    eNaN(this.pk, "pk", "CamadaAtencao");
    this.bk.forEach((v,i)=>{if(!isFinite(v)) throw new Error(`bk contém NaN ou infinito em ${i}: ${v}`);});
    eNaN(this.pv, "pv", "CamadaAtencao");
    this.bv.forEach((v,i)=>{if(!isFinite(v)) throw new Error(`bv contém NaN ou infinito em ${i}: ${v}`);});
    eNaN(this.ps, "ps", "CamadaAtencao");
    this.bs.forEach((v, i) => {if(!isFinite(v)) throw new Error(`bs contém NaN ou infinito em ${i}: ${v}`);});
    const seqTam = x.length;
    const q = x.map(seq=>somarVetores(aplicarMatriz(this.pq, seq), this.bq));
    const k = x.map(seq => somarVetores(aplicarMatriz(this.pk, seq), this.bk));
    const v = x.map(seq => somarVetores(aplicarMatriz(this.pv, seq), this.bv));
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
      pCabecas[o] = ponto.map(linha=>softmax(linha));
    }
    const atSCabecas = pCabecas.map((pp, o)=>{
      const vo = vCab[o];
      return vo.map((_, i) => {
        let ctx = zeros(this.dimCabeca);
        for(let j=0; j<seqTam; j++) {
          const mult = pp[i][j];
          const contrib = vo[j].map(vij=>vij*mult);
          ctx = somarVetores(ctx, contrib);
        }
        return ctx;
      });
    });
    const juntar = this.juntarCabecas(atSCabecas);
    eNaN(juntar, "juntar", "[CamadaAtencao]");
    let o = juntar.map(seq=>somarVetores(aplicarMatriz(this.ps, seq), this.bs));
    this.cache = { x, qCab, kCab, vCab, pCabecas, juntar };
    eNaN(o, "o", "CamadaAtencao");
    if(treino) o = dropout(o, this.taxaDrop);
    return o;
  }
  retropropagar(dO, taxa, lambda=0.001) {
    eNaN(dO, "dO", "CamadaAtencao");
    const { x, qCab, kCab, vCab, pCabecas, juntar } = this.cache;
    const seqTam = x.length;
    // dConcat = dO · po^T
    const dPoT = transpor(this.ps);
    eNaN(dPoT, "dPoT", "CamadaAtencao");
    if(!Array.isArray(dO) || dO.some(l => !Array.isArray(l))) throw new Error("[CamadaAtencao] dO contém item inválido ou undefined");
    
    let dConcat = dO.map(do_i => aplicarMatriz(dPoT, do_i));
    eNaN(dConcat, "dConcat", "CamadaAtencao");
    dConcat = dConcat.map(vetor => clipVetor(vetor, 1.0));
    // gradientes de ps e bs
    const dOmatT = transpor(dO);  
    // shape: [dimModelo×dimModelo]×[dimModelo×seqTam]→[dimModelo×seqTam]
    let dPs = matrizZeros(this.dimModelo, this.dimModelo);
    const juntarT = transpor(juntar); // [dimModelo, seqTam]
    for(let i=0; i<this.dimModelo; i++) {
      for(let j=0; j<this.dimModelo; j++) {
        for(let k=0; k<seqTam; k++) {
          dPs[i][j] += juntarT[i][k]*dO[k][j];
        }
      }
    }
    // gradiente de bias = soma de cada dimensão sobre todas as posições na sequência
    let dBo = dO.reduce((acc, vec)=>somarVetores(acc, vec), zeros(this.dimModelo));
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
    // prepara espaço pra dko e dVo
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
      const dPoLeve = po.map((_, i)=>{
        const linha = [];
        for(let j=0; j<seqTam; j++) {
          let soma = 0;
          for(let k=0; k<this.dimCabeca; k++) soma += dAo[i][k]*vo[j][k];
          linha.push(soma);
        }
        return linha;
      });
      // calcula gradiente da softmax
      const dPontos = po.map((linha, i)=>derivadaSoftmax(linha, dPoLeve[i]));
      eNaN(dPontos, "dPontos", "CamadaAtencao");
      // atualizar dX, dkoTodo e dVoTodo
      const invSqrt = 1/Math.sqrt(this.dimCabeca);
      if(isNaN(invSqrt)) console.error("[CamadaAtencao]: invSqrt é NaN");
      for(let i=0; i<qo.length; i++) {
        for(let j=0; j<ko.length; j++) {
          const grad = dPontos[i][j]*invSqrt;
          // dX
          const inicio = o*this.dimCabeca;
          for(let k=0; k<this.dimCabeca; k++) {
            dX[i][inicio+k] += ko[j][k]*grad;
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
          const fator = po[i][j];
          if(fator != 0) {
            const contrib = dAo[i].map(val=>val*fator);
            dVoTodo[o][j] = somarVetores(dVoTodo[o][j], contrib);
          }
        }
      }
      // prepara matrizes pra gradientes de pesos
      const xMat = transpor(x);
      const qCabecaPlana = transpor(qo);
      const kCabecaPlana = transpor(dkoTodo[o]);
      const vCabecaPlana = transpor(dVoTodo[o]);
      const base = o*this.dimCabeca;
      // valida entradas
      for(let idc=0; idc<qCabecaPlana.length; idc++) if(qCabecaPlana[idc].some(v=>!isFinite(v))) throw new Error(`qCabecaPlana corrompido`);
      for(let idc=0; idc<xMat.length; idc++) if(xMat[idc].some(v=>!isFinite(v))) throw new Error(`xMat corrompido`);
      // gradientes
      for(let i=0; i<this.dimCabeca; i++) {
        const lpq = multMatrizVetor(xMat, qCabecaPlana[i]);
        const lpk = multMatrizVetor(xMat, kCabecaPlana[i]);
        const lpv = multMatrizVetor(xMat, vCabecaPlana[i]);
        dPq[base+i] = somarVetores(dPq[base+i], lpq);
        dPk[base+i] = somarVetores(dPk[base+i], lpk);
        dPv[base+i] = somarVetores(dPv[base+i], lpv);
        dBq[base+i] += qCabecaPlana[i].reduce((s, v)=>s+v, 0);
        dBk[base+i] += kCabecaPlana[i].reduce((s, v)=>s+v, 0);
        dBv[base+i] += vCabecaPlana[i].reduce((s, v)=>s+v, 0);
      }
    }
    // recorte
    dPs = clipMatriz(dPs, 1);
    dBo = clipVetor(dBo, 1);
    dPq = clipMatriz(dPq, 1);
    dBq = clipVetor(dBq, 1);
    dPk = clipMatriz(dPk, 1);
    dBk = clipVetor(dBk, 1);
    dPv = clipMatriz(dPv, 1);
    dBv = clipVetor(dBv, 1);
    limparGrad(dPs);
    limparGrad(dPq);
    limparGrad(dPk);
    limparGrad(dPv); 
    this.ps = attPesosAdam(this.ps, dPs, this.ms, this.vs, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.pq = attPesosAdam(this.pq, dPq, this.mq, this.vq, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.pk = attPesosAdam(this.pk, dPk, this.mk, this.vk, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.pv = attPesosAdam(this.pv, dPv, this.mv, this.vv, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.bs = attPesosAdam1D(this.bs, dBo, this.mBs, this.vBs, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.bq = attPesosAdam1D(this.bq, dBq, this.mBq, this.vBq, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.bk = attPesosAdam1D(this.bk, dBk, this.mBk, this.vBk, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.bv = attPesosAdam1D(this.bv, dBv, this.mBv, this.vBv, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.iteracao++;
    eNaN(dX, "dX", "CamadaAtencao");
    return dX;
  }
  dividirCabecas(x) {
    const cabecas = [];
    for(let o=0; o<this.numCabecas; o++) cabecas[o] = x.map(seq=>seq.slice(o*this.dimCabeca, (o+1)*this.dimCabeca));
    return cabecas;
  }
  juntarCabecas(cabecas) {
    return cabecas[0].map((_, i)=>cabecas.reduce((seq, o)=>seq.concat(o[i]), []));
  }
}
class CamadaFFN {
  constructor(dimModelo, dimFFN, taxaDropout=0.1) {
    this.p1 = iniPesosHe(dimFFN, dimModelo);
    this.b1 = zeros(dimFFN);
    this.p2 = iniPesosHe(dimModelo, dimFFN);
    this.b2 = zeros(dimModelo);
    // buffers adam:
    this.m1 = matrizZeros(dimFFN, dimModelo);
    this.v1 = matrizZeros(dimFFN, dimModelo);
    this.m2 = matrizZeros(dimModelo, dimFFN);
    this.v2 = matrizZeros(dimModelo, dimFFN);
    this.mB1 = zeros(dimFFN);
    this.vB1 = zeros(dimFFN);
    this.mB2 = zeros(dimModelo);
    this.vB2 = zeros(dimModelo);
    this.iteracao = 1;
    this.cache = {};
    this.taxaDrop = taxaDropout;
  }
  propagar(x, treino=true) {
    const z1 = x.map(seq=>aplicarMatriz(this.p1, seq));
    const lin1 = z1.map((seq, i)=>somarVetores(seq, this.b1));
    const ativ1 = lin1.map(seq=>seq.map(ReLU));
    
    let camada1 = ativ1;
    if(treino) camada1 = dropout(ativ1, this.taxaDrop);
    
    const z2 = camada1.map(seq=>aplicarMatriz(this.p2, seq));
    const saida = z2.map((seq, i)=>somarVetores(seq, this.b2));
    
    this.cache = { x, ativ1, camada1 };
    return saida;
  }
  retropropagar(dY, taxa, lambda=0.001) {
    const { x, ativ1 } = this.cache; // usa ativ1(pré-dropout)
    const lote = x.length;
    const dimFFN = this.p1.length;
    const dimEntrada = this.p1[0].length;
    const dimSaida = this.p2.length;
    const dP2 = matrizZeros(dimSaida, dimFFN);
    const dB2 = zeros(dimSaida);
    const dP1 = matrizZeros(dimFFN, dimEntrada);
    const dB1 = zeros(dimFFN);
    const dX  = Array(lote).fill(0).map(_=>zeros(dimEntrada));
    // RETROPROPAGAÇÃO:
    for(let n=0; n<lote; n++) {
      const seqX  = x[n];
      const seqo1 = ativ1[n]; // usar ativação PRÉ-dropout
      const seqDY = dY[n];
      // camada de saída
      for(let j=0; j < dimSaida; j++) {
        const err = seqDY[j];
        dB2[j] += err;
        for(let k=0; k<dimFFN; k++) dP2[j][k] += seqo1[k]*err;
      }
      // retropropagação na camada de saída
      const p2T = transpor(this.p2);
      const dO = multMatrizVetor(p2T, seqDY);
      // gradientes da primeira camada(usa ativ1)
      for(let k=0; k<dimFFN; k++) {
        const deriv = seqo1[k]>0 ? 1 : 0; // derivada ReLU
        dO[k] = dO[k]*deriv;
      }
      for(let j=0; j<dimFFN; j++) {
        const err1 = dO[j];
        dB1[j] += err1;
        for(let k=0; k<dimEntrada; k++) dP1[j][k] += seqX[k]*err1;
      }
      const p1T = transpor(this.p1);
      dX[n] = aplicarMatriz(p1T, dO);
    }
    // ATUALIZAÇÃO DOS PESOS:
    this.p1 = attPesosAdam(this.p1, dP1, this.m1, this.v1, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.p2 = attPesosAdam(this.p2, dP2, this.m2, this.v2, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.b1 = attPesosAdam1D(this.b1, dB1, this.mB1, this.vB1, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.b2 = attPesosAdam1D(this.b2, dB2, this.mB2, this.vB2, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.iteracao++;
    return dX;
  }
}
class CamadaNormalizacao {
  constructor(dimModelo, epsilon=1e-5, taxaDrop=0.1) {
    this.gamma = uns(dimModelo);
    this.beta = zeros(dimModelo);
    this.mG = zeros(dimModelo);
    this.vG = zeros(dimModelo);
    this.mB = zeros(dimModelo);
    this.vB = zeros(dimModelo);
    
    this.iteracao = 1;
    this.epsilon = epsilon;
    this.cache = {};
    this.taxaDrop = taxaDrop;
  }
  propagar(x) {
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
    eNaN(saida, "saida", "CamadaNormalizacao");
    return saida;
  }
  retropropagar(dY, taxa, lambda=0.001) {
    eNaN(dY, "dY", "CamadaNormalizacao");
    const { x, medias, vars } = this.cache;
    const seqTam = x.length;
    const dim = this.gamma.length;
    const dGamma = zeros(dim);
    const dBeta = zeros(dim);
    const dX = Array(seqTam).fill(0).map(()=>zeros(dim));
    for(let i=0; i<seqTam; i++) {
      const seq = x[i];
      const dYi = dY[i];
      const media = medias[i];
      const vari = vars[i];
      const std = Math.sqrt(vari+this.epsilon);
      const invStd = 1/std;
      
      const xCentrado = seq.map(val=>val-media);
      for(let j=0; j<dim; j++) {
        dGamma[j] += dYi[j]*xCentrado[j]*invStd;
        dBeta[j] += dYi[j];
      }
      const N = dim;
      const termoComum = dYi.map((dy, j)=>dy*this.gamma[j]*invStd);
      const somaTermo1 = termoComum.reduce((s, val)=>s+val, 0);
      const somaTermo2 = xCentrado.reduce((s, xc, j)=>s+xc*termoComum[j], 0);
      
      for(let j=0; j<dim; j++) dX[i][j] = termoComum[j]-(1/N)*somaTermo1-(1/N)*xCentrado[j]*somaTermo2*invStd*invStd;
    }
    this.gamma = attPesosAdam1D(this.gamma, dGamma, this.mG, this.vG, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.beta = attPesosAdam1D(this.beta, dBeta, this.mB, this.vB, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.iteracao++;
    eNaN(dX, "dX", "CamadaNormalizacao");
    return dX;
  }
}
class BlocoTransformer {
  constructor(dimModelo, numCabecas, dimFFN, taxaDrop) {
    this.atencao = new CamadaAtencao(dimModelo, numCabecas, taxaDrop);
    this.ffn = new CamadaFFN(dimModelo, dimFFN, taxaDrop);
    this.norm1 = new CamadaNormalizacao(dimModelo, 1e-6, taxaDrop);
    this.norm2 = new CamadaNormalizacao(dimModelo, 1e-6, taxaDrop);
  }
  propagar(x, mascara=null, treino=true) {
    eNaN(x, "x", "BlocoTransformer");
    const a = this.atencao.propagar(x, mascara, treino);
    const r1 = x.map((v, i)=>somarVetores(v, a[i]));
    const n1 = this.norm1.propagar(r1, treino);
    const f = this.ffn.propagar(n1, treino);
    const r2 = r1.map((v, i)=>somarVetores(v, f[i]));
    const r3 = this.norm2.propagar(r2, treino);
    this.cache = { x, a, r1, n1, f, r2, r3, mascara };
    eNaN(r3, "r3", "BlocoTransformer");
    return r3;
  }
  retropropagar(dY, taxa) {
    eNaN(dY, "dY", "BlocoTransformer");
    const { x, a, r1, n1, f, r2 } = this.cache;
    const dNorm2 = this.norm2.retropropagar(dY, taxa);
    const dF = dNorm2;
    const dN1_daF = this.ffn.retropropagar(dF, taxa);
    const dR1_from_residual = dNorm2;
    const dN1 = dR1_from_residual.map((v,i)=>somarVetores(v, dN1_daF[i]));
    const dR1 = this.norm1.retropropagar(dN1, taxa);
    const dX_from_residual = dR1;
    const dA = dR1;
    const dX_daAtencao = this.atencao.retropropagar(dA, taxa);
    const res = dX_from_residual.map((v,i)=>somarVetores(v, dX_daAtencao[i]));
    eNaN(res, "res", "BlocoTransformer");
    return res;
  }
}
class CamadaEmbedding {
  constructor(vocabTam, dimEmbedding, lambda=0.001) {
    this.vocabTam = vocabTam;
    this.dimEmbedding = dimEmbedding;
    this.lambda = lambda;
    this.embeddings = matriz(vocabTam, dimEmbedding);
    this.m = matrizZeros(vocabTam, dimEmbedding);
    this.v = matrizZeros(vocabTam, dimEmbedding);
    this.iteracao = 1;
  }
  propagar(indices){return indices.map(idc=>[...this.embeddings[idc]]);}
  retropropagar(gradSaida, indices, taxa) {
    const gradEmbeddings = matrizZeros(this.vocabTam, this.dimEmbedding);
    indices.forEach((idc, i)=>{
      for(let j=0; j<this.dimEmbedding; j++) gradEmbeddings[idc][j] += gradSaida[i][j];
    });
    this.embeddings = attPesosAdam(this.embeddings, gradEmbeddings, this.m, this.v, taxa, 0.9, 0.999, 1e-8, this.iteracao, this.lambda);
    this.iteracao++;
  }
}
class CamadaSaida {
  constructor(dimModelo, vocabTam) {
    this.p = matriz(vocabTam, dimModelo, 0.1);
    this.b = zeros(vocabTam);
    this.m = matrizZeros(vocabTam, dimModelo);
    this.v = matrizZeros(vocabTam, dimModelo);
    this.mb = zeros(vocabTam);
    this.vb = zeros(vocabTam);
    this.iteracao = 1;
  }
  propagar(x){return x.map(seq=>somarVetores(aplicarMatriz(this.p, seq), this.b));}
  retropropagar(gradSaida, entrada, taxa, lambda=0.001) {
    const gradP = matrizZeros(this.p.length,this.p[0].length);
    const gradB = zeros(this.b.length);
    const dEntrada = [];
    for(let i=0; i<entrada.length; i++) {
      const en = entrada[i]; // [dimModelo]
      const grad = gradSaida[i]; // [vocabTam]
      const dx = aplicarMatriz(transpor(this.p), grad); // gradiente da entrada
      dEntrada.push(dx);
      for(let j=0; j<grad.length; j++) {
        gradB[j] += grad[j];
        for(let k=0;k<en.length;k++)gradP[j][k]+=grad[j]*en[k];
      }
    }
    this.p = attPesosAdam(this.p, gradP, this.m, this.v, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.b = attPesosAdam1D(this.b, gradB, this.mb, this.vb, taxa, 0.9, 0.999, 1e-8, this.iteracao, lambda);
    this.iteracao++;
    return dEntrada;
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
      const v = [];
      for(let i=0; i<this.dimModelo; i++) {
        const expoente = i/this.dimModelo;
        const valor = pos/Math.pow(10000, expoente);
        v.push(i%2==0 ? Math.sin(valor) : Math.cos(valor));
      }
      codificacao.push(v);
    }
    return codificacao;
  }
  aplicar(x) {
    return x.map((seq, pos)=>somarVetores(seq, this.codificacao[pos]));
  }
}
class TokenizadorBPE {
  constructor(merges=[]) {
    this.vocab = {};
    this.bpeRanks = {};
    this.cache = new Map();
    // inicializa os merges
    for(let i=0; i<merges.length; i++) {
      const chave = merges[i].join(' ');
      this.bpeRanks[chave] = i;
    }
    // tokens especiais
    this.tokenPraId = new Map([['<ALMO>', 0], ['<DES>', 1], ['<FIM>', 2]]);
    this.idPraToken = new Map([[0, '<ALMO>'], [1, '<DES>'], [2, '<FIM>']]);
    this.proximoId = 3;
  }
  construirVocabulario(textos) {
    const todosTokens = new Set();
    const todosCaracteres = new Set();
    // add tokens especiais
    todosTokens.add('<ALMO>');
    todosTokens.add('<DES>');
    todosTokens.add('<FIM>');
    // coleta todos os caracteres únicos primeiro
    for(const texto of textos)for(const carac of texto)if(carac.trim() !== '')todosCaracteres.add(carac);
    for(const carac of todosCaracteres)todosTokens.add(carac);
    // processa todos os textos pra tokens BPE
    for(const texto of textos) {
      const tokens = this.encode(texto);
      tokens.forEach(token=>todosTokens.add(token));
    }
    // mapeia tokens para IDs
    let id = 3;
    for(const token of todosTokens) {
      if(!this.tokenPraId.has(token)) {
        this.tokenPraId.set(token, id);
        this.idPraToken.set(id, token);
        id++;
      }
    }
    this.proximoId = id;
    console.log(`Vocabulário construído: ${this.proximoId} tokens`);
  }
  codificar(texto) {
    const tokensBPE = this.encode(texto);
    return tokensBPE.flatMap(token => {
      let id = this.tokenPraId.get(token);
      if(id !== undefined) return [id];
      // fallback por caractere
      const saida = [];
      for(const c of token) {
        const cid = this.tokenPraId.get(c);
        if(cid !== undefined) saida.push(cid);
        else saida.push(1); // <DES>
      }
      return saida;
    });
  }
  decodificar(ids) {
    const tokens = ids.map(id=>{
      if(id===2) return ''; // <FIM>
      const token = this.idPraToken.get(id);
      return token !== undefined ? token : '<DES>';
    }).filter(token => token !== ''); // remove tokens vazios
    return this.decode(tokens);
  }
  get vocabTam(){return this.proximoId;}
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
        if(rank !== undefined && rank<minRank) {
          minRank = rank;
          melhorPar = par;
        }
      }
      if(melhorPar===null) break;
      const [primeiro, segundo] = melhorPar.split(' ');
      const novaPalavra = [];
      let i = 0;
      while(i<palavra.length) {
        let j = palavra.indexOf(primeiro, i);
        if(j===-1) {
          novaPalavra.push(...palavra.slice(i));
          break;
        }
        novaPalavra.push(...palavra.slice(i, j));
        if(j<palavra.length-1 && palavra[j+1]===segundo) {
          novaPalavra.push(primeiro+segundo);
          i = j+2;
        } else {
          novaPalavra.push(palavra[j]);
          i = j+1;
        }
      }
      palavra = novaPalavra;
      pares = this.obterPares(palavra);
    }
    this.cache.set(token, palavra);
    return palavra;
  }
  encode(texto) {
    // divide em palavras mantendo espaços
    const palavras = texto.split(/(\s+)/);
    const tokens = [];
    for(const palavra of palavras) {
      if(palavra.trim()=='') {
        // é um espaço - tratar como token especial
        if(palavra==' ') tokens.push('Ġ');
        else tokens.push('Ġ');
      } else {
        // é uma palavra real - tentar BPE primeiro
        const bpeTokens = this.bpe(palavra);
        // se BPE falhou(retornou a palavra original)
        if(bpeTokens.length===1 && bpeTokens[0]===palavra&&!this.tokenPraId.has(palavra))for(const carac of palavra)tokens.push(carac);
        else tokens.push(...bpeTokens);
      }
    }
    return tokens;
  }
  decode(tokens) {
    let texto = '';
    for(const token of tokens) {
      if(token=='Ġ') texto += ' ';
      else if(token.startsWith('Ġ'))texto += ' '+token.slice(1);
      else texto += token;
    }
    return texto;
  }
}
class GSA {
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
    this.embedding = new CamadaEmbedding(config.vocabTam, config.dimModelo);
    this.codificadorPos = new CodificadorPosicional(config.dimModelo, config.seqMaxima);
    this.camadas = [];
    for(let i=0; i<config.numCamadas; i++) this.camadas.push(new BlocoTransformer(config.dimModelo, config.numCabecas, config.dimFFN, this.taxaDropout));
    this.normFinal = new CamadaNormalizacao(config.dimModelo, 1e-6);
    this.cabecaSaida = new CamadaSaida(config.dimModelo, config.vocabTam);
    this.cache = {};
  }
  propagar(tokens, treino=true) {
    const seqTam = tokens.length
    let xNoPos = this.embedding.propagar(tokens);
    if(treino) xNoPos = dropout(xNoPos, this.taxaDropoutEmbed);
    let xPos = this.codificadorPos.aplicar(xNoPos);
    let x = xPos;
    const saidas = [];
    for(const camada of this.camadas) {
      x = camada.propagar(x, this.gerarMascaraCausal(seqTam), treino);
      saidas.push(clonarMatriz(x));
    }
    const normSaida = this.normFinal.propagar(x, treino);
    const logits = this.cabecaSaida.propagar(normSaida);
    eNaN(x, "x", "GSATransformer");
    eNaN(logits, "logits", "GSATransformer");
    eNaN(normSaida, "normSaida", "GSATransformer");
    this.cache = { tokens, xNoPos, xPos, saidas, normSaida, logits };
    return softmaxLote(logits);
  }
  retropropagar(dLogits) {
    eNaN(dLogits, "dLogits", "GSATransformer");
    heapUsado();
    const { tokens, xPos, saidas, normSaida } = this.cache;
    const seqTam = tokens.length;
    const dCabecaEntrada = this.cabecaSaida.retropropagar(dLogits, normSaida, this.taxa);
    let dX = dCabecaEntrada;
    dX = dCabecaEntrada.map(v=>clipVetor(v, 1.0));
    dX = this.normFinal.retropropagar(dX, this.taxa);
    for(let i=this.camadas.length-1; i >= 0; i--) {
      const ent = i==0 ? xPos : saidas[i-1];
      dX = this.camadas[i].retropropagar(dX, this.taxa);
    }
    eNaN(dX, "dX", "GSATransformer");
    this.embedding.retropropagar(dX, tokens, this.taxa);
  }
  gerarMascaraCausal(n) {
    const m = [];
    for(let i=0; i<n; i++) {
      m[i] = [];
      for(let j=0; j<n; j++) m[i][j] = j <= i ? 1 : 0;
    }
    return m;
  }
  buscaFeixe(tokensIniciais, maxTokens=50, temperatura=1.0, raioTam=5, repeticaoPenal=1.2) {
    const idFIM = gsa.tokenizador.tokenPraId.get('<FIM>') || 2;
    let feixe = [{
        sequencia: tokensIniciais,
        pontuacao: 0.0, // log-probabilidade acumulada
        finalizada: false
    }];
    for(let passo=0; passo<maxTokens; passo++) {
      let candidatos = [];
      for(const hipotese of feixe) {
        if(hipotese.finalizada) {
          candidatos.push(hipotese);
          continue;
        }
        // obtem logits para sequência atual
        const logits = this.propagar(hipotese.sequencia, false);
        const ultimosLogits = logits[logits.length-1];
        // aplica penalidade por repetição
        const tokensUnicos = [...new Set(hipotese.sequencia)];
        tokensUnicos.forEach(token => {
          ultimosLogits[token] /= repeticaoPenal;
        });
        // calcula probabilidades com temperatura
        const probs = softmax(ultimosLogits, temperatura);
        const logProbs = probs.map(p => Math.log(p+1e-10));
        // top-k tokens para considerar (k = raioTam*2)
        const topK = Math.min(raioTam*2, probs.length);
        const probsComIndice = probs.map((prob, idc)=>({token: idc, prob, logProb: logProbs[idc]}));
        probsComIndice.sort((a, b) => b.prob-a.prob);
        const topProbs = probsComIndice.slice(0, topK);
        for(const {token, logProb} of topProbs) {
          const novaSequencia = [...hipotese.sequencia, token];
          const novaPontuacao = hipotese.pontuacao+logProb;
          const finalizada = (token===idFIM);
          candidatos.push({
            sequencia: novaSequencia,
            pontuacao: novaPontuacao,
            finalizada: finalizada
          });
        }
      }
      // ordena candidatos por pontuação
      candidatos.sort((a, b)=>b.pontuacao-a.pontuacao);
      feixe = candidatos.slice(0, raioTam);
      // para se todas as hipóteses finalizaram
      if(feixe.every(h=>h.finalizada)) break;
    }
    // seleciona hipótese com maior pontuação
    if(feixe.length>0) {
      feixe.sort((a, b) => b.pontuacao-a.pontuacao);
      return feixe[0].sequencia;
    } else {
      console.error("Busca em feixe falhou - retornando tokens iniciais");
      return tokensIniciais;
    }
  }
  gerar(prompt, maxTokens=50, temperatura=0.8, metodo='top-p', raioFeixe=5, top_p=0.9, repeticaoPenal=1.5) {
    const tokens = gsa.tokenizador.codificar(prompt);
    const idFIM = gsa.tokenizador.tokenPraId.get('<FIM>') || 2;
    let tokensGerados = [];
    if(metodo=='feixe') {
      return gsa.tokenizador.decodificar(
        this.buscaFeixe(tokens, maxTokens, temperatura, raioFeixe, repeticaoPenal)
          .filter(t => t !== idFIM)
      );
    }
    // geração token por token
    for(let i=0; i<maxTokens; i++) {
      const contexto = tokens.slice(-this.seqMaxima); // janela deslizante
      const logits = this.propagar(contexto, false);
      const ultimosLogits = logits[logits.length-1];
      // aplicar penalidade por repetição
      const tokensUnicos = [...new Set(tokens)];
      tokensUnicos.forEach(token=>{ultimosLogits[token] /= repeticaoPenal;});

      const probs = softmax(ultimosLogits, temperatura);
      
      let proximoToken;
      switch(metodo) {
        case 'top-p':
          proximoToken = amostrarTopP(probs, top_p);
          break;
        case 'greedy':
          proximoToken = argmax(probs);
          break;
        default: // amostragem padrão
          proximoToken = this.exemploProb(probs);
      }
      if(proximoToken===idFIM) break;
      tokens.push(proximoToken);
      tokensGerados.push(proximoToken);
      if(tokens.length >= this.seqMaxima) break;
    }
    return gsa.tokenizador.decodificar(tokensGerados);
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
    const logits = this.propagar(tokens.slice(0, -1), false);
    const alvos = tokens.slice(1);
    let perdaTotal = 0;
    for(let i=0; i<logits.length; i++) {
      const probs = softmax(logits[i]);
      const alvo = alvos[i];
      perdaTotal += -Math.log(probs[alvo]+1e-10);
    }
    return perdaTotal/logits.length;
  }
}
class Treinador {
  constructor(modelo, taxaAprendizado=0.0001) {
    this.modelo = modelo;
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
        console.log(`Época ${epoca+1}/${epocas}, Lote ${numLotes}, Perda: ${perdaLote.toFixed(4)}, Taxa: ${this.modelo.taxa.toFixed(4)}`);
        console.log("\nAmostra \"Olá\": ", gsa.gerar("Olá", 20));
        console.log("\nAmostra \"Oi\": ", gsa.gerar("Oi", 20));
        console.log("Amostra \"O plasma\": ", gsa.gerar("O plasma", 20));
        console.log("Amostra \"O hidrogênio é\": ", gsa.gerar("O hidrogênio é", 20)+"\n");
        salvar(gsa, "modelo-"+epoca+"-"+numLotes+"_"+perdaLote.toFixed(4)+".gsa");
        numLotes++;
      }
      const perdaMedia=totalTokensEpoca>0?perdaTotalEpoca/totalTokensEpoca:0;
      this.historico.push(perdaMedia);
      console.log(`Época ${epoca+1}/${epocas}, Perda média: ${perdaMedia.toFixed(4)}, Taxa: ${this.modelo.taxa.toFixed(4)}`);
      if(epoca%1==0) {
        salvar(gsa, "modelo-"+epoca+".gsa");
        console.log("\nAmostra \"olá\": ", gsa.gerar("olá", 20));
        console.log("\nAmostra \"oi\": ", gsa.gerar("oi", 20));
        console.log("Amostra \"tudo bem?\": ", gsa.gerar("tudo bem?", 20));
        console.log("Amostra \"quem é você?\": ", gsa.gerar("quem é você?", 20)+"\n");
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
    const idFIM = gsa.tokenizador.tokenPraId.get('<FIM>');
    let perdaTotal = 0;
    let totalTokens = 0;
    for(const seq of lote) {
      const posFIM = seq.indexOf(this.idFIM);
      const seqValida = posFIM >= 0 ? seq.slice(0, posFIM+1) : seq;
      if(seqValida.length<2) continue;
      const entrada = seqValida.slice(0, -1);
      const esperado=seqValida.slice(1).map(t=>t==idFIM ? 0 : t); 
      // previsões
      const logits = this.modelo.propagar(entrada, true);
      const probs = logits.map(l => softmax(l));
      const sVerdade = esperado.map((t, i)=>oneHot(t, this.modelo.vocabTam));
      // gradientes
      let dLogits=sVerdade.map((s,i)=>derivadaEntropiaCruzada(s,probs[i]));
      dLogits = dLogits.map(vetor=>clipVetor(vetor, 1.0));
      this.modelo.retropropagar(dLogits);
      // calcula perda pra cada posição
      for(let pos=0; pos<sVerdade.length; pos++) {
        if(esperado[pos]===idFIM) continue; // Ignorar token FIM
        perdaTotal += entropiaCruzada(sVerdade[pos], probs[pos]);
      }
      totalTokens += sVerdade.length;
    }
    const perdaMedia = totalTokens>0 ? perdaTotal/totalTokens : 0;
    return {perdaLote: perdaMedia, tokensNoLote: totalTokens};
  }
}
class TreinadorBPE {
  constructor(numMerges=500, logIntervalo=0.1) {
    this.numMerges = numMerges;
    this.logIntervalo = logIntervalo;
    this.vocab = {};
    this.merges = [];
  }
  treinar(textos) {
    console.log("Iniciando treinamento BPE");
    console.log(`Textos: ${textos.length} | Merges: ${this.numMerges}`);
    this.construirVocab(textos);
    console.log(`Tokens iniciais: ${Object.keys(this.vocab).length}`);
    const pontosLog = [];
    for(let i=1; i <= 10; i++) pontosLog.push(Math.floor(this.numMerges*i*this.logIntervalo));
    for(let i=0; i<this.numMerges; i++) {
      if(pontosLog.includes(i)) console.log(`Progresso: ${Math.round((i/this.numMerges)*100)}% (${i}/${this.numMerges})`);
      const pares = this.contarPares();
      if(Object.keys(pares).length==0) {
        console.log("Sem pares disponíveis - parando antecipadamente");
        break;
      }
      const [melhorPar, freq] = Object.entries(pares).reduce((melhor, [par, f])=>f>melhor[1] ? [par, f] : melhor, ['', 0]);
      this.aplicarMerge(melhorPar);
      this.merges.push(melhorPar.split(' '));
    }
    console.log("Treinamento concluído");
    console.log(`Merges realizados: ${this.merges.length}`);
    return this.merges;
  }
  construirVocab(textos) {
    this.vocab = {};
    textos.forEach(txt => {
      const palavras = txt.split(/(\s+)/);
      palavras.forEach(palavra=>{
        if(palavra.trim()==='') {
          // é um espaço
          const tokens = ['Ġ', '</w>'];
          const chave = tokens.join(' ');
          this.vocab[chave] = (this.vocab[chave] || 0)+1;
        } else {
          // é uma palavra real
          const tokens = [...palavra, '</w>'];
          const chave = tokens.join(' ');
          this.vocab[chave] = (this.vocab[chave] || 0)+1;
        }
      });
    });
  }
  contarPares() {
    const pares = {};
    for(const [seq, freq] of Object.entries(this.vocab)) {
      const tokens = seq.split(' ');
      for(let i=0; i<tokens.length-1; i++) {
        const par = tokens[i]+' '+tokens[i+1];
        pares[par] = (pares[par] || 0)+freq;
      }
    }
    return pares;
  }
  aplicarMerge(par) {
    const [a, b] = par.split(' ');
    const novoToken = a+b;
    const regex = new RegExp(`${this.escapeRegex(a)} ${this.escapeRegex(b)}`, 'g');
    const novoVocab = {};
    for(const [seq, freq] of Object.entries(this.vocab)) {
      novoVocab[seq.replace(regex, novoToken)] = freq;
    }
    this.vocab = novoVocab;
  }
  escapeRegex(str) {
    return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
}
function criarGSA(textoTreinamento, config={}) {
  const caminhoArquivo = "margesBPE.gsa";
  let tokenizador = null;
  let merges = [];
  if(fs.existsSync(caminhoArquivo)) {
    merges = JSON.parse(fs.readFileSync(caminhoArquivo).toString('utf8'));
    console.log("Merges carregadas de: "+caminhoArquivo);
    tokenizador = config.tokenizador || new TokenizadorBPE(merges);
  } else {
    console.log("Treinando tokenizador...");
    const treinoBPE = new TreinadorBPE(100, 0.1);
    merges = treinoBPE.treinar(textos);
    tokenizador = config.tokenizador || new TokenizadorBPE(merges);
     console.log("\nVocabulário construído:");
     for(let i=0; i<Math.min(tokenizador.vocabTam, 20); i++) {
       const token = tokenizador.idPraToken.get(i);
       console.log(`${i}: "${token}"`);
     }
     const buffer = Buffer.from(JSON.stringify(merges));
     fs.writeFileSync(caminhoArquivo, buffer, "utf8");
     console.log("Merges salvas em: "+caminhoArquivo+` (${(buffer.byteLength/1024/1024).toFixed(3)} MB)`);
  }
  tokenizador.construirVocabulario(textos);
  console.log("\nTestando tokenização:");
  const testes = [
    "o sol ilumina a água",
    "céu azul",
    "transparência da água"
  ];
  testes.forEach(teste=>{
    console.log(`\nTexto: "${teste}"`);
    const ids = tokenizador.codificar(teste);
    console.log("IDs:", ids);
    const reconstruido = tokenizador.decodificar(ids);
    console.log("Reconstruído:", `"${reconstruido}"`);
    console.log(teste==reconstruido ? "Certo (√)" : "Errado (X)");
  });
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
  const modelo = config.modelo || new GSA(configuracao);
  const treinador = config.treinador || new Treinador(modelo, config.taxaAprendizado);
  return {
    modelo,tokenizador,treinador,
    gerar:function(prompt, maxTokens, temperatura, metodo, raio, p, repeticaoPenal) {
      return modelo.gerar(prompt, maxTokens, temperatura, metodo, p, repeticaoPenal);
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
const mConfig = {
  dimModelo: 128,
  numCamadas: 3,
  numCabecas: 4,
  dimFFN: 128*2,
  seqMaxima: 526,
  taxaAprendizado: 0.0001,
  taxaDropout: 0.05
};
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
  "wik",
  "conversa"
];
let treino = "";
const textos = [];
for(let i=0; i<estudo.length; i++) {
  if(fs.existsSync("treino/"+estudo[i]+".txt")) {
    const texto = fs.readFileSync("treino/"+estudo[i]+".txt","utf-8");
    textos.push(texto)
    treino += texto+" <FIM> ";
    console.log("[MATERIA]: +"+estudo[i]);
  } else console.log("[MATERIA]: NÃO EXISTE "+estudo[i]+".txt");
}
// fs.writeFileSync("treino.txt", treino, "utf-8");
console.log("[DADOS DE TREINO]: Concluídos");
// treino = fs.readFileSync("treino.txt", "utf-8");
function prepararDados(texto, tokenizador, seqTam=32) {
  texto = texto
  .replace(/https?:\/\/\S+/g, '') // remove ulrs
  .replace(/[\u200B-\u200D\uFEFF]/g, "") // zero-width
  .replace(/```[\s\S]*?```/g, '') // remove blocos de código
  .replace(/[\u{1F600}-\u{1F6FF}]/gu, ''); // remove emojis
  let frases = texto.split("<FIM>").map(s=>s.trim()).filter(x=>x.length>0);
  frases = embaralhar(frases);
  texto = frases.join(" <FIM> ");
  const tokens = tokenizador.codificar(texto);
  const seqs = [];
  const idFIM = tokenizador.tokenPraId.get('<FIM>');
  for(let i=0; i<tokens.length-seqTam; i += seqTam) {
    let seq = tokens.slice(i, i+seqTam);
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
    const resposta = gsa.gerar(pergunta, 30, 0.7);
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
    rl.question('Você: ',(entrada)=>{
      if(entrada.toLowerCase()=='/pr'){rl.close();return;} else if(entrada.startsWith("$")) eval(entrada.replace("$", ""));
      const resposta = gsa.gerar(entrada, 30);
      console.log(`ALVA GSA-1: ${resposta}`);
      perguntarUsuario();});
  }
  perguntarUsuario();
}
function executarTeste() {
  rl.question('> Treinar novo? (s/n) ', (entrada) => {
    if(entrada.toLowerCase()=='s'){console.log('> Criando modelo...');
      gsa = criarGSA(treino, mConfig);
      treinarNovo();
    } else{gsa = carregar("modelo.gsa", mConfig);conversa();}});
}
function treinarNovo() {
  console.log(`> Vocabulário criado com ${gsa.tokenizador.vocabTam} tokens`);
  console.log('> Preparando dados de treinamento...');
  const dados = prepararDados(treino, gsa.tokenizador, 64+16);
  console.log(`   ${dados.length} sequências de treinamento ${dados.length > 0 ? "preparadas (√)" : "erradas (X)"}`);
  console.log("==== ESTATÍSTICAS ====");
  console.log(`  Dimensão do modelo: ${gsa.modelo.dimModelo}`);
  console.log(`  Número de camadas: ${gsa.modelo.numCamadas}`);
  console.log(`  Número de cabeças: ${gsa.modelo.numCabecas}`);
  console.log(`  Dimensão FFN: ${gsa.modelo.dimFFN}`);
  console.log(`  Sequência máxima: ${gsa.modelo.seqMaxima}`);
  console.log(`  Taxa Dropout: ${gsa.modelo.taxaDropout}`);
  console.log(`  Taxa de aprendizado: ${gsa.modelo.taxa}`);
  console.log('  ★Parâmetros estimados★:', obterParams(gsa));
  console.log('\n> Iniciando treinamento...');
  // TREINAMENTO:
  gsa.treinador.treinar(dados, 20, 32);
  console.log('\n> Testando geração de texto...');
  const perguntasTeste = ['Como você está',
    'Qual é o seu nome',
    'O que você gosta',
    'Conte uma história',
    'Qual seu conselho'];
  avaliarModelo(gsa, perguntasTeste);
  console.log('\n> Calculando perplexidade...');
  const textoTeste = 'Olá, como você está? Qual é o seu nome?';
  const perplexidade = calcularPerplexidade(gsa.modelo, gsa.tokenizador, textoTeste);
  console.log(`   Perplexidade: ${perplexidade.toFixed(2)}`);
  console.log('\n> Exemplos de geração livre:');
  const prompts = ['Era uma vez', 'A tecnologia', 'Amizade é'];
  prompts.forEach(prompt=>{
    console.log(`   "${prompt}": "${gsa.gerar(prompt, 20, 0.8)}"`);
  });
  console.log('\n=== TREINO CONCLUÍDO ===');
  console.log('Modelo treinado e testado com sucesso!');
  salvar(gsa, "modelo.gsa");
  conversa(gsa);
}
const SALVAMENTO_VERSAO = 1;
function salvar(gsa, caminhoArquivo) {
    const modelo = gsa.modelo;
    const floats = [];
    const leitor = new Int32Array([
        SALVAMENTO_VERSAO,
        modelo.vocabTam,
        modelo.dimModelo,
        modelo.numCamadas,
        modelo.numCabecas,
        modelo.dimFFN,
        modelo.seqMaxima
    ]);
    // embeddings
    for(let i=0; i<modelo.embedding.embeddings.length; i++) {
      for(let j=0;j<modelo.embedding.embeddings[i].length;j++)floats.push(modelo.embedding.embeddings[i][j]);
    }
    // camadas
    modelo.camadas.forEach(bloco => {
        // atenção
        salvarMatriz(bloco.atencao.pq, floats);
        salvarMatriz(bloco.atencao.pk, floats);
        salvarMatriz(bloco.atencao.pv, floats);
        salvarMatriz(bloco.atencao.ps, floats);
        salvarVetor(bloco.atencao.bq, floats);
        salvarVetor(bloco.atencao.bk, floats);
        salvarVetor(bloco.atencao.bv, floats);
        salvarVetor(bloco.atencao.bs, floats);
        // FFN
        salvarMatriz(bloco.ffn.p1, floats);
        salvarMatriz(bloco.ffn.p2, floats);
        salvarVetor(bloco.ffn.b1, floats);
        salvarVetor(bloco.ffn.b2, floats);
        // normalização(gamma antes de beta)
        salvarVetor(bloco.norm1.gamma, floats);
        salvarVetor(bloco.norm1.beta, floats);
        salvarVetor(bloco.norm2.gamma, floats);
        salvarVetor(bloco.norm2.beta, floats);
    });
    // normalização final
    salvarVetor(modelo.normFinal.gamma, floats);
    salvarVetor(modelo.normFinal.beta, floats);
    // camada de saída
    salvarMatriz(modelo.cabecaSaida.p, floats);
    salvarVetor(modelo.cabecaSaida.b, floats);
    const floatBuffer = new Float32Array(floats);
    const combinado = new Uint8Array(leitor.byteLength+floatBuffer.byteLength);
    combinado.set(new Uint8Array(leitor.buffer),0);
    combinado.set(new Uint8Array(floatBuffer.buffer),leitor.byteLength);
    fs.writeFileSync(caminhoArquivo, combinado);
    console.log(`Modelo salvo (v${SALVAMENTO_VERSAO}): ${caminhoArquivo} [${floatBuffer.length} parâmetros]`);
}
function carregar(caminhoArquivo, config) {
    const buffer = fs.readFileSync(caminhoArquivo);
    const leitor = new Int32Array(buffer.buffer, 0, 7);
    const [versao, vocabTam, dimModelo, numCamadas, numCabecas, dimFFN, seqMaxima] = leitor;
    if(versao != SALVAMENTO_VERSAO)throw new Error(`Versão incompatível: arquivo v${versao} | esperado v${SALVAMENTO_VERSAO}`);
    const configCompativel = (config.dimModelo==dimModelo&&config.numCamadas==numCamadas&&config.numCabecas==numCabecas&&config.dimFFN==dimFFN);
    if(!configCompativel) {
        console.warn("Aviso: Configuração do modelo diferente do arquivo!");
        console.warn(`Arquivo: ${dimModelo}D ${numCamadas}L ${numCabecas}H`);
        console.warn(`Config:  ${config.dimModelo}D ${config.numCamadas}L ${config.numCabecas}H`);
    }
    const floatBuffer = new Float32Array(buffer.buffer,leitor.byteLength,(buffer.byteLength-leitor.byteLength)/Float32Array.BYTES_PER_ELEMENT
    );
    let pos = 0;
    const modelo = new GSA({...config, vocabTam, seqMaxima});
    // embeddings
    for(let i=0; i<vocabTam; i++) {
      for(let j=0;j<dimModelo;j++)modelo.embedding.embeddings[i][j]=floatBuffer[pos++];
    }
    // carrega camadas
    for(let i=0; i<numCamadas; i++) {
        const bloco = modelo.camadas[i];
        // atenção
        carregarMatriz(bloco.atencao.pq, floatBuffer, pos); pos += dimModelo*dimModelo;
        carregarMatriz(bloco.atencao.pk, floatBuffer, pos); pos += dimModelo*dimModelo;
        carregarMatriz(bloco.atencao.pv, floatBuffer, pos); pos += dimModelo*dimModelo;
        carregarMatriz(bloco.atencao.ps, floatBuffer, pos); pos += dimModelo*dimModelo;
        carregarVetor(bloco.atencao.bq, floatBuffer, pos); pos += dimModelo;
        carregarVetor(bloco.atencao.bk, floatBuffer, pos); pos += dimModelo;
        carregarVetor(bloco.atencao.bv, floatBuffer, pos); pos += dimModelo;
        carregarVetor(bloco.atencao.bs, floatBuffer, pos); pos += dimModelo;
        // FFN
        carregarMatriz(bloco.ffn.p1, floatBuffer, pos); pos += dimFFN*dimModelo;
        carregarMatriz(bloco.ffn.p2, floatBuffer, pos); pos += dimModelo*dimFFN;
        carregarVetor(bloco.ffn.b1, floatBuffer, pos); pos += dimFFN;
        carregarVetor(bloco.ffn.b2, floatBuffer, pos); pos += dimModelo;
        // normalização
        carregarVetor(bloco.norm1.gamma, floatBuffer, pos); pos += dimModelo;
        carregarVetor(bloco.norm1.beta, floatBuffer, pos); pos += dimModelo;
        carregarVetor(bloco.norm2.gamma, floatBuffer, pos); pos += dimModelo;
        carregarVetor(bloco.norm2.beta, floatBuffer, pos); pos += dimModelo;
    }
    // normalização final
    carregarVetor(modelo.normFinal.gamma, floatBuffer, pos); pos += dimModelo;
    carregarVetor(modelo.normFinal.beta, floatBuffer, pos); pos += dimModelo;
    // camada de saída
    carregarMatriz(modelo.cabecaSaida.p, floatBuffer, pos); pos += vocabTam * dimModelo;
    carregarVetor(modelo.cabecaSaida.b, floatBuffer, pos); pos += vocabTam;
    console.log(`Modelo carregado: ${caminhoArquivo} [${floatBuffer.length} parâmetros]`);
    return criarGSA(null, { modelo });
}
function salvarMatriz(m, floats) {
  for(let i=0; i<m.length; i++) {
    for(let j=0;j<m[i].length;j++)floats.push(m[i][j]);
  }
}
function salvarVetor(v,floats){for(let i=0; i<v.length;i++)floats.push(v[i]);}
function carregarMatriz(m, buffer, ini) {
  let pos = ini;
  for(let i=0; i<m.length; i++) {
    for(let j=0;j<m[i].length;j++)m[i][j]=buffer[pos++];
  }
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
