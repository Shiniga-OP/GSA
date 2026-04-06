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
    this.ps = attPesosAdam(this.ps, dPs, this.ms, this.vs, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.pq = attPesosAdam(this.pq, dPq, this.mq, this.vq, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.pk = attPesosAdam(this.pk, dPk, this.mk, this.vk, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.pv = attPesosAdam(this.pv, dPv, this.mv, this.vv, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.bs = attPesosAdam1D(this.bs, dBo, this.mBs, this.vBs, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.bq = attPesosAdam1D(this.bq, dBq, this.mBq, this.vBq, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.bk = attPesosAdam1D(this.bk, dBk, this.mBk, this.vBk, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.bv = attPesosAdam1D(this.bv, dBv, this.mBv, this.vBv, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
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
    if(treino) camada1 = ativ1.map(seq=>seq.map(v=>dropout(v, this.taxaDrop)));
    
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
    this.p1 = attPesosAdam(this.p1, dP1, this.m1, this.v1, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.p2 = attPesosAdam(this.p2, dP2, this.m2, this.v2, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.b1 = attPesosAdam1D(this.b1, dB1, this.mB1, this.vB1, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.b2 = attPesosAdam1D(this.b2, dB2, this.mB2, this.vB2, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
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
    this.gamma = attPesosAdam1D(this.gamma, dGamma, this.mG, this.vG, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.beta = attPesosAdam1D(this.beta, dBeta, this.mB, this.vB, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
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
    this.embeddings = attPesosAdam(this.embeddings, gradEmbeddings, this.m, this.v, taxa, 0.9, null, 1e-8, this.iteracao, this.lambda);
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
    this.p = attPesosAdam(this.p, gradP, this.m, this.v, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.b = attPesosAdam1D(this.b, gradB, this.mb, this.vb, taxa, 0.9, null, 1e-8, this.iteracao, lambda);
    this.iteracao++;
    return dEntrada;
  }
}