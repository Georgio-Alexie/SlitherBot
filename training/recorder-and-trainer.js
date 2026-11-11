// Training side of the V5 network (Claude, 2026-11-08). Load bot/slither-bot-v6.js first (it defines V5).
// Flow in the slither.io page console:
//   1. V5restoreData()          load the 3,073 recorded samples from this page's localStorage (or V5loadData(json) from data/dataset.json)
//   2. V5record()               while YOU play (__mode='human'), log features + your turn/boost every 100 ms
//   3. await V5.train({epochs:4,lr:2e-3})   run-wise holdout; prints V5.trainInfo vs a majority baseline
//   4. V5exportWeights()        downloads a new model/weights.json; save it over the old one
// Notes: input 81 (own speed) is zeroed: it leaked the boost label. More than ~4 epochs overfits this little data.
(()=>{
const {NS,NF,NB,wrap,binOf}=V5;
// dataset
V5.X=V5.X||[]; V5.Yt=V5.Yt||[]; V5.Yb=V5.Yb||[]; V5.run=V5.run||[];
V5.runId=V5.runId||0; V5.wasAlive=false;
window.V5record=function(){ if(window.__rec5)clearInterval(window.__rec5); window.__mode='human'; try{setAcceleration(0)}catch(e){}
window.__rec5=setInterval(()=>{
 const s=window.slither; if(window.__mode!=='human'||!s){V5.wasAlive=false;return;}
 if(!V5.wasAlive){V5.wasAlive=true;V5.runId++;}
 const {F}=V5.feats(s); const tgt=Math.atan2(window.ym,window.xm); const rel=wrap(tgt-s.ang);
 V5.X.push(F); V5.Yt.push(binOf(rel)); V5.Yb.push((s.sp||0)>9?1:0); V5.run.push(V5.runId);
},100); };
V5.nParams=P=>Object.values(P).reduce((a,v)=>a+v.length,0);
V5.train=async function({epochs=30,lr=3e-3,bs=64,boostW=0.5}={}){
 const n=V5.X.length; const runs=[...new Set(V5.run)]; const valRuns=new Set(runs.length>=3?[runs[runs.length-1]]:[]);
 let tr=[],va=[]; for(let i=0;i<n;i++)(valRuns.has(V5.run[i])?va:tr).push(i); if(!va.length){const k=Math.floor(n*0.85);va=tr.slice(k);tr=tr.slice(0,k);}
 const cnt=new Float32Array(NB).fill(1); for(const i of tr)cnt[V5.Yt[i]]++; const cw=Array.from(cnt,c=>Math.pow(tr.length/NB/c,0.5));
 const P=V5.init(7), M={},Vv={}; for(const k in P){M[k]=new Float32Array(P[k].length);Vv[k]=new Float32Array(P[k].length);} let t=0;
 const H=V5.H,NO=NB+1; const log=[];
 for(let ep=0;ep<epochs;ep++){
  for(let i=tr.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[tr[i],tr[j]]=[tr[j],tr[i]];}
  let loss=0;
  for(let b0=0;b0<tr.length;b0+=bs){
   const G={};for(const k in P)G[k]=new Float32Array(P[k].length); const B=Math.min(bs,tr.length-b0);
   for(let q=0;q<B;q++){ const idx=tr[b0+q]; let x=V5.X[idx],yt=V5.Yt[idx]; if(Math.random()<0.5){x=V5.mirror(x);yt=NB-1-yt;} const yb=V5.Yb[idx];
     const f=V5.fwd(P,x); const w=cw[yt]; loss+=-Math.log(f.p[yt]+1e-9)*w;
     const d3=new Float32Array(NO); for(let j=0;j<NB;j++)d3[j]=(f.p[j]-(j===yt?1:0))*w/B; d3[NB]=(f.pb-yb)*boostW/B;
     const d2=new Float32Array(H); for(let i=0;i<H;i++){let s=0;for(let j=0;j<NO;j++){G.W3[i*NO+j]+=f.h2[i]*d3[j];s+=P.W3[i*NO+j]*d3[j];}d2[i]=f.h2[i]>0?s:0;} for(let j=0;j<NO;j++)G.b3[j]+=d3[j];
     const d1=new Float32Array(H); for(let i=0;i<H;i++){let s=0;for(let j=0;j<H;j++){G.W2[i*H+j]+=f.h1[i]*d2[j];s+=P.W2[i*H+j]*d2[j];}d1[i]=f.h1[i]>0?s:0;} for(let j=0;j<H;j++)G.b2[j]+=d2[j];
     for(let i=0;i<NF;i++){const xi=x[i];if(!xi)continue;for(let j=0;j<H;j++)G.W1[i*H+j]+=xi*d1[j];} for(let j=0;j<H;j++)G.b1[j]+=d1[j]; }
   t++; const b1=0.9,b2=0.999; for(const k in P){const p=P[k],g=G[k],m=M[k],v=Vv[k]; for(let i=0;i<p.length;i++){const gi=g[i]+(k[0]==='W'?1e-4*p[i]:0); m[i]=b1*m[i]+(1-b1)*gi; v[i]=b2*v[i]+(1-b2)*gi*gi; p[i]-=lr*(m[i]/(1-Math.pow(b1,t)))/(Math.sqrt(v[i]/(1-Math.pow(b2,t)))+1e-8);} }
   if((b0/bs)%40===0)await new Promise(r=>setTimeout(r,0));
  }
  let acc=0,near=0,bacc=0; for(const i of va){const f=V5.fwd(P,V5.X[i]); let a=0;for(let j=1;j<NB;j++)if(f.p[j]>f.p[a])a=j; if(a===V5.Yt[i])acc++; if(Math.abs(a-V5.Yt[i])<=1)near++; if((f.pb>0.5?1:0)===V5.Yb[i])bacc++;}
  log.push({ep,loss:+(loss/tr.length).toFixed(3),valTurnAcc:+(acc/va.length).toFixed(3),valTurnWithin1:+(near/va.length).toFixed(3),valBoostAcc:+(bacc/va.length).toFixed(3)});
  V5.trainLog=log; await new Promise(r=>setTimeout(r,0));
 }
 let maj=new Float32Array(NB); for(const i of tr)maj[V5.Yt[i]]++; let mb=0;for(let j=1;j<NB;j++)if(maj[j]>maj[mb])mb=j; let base=0,bb=0,boostRate=0; for(const i of tr)boostRate+=V5.Yb[i]; boostRate/=tr.length; for(const i of va){if(V5.Yt[i]===mb)base++; if((boostRate>0.5?1:0)===V5.Yb[i])bb++;}
 V5.P=P; V5.trainInfo={n,train:tr.length,val:va.length,params:V5.nParams(P),baselineTurnAcc:+(base/va.length).toFixed(3),baselineBoostAcc:+(bb/va.length).toFixed(3),last:log[log.length-1]};
 return V5.trainInfo;
};
V5.backup=function(){ const n=V5.X.length,NF=V5.NF; const u=new Uint8Array(n*(NF+3)); for(let i=0;i<n;i++){const o=i*(NF+3); for(let j=0;j<NF;j++)u[o+j]=Math.round((Math.max(-1,Math.min(1,V5.X[i][j]))+1)/2*255); u[o+NF]=V5.Yt[i]; u[o+NF+1]=V5.Yb[i]; u[o+NF+2]=V5.run[i]%256;}
 let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192)); try{localStorage.__v5data=btoa(s);localStorage.__v5n=n;return n;}catch(e){return 'fail:'+e.message;} };
V5.restore=function(b64,n){ const s=atob(b64),R=NF+3; V5.X=[];V5.Yt=[];V5.Yb=[];V5.run=[];
 for(let i=0;i<n;i++){const o=i*R,x=new Float32Array(NF); for(let j=0;j<NF;j++)x[j]=s.charCodeAt(o+j)/255*2-1; x[NS*5+1]=0; V5.X.push(x); V5.Yt.push(s.charCodeAt(o+NF)); V5.Yb.push(s.charCodeAt(o+NF+1)); V5.run.push(s.charCodeAt(o+NF+2));}
 V5.runId=Math.max(0,...V5.run); return V5.X.length; };
window.V5restoreData=()=>V5.restore(localStorage.__v5data||'',+localStorage.__v5n||0);
window.V5loadData=json=>V5.restore(json.dataset.b64,json.dataset.n);
window.V5exportWeights=function(){ const b64=a=>{const u=new Uint8Array(a.buffer,a.byteOffset,a.byteLength);let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192));return btoa(s);};
 const H=V5.H,NO=NB+1,shapes={W1:[NF,H],b1:[H],W2:[H,H],b2:[H],W3:[H,NO],b3:[NO]}; const tensors={}; for(const k in V5.P)tensors[k]={shape:shapes[k],dtype:'float32-le',b64:b64(V5.P[k])};
 const out={format:'slither-v5-mlp/1',exportedAt:new Date().toISOString(),arch:{inputs:NF,hidden:[H,H],outputs:{turnBins:NB,boostLogit:1},activation:'relu',layout:'W[i*cols+j], i=input, j=output',bins:V5.BINS,maskedInputs:[NS*5+1]},tensors,hand:V5.hand,trainInfo:V5.trainInfo,v6cfg:window.V6&&V6.cfg};
 const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'})); a.download='weights.json'; a.click(); };
})();
