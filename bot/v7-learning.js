// V7 self-improvement layer (Claude, 2026-11-08). Load after bot/slither-bot-v6.js, or use bot/slither-bot-v7.bundle.js.
// After every life it:
//  1. labels the life: headings taken in the 4 s before death -> danger (soft label), headings followed by
//     above-median growth over the next 10 s, or by a kill within 3 s -> good;
//  2. retrains a larger network (85 -> 64 -> 64 -> 23, 11,159 weights): 11 steering logits, 1 boost logit and
//     11 death-risk outputs (one per heading bin), on the replay buffer plus a slice of the human data;
//  3. runs an evolution strategy on the hand weights: every life tries a perturbed set (antithetic within a tab);
//     after a population of lives (2 in one tab, up to 20 across tabs) the mean moves toward the better-ranked ones
//     (fitness = ln(length+50) + 0.2*kills);
//  4. saves weights, ES state and the life log to localStorage (V7.export() downloads them for the repo).
// The planner subtracts deathW * risk(heading) once the critic has seen 2 deaths.
(()=>{
const {NS,NF,NB,wrap,binOf}=V5;
const H=64, NO=2*NB+1;
const V7=window.V7={H,NO,buf:[],life:[],log:[],nDeaths:0,policyReady:false,busy:false,tick:0,human:[]};
const b64e=a=>{const u=new Uint8Array(a.buffer,a.byteOffset,a.byteLength);let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192));return btoa(s);};
const b64d=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);};
const sig=z=>1/(1+Math.exp(-z));
const gauss=()=>{let u=Math.random()||1e-9,v=Math.random();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};

// ---- network ----
V7.init=function(seed=11){ let r=seed; const rnd=()=>{r=(r*1103515245+12345)%2147483648; return r/2147483648;}; const g=()=>{let u=rnd()||1e-9,v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
 const mk=(n,m)=>{const W=new Float32Array(n*m);const s=Math.sqrt(2/n);for(let i=0;i<W.length;i++)W[i]=g()*s;return W;};
 const P={W1:mk(NF,H),b1:new Float32Array(H),W2:mk(H,H),b2:new Float32Array(H),W3:mk(H,NO),b3:new Float32Array(NO)};
 for(let j=0;j<NO;j++)if(j>NB)for(let i=0;i<H;i++)P.W3[i*NO+j]*=0.1;   // death head starts quiet...
 for(let j=0;j<NB;j++)P.b3[NB+1+j]=-3;                                   // ...at about 5% risk
 return P; };
V7.fwd=function(P,x){ const h1=new Float32Array(H),h2=new Float32Array(H),o=new Float32Array(NO);
 for(let j=0;j<H;j++){let s=P.b1[j];for(let i=0;i<NF;i++)s+=x[i]*P.W1[i*H+j];h1[j]=s>0?s:0;}
 for(let j=0;j<H;j++){let s=P.b2[j];for(let i=0;i<H;i++)s+=h1[i]*P.W2[i*H+j];h2[j]=s>0?s:0;}
 for(let j=0;j<NO;j++){let s=P.b3[j];for(let i=0;i<H;i++)s+=h2[i]*P.W3[i*NO+j];o[j]=s;}
 let m=-1e9;for(let j=0;j<NB;j++)m=Math.max(m,o[j]); let z=0; const p=new Float32Array(NB); for(let j=0;j<NB;j++){p[j]=Math.exp(o[j]-m);z+=p[j];} for(let j=0;j<NB;j++)p[j]/=z;
 const pd=new Float32Array(NB); for(let j=0;j<NB;j++)pd[j]=sig(o[NB+1+j]);
 return {h1,h2,o,p,pb:sig(o[NB]),pd}; };
V7.nParams=()=>Object.values(V7.P).reduce((a,v)=>a+v.length,0);
// samples: {F,b,bo,human} (your play) or {F,b,bo,yd,w} (bot's own play: yd = death label 0..1, w = self-imitation weight)
V7.trainOn=async function(S,{epochs=2,lr=1e-3,bs=64}={}){
 const P=V7.P; if(!V7.opt){V7.opt={t:0,M:{},V:{}}; for(const k in P){V7.opt.M[k]=new Float32Array(P[k].length);V7.opt.V[k]=new Float32Array(P[k].length);}}
 const O=V7.opt, idx=S.map((_,i)=>i), posW=V7.posW||1; let loss=0,cnt=0;
 for(let ep=0;ep<epochs;ep++){
  for(let i=idx.length-1;i>0;i--){const j=Math.floor(Math.random()*(i+1));[idx[i],idx[j]]=[idx[j],idx[i]];}
  for(let b0=0;b0<idx.length;b0+=bs){
   const G={};for(const k in P)G[k]=new Float32Array(P[k].length); const B=Math.min(bs,idx.length-b0);
   for(let q=0;q<B;q++){ const sm=S[idx[b0+q]]; let x=sm.F,b=sm.b; if(Math.random()<0.5){x=V5.mirror(x);b=NB-1-b;}
     const f=V7.fwd(P,x); const d3=new Float32Array(NO);
     if(sm.human){ for(let j=0;j<NB;j++)d3[j]=(f.p[j]-(j===b?1:0))/B; d3[NB]=(f.pb-sm.bo)*0.5/B; loss+=-Math.log(f.p[b]+1e-9); cnt++; }
     else { if(sm.w>0)for(let j=0;j<NB;j++)d3[j]=(f.p[j]-(j===b?1:0))*sm.w*0.5/B;
       const k=NB+1+b, pd=f.pd[b], wd=sm.yd>0.3?posW:1; d3[k]=(pd-sm.yd)*wd/B; loss+=-(sm.yd*Math.log(pd+1e-9)+(1-sm.yd)*Math.log(1-pd+1e-9)); cnt++; }
     const d2=new Float32Array(H); for(let i=0;i<H;i++){let s=0;for(let j=0;j<NO;j++){if(!d3[j])continue;G.W3[i*NO+j]+=f.h2[i]*d3[j];s+=P.W3[i*NO+j]*d3[j];}d2[i]=f.h2[i]>0?s:0;} for(let j=0;j<NO;j++)G.b3[j]+=d3[j];
     const d1=new Float32Array(H); for(let i=0;i<H;i++){if(!f.h1[i])continue;let s=0;for(let j=0;j<H;j++){G.W2[i*H+j]+=f.h1[i]*d2[j];s+=P.W2[i*H+j]*d2[j];}d1[i]=s;} for(let j=0;j<H;j++)G.b2[j]+=d2[j];
     for(let i=0;i<NF;i++){const xi=x[i];if(!xi)continue;for(let j=0;j<H;j++)G.W1[i*H+j]+=xi*d1[j];} for(let j=0;j<H;j++)G.b1[j]+=d1[j]; }
   O.t++; const b1=0.9,b2=0.999; for(const k in P){const p=P[k],g=G[k],m=O.M[k],v=O.V[k]; for(let i=0;i<p.length;i++){const gi=g[i]+(k[0]==='W'?1e-4*p[i]:0); m[i]=b1*m[i]+(1-b1)*gi; v[i]=b2*v[i]+(1-b2)*gi*gi; p[i]-=lr*(m[i]/(1-Math.pow(b1,O.t)))/(Math.sqrt(v[i]/(1-Math.pow(b2,O.t)))+1e-8);} }
   if((b0/bs)%25===0)await new Promise(r=>{ if(typeof MessageChannel!=='undefined'){const c=new MessageChannel(); c.port1.onmessage=()=>{c.port1.close();r();}; c.port2.postMessage(0);} else setTimeout(r,0); });   // not throttled in hidden tabs
  }
 }
 return +(loss/Math.max(1,cnt)).toFixed(3); };

// ---- evolution strategy over the hand weights ----
if(V6.cfg.deathW==null)V6.cfg.deathW=400;
if(!V6.base)V6.base={...V6.cfg};
const TUNE=['big','near','prey','pathFood','gapW','rearW','raceW','cutW','coilW','netW','deathW','orbWMax','orbBias','safeW','enclosedFrac','minBoostL','stakeL'].filter(k=>V6.base[k]!=null);
V7.TUNE=TUNE;
V7.es={mu:Object.fromEntries(TUNE.map(k=>[k,0])),sigma:0.25,alpha:0.3,pool:[],popN:2,gen:0,pend:null,cur:null,best:null};
V7.applyCfg=function(eps,sign){ const E=V7.es; for(const k of TUNE){ if(eps[k]==null)eps[k]=gauss(); const th=Math.max(-1.5,Math.min(1.5,(E.mu[k]||0)+sign*E.sigma*eps[k])); let v=V6.base[k]*Math.exp(th); if(k==='enclosedFrac')v=Math.max(0.3,Math.min(0.75,v)); V6.cfg[k]=v; } };
// each life tries mu + sign*sigma*eps; the next life in the same tab tries the opposite sign (antithetic)
V7.onStart=function(){ V7.life=[]; V7.tick=0; V7.kc=0; const E=V7.es; let eps,sign;
 if(E.pend){eps=E.pend;sign=-1;E.pend=null;} else {eps={};for(const k of TUNE)eps[k]=gauss();sign=1;E.pend=eps;}
 E.cur={eps,sign}; V7.applyCfg(eps,sign); };
V7.record=function(F,b,bo,L){ const kc=(window.slither&&window.slither.kill_count)||0;
 if(kc>(V7.kc||0)){ const n=V7.life.length; for(let i=Math.max(0,n-25);i<n;i++)V7.life[i].killed=1; }  // the 3 s before a kill count as good play
 V7.kc=kc; if((V7.tick++)%3)return; V7.life.push({F,b,bo,L,t:Date.now()}); };
V7.deathOn=()=>V7.nDeaths>=2;

// ---- after each life: label it, then learn here (single tab) or send it to the hub (multi-tab, see v8) ----
V7.onDeath=function(len,kills,secs){
 const tD=Date.now(), life=V7.life; V7.life=[];
 for(const q of life){ const dt=tD-q.t; q.yd=dt<1000?1:dt<4000?Math.exp(-(dt-1000)/1200):0; }
 const gains=[]; let j=0; for(let i=0;i<life.length;i++){ if(j<i)j=i; while(j<life.length-1&&life[j+1].t<=life[i].t+10000)j++; gains.push(life[j].L-life[i].L); }
 const med=gains.slice().sort((a,b)=>a-b)[gains.length>>1]||0;
 for(let i=0;i<life.length;i++){ const q=life[i]; q.w=q.yd>0.05?0:Math.max(q.killed?1:0,Math.min(1,(gains[i]-med)/60)); }
 const cur=V7.es.cur; V7.es.cur=null;
 const pk={life,len,kills,secs,eps:cur&&cur.eps,sign:cur&&cur.sign,cfg:{...V6.cfg}};
 if(V7.remote)V7.remote(pk); else V7.ingest(pk);
};
V7.ingest=function(pk){
 const life=pk.life, len=pk.len, kills=pk.kills;
 // honest check: how the critic rated this life's moves BEFORE learning from it
 let sd=0,nd=0,ss=0,ns=0; for(const q of life){ const pd=V7.fwd(V7.P,q.F).pd[q.b]; if(q.yd>0.5){sd+=pd;nd++;} else if(q.yd===0){ss+=pd;ns++;} }
 for(const q of life)V7.buf.push(q); if(V7.buf.length>60000)V7.buf.splice(0,V7.buf.length-60000);
 V7.nDeaths++;
 const fit=Math.log(len+50)+0.2*kills, E=V7.es;
 if(pk.eps){ E.pool.push({eps:pk.eps,sign:pk.sign,fit});
   if(E.pool.length>=E.popN){ // rank-based ES step over the population of lives
     const P=E.pool.slice().sort((a,b)=>a.fit-b.fit), n=P.length; const step={};
     for(const k of TUNE){ let g=0; P.forEach((r,i)=>{ const u=n>1?i/(n-1)-0.5:0; g+=u*r.sign*r.eps[k]; }); E.mu[k]=Math.max(-1,Math.min(1,(E.mu[k]||0)+E.alpha*E.sigma*g/n*4)); }
     E.pool=[]; E.gen++; } }
 if(!E.best||fit>E.best.fit)E.best={fit:+fit.toFixed(2),len,kills,cfg:pk.cfg||{...V6.cfg},tab:pk.from||'here'};
 V7.log.push({life:V7.log.length+1,at:new Date().toISOString(),tab:pk.from||'here',len,kills,secs:pk.secs,fit:+fit.toFixed(2),es:pk.sign>0?'+':pk.sign<0?'-':'',gen:E.gen,
   critic:nd?{deathWindow:+(sd/nd).toFixed(3),safe:+(ss/Math.max(1,ns)).toFixed(3)}:null,samples:life.length});
 if(V7.log.length>3000)V7.log.splice(0,V7.log.length-3000);
 V7.learnSoon();
};
V7.learnSoon=function(){ if(V7.busy){V7.pendingLearn=true;return;}
 V7.learn().then(()=>{ V7.save(); if(V7.onLearned)V7.onLearned(); if(V7.pendingLearn){V7.pendingLearn=false;V7.learnSoon();} }).catch(e=>{V7.err=String(e);}); };
V7.learn=async function(){ if(V7.busy)return; V7.busy=true; const t0=Date.now();
 try{ const B=V7.buf, S=B.slice(-8000); for(let i=0;i<4000&&B.length>8000;i++)S.push(B[Math.floor(Math.random()*(B.length-8000))]);
   let nPos=0; for(const q of S)if(q.yd>0.3)nPos++;
   V7.posW=Math.max(1,Math.min(15,Math.sqrt((S.length-nPos)/Math.max(1,nPos))));
   if(V7.human.length)for(let i=0;i<Math.min(1500,V7.human.length);i++)S.push(V7.human[Math.floor(Math.random()*V7.human.length)]);
   const loss=await V7.trainOn(S,{epochs:2,lr:1e-3});
   V7.lastLearn={n:S.length,deathSamples:nPos,posW:+V7.posW.toFixed(1),loss,ms:Date.now()-t0};
   const last=V7.log[V7.log.length-1]; if(last)last.learn=V7.lastLearn;
 } finally { V7.busy=false; } };

// ---- persistence ----
V7.state=function(){ const P={}; for(const k in V7.P)P[k]=b64e(V7.P[k]);
 return {format:'slither-v7/1',savedAt:new Date().toISOString(),arch:{inputs:NF,hidden:[H,H],outputs:{policy:NB,boost:1,deathRisk:NB},params:V7.nParams()},
   P,es:{...V7.es,cur:null},base:V6.base,cfgNow:{...V6.cfg},log:V7.log,nDeaths:V7.nDeaths,policyReady:V7.policyReady}; };
V7.save=function(){ try{localStorage.__v7state=JSON.stringify(V7.state());return true;}catch(e){V7.saveErr=String(e);return false;} };
V7.export=function(withBuffer=true){ const st=V7.state();
 if(withBuffer&&V7.buf.length){ const R=NF+4,n=V7.buf.length,u=new Uint8Array(n*R); for(let i=0;i<n;i++){const q=V7.buf[i],o=i*R; for(let j=0;j<NF;j++)u[o+j]=Math.round((Math.max(-1,Math.min(1,q.F[j]))+1)/2*255); u[o+NF]=q.b; u[o+NF+1]=q.bo; u[o+NF+2]=Math.round(q.yd*255); u[o+NF+3]=Math.round(q.w*255);}
   st.buffer={n,rowBytes:R,encoding:'uint8 per row: NF features (x+1)/2*255, heading bin, boost, death label*255, self-imitation weight*255',b64:b64e(u)}; }
 const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(st)],{type:'application/json'})); a.download='v7-state.json'; document.body.appendChild(a); a.click(); a.remove(); return st.log.length; };

// ---- human data (your recorded play) ----
try{ const s=atob(localStorage.__v5data||''), n=+localStorage.__v5n||0, R=NF+3;
 for(let i=0;i<n;i++){const o=i*R,x=new Float32Array(NF); for(let j=0;j<NF;j++)x[j]=s.charCodeAt(o+j)/255*2-1; x[NS*5+1]=0; V7.human.push({F:x,b:s.charCodeAt(o+NF),bo:s.charCodeAt(o+NF+1),human:true});} }catch(e){}

// ---- load saved state, or start fresh (steering head pre-trained on your play) ----
let st=window.SLITHER_V7_STATE||null; if(!st){try{st=JSON.parse(localStorage.__v7state||'null')}catch(e){}}
if(st&&st.P){ V7.P={}; for(const k in st.P)V7.P[k]=b64d(st.P[k]); const es=st.es||{}; Object.assign(V7.es,es,{cur:null}); V7.log=st.log||[]; V7.nDeaths=st.nDeaths||0; V7.policyReady=!!st.policyReady; V7.src=window.SLITHER_V7_STATE?'file':'localStorage'; }
else { V7.P=V7.init(11); V7.src='fresh'; }
if(!V7.policyReady&&V7.human.length){ V7.busy=true; V7.trainOn(V7.human,{epochs:4,lr:2e-3}).then(l=>{V7.policyReady=true;V7.pretrainLoss=l;V7.busy=false;V7.save();}).catch(e=>{V7.err=String(e);V7.busy=false;}); }

// ---- wire into the V6 planner ----
let src=window.__botStep6.toString();
const rep=(a,b)=>{ if(src.split(a).length!==2)throw new Error('V7 patch target not found once: '+a.slice(0,50)); src=src.replace(a,b); };
rep('const net=V5.fwd(V5.P,F);','const v7=V7.fwd(V7.P,F); const net=V7.policyReady?v7:V5.fwd(V5.P,F);');
rep('V6.diag=[];}','V6.diag=[];V7.onStart();}');
rep('V6.deaths.push({len:window.__bMax,last:d});','V6.deaths.push({len:window.__bMax,last:d}); V7.onDeath(window.__bMax,window.__bKills,Math.round((Date.now()-window.__bStart)/1000));');
rep('if(goalA!==null&&steps>=NSTEP)sc+=goalW','if(V7.deathOn())sc-=C.deathW*v7.pd[binOf(rel)];\n   if(goalA!==null&&steps>=NSTEP)sc+=goalW');
rep('try{setAcceleration(why?1:0)}catch(e){}','try{setAcceleration(why?1:0)}catch(e){} V7.record(F,binOf(wrap(bestA-ha)),why?1:0,L);');
rep('rear:rearPush.length,','rear:rearPush.length,risk:V7.deathOn()?+v7.pd[binOf(wrap(bestA-ha))].toFixed(2):null,');
window.__botStep6=eval('('+src+')');
V7.ver='v7';
})();
