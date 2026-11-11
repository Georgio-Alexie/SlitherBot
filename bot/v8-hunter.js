// V8/V9 hunter (Claude, 2026-11-08; v9.0 2026-11-11). Load after v7-learning.js, or use bot/slither-bot-v8.bundle.js (same file name, contains v9.0).
//
// 1. Behaviour model, 31,500 weights (104 -> 128 -> 128 -> 12; v9, was 1M): from the last 2 s of an enemy's
//    movement, what surrounds it (bodies, food, wall, me) it predicts where its head will be 0.12-1.44 s ahead.
//    Every visible snake supplies about 8 training examples a second, so the model can keep learning without
//    running out of data. It trains in background threads and is only trusted once it beats a plain
//    "keeps going straight at the same speed" guess on held-out examples.
// 2. Hunting: for each candidate heading the planner checks whether a predicted enemy head will run into the
//    body we lay down, before it gets there. Predictable snakes (low recent error, or detected circling)
//    get a bigger reward. A reachable cut-off can trigger a boost.
//    Predicted head positions are also added as time-matched obstacles.
// 3. Minimap: when nothing nearby needs doing, it heads for a region the minimap shows has the right number
//    of snakes: few while small, a few more once big enough to hunt. It stays inside the red wall.
// 4. Many tabs: one hub tab trains (3 threads, weights averaged every 20 s) and runs the population ES;
//    every bot tab plays, streams what it sees to the hub over a BroadcastChannel, and gets new weights back.
//    The bot tick runs from a worker timer so background tabs aren't slowed down.
//
// Use: single tab  -> V8.solo()      (plays and learns in one tab)
//      many tabs   -> V8.hub() in one tab, V8.bot() in each of the others
(()=>{
const {NS,NF,NB,wrap,binOf}=V5;
const IN=104, OUT=12, HZ=[1,2,4,6,8,12], DT=120, LAYERS=[IN,128,128,OUT];   // 100 shape/context inputs + 4 temperament (greed, fear, aggr, panic)

// ================= worker: the big network =================
function enemyWorker(self){
 const NL=LAYERS.length-1; let P=null,M=null,V=null,G=null,t=0,lr=3e-4,train=false;
 const CAP=60000,VCAP=4000; let BX=null,BY=null,VX=null,VY=null,bn=0,bh=0,vn=0,vh=0;   // buffers only in training threads
 const st={seen:0,steps:0,loss:null,err096:null,base096:null,err144:null,base144:null,msPerSample:null,mode:'full',rateMs:0,paused:false};
 let paused=false,mode='full',evals=[],bestErr=1e9,sinceEval=0;
 const RATE={full:0,throttled:50,idle:500}, EVERY={full:150,throttled:40,idle:10};   // ms between 16-sample steps; steps between validations
 const okShape=Q=>Array.isArray(Q)&&Q.length===LAYERS.length-1&&Q.every((p,l)=>p&&p.W&&p.b&&p.W.length===LAYERS[l]*LAYERS[l+1]&&p.b.length===LAYERS[l+1]);
 let r=Math.floor(Math.random()*1e9)+1; const rnd=()=>{r=(r*1103515245+12345)%2147483648;return r/2147483648;};
 const gauss=()=>{let u=rnd()||1e-9,v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
 const zeros=()=>P.map(p=>({W:new Float32Array(p.W.length),b:new Float32Array(p.b.length)}));
 function init(){ P=[]; for(let l=0;l<NL;l++){const a=LAYERS[l],b=LAYERS[l+1],W=new Float32Array(a*b),s=Math.sqrt(2/a)*(l===NL-1?0.1:1); for(let i=0;i<W.length;i++)W[i]=gauss()*s; P.push({W,b:new Float32Array(b)});} }
 const A=LAYERS.map(n=>new Float32Array(n)), D=LAYERS.map(n=>new Float32Array(n));
 function fwd(x,xo){ const a0=A[0]; for(let i=0;i<IN;i++)a0[i]=x[xo+i];
  for(let l=0;l<NL;l++){ const X=A[l],Z=A[l+1],W=P[l].W,na=LAYERS[l],nz=LAYERS[l+1]; Z.set(P[l].b);
   for(let i=0;i<na;i++){const xi=X[i]; if(xi===0)continue; const o=i*nz; for(let j=0;j<nz;j++)Z[j]+=xi*W[o+j];}
   if(l<NL-1)for(let j=0;j<nz;j++)if(Z[j]<0)Z[j]=0; }
  return A[NL]; }
 function step(B){
  for(const g of G){g.W.fill(0);g.b.fill(0);} let loss=0;
  for(let q=0;q<B;q++){ const k=Math.floor(rnd()*bn), y=fwd(BX,k*IN), d=D[NL];
   for(let j=0;j<OUT;j++){const e=y[j]-BY[k*OUT+j]; loss+=e*e; d[j]=(e>1?1:e<-1?-1:e)/B;}          // Huber loss
   for(let l=NL-1;l>=0;l--){ const X=A[l],Dz=D[l+1],W=P[l].W,gW=G[l].W,gb=G[l].b,na=LAYERS[l],nz=LAYERS[l+1],Dx=D[l];
    for(let j=0;j<nz;j++)gb[j]+=Dz[j];
    for(let i=0;i<na;i++){ const xi=X[i]; if(xi===0){Dx[i]=0;continue;} const o=i*nz; let s=0;
     if(l>0){ for(let j=0;j<nz;j++){gW[o+j]+=xi*Dz[j]; s+=W[o+j]*Dz[j];} } else { for(let j=0;j<nz;j++)gW[o+j]+=xi*Dz[j]; }
     Dx[i]=s; } } }
  t++; const b1=0.9,b2=0.999,c1=1-Math.pow(b1,t),c2=1-Math.pow(b2,t);
  for(let l=0;l<NL;l++)for(const key of ['W','b']){const p=P[l][key],g=G[l][key],m=M[l][key],v=V[l][key],wd=key==='W'?1e-5:0;
   for(let i=0;i<p.length;i++){const gi=g[i]+wd*p[i]; m[i]=b1*m[i]+(1-b1)*gi; v[i]=b2*v[i]+(1-b2)*gi*gi; p[i]-=lr*(m[i]/c1)/(Math.sqrt(v[i]/c2)+1e-8);} }
  return loss/B; }
 // duty cycle: flat out until the model is trusted, then 1 step / 50 ms, then 1 step / 500 ms once validation error has stalled
 // (< 2% better over the last 10 validations); back to 50 ms if it rises > 10% above its best
 function ctl(rec){ const e=st.err096; if(e==null)return; if(rec){evals.push(e); if(evals.length>40)evals.shift();} if(e<bestErr)bestErr=e;
  const trusted=st.valN>=200&&e<0.85*st.base096;
  if(!trusted)mode='full';
  else if(mode==='full'){mode='throttled';evals=[e];}
  else if(mode==='throttled'){ if(rec&&evals.length>=11&&Math.min(...evals.slice(-10))>=Math.min(...evals.slice(0,-10))*0.98)mode='idle'; }
  else if(mode==='idle'&&e>bestErr*1.1){mode='throttled';evals=[e];}
  st.mode=mode; st.rateMs=paused?null:RATE[mode]; }
 function evalVal(rec){ if(!VX)return; const n=Math.min(vn,600); if(n<60)return; let e8=0,e12=0,b8=0,b12=0;
  for(let k=0;k<n;k++){ const y=fwd(VX,k*IN), o=k*OUT, vx=-VX[k*IN], vy=-VX[k*IN+1];
   const c=v=>Math.min(500,v*200);
   e8+=c(Math.hypot(y[8]-VY[o+8],y[9]-VY[o+9])); e12+=c(Math.hypot(y[10]-VY[o+10],y[11]-VY[o+11]));
   b8+=c(Math.hypot(vx*8-VY[o+8],vy*8-VY[o+9])); b12+=c(Math.hypot(vx*12-VY[o+10],vy*12-VY[o+11])); }
  st.err096=+(e8/n).toFixed(1); st.err144=+(e12/n).toFixed(1); st.base096=+(b8/n).toFixed(1); st.base144=+(b12/n).toFixed(1); st.valN=n; ctl(rec); }
 const nParams=()=>P.reduce((a,p)=>a+p.W.length+p.b.length,0);
 self.onmessage=e=>{ const m=e.data;
  if(m.t==='init'){ train=!!m.train; if(m.P&&okShape(m.P))P=m.P; else init(); if(train){M=zeros();V=zeros();G=zeros();BX=new Float32Array(CAP*IN);BY=new Float32Array(CAP*OUT);VX=new Float32Array(VCAP*IN);VY=new Float32Array(VCAP*OUT);} if(m.stats)Object.assign(st,m.stats); self.postMessage({t:'ready',params:nParams()}); }
  else if(m.t==='pause'){ paused=!!m.on; st.paused=paused; st.rateMs=paused?null:RATE[mode]; }
  else if(m.t==='setP'){ if(m.P&&!okShape(m.P)){} else if(m.P&&P){ for(let l=0;l<NL;l++){P[l].W.set(m.P[l].W);P[l].b.set(m.P[l].b);} } else if(m.P){P=m.P;if(train){M=zeros();V=zeros();G=zeros();}} if(m.stats)Object.assign(st,m.stats); }
  else if(m.t==='data'){ if(!BX)return; for(let k=0;k<m.n;k++){ const x=m.X.subarray(k*IN,(k+1)*IN), y=m.Y.subarray(k*OUT,(k+1)*OUT);
    if(rnd()<0.08){VX.set(x,vh*IN);VY.set(y,vh*OUT);vh=(vh+1)%VCAP;vn=Math.min(VCAP,vn+1);} else {BX.set(x,bh*IN);BY.set(y,bh*OUT);bh=(bh+1)%CAP;bn=Math.min(CAP,bn+1);} st.seen++; } }
  else if(m.t==='predict'){ if(!P)return; const Y=new Float32Array(m.n*OUT); for(let k=0;k<m.n;k++)Y.set(fwd(m.X,k*IN),k*OUT); self.postMessage({t:'pred',meta:m.meta,Y},[Y.buffer]); }
  else if(m.t==='get'){ evalVal(false); self.postMessage({t:'weights',P:P.map(p=>({W:p.W.slice(),b:p.b.slice()})),stats:{...st,params:nParams()}}); }
  else if(m.t==='stats'){ self.postMessage({t:'stats',stats:{...st,params:P?nParams():0,buf:bn}}); } };
 function loop(){ const go=train&&P&&bn>=400&&!paused; if(go){ const t0=Date.now(); const l=step(16); st.steps++; st.loss=st.loss==null?l:st.loss*0.98+l*0.02; st.msPerSample=+((Date.now()-t0)/16).toFixed(2); if(++sinceEval>=EVERY[mode]){sinceEval=0;evalVal(true);} } setTimeout(loop,go?RATE[mode]:250); }
 loop();
}
function makeWorker(){
 try{ if(typeof Worker!=='undefined'&&typeof Blob!=='undefined'&&typeof URL!=='undefined'&&URL.createObjectURL){
   const src='const LAYERS='+JSON.stringify(LAYERS)+',IN='+IN+',OUT='+OUT+';('+enemyWorker.toString()+')(self);';
   const u=URL.createObjectURL(new Blob([src],{type:'text/javascript'})); if(u)return new Worker(u); } }catch(e){}
 // in-process fallback (offline test only: it would stall a real game)
 const w={inproc:true,onmessage:null}, inner={onmessage:null,postMessage:d=>setTimeout(()=>w.onmessage&&w.onmessage({data:d}),0)};
 w.postMessage=d=>setTimeout(()=>inner.onmessage&&inner.onmessage({data:d}),0); w.terminate=()=>{}; enemyWorker(inner); return w; }

const V8=window.V8={IN,OUT,HZ,LAYERS,role:null,hist:new Map(),pend:[],pred:new Map(),ph:new Map(),err:new Map(),circ:new Set(),temper:new Map(),_orbs:[],
 meV:185,ready:false,trusted:false,stats:null,PXk:new Float64Array(16),PYk:new Float64Array(16),_A:[],_byId:new Map(),out:[],tabs:new Map(),id:Math.random().toString(36).slice(2,8)};

// ================= behaviour features for one enemy =================
V8.context=function(s){ const xs=[],ys=[],own=[]; for(const o of slithers){ if(!o)continue; if(Math.abs(o.xx-s.xx)>2500||Math.abs(o.yy-s.yy)>2500)continue;
  xs.push(o.xx);ys.push(o.yy);own.push(o.id); const pts=o.pts||[]; for(let i=0;i<pts.length;i+=2){const p=pts[i]; if(!p||p.dying)continue; if(Math.abs(p.xx-s.xx)>2500||Math.abs(p.yy-s.yy)>2500)continue; xs.push(p.xx);ys.push(p.yy);own.push(o.id);} }
 const food=[]; for(const f of foods){ if(f&&Math.abs(f.xx-s.xx)<2300&&Math.abs(f.yy-s.yy)<2300)food.push(f); } return {x:xs,y:ys,own,n:xs.length,food}; };
V8.encode=function(o,H,s,ctx,X,off){
 const n=H.length, cur=H[n-1], ca=Math.cos(-cur.a), sa=Math.sin(-cur.a);
 for(let k=1;k<=16;k++){ const i=n-1-k; if(i<0)break; const e=H[i], dx=e.x-cur.x, dy=e.y-cur.y;
  X[off+2*(k-1)]=(dx*ca-dy*sa)/200; X[off+2*(k-1)+1]=(dx*sa+dy*ca)/200; X[off+32+k-1]=wrap(H[i+1].a-e.a); X[off+48+k-1]=H[i+1].sp/14; }
 X[off+64]=cur.sc/4; X[off+65]=Math.min(16,n-1)/16;
 const mx=s.xx-cur.x,my=s.yy-cur.y; X[off+66]=Math.max(-1.5,Math.min(1.5,(mx*ca-my*sa)/1000)); X[off+67]=Math.max(-1.5,Math.min(1.5,(mx*sa+my*ca)/1000));
 const ra=wrap(s.ang-cur.a); X[off+68]=Math.cos(ra); X[off+69]=Math.sin(ra); X[off+70]=Math.min(3,s.sc/cur.sc)/3; X[off+71]=(s.sp||5.78)/14;
 const sec=(dx,dy)=>{const rx=dx*ca-dy*sa, ry=dx*sa+dy*ca; return ((Math.floor((Math.atan2(ry,rx)+Math.PI)/(2*Math.PI)*12))%12+12)%12;};
 for(let i=0;i<ctx.n;i++){ const dx=ctx.x[i]-cur.x,dy=ctx.y[i]-cur.y; if(dx>800||dx<-800||dy>800||dy<-800)continue; const d=Math.hypot(dx,dy); if(d>=800||(ctx.own[i]===o.id&&d<150))continue; const j=sec(dx,dy),p=1-d/800; if(p>X[off+72+j])X[off+72+j]=p; }
 const fm=new Float32Array(12); for(const f of ctx.food){const dx=f.xx-cur.x,dy=f.yy-cur.y; if(dx>700||dx<-700||dy>700||dy<-700||dx*dx+dy*dy>490000)continue; fm[sec(dx,dy)]+=(f.sz||1)*(f.sz||1)/5;}
 for(let j=0;j<12;j++)X[off+84+j]=Math.min(1,Math.log1p(fm[j])/5);
 const R=window.__wallR(), ex=cur.x-grd, ey=cur.y-grd, ed=Math.hypot(ex,ey)||1, wp=Math.max(0,Math.min(1,1-(R-ed)/2000));
 X[off+96]=wp; X[off+97]=(ex/ed*ca-ey/ed*sa)*wp; X[off+98]=(ex/ed*sa+ey/ed*ca)*wp;
 let S=0; for(let k=Math.max(1,n-16);k<n;k++)S+=wrap(H[k].a-H[k-1].a); X[off+99]=Math.max(-1.5,Math.min(1.5,S/(2*Math.PI)));
 const tp=V8.temper.get(o.id); if(tp){X[off+100]=tp.greed;X[off+101]=tp.fear;X[off+102]=tp.aggr;X[off+103]=tp.panic;} };

// ================= enemy temperament: running averages (0..1) of how each snake behaves =================
// greed: heading points at the nearest big orb within 600.  fear: turns away / boosts when a bigger head (ours too) closes in within 500.
// aggr: boosting along a line toward another snake's head path within 500.  panic: erratic heading + boosting.
const ZT={greed:0,fear:0,aggr:0,panic:0,n:0};
V8.tp=id=>V8.temper.get(id)||null;
V8.tsum=()=>{ const r={greed:0,fear:0,aggr:0,panic:0}; for(const t of V8.temper.values()){ if(t.greed>0.6)r.greed++; if(t.fear>0.6)r.fear++; if(t.aggr>0.6)r.aggr++; if(t.panic>0.6)r.panic++; } return r; };
V8.updTemper=function(o,H,s,orbs){ let T=V8.temper.get(o.id); if(!T){T={greed:0,fear:0,aggr:0,panic:0,n:0};V8.temper.set(o.id,T);}
 const A=0.12, ema=(k,v)=>{T[k]+=(v-T[k])*A;}, boost=(o.sp||0)>9; T.n++;
 let bo=null,bd=600; for(const f of orbs){ const dx=f.xx-o.xx,dy=f.yy-o.yy; if(dx>600||dx<-600||dy>600||dy<-600)continue; const d=Math.hypot(dx,dy); if(d<bd){bd=d;bo=f;} }
 if(bo)ema('greed',Math.cos(o.ang-Math.atan2(bo.yy-o.yy,bo.xx-o.xx))>0.85?1:0);
 let bg=null,bgd=500,near=false,ah=null,ahd=500;
 for(const b of slithers){ if(!b||b===o||b.dead)continue; const dx=b.xx-o.xx,dy=b.yy-o.yy; if(dx>500||dx<-500||dy>500||dy<-500)continue; const d=Math.hypot(dx,dy); if(d>=500)continue; near=true;
  if(d<ahd){ const px=b.xx+Math.cos(b.ang)*150,py=b.yy+Math.sin(b.ang)*150; ahd=d; ah={x:px,y:py}; }   // aim point: ahead of that snake's head
  if(b.sc>o.sc*1.05&&d<bgd&&Math.cos(b.ang-Math.atan2(-dy,-dx))>0.3){bgd=d;bg=b;} }
 if(bg&&H.length>=2){ const A0=Math.atan2(o.yy-bg.yy,o.xx-bg.xx), before=Math.abs(wrap(H[H.length-2].a-A0)), now=Math.abs(wrap(o.ang-A0)); ema('fear',(now<before-0.02||(boost&&now<1.2))?1:0); }
 if(near&&ah)ema('aggr',(boost&&Math.cos(o.ang-Math.atan2(ah.y-o.yy,ah.x-o.xx))>0.9)?1:0);
 if(H.length>=8){ let m=0,q=0; const n=7; for(let k=H.length-n;k<H.length;k++)m+=wrap(H[k].a-H[k-1].a); m/=n; for(let k=H.length-n;k<H.length;k++){const d=wrap(H[k].a-H[k-1].a)-m; q+=d*d;} ema('panic',(q/n>0.03&&boost)?1:0); } };

// ================= every 120 ms: track snakes, make training pairs, ask for predictions =================
V8.sample=function(){
 const s=window.slither, now=Date.now(); if(!s){ if(V8.hist.size){V8.hist.clear();V8.pend=[];V8.pred.clear();V8.ph.clear();V8.err.clear();V8.circ.clear();V8.temper.clear();} V8.me=[]; V8.pauseCheck(false,now); return; }
 V8.pauseCheck(true,now);
 const mh=V8.me||(V8.me=[]); mh.push({x:s.xx,y:s.yy,t:now}); if(mh.length>4)mh.shift();
 if(mh.length>=3){const a=mh[0],b=mh[mh.length-1],v=Math.hypot(b.x-a.x,b.y-a.y)/(((b.t-a.t)/1000)||1); if(v>40&&v<1500)V8.meV=V8.meV*0.5+v*0.5;}
 const seen=new Set(), near=[], orbs=V8._orbs; orbs.length=0; for(const f of foods){ if(f&&f.sz>=8&&Math.abs(f.xx-s.xx)<2500&&Math.abs(f.yy-s.yy)<2500){orbs.push(f); if(orbs.length>=250)break;} }
 for(const o of slithers){ if(!o||o===s||o.id==null)continue; seen.add(o.id); let H=V8.hist.get(o.id); if(!H){H=[];V8.hist.set(o.id,H);}
  const last=H[H.length-1], c=last?last.c+1:0;
  if(last&&(Math.hypot(o.xx-last.x,o.yy-last.y)>160||now-last.t>320)){ H.length=0; V8.pend=V8.pend.filter(p=>p.id!==o.id); V8.jumps=(V8.jumps||0)+1; }
  H.push({c,x:o.xx,y:o.yy,a:o.ang,sp:o.sp||5.78,sc:o.sc||1,t:now}); if(H.length>40)H.shift();
  if(H.length>=30){ let S=0,pos=0; for(let k=H.length-29;k<H.length;k++){const d=wrap(H[k].a-H[k-1].a); S+=d; if(d*Math.sign(S||1)>0)pos++;} if(Math.abs(S)>1.6*Math.PI&&pos/29>0.8)V8.circ.add(o.id); else V8.circ.delete(o.id); }
  V8.updTemper(o,H,s,orbs);
  const d=Math.hypot(o.xx-s.xx,o.yy-s.yy); if(d<1700&&H.length>=4)near.push({o,H,d}); }
 for(const id of [...V8.hist.keys()])if(!seen.has(id)){V8.hist.delete(id);V8.pred.delete(id);V8.ph.delete(id);V8.err.delete(id);V8.circ.delete(id);V8.temper.delete(id);}
 near.sort((a,b)=>a.d-b.d); const use=near.slice(0,14);
 if(use.length){ const ctx=V8.context(s), X=new Float32Array(use.length*IN), meta=[];
  use.forEach((e,k)=>{ V8.encode(e.o,e.H,s,ctx,X,k*IN); const c=e.H[e.H.length-1]; meta.push({id:e.o.id,c:c.c,x:c.x,y:c.y,a:c.a,t:now}); V8.pend.push({id:e.o.id,c:c.c,x:c.x,y:c.y,a:c.a,t:now,X:X.slice(k*IN,(k+1)*IN)}); });
  const np=use.filter(e=>e.d<1500).slice(0,8).length;   // predictions only where the planner can use them
  if(np&&V8.ready&&V8.w)V8.w.postMessage({t:'predict',n:np,X:X.slice(0,np*IN),meta:meta.slice(0,np)}); }
 // resolve training pairs whose future (1.44 s) is now known
 const keep=[]; for(const p of V8.pend){ const H=V8.hist.get(p.id); if(!H||now-p.t>4000)continue; const last=H[H.length-1]; if(last.c<p.c+12){keep.push(p);continue;}
  const Y=new Float32Array(OUT), ca=Math.cos(-p.a), sa=Math.sin(-p.a); let ok=true;
  for(let h=0;h<HZ.length;h++){ const e=H.find(q=>q.c===p.c+HZ[h]); if(!e||e.t<=p.t){ok=false;break;} const dx=e.x-p.x,dy=e.y-p.y; Y[2*h]=(dx*ca-dy*sa)/200; Y[2*h+1]=(dx*sa+dy*ca)/200; }
  if(!ok)continue; V8.out.push(p.X,Y);
  const ph=V8.ph.get(p.id), py=ph&&ph.get(p.c); if(py){ const er=Math.hypot(py[8]-Y[8],py[9]-Y[9])*200; V8.err.set(p.id,V8.err.has(p.id)?V8.err.get(p.id)*0.7+er*0.3:er); } }
 V8.pend=keep;
 if(V8.out.length>=40||(V8.out.length&&now-(V8.lastSend||0)>1000)){ const n=V8.out.length/2, X=new Float32Array(n*IN), Y=new Float32Array(n*OUT);
  for(let k=0;k<n;k++){X.set(V8.out[2*k],k*IN);Y.set(V8.out[2*k+1],k*OUT);} V8.out=[]; V8.lastSend=now; V8.sink&&V8.sink(X,Y,n); }
};
// training pauses while we have had no snake for > 5 s (a hidden tab that is still playing is NOT a reason to pause)
V8._noSnake=0; V8.paused=false;
V8.pauseCheck=function(alive,now){ if(alive){V8._noSnake=0; if(V8.paused)V8.setPause(false);} else { if(!V8._noSnake)V8._noSnake=now; if(!V8.paused&&now-V8._noSnake>5000)V8.setPause(true); } };
V8.setPause=function(on){ V8.paused=!!on; if(V8.role==='solo'&&V8.w)V8.w.postMessage({t:'pause',on:!!on}); else if(V8.role==='hub'&&V8.trainers)V8.trainers.forEach(w=>w.postMessage({t:'pause',on:!!on})); };
V8.onPred=function(m){ for(let k=0;k<m.meta.length;k++){ const mt=m.meta[k], Y=m.Y.slice(k*OUT,(k+1)*OUT);
 V8.pred.set(mt.id,{t:mt.t,x:mt.x,y:mt.y,a:mt.a,Y}); let ph=V8.ph.get(mt.id); if(!ph){ph=new Map();V8.ph.set(mt.id,ph);} ph.set(mt.c,Y); if(ph.size>30)ph.delete(ph.keys().next().value); } };

// ================= planner helpers (called directly from V6's __botStep6 when V8 is loaded) =================
V8.at0=(T,PX,PY,tau)=>{ if(tau<T[0]||tau>T[6])return false; let i=0; while(i<5&&T[i+1]<tau)i++; const f=(tau-T[i])/((T[i+1]-T[i])||1); V8._x=PX[i]+(PX[i+1]-PX[i])*f; V8._y=PY[i]+(PY[i+1]-PY[i])*f; return true; };
V8.at=(e,tau)=>V8.at0(e.T,e.PX,e.PY,tau);
V8.active=function(s,myW){ const A=V8._A; A.length=0; V8.tickMaxKill=0; if(!V8.trusted||!V8.pred.size)return A; const now=Date.now(), byId=V8._byId; byId.clear();
 for(const o of slithers)if(o&&o!==s)byId.set(o.id,o);
 for(const [id,p] of V8.pred){ const age=(now-p.t)/1000; if(age>0.5)continue; const o=byId.get(id); if(!o||o.dead||Math.abs(o.xx-s.xx)>1500||Math.abs(o.yy-s.yy)>1500)continue;
  const ca=Math.cos(p.a),sa=Math.sin(p.a), T=[-age],PX=[p.x],PY=[p.y];
  for(let h=0;h<6;h++){const ox=p.Y[2*h]*200,oy=p.Y[2*h+1]*200; T.push(HZ[h]*DT/1000-age); PX.push(p.x+ox*ca-oy*sa); PY.push(p.y+ox*sa+oy*ca);}
  const Dn=[]; for(let tt=0.12;tt<=T[6]+1e-6;tt+=0.12){ if(V8.at0(T,PX,PY,tt))Dn.push(tt,V8._x,V8._y); }
  const tp=V8.temper.get(id)||ZT, ln=V6.lenOf(o), tr=V6.turnRadius(o.sc||1,o.sp>0?o.sp:V6.normSp(o.sc||1),o);
  const err=V8.err.has(id)?V8.err.get(id):220, circ=V8.circ.has(id); let conf=Math.exp(-err/150); if(circ)conf=Math.min(1,conf*1.5+0.2);
  conf*=1-0.5*tp.panic;   // a panicking snake is less predictable
  const lenN=Math.min(1,Math.log1p(ln.len/150)/Math.log1p(20));   // log-scaled victim length (3000 -> 1): more length = more loot
  const e={id,T,PX,PY,D:Dn,hw:29*(o.sc||1)*0.5+myW*0.5+20+8/(1+tr/100),kw:29*(o.sc||1)*0.5+myW*0.5,conf,circ,len:ln.len,tr,greed:tp.greed,line:null,
   val:Math.min(3,0.4+(o.sc||1)/(s.sc||1)+0.6*lenN)};   // hw: a tighter-turning (small) snake can swerve more, so a slightly wider margin
  if(tp.greed>0.6){ let bf=null,bd=600; for(const f of V8._orbs){ const dx=f.xx-o.xx,dy=f.yy-o.yy; if(dx>600||dx<-600||dy>600||dy<-600)continue; const d=Math.hypot(dx,dy); if(d<bd){bd=d;bf=f;} } if(bf)e.line={x1:o.xx,y1:o.yy,x2:bf.xx,y2:bf.yy,tt:bd/(V8.meV||185)+0.2}; }
  A.push(e); }
 return A; };
V8.kill=function(A,PXk,PYk,STEP){ let best=0,bs=9,bl=0; const v=V8.meV;
 for(const e of A){ const Dn=e.D; let hit=false,slack=9;
  for(let j=0;j<Dn.length&&!hit;j+=3){ const th=Dn[j],ex=Dn[j+1],ey=Dn[j+2];
   for(let k=1;k<=14;k++){ const tk=k*STEP/v; if(tk+0.2>th)break; const dx=PXk[k]-ex,dy=PYk[k]-ey; if(dx*dx+dy*dy<e.kw*e.kw){hit=true;slack=th-tk;break;} } }
  let sc=0; if(hit)sc=e.val*e.conf*(e.circ?1.5:1);
  else if(e.line){ // a greedy snake is heading for loot: our path crossing the line between it and the orb, before it gets there, counts partly
   const L=e.line,lx=L.x2-L.x1,ly=L.y2-L.y1,l2=lx*lx+ly*ly||1, r=e.kw+20;
   for(let k=1;k<=14;k++){ const tk=k*STEP/v; if(tk>L.tt)break; let u=((PXk[k]-L.x1)*lx+(PYk[k]-L.y1)*ly)/l2; u=u<0?0:u>1?1:u; const dx=PXk[k]-(L.x1+lx*u),dy=PYk[k]-(L.y1+ly*u); if(dx*dx+dy*dy<r*r){sc=e.val*e.conf*0.6*e.greed;break;} } }
  if(sc>best){best=sc;bs=slack;bl=e.len;} }
 V8._ks=bs; V8._kLen=bl; if(best>V8.tickMaxKill)V8.tickMaxKill=best; return best; };

// ================= minimap: pick a region to roam toward =================
V8.mapGoal=function(s,L){ const now=Date.now(); if(V8._mg&&now-V8._mg.t<1000)return V8._mg.a;
 let best=null,bs=0; const N=136, d=window.mmdata, R=window.__wallR(); if(!d||d.length!==N*N){V8._mg={t:now,a:null};return null;}
 const cell=grd/68, want=L<500?1:L<2000?3:5;
 for(let cy=2;cy<N-2;cy+=2)for(let cx=2;cx<N-2;cx+=2){ const wx=grd+(cx+0.5-68)*cell, wy=grd+(cy+0.5-68)*cell; if(Math.hypot(wx-grd,wy-grd)>R-1800)continue;
  const dist=Math.hypot(wx-s.xx,wy-s.yy); if(dist<900||dist>9000)continue; let occ=0; for(let y=cy-2;y<=cy+2;y++)for(let x=cx-2;x<=cx+2;x++)occ+=d[y*N+x]?1:0;
  const sc=Math.exp(-Math.abs(occ-want)/2)/(1+dist/4000); if(sc>bs){bs=sc;best=Math.atan2(wy-s.yy,wx-s.xx);} }
 V8._mg={t:now,a:best}; return best; };

// ================= config defaults for the hooks the planner calls =================
if(V6.cfg.killW==null)V6.cfg.killW=500; if(V6.cfg.mapW==null)V6.cfg.mapW=180;
// loot: the trail of big orbs a snake leaves when it dies. It outranks hunting, coiling, roaming and plain food;
// only escaping an enclosure comes first. Straight after one of our kills even a small trail is worth going back for.
for(const [k,v] of Object.entries({lootSz:8,lootR:1400,lootMin:150,lootKillMin:40,lootWMax:900}))if(V6.cfg[k]==null)V6.cfg[k]=v;
V6.base.lootWMax=V6.base.lootWMax??900;
V8.lootUntil=0; V8._kc=0;
// big orbs are bucketed into a 300-unit grid, so each orb sums only its neighbours (was an O(n^2) pair loop); result cached 160 ms
V8.loot=function(s,hx,hy,R,wallM,C){ const now=Date.now(), kc=s.kill_count||0; const killed=kc>V8._kc; if(killed)V8.lootUntil=now+10000; V8._kc=kc;
 const c=V8._lc; if(!killed&&c&&c.foods===foods&&c.sl===slithers&&c.s===s&&now-c.t<160&&c.win===(now<V8.lootUntil)){ return c.r?{x:c.r.x,y:c.r.y,v:c.r.v,d:Math.hypot(c.r.x-hx,c.r.y-hy),rival:c.r.rival}:null; }
 const r=V8._loot(s,hx,hy,R,wallM,C,now); V8._lc={foods,sl:slithers,s,t:now,win:now<V8.lootUntil,r}; return r; };
V8._loot=function(s,hx,hy,R,wallM,C,now){
 const big=[]; for(const f of foods){ if(!f||(f.sz||0)<C.lootSz)continue; const dx=f.xx-hx,dy=f.yy-hy; if(dx>C.lootR||dx<-C.lootR||dy>C.lootR||dy<-C.lootR)continue;
  if(R-Math.hypot(f.xx-grd,f.yy-grd)<wallM+150)continue; big.push(f); if(big.length>=300)break; }
 if(big.length<2)return null;
 const G=new Map(), key=(cx,cy)=>(cx+8)*1024+(cy+8); for(const f of big){ const k=key(Math.floor(f.xx/300),Math.floor(f.yy/300)); const a=G.get(k); if(a)a.push(f); else G.set(k,[f]); }
 const th=[],all=[]; for(const o of slithers){ if(!o||o===s||o.dead)continue; all.push(o); if(o.sc>=s.sc*0.95)th.push(o); }
 const min=now<V8.lootUntil?C.lootKillMin:C.lootMin; let best=null,bs=0;
 for(const f of big){ let v=0; const cx=Math.floor(f.xx/300),cy=Math.floor(f.yy/300);
  for(let i=-1;i<=1;i++)for(let j=-1;j<=1;j++){ const a=G.get(key(cx+i,cy+j)); if(!a)continue; for(const g of a){ if(Math.abs(g.xx-f.xx)+Math.abs(g.yy-f.yy)<300)v+=g.sz*g.sz/5; } }
  if(v<min)continue;
  let danger=false; for(const o of th){ if(Math.hypot(o.xx-f.xx,o.yy-f.yy)<220){danger=true;break;} } if(danger)continue;
  const d=Math.hypot(f.xx-hx,f.yy-hy), sc=v/(1+d/400); if(sc>bs){ let rival=false; for(const o of all){ if(Math.hypot(o.xx-f.xx,o.yy-f.yy)<Math.min(d,600)){rival=true;break;} } bs=sc; best={x:f.xx,y:f.yy,v,d,rival}; } }
 return best; };
V6.base.killW=V6.base.killW??500; V6.base.mapW=V6.base.mapW??180;
for(const k of ['killW','mapW','lootWMax'])if(!V7.TUNE.includes(k)){V7.TUNE.push(k); if(V7.es.mu[k]==null)V7.es.mu[k]=0;}

// ================= tick driver that background tabs don't throttle =================
// Timers on a hidden tab's main thread are slowed to once a second (or once a minute after 5 minutes), and the
// game's own loop (oef, run by requestAnimationFrame) stops altogether, so steering never reaches the server.
// A worker's timer is not slowed, so the bot ticks from one, and in a hidden tab it also runs one game frame per tick.
// raf and redraw are stubbed for that call so it neither queues extra loops nor draws.
const wTimer=(ms,fn)=>{ try{ const w=new Worker(URL.createObjectURL(new Blob(['setInterval(()=>postMessage(0),'+ms+')'],{type:'text/javascript'}))); w.onmessage=()=>{try{fn()}catch(e){}}; return {stop:()=>w.terminate()}; }
 catch(e){ const id=setInterval(fn,ms); return {stop:()=>clearInterval(id)}; } };
V8.wTimer=wTimer;
V8.pump=function(){ if(typeof document==='undefined'||!document.hidden||typeof window.oef!=='function')return; const r=window.raf, d=window.redraw;
 window.raf=()=>{}; window.redraw=()=>{}; try{window.oef()}catch(e){V8.pumpErr=String(e)} finally{window.raf=r; window.redraw=d;} V8.pumps=(V8.pumps||0)+1; };
V8.startTicks=function(){ V8.stopTicks(); window.__mode='bot'; window.__respawnAt=0;
 try{ V8.tw=new Worker(URL.createObjectURL(new Blob(['setInterval(()=>postMessage(0),40)'],{type:'text/javascript'})));
  let n=0; V8.tw.onmessage=()=>{ V8.pump(); try{window.__botStep6()}catch(e){window.__v6err=String(e)} if(++n%3===0){try{V8.sample()}catch(e){V8.err8=String(e)}} if(n%75===0&&V8.onTick3s)V8.onTick3s(); }; }
 catch(e){ window.__bot=setInterval(()=>{try{window.__botStep6()}catch(e){window.__v6err=String(e)}},40); window.__trk8=setInterval(()=>{try{V8.sample()}catch(e){V8.err8=String(e)}},DT); } };
V8.stopTicks=function(){ if(V8.tw){V8.tw.terminate();V8.tw=null;} for(const k of ['__bot','__trk8'])if(window[k]){clearInterval(window[k]);window[k]=null;} };
V8.stop=function(){ V8.stopTicks(); window.__mode='off'; try{setAcceleration(0)}catch(e){} };
const trustCheck=st=>{ V8.stats=st; V8.trusted=!!(st&&st.err096!=null&&st.valN>=200&&st.err096<0.85*st.base096); };

// ================= persistence of the big model (IndexedDB) =================
const idb=(mode,fn)=>new Promise((res,rej)=>{ if(typeof indexedDB==='undefined')return res(null); const q=indexedDB.open('slitherV8',1);
 q.onupgradeneeded=()=>q.result.createObjectStore('m'); q.onerror=()=>rej(q.error);
 q.onsuccess=()=>{ const db=q.result, tx=db.transaction('m',mode), st=tx.objectStore('m'), r=fn(st); tx.oncomplete=()=>{db.close();res(r&&r.result);}; tx.onerror=()=>rej(tx.error); }; });
// key 'enemy9' (the v8 key 'enemy' holds the old 1M-weight shape and is left untouched); a record whose layer sizes differ is ignored
V8.validModel=rec=>!!(rec&&Array.isArray(rec.P)&&rec.P.length===LAYERS.length-1&&rec.P.every((p,l)=>p&&p.W&&p.b&&p.W.length===LAYERS[l]*LAYERS[l+1]&&p.b.length===LAYERS[l+1]));
V8.loadModel=()=>idb('readonly',st=>st.get('enemy9')).then(rec=>{ if(rec&&!V8.validModel(rec)){V8.ignoredModel=true;return null;} return rec||null; }).catch(()=>null);
V8.storeModel=(P,stats)=>idb('readwrite',st=>st.put({P,stats,savedAt:new Date().toISOString()},'enemy9')).catch(e=>{V8.saveErr=String(e);});
V8.exportModel=async function(){ const rec=await V8.loadModel(); if(!rec)return 'no saved model yet';
 const b64=a=>{const u=new Uint8Array(a.buffer,a.byteOffset,a.byteLength);let s='';for(let i=0;i<u.length;i+=8192)s+=String.fromCharCode.apply(null,u.subarray(i,i+8192));return btoa(s);};
 const out={format:'slither-v8-enemy/1',layers:LAYERS,horizonsSec:HZ.map(h=>h*DT/1000),savedAt:rec.savedAt,stats:rec.stats,P:rec.P.map(p=>({W:b64(p.W),b:b64(p.b)}))};
 const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([JSON.stringify(out)],{type:'application/json'})); a.download='enemy-model.json'; document.body.appendChild(a); a.click(); a.remove(); return 'downloaded'; };

// ================= roles =================
const CH='slither-v8';
function packLife(pk){ const n=pk.life.length, F=new Float32Array(n*NF), b=new Uint8Array(n), bo=new Uint8Array(n), yd=new Float32Array(n), w=new Float32Array(n);
 pk.life.forEach((q,i)=>{F.set(q.F,i*NF);b[i]=q.b;bo[i]=q.bo;yd[i]=q.yd;w[i]=q.w;}); return {n,F,b,bo,yd,w,len:pk.len,kills:pk.kills,secs:pk.secs,eps:pk.eps,sign:pk.sign,cfg:pk.cfg}; }
function unpackLife(m){ const life=[]; for(let i=0;i<m.n;i++)life.push({F:m.F.slice(i*NF,(i+1)*NF),b:m.b[i],bo:m.bo[i],yd:m.yd[i],w:m.w[i]}); return {life,len:m.len,kills:m.kills,secs:m.secs,eps:m.eps,sign:m.sign,cfg:m.cfg,from:m.from}; }
const v7msg=()=>({k:'v7',P:V7.P,mu:{...V7.es.mu},sigma:V7.es.sigma,nDeaths:V7.nDeaths,policyReady:V7.policyReady});

// a tab that plays: its own small worker for predictions, sends data/lives to the hub
V8.bot=function(){ V8.role='bot'; V7.save=()=>false; const bc=V8.bc=new BroadcastChannel(CH);   // only the hub writes the saved V7 state
 V8.w=makeWorker(); V8.w.onmessage=e=>{const m=e.data; if(m.t==='pred')V8.onPred(m); else if(m.t==='ready')V8.ready=true;};
 V8.sink=(X,Y,n)=>bc.postMessage({k:'em',from:V8.id,X,Y,n});
 V7.remote=pk=>{ const m=packLife(pk); m.k='life'; m.from=V8.id; bc.postMessage(m); };
 bc.onmessage=e=>{ const m=e.data;
  if(m.k==='v7'){ V7.P=m.P; Object.assign(V7.es.mu,m.mu); V7.es.sigma=m.sigma; V7.nDeaths=m.nDeaths; V7.policyReady=m.policyReady; }
  else if(m.k==='emP'){ if(!V8.wInit){V8.w.postMessage({t:'init',P:m.P,train:false});V8.wInit=true;} else V8.w.postMessage({t:'setP',P:m.P}); trustCheck(m.stats); }
  else if(m.k==='hubHello')bc.postMessage({k:'hello',from:V8.id}); };
 bc.postMessage({k:'hello',from:V8.id});
 V8.onTick3s=()=>{ const s=window.slither; bc.postMessage({k:'stat',from:V8.id,alive:!!s,L:window.__dbg&&__dbg.L,kills:s&&s.kill_count,mode:window.__dbg&&__dbg.mode,why:window.__dbg&&__dbg.why,meV:Math.round(V8.meV),hidden:document.hidden,pumps:V8.pumps||0,err:window.__v6err||V8.err8||V8.pumpErr||null,t:Date.now()}); };
 V8.startTicks(); return 'bot '+V8.id; };

// the hub: trains (3 threads, averaged), runs the population ES, sends weights out; does not play
V8.hub=async function(nTrainers=3){ V8.role='hub'; V8.stop(); const bc=V8.bc=new BroadcastChannel(CH);
 const rec=await V8.loadModel(); V8.src=rec?'indexedDB '+rec.savedAt:'fresh';
 V8.trainers=[]; for(let i=0;i<nTrainers;i++){ const w=makeWorker(); w.onmessage=e=>V8.onTrainerMsg(i,e.data); w.postMessage({t:'init',P:rec&&rec.P?rec.P.map(p=>({W:p.W.slice(),b:p.b.slice()})):null,train:true});   /* accuracy is measured afresh, never carried over */ V8.trainers.push(w); }
 V8._rr=0; V8.got=[];
 V8.onTrainerMsg=(i,m)=>{ if(m.t==='ready'&&i===0){V8.params=m.params; if(!rec)V8.sync(true);} else if(m.t==='weights'){ V8.got[i]=m; if(V8._first||V8.got.filter(Boolean).length===V8.trainers.length)V8.average(); } else if(m.t==='stats'&&i===0)V8.stats=m.stats; };
 V8.sync=(first)=>{ V8.got=[]; if(first){ V8.trainers[0].postMessage({t:'get'}); V8._first=true; return; } V8.trainers.forEach(w=>w.postMessage({t:'get'})); };
 V8.average=()=>{ const G=V8.got, n=G.length; let P;
  if(V8._first){ P=G[0].P; V8._first=false; } else { P=G[0].P; for(let l=0;l<P.length;l++)for(const key of ['W','b']){const a=P[l][key]; for(let k=1;k<n;k++){const b=G[k].P[l][key]; for(let i=0;i<a.length;i++)a[i]+=b[i];} for(let i=0;i<a.length;i++)a[i]/=n;} }
  const stats={...G[0].stats,seen:G.reduce((a,g)=>a+(g&&g.stats.seen||0),0)}; trustCheck(stats); V8.got=[];
  V8.trainers.forEach(w=>w.postMessage({t:'setP',P:P.map(p=>({W:p.W.slice(),b:p.b.slice()}))}));
  V8.lastP=P; V8.syncs=(V8.syncs||0)+1; if(V8.syncs%3===1)bc.postMessage({k:'emP',P,stats});   // to the bot tabs once a minute
  if(V8.syncs%3===0)V8.storeModel(P,stats).then(()=>{V8.savedAt=new Date().toISOString();}); };
 V8._syncT=wTimer(20000,()=>V8.sync(false));
 V8._pzT=wTimer(3000,()=>{ const now=Date.now(), any=[...V8.tabs.values()].some(t=>t.from&&t.alive&&now-t.t<15000); V8.pauseCheck(any,now); });   // no playing tab for > 5 s: pause training
 bc.onmessage=e=>{ const m=e.data;
  if(m.k==='em'){ V8.trainers[V8._rr++%V8.trainers.length].postMessage({t:'data',X:m.X,Y:m.Y,n:m.n}); V8.emIn=(V8.emIn||0)+m.n; }
  else if(m.k==='life'){ const pk=unpackLife(m); V7.es.popN=Math.max(2,Math.min(20,[...V8.tabs.values()].filter(t=>Date.now()-t.t<15000).length)); V7.ingest(pk); }
  else if(m.k==='hello'){ V8.tabs.set(m.from,{...(V8.tabs.get(m.from)||{}),t:Date.now()}); bc.postMessage(v7msg()); if(V8.lastP)bc.postMessage({k:'emP',P:V8.lastP,stats:V8.stats}); }
  else if(m.k==='stat'){ V8.tabs.set(m.from,m); } };
 V7.onLearned=()=>bc.postMessage(v7msg());
 bc.postMessage({k:'hubHello'});
 return 'hub: '+nTrainers+' trainer threads, model '+V8.src; };

// one tab doing everything (plays, trains one thread, learns V7 locally)
V8.solo=async function(){ V8.role='solo'; const rec=await V8.loadModel(); V8.src=rec?'indexedDB '+rec.savedAt:'fresh';
 V8.w=makeWorker(); V8.w.onmessage=e=>{ const m=e.data; if(m.t==='pred')V8.onPred(m); else if(m.t==='ready')V8.ready=true; else if(m.t==='stats')trustCheck(m.stats); else if(m.t==='weights'){trustCheck(m.stats);V8.storeModel(m.P,m.stats);} };
 V8.w.postMessage({t:'init',P:rec&&rec.P,train:true}); V8.sink=(X,Y,n)=>V8.w.postMessage({t:'data',X,Y,n});
 V8._st=wTimer(5000,()=>V8.w.postMessage({t:'stats'})); V8._sv=wTimer(120000,()=>V8.w.postMessage({t:'get'}));
 V8.startTicks(); return 'solo, model '+V8.src; };

V8.report=function(){ const st=V8.stats||{}, now=Date.now(), tabs=[...V8.tabs.values()].filter(t=>t.from&&now-t.t<15000);
 return {role:V8.role,model:{params:V8.params||st.params,trusted:V8.trusted,seen:st.seen,steps:st.steps,err096:st.err096,base096:st.base096,err144:st.err144,base144:st.base144,msPerSample:st.msPerSample,trainMode:st.mode,rateMs:st.rateMs,paused:V8.paused,syncs:V8.syncs,savedAt:V8.savedAt},
  tabs:tabs.length,alive:tabs.filter(t=>t.alive).length,tabL:tabs.map(t=>t.L||0).sort((a,b)=>b-a),modes:tabs.map(t=>t.mode+(t.why?'/'+t.why:'')),errs:tabs.filter(t=>t.err).map(t=>t.err).slice(0,3),
  lives:V7.log.length,last:V7.log.slice(-8).map(r=>r.len+'/'+r.kills+'k'),best:V7.es.best&&{len:V7.es.best.len,kills:V7.es.best.kills},esGen:V7.es.gen,
  mu:Object.fromEntries(Object.entries(V7.es.mu).map(([k,v])=>[k,+v.toFixed(2)]).filter(([k,v])=>v!==0)),deathsSeen:V7.nDeaths,learn:V7.lastLearn}; };
// stop everything this tab runs for V8 (before loading a newer bundle into the same tab)
V8.shutdown=function(){ V8.stop(); for(const k of ['_syncT','_st','_sv','_pzT'])if(V8[k])V8[k].stop(); V8.onTick3s=null; try{V8.bc&&V8.bc.close()}catch(e){} try{V8.w&&V8.w.terminate()}catch(e){} (V8.trainers||[]).forEach(w=>{try{w.terminate()}catch(e){}}); V7.remote=null; V7.onLearned=null; return 'v8 stopped'; };
// GPU: deliberately NOT implemented. A 31.5k-weight net at batch 16 costs more power on WebGPU (upload, dispatch, readback,
// no browser here to test it) than it saves on the CPU, which the duty-cycled trainer now barely uses. See README, "GPU decision".
V8.gpu={enabled:false,reason:'not shipped: tiny model, untestable offline; CPU path only'};
V8.ver='v9.0';
})();
