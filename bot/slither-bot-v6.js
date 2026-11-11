// slither.io bot V6.1 (Claude, 2026-11-08; V6.1 seeks big orbs). Paste into the slither.io page console, then run V6start(); V6stop() stops it.
// Use bot/slither-bot-v6.bundle.js instead if this browser has no saved weights (it embeds model/weights.json).
// Pieces: V5 sensors (16 sectors x 5 channels + 5 globals), a tiny MLP (85->32->32->12) trained on the user's play,
// hand weights, 56-heading lookahead with a safety shield, enclosure/gap escape, rear-head and race avoidance,
// coil/cut-off hunting, crowd-aware food clusters, red-wall avoidance (flux_grd).
(()=>{
const NS=16, NF=NS*5+5, BINS=[-Math.PI,-1.6,-0.9,-0.5,-0.25,-0.1,0.1,0.25,0.5,0.9,1.6,Math.PI], NB=11;
const CENT=[-2.2,-1.2,-0.7,-0.37,-0.17,0,0.17,0.37,0.7,1.2,2.2];
const wrap=a=>((a+Math.PI*3)%(Math.PI*2))-Math.PI;
const binOf=r=>{for(let i=0;i<NB;i++)if(r<BINS[i+1])return i;return NB-1};
window.V5={NS,NF,NB,BINS,CENT,wrap,binOf};
V5.feats=function(s){
 const F=new Float32Array(NF), hx=s.xx,hy=s.yy, ca=Math.cos(-s.ang), sa=Math.sin(-s.ang);
 const sec=(dx,dy)=>{const rx=dx*ca-dy*sa, ry=dx*sa+dy*ca; const a=Math.atan2(ry,rx); let i=Math.floor((a+Math.PI)/(2*Math.PI)*NS); return Math.min(NS-1,Math.max(0,i));};
 const prox=d=>Math.max(0,Math.min(1,1-d/1000));
 for(const o of slithers){ if(!o||o===s)continue; const w=29*(o.sc||1); const bigger=o.sc>=s.sc*0.95;
   let d=Math.hypot(o.xx-hx,o.yy-hy)-w; if(d<1000){const i=sec(o.xx-hx,o.yy-hy),p=prox(d); F[NS*2+i]=Math.max(F[NS*2+i],p); if(!bigger)F[NS*3+i]=Math.max(F[NS*3+i],p); F[i]=Math.max(F[i],p); if(bigger)F[NS+i]=Math.max(F[NS+i],p);}
   for(const q of (o.pts||[])){ if(q.dying)continue; const dx=q.xx-hx,dy=q.yy-hy; if(Math.abs(dx)>1000||Math.abs(dy)>1000)continue; d=Math.hypot(dx,dy)-w; if(d>=1000)continue; const i=sec(dx,dy),p=prox(d); if(p>F[i])F[i]=p; if(bigger&&p>F[NS+i])F[NS+i]=p; } }
 const fm=new Float32Array(NS); for(const f of foods){ if(!f)continue; const dx=f.xx-hx,dy=f.yy-hy; if(Math.abs(dx)>900||Math.abs(dy)>900)continue; if(dx*dx+dy*dy>810000)continue; fm[sec(dx,dy)]+=(f.sz||1); }
 for(let i=0;i<NS;i++)F[NS*4+i]=Math.min(1,Math.log1p(fm[i])/6);
 const L=+(document.body.innerText.match(/Your length:\s*(\d+)/)||[0,0])[1];
 const ex=hx-grd,ey=hy-grd,ed=Math.hypot(ex,ey)||1, e=grd*0.98-ed, ep=Math.max(0,1-e/2500);
 const ox=ex/ed,oy=ey/ed, rx=ox*ca-oy*sa, ry=ox*sa+oy*ca;
 F[NS*5]=Math.min(1,Math.log1p(L)/9); F[NS*5+1]=Math.min(1,(s.sp||6)/14); F[NS*5+2]=ep; F[NS*5+3]=rx*ep; F[NS*5+4]=ry*ep;
 return {F,L};
};
// mirror (left-right) for augmentation
V5.mirror=function(F){const G=new Float32Array(NF); for(let c=0;c<5;c++)for(let i=0;i<NS;i++)G[c*NS+i]=F[c*NS+(NS-1-i)]; for(let j=0;j<5;j++)G[NS*5+j]=F[NS*5+j]; G[NS*5+4]=-F[NS*5+4]; return G;};
V5.H=32;
V5.init=function(seed=1){ let r=seed; const rnd=()=>{r=(r*1103515245+12345)%2147483648; return r/2147483648;}; const g=()=>{let u=rnd()||1e-9,v=rnd();return Math.sqrt(-2*Math.log(u))*Math.cos(2*Math.PI*v);};
 const mk=(n,m)=>{const W=new Float32Array(n*m);const s=Math.sqrt(2/n);for(let i=0;i<W.length;i++)W[i]=g()*s;return W;};
 const H=V5.H,NO=NB+1; return {W1:mk(NF,H),b1:new Float32Array(H),W2:mk(H,H),b2:new Float32Array(H),W3:mk(H,NO),b3:new Float32Array(NO)}; };
V5.fwd=function(P,x){ const H=V5.H,NO=NB+1; const h1=new Float32Array(H),h2=new Float32Array(H),o=new Float32Array(NO);
 for(let j=0;j<H;j++){let s=P.b1[j];for(let i=0;i<NF;i++)s+=x[i]*P.W1[i*H+j];h1[j]=s>0?s:0;}
 for(let j=0;j<H;j++){let s=P.b2[j];for(let i=0;i<H;i++)s+=h1[i]*P.W2[i*H+j];h2[j]=s>0?s:0;}
 for(let j=0;j<NO;j++){let s=P.b3[j];for(let i=0;i<H;i++)s+=h2[i]*P.W3[i*NO+j];o[j]=s;}
 let m=-1e9;for(let j=0;j<NB;j++)m=Math.max(m,o[j]); let z=0; const p=new Float32Array(NB); for(let j=0;j<NB;j++){p[j]=Math.exp(o[j]-m);z+=p[j];} for(let j=0;j<NB;j++)p[j]/=z;
 return {h1,h2,o,p,pb:1/(1+Math.exp(-o[NB]))}; };
// speed input is masked (it leaked the boost label during training)
V5._featsRaw=V5.feats; V5.feats=function(s){const r=V5._featsRaw(s); r.F[NS*5+1]=0; return r;};
// weights: window.SLITHER_MODEL (contents of model/weights.json) if set, else this page's localStorage, else none
const unb64=s=>{const b=atob(s),u=new Uint8Array(b.length);for(let i=0;i<b.length;i++)u[i]=b.charCodeAt(i);return new Float32Array(u.buffer);};
V5.P=null;
try{ const M=window.SLITHER_MODEL; if(M&&M.tensors){ V5.P={}; for(const k in M.tensors)V5.P[k]=unb64(M.tensors[k].b64); V5.hand=M.hand; V5.trainInfo=M.trainInfo; V5.src='SLITHER_MODEL'; } }catch(e){ V5.P=null; }
if(!V5.P){ try{ const sv=JSON.parse(localStorage.__v5weights); V5.P={}; for(const k in sv.w)V5.P[k]=Float32Array.from(sv.w[k]); V5.hand=sv.hand; V5.trainInfo=sv.trainInfo; V5.src='localStorage'; }catch(e){ V5.P=null; } }
if(!V5.P){ V5.P=V5.init(7); V5.noWeights=true; }  // fallback: random net, its weight is zeroed below
window.__wallR=()=>Math.min(window.real_flux_grd||1e9, window.flux_grd||1e9, grd*0.98);
window.__botLog=window.__botLog||[];
window.V6={deaths:[],diag:[]};
// cfg notes: bodyRad = factor on 29*sc for enemy BODY points (1 = old behaviour; tune live: V6.cfg.bodyRad=0.7).
// boostK: multiplier on all boost clearance bars (tests/tuning). stakeL: length at which the "stake" factor reaches 2 (stake = 1+min(1,L/stakeL): wider margins, higher boost bar).
// boostCost: value (food units, sz*sz/5) a boost is expected to burn at stake 1; killGain: share of a victim's length counted as loot.
// growK: APPROXIMATE score gained per food-value unit (5 = a normal orb ~ 1 point); only used to project post-meal width.
// coilMinL is legacy (coiling is now gated by body-length geometry), kept so saved V7 state loads.
V6.cfg={netW:50,safeW:2,big:-320,near:-110,prey:130,pathFood:5,wall:-900,
 enclosedR:1300,enclosedFrac:0.45,gapW:650, rearW:350, raceLat:450,raceAng:0.6,raceW:-500,
 cutW:350, coilMinL:1200, coilR:260, coilW:700, clusterBoostMin:250, crowdMax:2, minBoostL:250,
 orbMinSz:8, orbBoostSz:11, orbR:1200, orbBias:1.5, orbWMax:600,
 bodyRad:1, boostK:1, stakeL:5000, boostCost:20, killGain:0.5, growK:0.2, segLen:42, coilTurnK:1.15, coilCircK:1.25};
// ---------- shared helpers (exported so v8 and the tests use the same formulas) ----------
// Game facts: sc=min(6,1+(sct-2)/106); turn per 8 ms frame <= 0.033*scang*spang; world speed = sp/4 units per frame.
V6.scang=sc=>{ const k=(7-Math.max(1,Math.min(6,sc)))/6; return 0.13+0.87*k*k; };
V6.turnRadius=(sc,sp,sn)=>{ const own=sn&&isFinite(sn.scang)&&sn.scang>0, scg=own?sn.scang:V6.scang(sc);
 const spg=(sn&&isFinite(sn.spang)&&sn.spang>0&&Math.abs((sn.sp||0)-sp)<0.05)?sn.spang:Math.min(1,sp/4.8);
 return (sp/4)/(0.033*scg*Math.max(0.05,spg)); };
V6.normSp=sc=>5.39+0.4*sc;
V6.lenOf=o=>{ if(!o)return {len:0,sct:0,exact:false,lowerBound:false}; const fp=window.fpsls,fm=window.fmlts;
 if(fp&&fm&&isFinite(o.sct)&&isFinite(o.fam)&&isFinite(fp[o.sct])&&fm[o.sct]>0)return {len:Math.max(0,Math.floor(15*(fp[o.sct]+o.fam/fm[o.sct]-1)-5)),sct:o.sct,exact:true,lowerBound:false};
 const sc=Math.max(1,Math.min(6,o.sc||1)), sct=2+106*(sc-1); let len=0;   // from width: sc=width/29; fam unknown, assume 0.5
 if(fp&&fm){ const i=Math.max(0,Math.min(fp.length-2,Math.floor(sct))), fr=sct-i, a=fp[i], b=fp[i+1]; if(isFinite(a)&&isFinite(b))len=Math.max(0,Math.floor(15*(a+(b-a)*fr+0.5/(fm[i]||1)-1)-5)); }
 return {len,sct,exact:false,lowerBound:(o.sc||1)>=5.99}; };
V6.bodyLen=o=>{ const p=o&&o.pts; if(p&&p.length>1){ let L=0,px=null,py=null; for(const q of p){ if(!q||q.dying)continue; if(px!==null)L+=Math.hypot(q.xx-px,q.yy-py); px=q.xx;py=q.yy; } if(L>0)return L; }
 return V6.lenOf(o).sct*V6.cfg.segLen; };
// project sct/fam/width/turn radius after eating food of total value `value` (approximate: score += value*growK)
V6.grow=(sct,fam,value,sc0)=>{ const fm=window.fmlts; if(!isFinite(sct)||!isFinite(fam)||!fm)return null; let dL=Math.max(0,value||0)*V6.cfg.growK,n=sct,f=fam,g=0;
 while(dL>0&&g++<400){ const m=fm[n]>0?fm[n]:1, room=(1-f)*15/m; if(dL<room){f+=dL*m/15;dL=0;} else {dL-=room;n++;f=0;} }
 const sc=Math.min(6,1+(n-2)/106); return {sct:n,fam:f,sc,width:29*sc,turnR:V6.turnRadius(sc,V6.normSp(sc))}; };
// self-check only (never steers): fastest turn seen per distance travelled vs the formula
V6.turnFit={n:0,maxCurv:0,measuredR:null,formulaR:null,ratio:null};
V6._fit=(s,formulaR)=>{ const t=Date.now(),p=V6._tf; V6._tf={x:s.xx,y:s.yy,a:s.ang,t}; if(!p)return; const d=Math.hypot(s.xx-p.x,s.yy-p.y);
 if(d<3||d>120||t-p.t>200)return; const F=V6.turnFit,c=Math.abs(wrap(s.ang-p.a))/d; F.maxCurv=Math.max(c,F.maxCurv*0.995); F.n++;
 F.measuredR=F.maxCurv>1e-6?+(1/F.maxCurv).toFixed(1):null; F.formulaR=+formulaR.toFixed(1); F.ratio=F.measuredR?+(F.measuredR/formulaR).toFixed(2):null; };
V6.coil=null; V6._pb=false;
window.__botStep6=function(){
 const s=window.slither, C=V6.cfg, V8=(window.V8&&window.V8.active)?window.V8:null;
 if(!s){ if(window.__bAlive){window.__bAlive=false; const d=V6.diag.filter((_,i)=>i%5===0).slice(-10); window.__botLog.push({len:window.__bMax,secs:Math.round((Date.now()-window.__bStart)/1000),kills:window.__bKills,ver:'v6'}); V6.deaths.push({len:window.__bMax,last:d}); try{setAcceleration(0)}catch(e){}}
   V6.coil=null; V6._tf=null; V6._pb=false;
   if(window.__mode==='bot'){ if(!window.__respawnAt)window.__respawnAt=Date.now()+3000; if(Date.now()>window.__respawnAt){const i=document.querySelector('input');if(i&&!i.value)i.value='Claude';const b=document.getElementById('playh');const c=b&&(b.querySelector('.btnt')||b);if(c&&c.offsetParent)c.click();window.__respawnAt=Date.now()+5000;} } return; }
 window.__respawnAt=0; if(window.__mode!=='bot'||!V5.P)return;
 const {F}=V5.feats(s); const L=V6.lenOf(s).len; const net=V5.fwd(V5.P,F);
 if(!window.__bAlive){window.__bAlive=true;window.__bStart=Date.now();window.__bMax=0;window.__bKills=0;V6.diag=[];}
 if(Date.now()-window.__bStart>2000)window.__bMax=Math.max(window.__bMax,L); window.__bKills=s.kill_count||0;
 const near=F.subarray(0,NS),big=F.subarray(NS,2*NS),prey=F.subarray(3*NS,4*NS),food=F.subarray(4*NS,5*NS);
 const hx=s.xx,hy=s.yy,ha=s.ang,sc1=s.sc||1,myW=29*sc1,boosting=(s.sp||0)>9,NA=56,STEP=50,NSTEP=14,nowT=Date.now();
 // stake: the more length we hold, the more we have to lose
 const stake=1+Math.min(1,Math.max(0,L)/C.stakeL), bigR=1000*(1+0.5*(stake-1));
 // turn radius at the speed we will actually have (boost if we are, or decided to last tick)
 const spN=(s.sp>0&&isFinite(s.sp))?s.sp:V6.normSp(sc1), boostNow=boosting||V6._pb;
 const trN=V6.turnRadius(sc1,boostNow?14:spN,s), trB=V6.turnRadius(sc1,14,s), maxT=Math.min(2,STEP/trN), maxTB=Math.min(2,STEP/trB);
 V6._fit(s,V6.turnRadius(sc1,boosting?14:spN,s));
 const PRd=V8?V8.active(s,myW):[], PIds=new Set(); for(const e of PRd)if(e.conf>=0.4)PIds.add(e.id);   // snakes with a trusted predicted track
 const obs=[]; let bigBodyNear=1e9,nearestBody=1e9,nearestHead=1e9,headEsc=false,raceBig=false; const rearPush=[],racePen=[];
 const blocked=new Uint8Array(32); let preyO=null,preyD=1e9,bigNearC=0; const preyList=[];
 for(const o of slithers){ if(!o||o===s||o.dead)continue; const tpo=V8?V8.tp(o.id):null, aggr=!!(tpo&&tpo.aggr>0.6);   // a dying snake's body is loot, not a wall
   const w=29*(o.sc||1)+myW*0.5, wb=29*(o.sc||1)*C.bodyRad+myW*0.5, bigger=o.sc>=s.sc*0.95, osm=(o.sp||5.78)/5.78;
   const dx=o.xx-hx,dy=o.yy-hy,dh=Math.hypot(dx,dy); if(dh<nearestHead)nearestHead=dh;
   obs.push({x:o.xx,y:o.yy,w:w+(bigger?45:30)+15*osm+(boosting?myW*0.5:0)+(bigger&&aggr?40:0)});
   if(bigger||!PIds.has(o.id))for(let k=1;k<=3;k++)obs.push({x:o.xx+Math.cos(o.ang)*60*k*osm,y:o.yy+Math.sin(o.ang)*60*k*osm,w:w+15});   // guessed head path; a smaller snake with a predicted track is covered by the time-matched obstacle instead
   if(bigger&&dh<bigR+(aggr?300:0))bigNearC++;
   // enemy head behind me & coming -> turn 90deg away; boosting head close -> escape boost
   const toH=Math.atan2(dy,dx), relH=wrap(toH-ha), hTow=Math.cos(Math.atan2(-dy,-dx)-o.ang);
   if(dh<(5*myW+250)*(aggr?1.4:1)){ if(Math.abs(relH)>3*Math.PI/4&&hTow>0.4)rearPush.push(ha-Math.sign(relH)*Math.PI/2); if((o.sp||0)>10&&hTow>0.3&&bigger)headEsc=true; }
   // side-by-side race with similar/bigger snake: peel away
   const along=-(dx*Math.cos(ha)+dy*Math.sin(ha)), lat=Math.abs(-dx*Math.sin(ha)+dy*Math.cos(ha));
   if(o.sc>=s.sc*0.8&&Math.abs(wrap(o.ang-ha))<C.raceAng&&lat<C.raceLat&&Math.abs(along)<600){racePen.push(toH);raceBig=true;}
   if(o.sc<s.sc*0.9&&dh>60&&dh<900){ preyList.push({o,d:dh}); if(dh<preyD){preyD=dh;preyO=o;} }
   for(const p of (o.pts||[])){ if(p.dying)continue; const px=p.xx-hx,py=p.yy-hy; if(Math.abs(px)>C.enclosedR||Math.abs(py)>C.enclosedR)continue; const d=Math.hypot(px,py); if(d>C.enclosedR)continue;
     const dd=d-wb; if(dd<nearestBody)nearestBody=dd; if(bigger&&dd<bigBodyNear)bigBodyNear=dd;
     if(o.sc>=s.sc*0.6){const bi=((Math.floor((Math.atan2(py,px)+Math.PI)/(2*Math.PI)*32))%32+32)%32; blocked[bi]=1;}
     if(d<1100)obs.push({x:p.xx,y:p.yy,w:wb-5+(bigger?10:0)+(boosting?myW*0.3:0)}); } }
 // enclosure: fraction of directions blocked, and the biggest gap
 let nb=0;for(let i=0;i<32;i++)nb+=blocked[i]; const encFrac=nb/32; let gapA=null;
 if(encFrac>=C.enclosedFrac&&nb<32){ let best=0,bs=0; for(let st=0;st<32;st++){ if(blocked[st]||!blocked[(st+31)%32])continue; let l=0; while(l<32&&!blocked[(st+l)%32])l++; if(l>best){best=l;bs=st;} } gapA=-Math.PI+((bs+best/2)/32)*2*Math.PI; }
 const enclosed=gapA!==null;
 // goals: coil (geometry-feasible vs a smaller snake), cut-off (mid), cluster
 const R=window.__wallR(), wallM=300+myW, wd=R-Math.hypot(hx-grd,hy-grd), outA=Math.atan2(hy-grd,hx-grd);
 let goalA=null,goalW=0,mode='roam',feast=false,hunt=false,coilOn=false;
 if(preyO&&!enclosed&&!raceBig){
   // coil: our body must be long enough to close a ring of radius Rc (Rc >= our own turn radius), the target no bigger than 0.9x us, no bigger snake near
   const myLen=V6.bodyLen(s), trC=V6.turnRadius(sc1,V6.normSp(sc1),s)*C.coilTurnK, circ=id=>!!(V8&&V8.circ.has(id));
   const feas=(d,ck)=>{ if(ck&&ck.sweep>5.3)return true; const Rc=Math.max(trC,Math.min(900,d*0.8),C.coilR); return myLen>=2*Math.PI*Rc*C.coilCircK; };
   const qual=(p,maxD,ck)=>p.d<maxD&&!p.o.dead&&p.o.sc<s.sc*0.9&&bigNearC===0&&L>=150&&feas(p.d,ck);
   const same=(ck,o)=>ck&&(ck.o===o||(ck.id!=null&&ck.id===o.id));
   let ck=V6.coil,cT=null;
   if(ck){ const p=preyList.find(q=>same(ck,q.o)); if(p&&qual(p,900,ck)&&nowT-ck.t0<6000)cT=p; else if(!p||!qual(p,900,ck))V6.coil=ck=null; }
   if(!cT){ let bs=1e9; for(const p of preyList){ if(!qual(p,700,same(ck,p.o)?ck:null))continue; const tq=V8?V8.tp(p.o.id):null, sc2=p.d-120*(circ(p.o.id)?1:0)-160*(tq?tq.fear:0); if(sc2<bs){bs=sc2;cT=p;} }
     if(cT){ if(same(ck,cT.o))ck.t0=nowT; else V6.coil=ck={o:cT.o,id:cT.o.id,sgn:Math.sign(wrap(Math.atan2(cT.o.yy-hy,cT.o.xx-hx)-ha))||1,t0:nowT,sweep:0,lb:null}; } else V6.coil=ck=null; }
   if(cT&&ck){ const o=cT.o, brg=Math.atan2(hy-o.yy,hx-o.xx); if(ck.lb!=null)ck.sweep+=ck.sgn*wrap(brg-ck.lb); ck.lb=brg;
     const tight=ck.sweep>5.3?Math.max(0.5,1-(ck.sweep-5.3)/(4*Math.PI)):1, Rc=Math.max(trC,C.coilR*tight), tp=Math.atan2(o.yy-hy,o.xx-hx);
     const inward=Math.max(-0.5,Math.min(0.8,(cT.d-Rc)/400)), tq=V8?V8.tp(o.id):null;
     ck.Rc=Rc; goalA=tp-ck.sgn*(Math.PI/2-inward); goalW=C.coilW*(1+0.3*(tq?tq.fear:0)+0.25*(circ(o.id)?1:0)); mode='coil'; coilOn=true; }
   else if(L>=150){ const lead=120+preyD*0.6; let ix=preyO.xx+Math.cos(preyO.ang)*lead, iy=preyO.yy+Math.sin(preyO.ang)*lead; const rx=hx-preyO.xx,ry=hy-preyO.yy, al=rx*Math.cos(preyO.ang)+ry*Math.sin(preyO.ang), la=-rx*Math.sin(preyO.ang)+ry*Math.cos(preyO.ang);
     if(al>40&&Math.abs(la)<350){const side=la>0?-1:1,pa=preyO.ang+side*Math.PI/2; ix=hx+Math.cos(pa)*300+Math.cos(preyO.ang)*80; iy=hy+Math.sin(pa)*300+Math.sin(preyO.ang)*80; mode='cut';} else mode='hunt';
     goalA=Math.atan2(iy-hy,ix-hx); goalW=C.cutW; hunt=preyD<500; } }
 if(!coilOn)V6.coil=null;
 // food: integer-keyed 300-unit grid (cluster choice cached 160 ms; the cache is dropped when foods/slithers/snake change)
 const fd=[],cell=300; for(const f of foods){ if(!f)continue; const dx=f.xx-hx,dy=f.yy-hy; if(dx<770&&dx>-770&&dy<770&&dy>-770)fd.push(f); }   // only orbs a 14x50 rollout can reach
 let cc=V6._cc; if(!cc||cc.foods!==foods||cc.sl!==slithers||cc.s!==s||nowT-cc.t>=160){
   const grid=new Map(); for(const f of foods){ if(!f)continue; const dx=f.xx-hx,dy=f.yy-hy; if(Math.abs(dx)>1600||Math.abs(dy)>1600)continue; const k=(Math.floor(f.xx/cell)+8)*1024+(Math.floor(f.yy/cell)+8); let g=grid.get(k); if(!g){g={m:0,x:0,y:0};grid.set(k,g);} const m=(f.sz||1)*(f.sz||1)/5; g.m+=m;g.x+=f.xx*m;g.y+=f.yy*m; }
   let cl0=null,clS0=0; for(const g of grid.values()){ const cx=g.x/g.m,cy=g.y/g.m, d=Math.hypot(cx-hx,cy-hy); if(R-Math.hypot(cx-grd,cy-grd)<wallM+200)continue; const sc=g.m/(1+d/600); if(g.m>120&&sc>clS0){ let crowd=0; for(const o of slithers){if(o&&o!==s){const ex=o.xx-cx,ey=o.yy-cy; if(ex*ex+ey*ey<360000)crowd++;}} if(crowd>C.crowdMax)continue; clS0=sc;cl0={x:cx,y:cy,m:g.m,d,crowd}; } }
   cc=V6._cc={foods,sl:slithers,s,t:nowT,cl:cl0}; }
 const cl=cc.cl?{x:cc.cl.x,y:cc.cl.y,m:cc.cl.m,d:Math.hypot(cc.cl.x-hx,cc.cl.y-hy),crowd:cc.cl.crowd}:null;
 let orb=null,orbS=0; if(!enclosed){ for(const f of foods){ if(!f||(f.sz||0)<C.orbMinSz)continue; const dx=f.xx-hx,dy=f.yy-hy; if(Math.abs(dx)>C.orbR||Math.abs(dy)>C.orbR)continue; const d=Math.hypot(dx,dy); if(d>C.orbR||R-Math.hypot(f.xx-grd,f.yy-grd)<wallM+150)continue; const v=f.sz*f.sz/5, sc=v/(1+d/350); if(sc<=orbS)continue; let danger=false,rival=false; for(const o of slithers){ if(!o||o===s)continue; const od=Math.hypot(o.xx-f.xx,o.yy-f.yy); if(od<200&&o.sc>=s.sc*0.95)danger=true; if(od<Math.min(d,500))rival=true; } if(danger)continue; orbS=sc; orb={x:f.xx,y:f.yy,sz:f.sz,v,d,rival}; } }
 const clScore=cl?cl.m/(1+cl.d/600):0;
 if(orb&&(goalA===null||mode==='hunt')&&orbS*C.orbBias>=clScore){ goalA=Math.atan2(orb.y-hy,orb.x-hx); goalW=Math.min(C.orbWMax,150+orb.v*15); mode='orb'; hunt=false; }
 if(goalA===null&&cl&&!enclosed){ goalA=Math.atan2(cl.y-hy,cl.x-hx); goalW=Math.min(500,cl.m*1.5); mode='feed'; }
 // loot: a dead snake's orb trail. Outranks hunting, coiling, roaming and plain food; only escaping comes first.
 const lt=(V8&&!enclosed)?V8.loot(s,hx,hy,R,wallM,C):null; if(lt){ goalA=Math.atan2(lt.y-hy,lt.x-hx); goalW=Math.min(C.lootWMax,250+lt.v*3); mode='loot'; hunt=false; }
 if(enclosed)mode='escape-enclosure';
 const PXk=V8?V8.PXk:(V6._PX||(V6._PX=new Float64Array(16))), PYk=V8?V8.PYk:(V6._PY||(V6._PY=new Float64Array(16))); let bestKill=0,bestKS=9,bestKLen=0;
 if(V8&&goalA===null&&!enclosed){ const mg=V8.mapGoal(s,L); if(mg!==null){goalA=mg;goalW=C.mapW;mode='travel';} }
 // post-meal body: widen the margins a little for food targets, so we don't plan through a gap we won't fit after eating
 const tgt=mode==='loot'?lt:mode==='orb'?orb:mode==='feed'?cl:null, gr=tgt?V6.grow(s.sct,s.fam,tgt.v||tgt.m||0):null, wGrow=gr?Math.max(0,gr.width-myW)*0.5:0;
 const no0=obs.length, reach=NSTEP*STEP+60, ob=[];   // only obstacles a rollout can reach
 for(let i=0;i<no0;i++){ const o=obs[i],dx=o.x-hx,dy=o.y-hy,r=reach+o.w; if(dx*dx+dy*dy<=r*r)ob.push(o); }
 const nob=ob.length, RR={steps:NSTEP,clear:1e9,pf:0};
 const roll=(a,mT,rec,hs)=>{ let x=hx,y=hy,h=ha,clear=1e9,steps=NSTEP,pf=0,da=wrap(a-ha);
   for(let k=1;k<=NSTEP;k++){ const t=Math.max(-mT,Math.min(mT,da)); h+=t; da-=t; x+=Math.cos(h)*STEP; y+=Math.sin(h)*STEP;
     let mc=1e9; for(let i=0;i<nob;i++){ const o=ob[i],lim=mc+o.w; if(lim<=0)continue; const dx=o.x-x,dy=o.y-y,d2=dx*dx+dy*dy; if(d2>=lim*lim)continue; const d=Math.sqrt(d2)-o.w; if(d<mc)mc=d; }
     const ed=R-Math.hypot(x-grd,y-grd)-wallM; if(ed<mc)mc=ed;
     if(rec){PXk[k]=x;PYk[k]=y;} if(hs)hs.push(h);
     if(PRd.length){const tk=k*STEP/V8.meV; for(const e of PRd){if(V8.at(e,tk)){const dd=Math.hypot(V8._x-x,V8._y-y)-e.hw; if(dd<mc)mc=dd;}}}
     if(wGrow>0)mc-=wGrow*Math.min(1,k/6);
     const wk=mc/(1+k*0.06); if(wk<clear)clear=wk; if(mc<0){steps=k;break;}
     for(const f of fd){const fx=f.xx-x,fy=f.yy-y; if(fx*fx+fy*fy<4900)pf+=(f.sz||1)*(f.sz||1)/5/(1+k*0.12);} }
   RR.steps=steps;RR.clear=clear;RR.pf=pf; };
 let bestA=ha,bestS=-1e9,bestClear=0;
 for(let i=0;i<NA;i++){ const a=ha+(i/NA)*Math.PI*2-Math.PI, rel=wrap(a-ha); roll(a,maxT,true); const steps=RR.steps,clear=RR.clear,pf=RR.pf;
   const si=Math.min(NS-1,Math.max(0,Math.floor((rel+Math.PI)/(2*Math.PI)*NS)));
   let sc=(steps<NSTEP?-5000+steps*300:Math.min(clear,250*stake)*C.safeW)+Math.log(net.p[binOf(rel)]+1e-3)*C.netW
     +C.big*big[si]+C.near*near[si]+(L>=150?C.prey*prey[si]:0)+pf*C.pathFood;
   if(wd<2000)sc+=C.wall*(1-wd/2000)*Math.max(0,Math.cos(a-outA));
   if(enclosed)sc+=C.gapW*Math.cos(a-gapA);
   for(const ra of rearPush)sc+=C.rearW*Math.cos(a-ra);
   for(const ta of racePen)sc+=C.raceW*Math.max(0,Math.cos(a-ta));
   let ks=0; if(PRd.length&&steps>=NSTEP){ks=V8.kill(PRd,PXk,PYk,STEP); sc+=C.killW*ks*(mode==='loot'?0.25:1);}   // loot outranks hunting
   if(goalA!==null&&steps>=NSTEP)sc+=goalW*Math.cos(a-goalA);
   if(sc>bestS){bestS=sc;bestA=a;bestClear=steps<NSTEP?-1:clear;bestKill=ks;bestKS=ks?V8._ks:9;bestKLen=ks?V8._kLen:0;} }
 window.xm=Math.cos(bestA)*220; window.ym=Math.sin(bestA)*220;
 V6.lastPlan={maxT,trN,hs:[]}; roll(bestA,maxT,false,V6.lastPlan.hs);
 const escape=bestClear>=0&&bestClear<45&&L>40;
 const encEsc=enclosed&&L>40; const trapped=bestClear<0&&L>40;
 // boosting burns length: the expected gain must beat the cost, and clearance bars scale with the stake
 const bk=stake*C.boostK, cost=C.boostCost*stake, cb=100*bk, bb=80*bk, hb=60*bk;
 const huntB=hunt&&bestClear>70*bk&&bigBodyNear>hb&&!raceBig&&L>C.minBoostL&&C.killGain*V6.lenOf(preyO).len>=cost;
 const orbB=mode==='orb'&&orb.rival&&orb.sz>=C.orbBoostSz&&orb.d<600&&bestClear>cb&&bigBodyNear>bb&&L>C.minBoostL&&orb.v>=cost;
 feast=mode==='feed'&&cl&&cl.m>C.clusterBoostMin&&cl.d<900&&cl.crowd<=1&&bestClear>cb&&bigBodyNear>bb&&L>C.minBoostL&&cl.m>=cost;
 const killB=bestKill>0.3&&bestKS<0.7&&bestClear>80*bk&&bigBodyNear>hb&&L>C.minBoostL&&C.killGain*bestKLen>=cost;
 const lootB=mode==='loot'&&lt&&lt.rival&&lt.d<700&&lt.v>=100&&lt.v>=cost&&bestClear>cb&&bigBodyNear>bb&&L>C.minBoostL;
 const hardWhy=trapped?'trapped':escape?'escape':encEsc?'enclosed':headEsc&&L>40?'headEsc':'';
 let soft=killB?'kill':lootB?'loot':huntB?'hunt':orbB?'orb':feast?'feast':'', bchk=null;
 if(soft&&!hardWhy&&!boostNow){ roll(bestA,maxTB,false); bchk=RR.steps>=NSTEP&&RR.clear>70*bk; if(!bchk)soft=''; }   // is the heading still clear with the (wider) boost turn radius?
 const why=hardWhy||soft; V6._pb=!!why;
 try{setAcceleration(why?1:0)}catch(e){}
 const dg={t:Date.now()-window.__bStart,L,mode,why,clear:Math.round(bestClear),enc:+encFrac.toFixed(2),body:Math.round(nearestBody),head:Math.round(nearestHead),wall:Math.round(wd),rear:rearPush.length,race:raceBig,orb:orb?+orb.sz.toFixed(1):0,
  tr:Math.round(trN),stake:+stake.toFixed(2),fit:V6.turnFit.ratio,bchk,coil:coilOn,growW:+(gr?gr.width:myW).toFixed(1),maxT:+maxT.toFixed(3),
  loot:lt?Math.round(lt.v):0,kill:bestKill?+bestKill.toFixed(2):0,circ:V8?V8.circ.size:0,temp:V8?V8.tsum():null};
 window.__dbg=dg; V6.diag.push(dg); if(V6.diag.length>60)V6.diag.shift();
};
if(V5.noWeights)V6.cfg.netW=0;
window.V6start=function(){ if(window.__bot)clearInterval(window.__bot); window.__bot=setInterval(()=>{try{window.__botStep6()}catch(e){window.__v6err=String(e)}},40); window.__mode='bot'; window.__respawnAt=0; };
window.V6stop=function(){ window.__mode='off'; if(window.__bot)clearInterval(window.__bot); try{setAcceleration(0)}catch(e){} };
})();
