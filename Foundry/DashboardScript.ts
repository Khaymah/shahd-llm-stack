// Control-panel client logic (M14 rebuild). One WebSocket (/ws) for realtime system/model/stats/learn/
// train + load-model; REST for /api/kinds, /api/checkpoints, /api/browse* , /api/system, /api/model.
// Seven views (Overview/Collect/Data/Train/Models/Chat/System) switched by a hash router; #chat is
// the trace-capable chat surface (the legacy /chat page has no trace). IMPORTANT: this is inlined into a backtick template in
// DashboardHtml.ts — it MUST stay free of backticks, ${ ... } and the closing script sequence; all
// strings use single quotes + concatenation. Kept in its own file so each file stays small.
export const DashboardScript = `
 var H=function(s){return String(s==null?'':s).replace(/[&<>"'/]/g,function(c){return {'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;','/':'&#47;'}[c];});};
 var Q=function(id){return document.getElementById(id);};
 var fmtN=function(n){return (Number(n)||0).toLocaleString();};
 var fmtB=function(n){n=Number(n)||0;return n>=1e9?(n/1e9).toFixed(2)+' GB':n>=1e6?(n/1e6).toFixed(1)+' MB':n>=1e3?(n/1e3).toFixed(0)+' KB':n+' B';};
 var fmtP=function(n){n=Number(n)||0;return n>=1e9?(n/1e9).toFixed(2)+'B':n>=1e6?(n/1e6).toFixed(2)+'M':n>=1e3?(n/1e3).toFixed(1)+'K':String(n);};
 var fmtDur=function(ms){var s=Math.round(ms/1000);if(s<90)return s+'s';var m=Math.round(s/60);if(m<90)return m+'m';var h=m/60;return h<48?h.toFixed(1)+'h':(h/24).toFixed(1)+'d';};
 var logLine=function(el,txt,cls){if(!el)return;var d=document.createElement('div');if(cls)d.className=cls;d.textContent='['+new Date().toTimeString().slice(0,8)+'] '+txt;el.appendChild(d);while(el.childElementCount>400)el.removeChild(el.firstChild);el.scrollTop=el.scrollHeight;};
 var pillKind=function(k){return '<span class="pill '+H(k)+'">'+H(k)+'</span>';};

 var WS=null, loadedName='', collecting=false, training=false, checkpoints=[], kindStats=[], lastSystem=null, lossHistory=[], sparkMeta=[], trainStart=0, trainFirstStep=-1;
 var chConv=null, chStreaming=false, chBubble=null, chGotTrace=false, chThinkSeg=-1, chThinkEl=null;
 var LANGS={oasst:[['all','All languages'],['en','English'],['ar','Arabic'],['es','Spanish'],['de','German'],['fr','French'],['ru','Russian'],['zh','Chinese']],
            wiki:[['en','English'],['ar','Arabic'],['es','Spanish'],['de','German'],['fr','French'],['ru','Russian'],['ja','Japanese']],
            gsm8k:[['train','Train (~7.5k)'],['test','Test (~1.3k)'],['all','All (~8.8k)']],
            wikidump:[['simple','Simple English (1 shard)'],['ar','Arabic (7 shards)'],['en','English (large)'],['es','Spanish'],['fr','French']],
            stackexchange:[['all','All SE sites (14 shards)']]};
 // MODEL-SCALING presets — a COMPLETE one-click config: [Embed,Layers,Heads,Context,Vocab,Batch,
 // Steps, CodeMb,KnowledgeMb (pretrain mix), ConvCount,CodeSamples (chat mix), Workers, Precision].
 // Picking a tier fills the architecture AND the data mix AND the run knobs for both modes; adjust
 // any field after. Micro+ defaults to F32 — memory is the binding constraint from that tier up
 // (see Docs/MODEL-SCALING.md, Run knobs).
 var PRESETS={Seed:[96,3,4,96,512,16,6000,2,0,3000,2000,4,'F64'],Nano:[128,4,4,256,512,16,5000,3,0,6000,3000,8,'F64'],Micro:[256,6,4,512,1024,16,16000,8,0,20000,8000,8,'F32'],Mini:[512,8,8,1024,4096,32,22000,30,0,100000,30000,8,'F32'],Small:[768,12,12,2048,16384,64,19000,80,0,300000,80000,8,'F32'],Base:[1024,24,16,4096,32000,128,17000,200,0,500000,150000,8,'F32'],Large:[2048,32,32,8192,50000,256,22000,500,0,1000000,300000,8,'F32']};

 // ── theme ──
 function applyTheme(t){document.documentElement.setAttribute('data-theme',t);try{localStorage.setItem('shahd.theme',t);}catch(e){}Q('themebtn').textContent=t==='dark'?'☀':'☾';}
 function toggleTheme(){var cur=document.documentElement.getAttribute('data-theme');applyTheme(cur==='dark'?'light':'dark');}

 // ── nav / router ──
 var TITLES={overview:['Overview','everything at a glance'],collect:['Collect data','sources land in per-kind tables'],data:['Data browser','review + clean the collected corpus'],train:['Train','new model, or resume a saved one'],models:['Models','load · add training · delete'],chat:['Chat','test the model + see its reasoning'],system:['System','host + compute']};
 function navTo(v){if(!TITLES[v])v='overview';
  var links=document.querySelectorAll('.nav a[data-v]');for(var i=0;i<links.length;i++)links[i].classList.toggle('on',links[i].getAttribute('data-v')===v);
  var views=document.querySelectorAll('.view');for(var j=0;j<views.length;j++)views[j].classList.remove('on');
  var el=Q('view-'+v);if(el)el.classList.add('on');
  Q('vtitle').textContent=TITLES[v][0];Q('vcrumb').textContent=TITLES[v][1];
  Q('side').classList.remove('open');window.scrollTo(0,0);
  if(v==='data')dLoad();else if(v==='models')loadCheckpoints();else if(v==='overview')loadOverview();else if(v==='system')renderSystem(lastSystem);else if(v==='chat')chInit();
 }
 window.addEventListener('hashchange',function(){navTo((location.hash||'#overview').slice(1));});

 // ── Overview ──
 async function loadOverview(){try{kindStats=await (await fetch('/api/kinds')).json();}catch(e){kindStats=[];}renderKinds();
  try{var m=await (await fetch('/api/model')).json();renderModelCard(m);}catch(e){}
  try{renderLedger(await (await fetch('/api/collection')).json());}catch(e){}}
 function renderLedger(rows){var el=Q('ov-ledger');if(!el)return;
  if(!rows||!rows.length){el.innerHTML='<div class="empty">nothing collected yet</div>';return;}
  el.innerHTML='<div style="overflow-x:auto"><table style="width:100%;border-collapse:collapse;font-size:12.5px">'
   +'<tr style="color:var(--mut);text-align:left"><th style="padding:4px 8px">Source</th><th>Kind</th><th class="tnum">Collected</th><th>State</th><th>Cursor</th></tr>'
   +rows.map(function(r){var st=r.Exhausted?'<span style="color:var(--mut)">complete</span>':'<span style="color:var(--conv)">growing</span>';
    var cur=(r.Cursor&&r.Cursor!=='{}')?H(r.Cursor):'—';
    return '<tr style="border-top:1px solid var(--line)"><td style="padding:4px 8px;font-family:var(--mono)">'+H(r.SourceKey)+'</td><td>'+pillKind(r.Kind)+'</td><td class="tnum">'+fmtN(r.Collected)+'</td><td>'+st+'</td><td style="font-family:var(--mono);color:var(--faint)">'+cur+'</td></tr>';}).join('')
   +'</table></div>';}
 function renderKinds(){
  var totalDocs=0,trainable=0;kindStats.forEach(function(k){totalDocs+=k.Count;trainable+=k.Filtered;});
  Q('ov-docs').textContent=fmtN(totalDocs);
  Q('ov-trainable').textContent=fmtN(trainable);
  var active=kindStats.filter(function(k){return k.Count>0;}).length;
  Q('ov-docs-s').textContent='across '+active+' active kind'+(active===1?'':'s');
  var maxCount=1;kindStats.forEach(function(k){if(k.Count>maxCount)maxCount=k.Count;});
  var COLOR={code:'var(--code)',conversation:'var(--conv)',knowledge:'var(--know)',books:'var(--books)',web:'var(--mut)',instruction:'var(--mut)'};
  Q('ov-kinds').innerHTML=kindStats.map(function(k){var off=k.Count===0;var c=COLOR[k.Kind]||'var(--mut)';
   var amt=off?'reserved':fmtN(k.Count)+' · '+fmtB(k.FilteredBytes)+' filtered';
   return '<div class="kind'+(off?' off':'')+'"><div class="nm"><span class="sw" style="background:'+c+'"></span>'+H(k.Kind)+'</div><div class="track"><div class="fill" style="width:'+Math.max(off?0:2,(k.Count/maxCount)*100)+'%;background:'+c+'"></div></div><div class="amt tnum">'+H(amt)+'</div></div>';}).join('');
 }
 function renderModelCard(m){
  Q('ov-models').textContent=fmtN(checkpoints.length);
  var chat=checkpoints.filter(function(c){return c.Format==='chat';}).length;
  Q('ov-models-s').textContent=(checkpoints.length-chat)+' base · '+chat+' chat';
  if(!m){Q('ov-model').innerHTML='<div class="empty">no model loaded — train one, then load it</div>';return;}
  Q('ov-model').innerHTML='<div style="font-family:var(--mono);font-size:20px;font-weight:680">'+H(loadedName||'model')+'</div>'
   +'<div style="color:var(--mut);font-size:12.5px;margin:3px 0 14px">'+fmtN(m.TotalParams)+' params · emb'+m.EmbedDim+' · L'+m.NumLayers+' · ctx'+m.BlockSize+'</div>'
   +'<div class="row c2"><div><div class="k" style="font-size:11px;color:var(--mut)">VOCAB</div><div class="mono" style="font-size:15px">'+fmtN(m.VocabSize)+'</div></div>'
   +'<div><div class="k" style="font-size:11px;color:var(--mut)">POSITION / NORM</div><div class="mono" style="font-size:15px">'+H(m.PositionScheme)+' / '+H(m.NormKind)+'</div></div></div>'
   +'<div style="display:flex;gap:8px;margin-top:16px"><button class="btn pri sm" data-act="open-chat">Open in Chat</button></div>';
 }
 function setJobs(){var t=collecting?'Collecting':training?'Training':'Idle';Q('ov-jobs').textContent=t;Q('ov-jobs-s').textContent=(collecting||training)?'run in progress':'no run in progress';}

 // ── Collect ──
 var SRC='github';
 function pickSource(s){SRC=s;var cards=document.querySelectorAll('#srccards .s');for(var i=0;i<cards.length;i++)cards[i].classList.toggle('on',cards[i].getAttribute('data-src')===s);syncCollectForm();}
 function syncCollectForm(){
  var general=(SRC==='oasst'||SRC==='oasst2'||SRC==='wikipedia'||SRC==='gsm8k'||SRC==='wikidump'||SRC==='stackexchange');
  var folder=(SRC==='folder');
  Q('c-ghbox').style.display=(SRC==='github'||SRC==='both')?'':'none';
  Q('c-localbox').style.display=(SRC==='local'||SRC==='both')?'':'none';
  Q('c-folderbox').style.display=folder?'':'none';
  Q('c-folderopts').style.display=folder?'':'none';
  Q('c-levelbox').style.display=(general||folder)?'none':'';
  Q('c-capbox').style.display=(general||folder)?'none':'';
  Q('c-genbox').style.display=general?'':'none';
  Q('c-maxlabel').textContent=general?'Max items':'Max repos';
  if(general){var opts=SRC==='wikipedia'?LANGS.wiki:SRC==='gsm8k'?LANGS.gsm8k:SRC==='wikidump'?LANGS.wikidump:SRC==='stackexchange'?LANGS.stackexchange:LANGS.oasst;Q('c-lang').innerHTML=opts.map(function(o){return '<option value="'+o[0]+'">'+o[1]+'</option>';}).join('');}
  var kind=folder?Q('c-folderkind').value:(SRC==='wikipedia'||SRC==='wikidump')?'knowledge':(SRC==='oasst'||SRC==='oasst2'||SRC==='stackexchange')?'conversation':SRC==='gsm8k'?'instruction':'code';
  // Collection semantics (mirrors each provider's Semantics): bounded = fixed dataset, exhausts after a
  // full collect; streaming = can keep producing fresh data, run again to grow.
  var streaming=(SRC==='github'||SRC==='both'||SRC==='wikipedia'||SRC==='wikidump'||SRC==='stackexchange');
  var semNote=streaming?'<span style="color:var(--conv)">streaming</span> — run again to collect more'
   :'<span style="color:var(--mut)">bounded</span> — a full collect exhausts it; re-runs only dedup';
  Q('c-kindhint').innerHTML='Stored in '+pillKind(kind)+' &nbsp;<span style="color:var(--faint)">documents_'+kind+'</span><br>'+semNote;
 }
 function collectSettings(){
  var general=(SRC==='oasst'||SRC==='oasst2'||SRC==='wikipedia'||SRC==='gsm8k'||SRC==='wikidump'||SRC==='stackexchange');
  var folder=(SRC==='folder');
  var query=general?Q('c-lang').value:(folder?'':Q('c-query').value);
  var repos=(folder?Q('c-folderpath').value:Q('c-repos').value).split(',').map(function(x){return x.trim();}).filter(Boolean);
  var s={Source:SRC,Query:query,Repos:repos,
   MinLevel:Q('c-minlevel').value,MaxRepos:+Q('c-maxrepos').value||1,MaxFilesPerRepo:+Q('c-maxfiles').value,
   MaxBytesPerRepo:(+Q('c-maxmb').value)*1e6,MaxContentBytes:(+Q('c-maxkb').value)*1e3,SkipLearned:Q('c-skip').checked};
  if(folder){s.Kind=Q('c-folderkind').value;s.License=Q('c-folderlicense').value;}
  return s;
 }
 function cStart(){if(!wsReady())return;save();WS.send(JSON.stringify({type:'learn',settings:collectSettings()}));}
 function cStop(){if(wsReady()){WS.send(JSON.stringify({type:'learn-stop'}));Q('c-start').textContent='stopping…';Q('c-start').disabled=true;}}
 function setCollectBtn(run){collecting=run;setJobs();var b=Q('c-start');b.disabled=false;b.textContent=run?'■ Stop':'▶ Start collection';b.className='btn '+(run?'danger':'pri');b.onclick=run?cStop:cStart;Q('c-stop').style.display='none';}
 var cMax=5,cSeen=0,cNew=0,cSkip=0,cErrs=0;
 function cProg(f,busy,label){var p=Q('c-prog');p.className='prog'+(busy?' busy':'');Q('c-progi').style.width=Math.max(0,Math.min(1,f))*100+'%';Q('c-plabel').textContent=label;}
 function onLearn(e){var log=Q('c-log');
  if(e.kind==='start'){cMax=e.repos||5;cSeen=0;cNew=0;cSkip=0;cErrs=0;setCollectBtn(true);Q('c-badge').className='pill run';Q('c-badge').textContent='collecting';log.innerHTML='';logLine(log,'collecting from '+e.source+' ('+(e.query||'own repos')+')','a');cProg(.02,false,'0 / '+cMax);}
  else if(e.kind==='scanning'){cProg(1,true,e.label);logLine(log,e.label,'a');}
  else if(e.kind==='repo'){cSeen++;if(e.ingested)cNew++;else cSkip++;logLine(log,(e.ingested?'stored ':'skip ')+e.repo+' ['+e.level+', '+e.files+' files]'+(e.ingested?'':(e.reason?' ('+e.reason+')':'')),e.ingested?'ok':'');cProg(cSeen/cMax,false,cSeen+' / '+cMax+' · '+cNew+' new · '+cSkip+' skipped');}
  else if(e.kind==='repo-progress'){cProg(e.filesTotal?e.filesDone/e.filesTotal:0,false,'ingesting '+e.repo+' — '+e.filesDone+' / '+e.filesTotal+' files');}
  else if(e.kind==='done'){setCollectBtn(false);Q('c-badge').className='pill '+(cErrs?'err':'done');Q('c-badge').textContent=cErrs?'done with errors':'done';
   var nw=(e['new']!=null?e['new']:cNew),dup=(e.duplicate!=null?e.duplicate:0);
   var life=(e.collected!=null?' · lifetime '+fmtN(e.collected):'');
   cProg(1,false,nw+' new · '+dup+' duplicate · '+e.ingested+' processed');
   logLine(log,'done — '+nw+' new · '+dup+' duplicate ('+e.ingested+' processed)'+life,nw?'ok':'w');
   if(e.exhausted){logLine(log,'this source is fully collected (bounded dataset, '+fmtN(e.collected||0)+' total) — re-running only re-checks it. Add a new source or query to grow.','w');}
   else if(nw===0&&e.ingested>0){logLine(log,e.semantics==='bounded'?'0 new: nothing new from this bounded source this run.':'0 new: everything fetched this run was already collected — try a different query/language, or run again for fresh items.','w');}
   else if(e.ingested===0){logLine(log,'nothing fetched this run — check the source/query (or the log above for rate-limit errors).','w');}
   loadOverview();}
  // error is NOT terminal: the server reports non-fatal failures mid-run and always ends with done.
  else if(e.kind==='error'){cErrs++;Q('c-badge').className='pill err';Q('c-badge').textContent='error';logLine(log,'error: '+e.message,'e');}
 }

 // ── Data browser ──
 var dPage=0,dTotal=0,dSize=50;
 async function dFacets(){try{var f=await (await fetch('/api/browse/facets?kind='+encodeURIComponent(Q('d-kind').value))).json();
   var langOpt='<option value="">any language</option>'+(f.Langs||[]).map(function(l){return '<option value="'+H(l)+'">'+H(l)+'</option>';}).join('');
   var licOpt='<option value="">any license</option>'+(f.Licenses||[]).map(function(l){return '<option value="'+H(l)+'">'+H(l)+'</option>';}).join('');
   var kl=Q('d-lang').value,kc=Q('d-license').value;Q('d-lang').innerHTML=langOpt;Q('d-license').innerHTML=licOpt;Q('d-lang').value=kl;Q('d-license').value=kc;
  }catch(e){}}
 function dQuery(){return 'kind='+encodeURIComponent(Q('d-kind').value)+'&tier='+encodeURIComponent(Q('d-tier').value)+'&lang='+encodeURIComponent(Q('d-lang').value)+'&license='+encodeURIComponent(Q('d-license').value)+'&q='+encodeURIComponent(Q('d-q').value);}
 async function dLoad(){await dFacets();dReload();}
 async function dReload(){var body=Q('d-tbody');body.innerHTML='<tr><td colspan="6" class="empty">loading…</td></tr>';
  try{var r=await (await fetch('/api/browse?'+dQuery()+'&page='+dPage+'&pageSize='+dSize)).json();dTotal=r.Total;var kind=Q('d-kind').value;
   if(!r.Rows.length){body.innerHTML='<tr><td colspan="6" class="empty">no documents match this filter.</td></tr>';}
   else body.innerHTML=r.Rows.map(function(d){return '<tr>'
    +'<td class="mono clip" title="'+H(d.provenance)+'">'+H(d.provenance)+'</td>'
    +'<td><span class="tier-'+H(d.tier)+'">'+H(d.tier)+'</span></td>'
    +'<td class="mono">'+H(d.lang)+'</td><td class="mono">'+H(d.license)+'</td><td class="mono tnum">'+fmtB(d.bytes)+'</td>'
    +'<td><div class="acts"><button class="btn sm" data-act="doc-view" data-kind="'+H(kind)+'" data-id="'+H(d.id)+'">View</button><button class="btn sm danger" data-act="doc-del" data-kind="'+H(kind)+'" data-id="'+H(d.id)+'">Delete</button></div></td></tr>';}).join('');
   var pages=Math.max(1,Math.ceil(dTotal/dSize));
   Q('d-total').textContent=fmtN(dTotal)+' document'+(dTotal===1?'':'s');
   Q('d-pageinfo').textContent='page '+(dPage+1)+' of '+pages;
   Q('d-prev').disabled=dPage<=0;Q('d-next').disabled=dPage>=pages-1;
   Q('d-bulk').disabled=dTotal===0;Q('d-bulk').textContent=dTotal===0?'Delete matching':'Delete '+fmtN(dTotal)+' matching';
  }catch(e){body.innerHTML='<tr><td colspan="6" class="empty">failed to load</td></tr>';}
 }
 function dSearch(){dPage=0;dReload();}
 function dPrev(){if(dPage>0){dPage--;dReload();}}
 function dNext(){dPage++;dReload();}
 async function dDelete(kind,id){if(!confirm('Delete this document permanently?'))return;
  await fetch('/api/browse/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:kind,id:id})});
  if(dTotal%dSize===1&&dPage>0)dPage--;dReload();loadOverview();}
 async function dBulk(){var kind=Q('d-kind').value;
  if(prompt('This permanently deletes ALL '+fmtN(dTotal)+' matching documents in "'+kind+'". Type DELETE to confirm.')!=='DELETE')return;
  var r=await (await fetch('/api/browse/delete-matching',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({kind:kind,tier:Q('d-tier').value,lang:Q('d-lang').value,license:Q('d-license').value,q:Q('d-q').value,confirm:true})})).json();
  if(r.error){alert('Delete failed: '+r.error);return;}
  alert('Deleted '+fmtN(r.deleted)+' documents.');dPage=0;dLoad();loadOverview();}
 async function openDoc(kind,id){var m=Q('modal');Q('m-title').textContent=id;Q('m-meta').textContent='';Q('m-body').textContent='loading…';m.classList.add('open');
  try{var f=await (await fetch('/api/browse/doc?kind='+encodeURIComponent(kind)+'&id='+encodeURIComponent(id))).json();if(f.error){Q('m-body').textContent='error: '+f.error;return;}
   Q('m-title').textContent=f.provenance;Q('m-meta').innerHTML='<span class="tier-'+H(f.tier)+'">'+H(f.tier)+'</span> · '+H(f.lang)+' · '+H(f.license)+' · '+H(f.origin)+' · '+fmtB(f.bytes)+(f.reason?' · '+H(f.reason):'');Q('m-body').textContent=f.content;
  }catch(e){Q('m-body').textContent='failed';}}
 function closeModal(){Q('modal').classList.remove('open');}

 // ── Train ──
 var tMode='pretrain';
 function setMode(m){tMode=m;Q('t-mode-pre').classList.toggle('on',m==='pretrain');Q('t-mode-chat').classList.toggle('on',m==='chat');Q('t-mix-pretrain').style.display=m==='pretrain'?'':'none';Q('t-mix-chat').style.display=m==='chat'?'':'none';}
 function onPreset(){var p=PRESETS[Q('t-preset').value];if(!p)return;Q('t-embed').value=p[0];Q('t-layers').value=p[1];Q('t-heads').value=p[2];Q('t-ctx').value=p[3];Q('t-vocab').value=p[4];Q('t-batch').value=p[5];Q('t-steps').value=p[6];Q('t-corpus').value=p[7];Q('t-know').value=p[8];Q('t-conv').value=p[9];Q('t-code').value=p[10];Q('t-workers').value=p[11];if(Q('t-prec'))Q('t-prec').value=p[12];}
 function renderResumeOptions(){var sel=Q('t-resume');var cur=sel.value;
  sel.innerHTML='<option value="">◇ New model</option>'+checkpoints.map(function(c){return '<option value="'+H(c.Name)+'">↻ '+H(c.Name)+' ('+H(c.Format)+', step '+fmtN(c.Step)+')</option>';}).join('');
  sel.value=cur;renderFromOptions();}
 // Warm-start picker: BASE checkpoints only (a chat run seeds from pretrained weights via --From).
 function renderFromOptions(){var sel=Q('t-from');if(!sel)return;var cur=sel.value;
  sel.innerHTML='<option value="">◇ from scratch</option>'+checkpoints.filter(function(c){return c.Format!=='chat';}).map(function(c){return '<option value="'+H(c.Name)+'">⤷ '+H(c.Name)+' ('+H(c.Arch)+', step '+fmtN(c.Step)+')</option>';}).join('');
  sel.value=cur;}
 function onResume(){var name=Q('t-resume').value;var note=Q('t-resumenote');
  if(!name){note.style.display='none';setArchLock(false);setTrainBtn(training);return;}
  var c=checkpoints.filter(function(x){return x.Name===name;})[0];if(!c){note.style.display='none';setArchLock(false);setTrainBtn(training);return;}
  Q('t-name').value=c.Name;setMode(c.Format==='chat'?'chat':'pretrain');prefillArch(c);
  if(c.Batch)Q('t-batch').value=c.Batch;
  if(c.MaxSteps)Q('t-steps').value=c.MaxSteps;
  setArchLock(true);setTrainBtn(training);
  note.style.display='';note.innerHTML='Will <b>continue</b> "'+H(c.Name)+'" from step '+fmtN(c.Step)+' — architecture + batch + optimizer are <b>inherited from the checkpoint</b> (locked here); raise Steps above '+fmtN(c.MaxSteps||c.Step)+' to train it further.';}
 function setArchLock(on){['t-embed','t-layers','t-heads','t-ctx','t-vocab','t-batch'].forEach(function(id){var el=Q(id);if(el)el.disabled=on;});}
 function prefillArch(c){
  if(c.Embed){Q('t-embed').value=c.Embed;Q('t-layers').value=c.Layers;Q('t-heads').value=c.Heads;Q('t-ctx').value=c.Block;}
  else{var mm=/emb(\\d+)\\s*L(\\d+)\\s*ctx(\\d+)/.exec(c.Arch||'');if(mm){Q('t-embed').value=mm[1];Q('t-layers').value=mm[2];Q('t-ctx').value=mm[3];}}
  if(c.Vocab)Q('t-vocab').value=c.Vocab;}
 function resumeModel(name){navTo('train');location.hash='#train';var sel=Q('t-resume');sel.value=name;onResume();}
 function trainSettings(){var vocab=+Q('t-vocab').value||512;
  return {Kind:tMode,Name:Q('t-name').value,Resume:!!Q('t-resume').value,Steps:+Q('t-steps').value,
   CorpusMb:+Q('t-corpus').value,EmbedDim:+Q('t-embed').value,NumLayers:+Q('t-layers').value,NumHeads:+Q('t-heads').value,
   BlockSize:+Q('t-ctx').value,Merges:Math.max(0,vocab-256),BatchSize:+Q('t-batch').value,Workers:+Q('t-workers').value,
   Precision:Q('t-prec')?Q('t-prec').value:'F64',
   From:(tMode==='chat'&&Q('t-from'))?Q('t-from').value:'',
   KnowledgeMb:+Q('t-know').value,ConvCount:+Q('t-conv').value,CodeSamples:+Q('t-code').value,
   MultiTurn:Q('t-multiturn')?+Q('t-multiturn').value:3000};}
 function tStart(){if(!wsReady())return;save();WS.send(JSON.stringify({type:'train',settings:trainSettings()}));}
 function tStop(){if(wsReady()){WS.send(JSON.stringify({type:'train-stop'}));Q('t-start').textContent='stopping…';Q('t-start').disabled=true;}}
 function setTrainBtn(run){training=run;setJobs();var b=Q('t-start');b.disabled=false;
  var rn=Q('t-resume')?Q('t-resume').value:'';var rc=rn?checkpoints.filter(function(x){return x.Name===rn;})[0]:null;
  b.textContent=run?'■ Stop (keeps last checkpoint)':(rc?'⏵ Resume from step '+fmtN(rc.Step):'▶ Train model');
  b.className='btn '+(run?'danger':'pri');b.onclick=run?tStop:tStart;}
 function sparkTipInit(){var w=Q('t-spark');var tip=Q('t-tip');if(!w||!tip||w.dataset.tip)return;w.dataset.tip='1';
  w.addEventListener('mousemove',function(ev){var n=lossHistory.length;if(!n){tip.style.display='none';return;}
   var r=w.getBoundingClientRect();var fx=Math.max(0,Math.min(1,(ev.clientX-r.left)/r.width));
   var i=Math.max(0,Math.min(n-1,Math.round(fx*(n-1))));var m=sparkMeta[i]||{};
   tip.textContent='step '+(m.s!=null?m.s:'?')+' · loss '+lossHistory[i].toFixed(4)+(m.ms!=null?' · '+m.ms+'ms':'');
   tip.style.display='block';tip.style.left=Math.max(0,Math.min(r.width-tip.offsetWidth-4,fx*r.width-tip.offsetWidth/2))+'px';});
  w.addEventListener('mouseleave',function(){tip.style.display='none';});}
 function drawSpark(){var el=Q('t-spark');if(!el)return;sparkTipInit();var L=lossHistory;if(!L.length){el.innerHTML='';return;}
  var W=300,Hh=74,min=Math.min.apply(null,L),max=Math.max.apply(null,L),rng=(max-min)||1,n=L.length;
  var pts=L.map(function(v,i){var x=n<2?W:(i/(n-1))*W;var y=6+(1-(v-min)/rng)*(Hh-14);return x.toFixed(1)+','+y.toFixed(1);});
  var d='M'+pts.join(' L');var last=pts[pts.length-1].split(',');
  el.innerHTML='<defs><linearGradient id="sg" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="var(--accent)" stop-opacity=".35"/><stop offset="1" stop-color="var(--accent)" stop-opacity="0"/></linearGradient></defs>'
   +'<path d="'+d+' L'+W+','+Hh+' L0,'+Hh+' Z" fill="url(#sg)"/><path d="'+d+'" fill="none" stroke="var(--accent)" stroke-width="2"/><circle cx="'+last[0]+'" cy="'+last[1]+'" r="3" fill="var(--accent)"/>';}
 function onTrain(e){var log=Q('t-log');
  if(e.kind==='train-start'){lossHistory=[];sparkMeta=[];trainStart=Date.now();trainFirstStep=-1;setTrainBtn(true);Q('t-badge').className='pill run';Q('t-badge').textContent='step 0 / '+e.steps;log.innerHTML='';logLine(log,'training '+e.steps+' steps…','a');drawSpark();}
  else if(e.kind==='train-info'){logLine(log,e.text);}
  else if(e.kind==='train-progress'){lossHistory.push(e.trainLoss);sparkMeta.push({s:e.step,ms:e.stepMs});if(lossHistory.length>240){lossHistory.shift();sparkMeta.shift();}drawSpark();
   Q('t-badge').textContent='step '+e.step+' / '+e.steps;Q('t-loss').textContent=e.trainLoss.toFixed(3)+(typeof e.valLoss==='number'?' (val '+e.valLoss.toFixed(3)+')':'');
   var el=e.elapsedMs||(Date.now()-trainStart);Q('t-elapsed').textContent=fmtDur(el);
   // ETA divides THIS session's elapsed by the steps done THIS session — a resumed run starts at a
   // high global step with a fresh clock, and dividing by the global step made 4h read as ~5s.
   if(trainFirstStep<0)trainFirstStep=e.step;var done=e.step-trainFirstStep+1;
   Q('t-eta').textContent='~'+fmtDur(el/done*(e.steps-e.step))+' left';
   logLine(log,'step '+e.step+'/'+e.steps+'  loss '+e.trainLoss.toFixed(3)+(typeof e.stepMs==='number'?'  '+e.stepMs+'ms':'')+(typeof e.valLoss==='number'?'  val '+e.valLoss.toFixed(3):''));}
  else if(e.kind==='train-done'){setTrainBtn(false);Q('t-badge').className='pill done';Q('t-badge').textContent='done';logLine(log,'saved to '+e.savedTo+' — reloaded into Chat','ok');}
  else if(e.kind==='train-error'){setTrainBtn(false);var stopped=e.message.indexOf('stop')>=0;Q('t-badge').className='pill '+(stopped?'idle':'err');Q('t-badge').textContent=stopped?'stopped':'error';logLine(log,e.message,stopped?'w':'e');}
 }

 // ── Models ──
 async function loadCheckpoints(){try{checkpoints=await (await fetch('/api/checkpoints')).json();}catch(e){checkpoints=[];}renderModels();renderResumeOptions();}
 function renderModels(){var b=Q('m-tbody');if(!checkpoints.length){b.innerHTML='<tr><td colspan="7" class="empty">no saved models yet — train one in the Train view.</td></tr>';Q('nav-models-tag').textContent='0';return;}
  Q('nav-models-tag').textContent=checkpoints.length;
  b.innerHTML=checkpoints.map(function(c){var loaded=c.Name===loadedName;
   return '<tr><td><b>'+H(c.Name)+'</b>'+(loaded?' <span class="pill done">loaded</span>':'')+'</td>'
    +'<td><span class="pill '+(c.Format==='chat'?'chat':'base')+'">'+H(c.Format)+'</span></td>'
    +'<td class="mono tnum">'+fmtP(c.Params)+'</td><td class="mono">'+H(c.Arch)+'</td>'
    +'<td class="mono tnum">step '+fmtN(c.Step)+'</td><td style="color:var(--mut)">'+H((c.CreatedAt||'').slice(0,10))+'</td>'
    +'<td><div class="acts"><button class="btn sm'+(loaded?'':' pri')+'" data-act="model-open" data-name="'+H(c.Name)+'">Open in Chat</button>'
    +'<button class="btn sm" data-act="model-resume" data-name="'+H(c.Name)+'">Resume training</button>'
    +'<button class="btn sm danger" data-act="model-del" data-name="'+H(c.Name)+'">Delete</button></div></td></tr>';}).join('');}
 // The SPA #chat view is the ONLY surface that renders the reasoning trace (the legacy /chat page has
 // no chat-trace handler) — so "Open in Chat" must land there, not on the trace-blind page.
 function openModel(name){if(wsReady())WS.send(JSON.stringify({type:'load-model',name:name}));location.hash='#chat';}
 async function deleteModel(name){if(!confirm('Delete model "'+name+'" permanently? This cannot be undone.'))return;
  await fetch('/api/checkpoint/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({name:name})});loadCheckpoints();loadOverview();}

 // ── Chat (embedded view) — reuses the /ws chat protocol + surfaces the reasoning trace ──
 function uuid(){return (window.crypto&&crypto.randomUUID)?crypto.randomUUID():'c-'+new Date().getTime().toString(16)+'-'+Math.floor(Math.random()*1e9).toString(16);}
 function chModelOptions(){var sel=Q('ch-model');if(!sel)return;var cur=sel.value||loadedName;
  if(!checkpoints.length){sel.innerHTML='<option value="">(no saved models — train one first)</option>';return;}
  sel.innerHTML=checkpoints.map(function(c){return '<option value="'+H(c.Name)+'">'+H(c.Name)+' — '+fmtP(c.Params)+' · '+H(c.Format)+'</option>';}).join('');if(cur)sel.value=cur;}
 function chInit(){chModelOptions();chLoadHistory();if(!chConv)chNew();}
 function chPick(name){if(name&&wsReady()){WS.send(JSON.stringify({type:'load-model',name:name}));Q('ch-stat').textContent='loading '+name+'…';}}
 function chBubbleEl(role,text,cls){var d=document.createElement('div');d.className='bub '+(cls||role);d.innerHTML='<div class="who">'+H(role)+'</div>';var t=document.createElement('div');t.textContent=text;d.appendChild(t);var box=Q('ch-bubbles');box.appendChild(d);box.scrollTop=box.scrollHeight;return {wrap:d,txt:t};}
 // Persisted per-message reasoning (Claude-Code style): a collapsible under each model bubble — any
 // past reply can be reopened to see HOW it decided (think -> tool -> answer).
 function chStepHtml(L){var k=(L.Kind==='think'||L.Kind==='tool'||L.Kind==='answer')?L.Kind:'answer';return '<div class="step"><span class="b '+k+'">'+k+'</span><span class="x">'+H(L.Text)+(L.Detail?' <span class="dt">→ '+H(L.Detail)+'</span>':'')+'</span></div>';}
 function chAttachTrace(wrap,lines){if(!lines||!lines.length)return;var old=wrap.querySelector('.trhist');if(old)old.remove();
  var d=document.createElement('details');d.className='trhist';d.innerHTML='<summary>⚙ reasoning · '+lines.length+' step'+(lines.length>1?'s':'')+'</summary><div class="trsteps">'+lines.map(chStepHtml).join('')+'</div>';wrap.appendChild(d);}
 function chNew(){chConv=uuid();Q('ch-bubbles').innerHTML='<div class="empty">say something to test the model</div>';Q('ch-trace').innerHTML='<div class="empty">the model steps (think → tool → answer) appear here after a reply</div>';Q('ch-stat').textContent='';var h=Q('ch-history');if(h)h.value='';}
 async function chLoadHistory(){try{var list=await (await fetch('/api/chat/conversations')).json();var sel=Q('ch-history');if(!sel)return;sel.innerHTML='<option value="">— '+list.length+' past chats —</option>'+list.map(function(c){return '<option value="'+H(c.Id)+'">'+H(c.Title)+'</option>';}).join('');if(chConv)sel.value=chConv;}catch(e){}}
 async function chOpen(id){if(!id||chStreaming)return;chConv=id;
  try{var msgs=await (await fetch('/api/chat/conversation?id='+encodeURIComponent(id))).json();Q('ch-bubbles').innerHTML='';
   if(!msgs.length)Q('ch-bubbles').innerHTML='<div class="empty">empty conversation</div>';
   else msgs.forEach(function(m){var b=chBubbleEl(m.Role==='assistant'?'model':'you',m.Content,m.Role==='assistant'?'model':'u');if(m.Role==='assistant'&&m.Trace)chAttachTrace(b.wrap,m.Trace);});
  }catch(e){Q('ch-bubbles').innerHTML='<div class="empty">failed to load</div>';}
  Q('ch-trace').innerHTML='<div class="empty">the reasoning trace shows for the NEXT reply</div>';}
 async function chDelCur(){if(!chConv)return;if(!confirm('Delete this conversation permanently?'))return;
  await fetch('/api/chat/delete',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({id:chConv})});chNew();chLoadHistory();}
 function chTrace(lines){var el=Q('ch-trace');if(!el)return;if(!lines||!lines.length){el.innerHTML='<div class="empty">no tool/think steps — the model replied directly</div>';return;}
  el.innerHTML=lines.map(function(L){var k=(L.Kind==='think'||L.Kind==='tool'||L.Kind==='answer')?L.Kind:'answer';var txt=H(L.Text)+(L.Detail?' <span class="dt">→ '+H(L.Detail)+'</span>':'');return '<div class="step"><span class="b '+k+'">'+k+'</span><span class="x">'+txt+'</span></div>';}).join('');}
 function chSend(){var box=Q('ch-box'),text=box.value.trim();if(!text||chStreaming)return;if(!wsReady())return;if(!loadedName){Q('ch-stat').textContent='load a model first (pick one above)';return;}if(!chConv)chConv=uuid();
  var em=Q('ch-bubbles').querySelector('.empty');if(em)Q('ch-bubbles').innerHTML='';
  box.value='';chBubbleEl('you',text,'u');chBubble=chBubbleEl('model','','');chBubble.wrap.classList.add('cursor');
  chStreaming=true;chGotTrace=false;chThinkSeg=-1;chThinkEl=null;Q('ch-send').disabled=true;var mx=Math.max(1,Math.min(4096,+Q('ch-max').value||512));Q('ch-stat').textContent='generating (max '+mx+' tokens)…';Q('ch-trace').innerHTML='<div class="empty">reasoning…</div>';
  WS.send(JSON.stringify({type:'chat',convId:chConv,message:text,temperature:+Q('ch-temp').value,maxTokens:mx}));}
 function chEnd(){chStreaming=false;Q('ch-send').disabled=false;if(chBubble)chBubble.wrap.classList.remove('cursor');chBubble=null;Q('ch-box').focus();}

 // ── System ──
 function renderSystem(s){if(!s){Q('sys').innerHTML='<div class="empty">connecting…</div>';return;}lastSystem=s;
  var gpu=(s.gpu&&s.gpu!=='none detected')?s.gpu:'none';
  var card=function(k,v,c){return '<div class="card stat"><div class="k">'+k+'</div><div class="v" style="font-size:18px'+(c?';color:'+c:'')+'">'+H(v)+'</div></div>';};
  Q('sys').innerHTML=card('Compute',s.gpuUsed?'GPU':'CPU')+card('Backend',s.computeBackend)+card('Go FFI',s.goFfiAvailable?'available':'no',s.goFfiAvailable?'var(--good)':'var(--mut)')
   +card('CPU',s.cpuModel)+card('Cores',String(s.cpuCount))+card('Memory',s.memGb+' GB')+card('GPU',gpu)+card('Runtime',s.runtime);}

 // ── WS ──
 function wsReady(){if(WS&&WS.readyState===1)return true;alert('not connected to the server');return false;}
 function connDot(on){Q('conndot').className='dt'+(on?' on':'');Q('connfoot').className='dot'+(on?' on':'');Q('connfoot-t').textContent=on?'online':'reconnecting…';}
 function connect(){WS=new WebSocket((location.protocol==='https:'?'wss://':'ws://')+location.host+'/ws');
  WS.onopen=function(){connDot(true);};
  WS.onclose=function(){connDot(false);setTimeout(connect,2000);};
  WS.onmessage=function(ev){var m=JSON.parse(ev.data);
   if(m.type==='system'){lastSystem=m.data;if(Q('view-system').classList.contains('on'))renderSystem(m.data);}
   else if(m.type==='model'){if(m.name)loadedName=m.name;Q('loadedchip').innerHTML=m.data?('Loaded: <b>'+H(loadedName)+'</b> · '+fmtP(m.data.TotalParams)):'No model loaded';renderModelCard(m.data);renderModels();chModelOptions();if(!chStreaming)Q('ch-stat').textContent='';}
   else if(m.type==='checkpoints'){checkpoints=m.data;renderModels();renderResumeOptions();renderModelCard(null);chModelOptions();if(checkpoints.length)loadOverview();}
   else if(m.type==='stats'){/* code-only aggregate; Overview uses /api/kinds instead */}
   else if(m.type==='learn')onLearn(m.event);
   else if(m.type==='train')onTrain(m.event);
   else if(m.type==='chat-delta'&&m.convId===chConv&&chBubble){chBubble.txt.textContent+=m.delta;var cb=Q('ch-bubbles');cb.scrollTop=cb.scrollHeight;}
   else if(m.type==='chat-think-delta'&&m.convId===chConv){chGotTrace=true;var tr=Q('ch-trace');
    if(m.seg!==chThinkSeg){if(chThinkSeg===-1)tr.innerHTML='';chThinkSeg=m.seg;chThinkEl=document.createElement('div');chThinkEl.className='step';chThinkEl.innerHTML='<span class="b think">think</span><span class="x"></span>';tr.appendChild(chThinkEl);}
    if(chThinkEl)chThinkEl.querySelector('.x').textContent+=m.delta;tr.scrollTop=tr.scrollHeight;}
   else if(m.type==='chat-tool'&&m.convId===chConv){chGotTrace=true;var tr2=Q('ch-trace');if(chThinkSeg===-1){tr2.innerHTML='';chThinkSeg=-2;}
    var td=document.createElement('div');td.className='step';td.innerHTML='<span class="b tool">tool</span><span class="x">'+H(m.line.Text)+(m.line.Detail?' <span class="dt">→ '+H(m.line.Detail)+'</span>':'')+'</span>';tr2.appendChild(td);tr2.scrollTop=tr2.scrollHeight;}
   else if(m.type==='chat-trace'&&m.convId===chConv){chGotTrace=true;chTrace(m.lines);if(chBubble)chAttachTrace(chBubble.wrap,m.lines);}
   else if(m.type==='chat-done'&&m.convId===chConv){Q('ch-stat').textContent='';if(!chGotTrace)Q('ch-trace').innerHTML='<div class="empty">this is a BASE model — it replies directly, with no think/tool steps. Train a Chat/SFT model (Train ▸ Chat) to see the reasoning trace.</div>';chEnd();chLoadHistory();}
   else if(m.type==='chat-error'&&m.convId===chConv){if(chBubble){chBubble.wrap.className='bub err';chBubble.txt.textContent='error: '+m.error;}Q('ch-stat').textContent='';chEnd();}
  };}

 // ── settings persistence ──
 var FIELDS=['c-query','c-repos','c-minlevel','c-maxrepos','c-maxfiles','c-maxmb','c-maxkb','c-skip','t-name','t-steps','t-embed','t-layers','t-heads','t-ctx','t-vocab','t-batch','t-workers','t-corpus','t-know','t-conv','t-code','t-from','t-prec','t-multiturn'];
 function save(){var o={};FIELDS.forEach(function(id){var el=Q(id);if(!el)return;o[id]=el.type==='checkbox'?el.checked:el.value;});try{localStorage.setItem('shahd.cfg',JSON.stringify(o));}catch(e){}}
 function restore(){try{var o=JSON.parse(localStorage.getItem('shahd.cfg')||'{}');FIELDS.forEach(function(id){if(o[id]===undefined)return;var el=Q(id);if(!el)return;if(el.type==='checkbox')el.checked=!!o[id];else el.value=o[id];});}catch(e){}}

 // ── row actions: ONE delegated listener reads data-* attributes (escaped by H, read back as plain
 // text) — generated rows carry no inline handlers, so a hostile id/name can never become script ──
 function onAction(e){var el=e.target&&e.target.closest?e.target.closest('[data-act]'):null;if(!el)return;
  var act=el.getAttribute('data-act'),kind=el.getAttribute('data-kind')||'',id=el.getAttribute('data-id')||'',name=el.getAttribute('data-name')||'';
  if(act==='open-chat')location.hash='#chat';
  else if(act==='doc-view')openDoc(kind,id);
  else if(act==='doc-del')dDelete(kind,id);
  else if(act==='model-open')openModel(name);
  else if(act==='model-resume')resumeModel(name);
  else if(act==='model-del')deleteModel(name);}

 // ── init ──
 function init(){
  applyTheme((function(){try{return localStorage.getItem('shahd.theme');}catch(e){return null;}})()||(window.matchMedia&&window.matchMedia('(prefers-color-scheme: light)').matches?'light':'dark'));
  var links=document.querySelectorAll('.nav a[data-v]');for(var i=0;i<links.length;i++)links[i].addEventListener('click',function(e){e.preventDefault();location.hash='#'+this.getAttribute('data-v');});
  document.addEventListener('keydown',function(e){if(e.key==='Escape')closeModal();});
  document.addEventListener('click',onAction);
  Q('d-kind').addEventListener('change',function(){dPage=0;dLoad();});
  Q('ch-box').addEventListener('keydown',function(e){if(e.key==='Enter'&&!e.shiftKey){e.preventDefault();chSend();}});
  restore();pickSource('github');setMode('pretrain');
  navTo((location.hash||'#overview').slice(1));
  connect();loadOverview();loadCheckpoints();
 }
 if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
`;
