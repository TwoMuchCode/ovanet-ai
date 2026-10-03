"""app/ovanet_ui/theme.py -- global CSS for the OvaNet AI interface (presentation only)."""

CSS = """
<style>
:root{
  --bg:#070b1f; --bg2:#0e1433; --panel:rgba(255,255,255,.045); --panel-2:rgba(255,255,255,.07);
  --line:rgba(148,163,255,.16); --text:#e6eaff; --mute:#93a0c8; --dim:#6b78a3;
  --blue:#4f7cff; --cyan:#22d3ee; --violet:#8b5cf6; --lav:#c4b5fd;
  --low:#2dd4bf; --mod:#fbbf24; --high:#fb7185;
  --grad:linear-gradient(120deg,#4f7cff 0%,#22d3ee 50%,#8b5cf6 100%);
}
html{scroll-behavior:smooth}
.stApp{
  color:var(--text);
  font-family:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;
  background:
    radial-gradient(900px 500px at 12% -5%,rgba(79,124,255,.20),transparent 60%),
    radial-gradient(800px 500px at 95% 10%,rgba(139,92,246,.18),transparent 60%),
    radial-gradient(700px 500px at 50% 105%,rgba(34,211,238,.10),transparent 60%),
    linear-gradient(rgba(148,163,255,.045) 1px,transparent 1px) 0 0/48px 48px,
    linear-gradient(90deg,rgba(148,163,255,.045) 1px,transparent 1px) 0 0/48px 48px,
    var(--bg);
  background-attachment:fixed;
}
.stApp::after{ /* drifting particles */
  content:"";position:fixed;inset:0;pointer-events:none;z-index:0;opacity:.5;
  background-image:
    radial-gradient(1.5px 1.5px at 10% 20%,#9fb4ff,transparent),radial-gradient(1.5px 1.5px at 30% 70%,#22d3ee,transparent),
    radial-gradient(1px 1px at 55% 35%,#c4b5fd,transparent),radial-gradient(1.5px 1.5px at 75% 80%,#9fb4ff,transparent),
    radial-gradient(1px 1px at 88% 15%,#22d3ee,transparent),radial-gradient(1px 1px at 20% 90%,#c4b5fd,transparent),
    radial-gradient(1.5px 1.5px at 65% 55%,#9fb4ff,transparent);
  animation:ov-drift 40s linear infinite alternate;
}
@keyframes ov-drift{from{transform:translate3d(0,0,0)}to{transform:translate3d(-24px,18px,0)}}
header[data-testid="stHeader"],footer,#MainMenu,[data-testid="stToolbar"],[data-testid="stDecoration"]{display:none!important}
.block-container{max-width:1180px!important;padding:6.2rem 1.25rem 3rem!important;position:relative;z-index:1}
html,body,.stApp{overflow-x:hidden}
h1,h2,h3,p,span,label,div{color:inherit}
a{color:var(--cyan)}
:focus-visible{outline:2px solid var(--cyan)!important;outline-offset:2px!important}

/* ---------- nav ---------- */
.ov-nav{position:fixed;top:12px;left:50%;transform:translateX(-50%);width:min(1160px,calc(100% - 24px));z-index:1000;
  display:flex;align-items:center;justify-content:space-between;gap:14px;padding:10px 14px 10px 16px;
  background:rgba(10,15,40,.62);backdrop-filter:blur(16px) saturate(140%);-webkit-backdrop-filter:blur(16px) saturate(140%);
  border:1px solid var(--line);border-radius:18px;box-shadow:0 10px 40px rgba(0,0,0,.35)}
.ov-brand{display:flex;align-items:center;gap:10px;text-decoration:none;color:var(--text)!important;font-weight:700;letter-spacing:.2px}
.ov-brand b{background:var(--grad);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.ov-links{display:flex;gap:4px}
.ov-links a,.ov-menu a{color:var(--mute)!important;text-decoration:none;font-size:.9rem;padding:7px 13px;border-radius:10px;transition:.2s}
.ov-links a:hover,.ov-menu a:hover{color:var(--text)!important;background:var(--panel-2)}
.ov-right{display:flex;align-items:center;gap:10px}
.ov-proto{font-size:.72rem;letter-spacing:.8px;text-transform:uppercase;color:var(--lav);border:1px solid rgba(196,181,253,.3);
  padding:5px 10px;border-radius:999px;background:rgba(139,92,246,.1);white-space:nowrap}
.ov-proto::before{content:"";display:inline-block;width:6px;height:6px;border-radius:50%;background:var(--violet);margin-right:7px;box-shadow:0 0 8px var(--violet);animation:ov-pulse 2.4s ease-in-out infinite}
.ov-cta{background:var(--grad);color:#041022!important;font-weight:700;font-size:.88rem;text-decoration:none;padding:8px 16px;border-radius:12px;transition:.25s;box-shadow:0 6px 22px rgba(79,124,255,.35)}
.ov-cta:hover{transform:translateY(-1px);box-shadow:0 10px 28px rgba(34,211,238,.4)}
.ov-menu{display:none;position:relative}
.ov-menu summary{list-style:none;cursor:pointer;padding:6px 12px;border:1px solid var(--line);border-radius:10px;color:var(--text)}
.ov-menu summary::-webkit-details-marker{display:none}
.ov-menu div{position:absolute;right:0;top:46px;min-width:200px;display:flex;flex-direction:column;padding:8px;border-radius:14px;
  background:rgba(10,15,40,.96);border:1px solid var(--line)}
@keyframes ov-pulse{0%,100%{opacity:1}50%{opacity:.35}}

/* ---------- hero ---------- */
.ov-anchor{position:relative;top:-90px;visibility:hidden;height:0}
.ov-hero{display:grid;grid-template-columns:1.05fr 1fr;gap:28px;align-items:center;padding:18px 0 30px}
.ov-eyebrow{font-size:.74rem;letter-spacing:2.2px;font-weight:700;color:var(--cyan);text-transform:uppercase}
.ov-h1{font-size:clamp(2.4rem,5.4vw,4.2rem);line-height:1.04;font-weight:800;margin:.5rem 0 .4rem;letter-spacing:-1.2px}
.ov-h1 span{background:var(--grad);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent;background-size:200% 100%;animation:ov-grad 10s ease-in-out infinite alternate}
@keyframes ov-grad{to{background-position:100% 0}}
.ov-sub{font-size:1.2rem;color:var(--lav);margin:.2rem 0 .9rem;font-weight:500}
.ov-desc{color:var(--mute);line-height:1.65;max-width:560px}
.ov-chips{display:flex;flex-wrap:wrap;gap:8px;margin:16px 0 22px}
.ov-chip{font-size:.78rem;padding:5px 11px;border-radius:999px;border:1px solid var(--line);background:var(--panel);color:var(--text)}
.ov-btns{display:flex;gap:12px;flex-wrap:wrap}
.ov-btn{display:inline-block;padding:12px 22px;border-radius:14px;font-weight:700;text-decoration:none;transition:.25s}
.ov-btn.p{background:var(--grad);color:#041022!important;box-shadow:0 10px 30px rgba(79,124,255,.4)}
.ov-btn.s{border:1px solid var(--line);color:var(--text)!important;background:var(--panel)}
.ov-btn:hover{transform:translateY(-2px)}
.ov-resp{margin-top:18px;font-size:.8rem;color:var(--dim);letter-spacing:.3px}
.ov-herovis{position:relative;border-radius:26px;border:1px solid var(--line);background:linear-gradient(160deg,rgba(255,255,255,.05),rgba(255,255,255,.015));
  box-shadow:0 30px 80px rgba(0,0,0,.45),inset 0 0 60px rgba(79,124,255,.07);padding:8px}
.ov-herovis svg{width:100%;height:auto;display:block}
.fl{animation:ov-float 7s ease-in-out infinite;transform-box:fill-box}
.fl2{animation-delay:-2s}.fl3{animation-delay:-4s}.fl4{animation-delay:-5.5s}.fl5{animation-delay:-1s}
@keyframes ov-float{0%,100%{transform:translateY(0)}50%{transform:translateY(-6px)}}
.flow{stroke-dasharray:4 6;animation:ov-dash 3s linear infinite}
@keyframes ov-dash{to{stroke-dashoffset:-40}}
.nd{animation:ov-pulse 3.2s ease-in-out infinite}

/* ---------- sections / cards ---------- */
.ov-sec{margin:56px 0 18px}
.ov-sec .k{font-size:.72rem;letter-spacing:2px;color:var(--cyan);font-weight:700;text-transform:uppercase}
.ov-sec h2{font-size:clamp(1.6rem,3vw,2.2rem);margin:.3rem 0 .3rem;font-weight:800;letter-spacing:-.5px}
.ov-sec p{color:var(--mute);max-width:720px;margin:0}
div[class*="st-key-glass"]{background:linear-gradient(160deg,var(--panel-2),var(--panel));border:1px solid var(--line);border-radius:22px;
  padding:22px;backdrop-filter:blur(10px);box-shadow:0 18px 50px rgba(0,0,0,.28);transition:border-color .3s,transform .3s}
div[class*="st-key-glass"]:hover{border-color:rgba(124,150,255,.34)}
.glass{background:linear-gradient(160deg,var(--panel-2),var(--panel));border:1px solid var(--line);border-radius:20px;padding:20px;
  box-shadow:0 14px 40px rgba(0,0,0,.25);transition:.3s;animation:ov-in .7s both}
.glass:hover{transform:translateY(-3px);border-color:rgba(124,150,255,.4)}
@keyframes ov-in{from{opacity:0;transform:translateY(14px)}to{opacity:1;transform:none}}
.grid{display:grid;gap:14px}
.g2{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(3,1fr)}.g4{grid-template-columns:repeat(4,1fr)}.g5{grid-template-columns:repeat(5,1fr)}
.metric .l{font-size:.72rem;letter-spacing:1.2px;text-transform:uppercase;color:var(--mute)}
.metric .v{font-size:1.9rem;font-weight:800;margin:4px 0 2px;letter-spacing:-.5px}
.metric .h{font-size:.8rem;color:var(--dim);line-height:1.4}
.bar{height:6px;border-radius:99px;background:rgba(255,255,255,.08);margin-top:10px;overflow:hidden}
.bar i{display:block;height:100%;border-radius:99px;background:var(--grad);animation:ov-grow 1.1s ease-out both;transform-origin:left}
@keyframes ov-grow{from{transform:scaleX(0)}}
.empty{border:1px dashed var(--line);border-radius:16px;padding:22px;color:var(--mute);text-align:center;font-size:.92rem;background:rgba(255,255,255,.02)}
.note{font-size:.8rem;color:var(--dim);margin-top:10px}

/* ---------- workflow strip + timeline ---------- */
.steps{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px}
.steps span{flex:1;min-width:130px;text-align:center;padding:9px 10px;border-radius:12px;background:var(--panel);border:1px solid var(--line);font-size:.85rem;color:var(--mute)}
.steps span b{color:var(--cyan);margin-right:6px}
.tl{display:grid;grid-template-columns:repeat(6,1fr);gap:8px;position:relative;margin:6px 0 4px}
.tl .s{position:relative;padding:12px 10px;border-radius:14px;background:var(--panel);border:1px solid var(--line);font-size:.8rem;color:var(--mute);overflow:hidden}
.tl .s b{display:block;color:var(--dim);font-size:.7rem;letter-spacing:1px;margin-bottom:3px}
.tl .s.done{color:var(--text);border-color:rgba(34,211,238,.35)}.tl .s.done b{color:var(--cyan)}
.tl .s.run::after{content:"";position:absolute;left:0;bottom:0;height:3px;width:40%;background:var(--grad);animation:ov-slide 1.4s ease-in-out infinite alternate}
.tl .s.run{border-color:rgba(124,150,255,.5);color:var(--text)}
.tl .s.done::after{content:"";position:absolute;left:0;bottom:0;height:3px;width:100%;background:var(--grad);opacity:.7}
@keyframes ov-slide{from{left:0}to{left:60%}}

/* ---------- viewer ---------- */
div[class*="st-key-viewer"]{background:#050816;border:1px solid var(--line);border-radius:20px;padding:14px}
div[data-testid="stImage"] img{border-radius:12px;max-height:520px;object-fit:contain;margin:0 auto}
button[data-baseweb="tab"]{color:var(--mute)!important;background:transparent!important;border-radius:10px 10px 0 0}
button[data-baseweb="tab"][aria-selected="true"]{color:var(--text)!important}
div[data-baseweb="tab-highlight"]{background:var(--grad)!important;height:3px!important}
.badge{display:inline-block;font-size:.72rem;letter-spacing:.8px;padding:4px 10px;border-radius:999px;border:1px solid var(--line);color:var(--mute);margin:6px 6px 0 0}
.badge.ok{color:var(--low);border-color:rgba(45,212,191,.4);background:rgba(45,212,191,.08)}
.badge.warn{color:var(--mod);border-color:rgba(251,191,36,.4);background:rgba(251,191,36,.08)}
.strip{display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:12px}
.strip div{background:var(--panel);border:1px solid var(--line);border-radius:14px;padding:10px 12px}
.strip small{display:block;color:var(--mute);font-size:.72rem;letter-spacing:.6px;text-transform:uppercase}
.strip b{font-size:1rem}

/* ---------- uploader ---------- */
[data-testid="stFileUploaderDropzone"]{background:
  radial-gradient(circle at 80% 20%,rgba(139,92,246,.18),transparent 50%),radial-gradient(circle at 15% 85%,rgba(34,211,238,.14),transparent 50%),rgba(255,255,255,.03)!important;
  border:1.5px dashed rgba(124,150,255,.4)!important;border-radius:20px!important;padding:34px 20px!important;transition:.3s}
[data-testid="stFileUploaderDropzone"]:hover{border-color:var(--cyan)!important;box-shadow:0 0 40px rgba(34,211,238,.12)}
[data-testid="stFileUploaderDropzone"] button{background:var(--grad)!important;color:#041022!important;border:0!important;font-weight:700!important;border-radius:12px!important}

/* ---------- risk ---------- */
.risk{display:grid;grid-template-columns:240px 1fr;gap:26px;align-items:center}
.risk svg{width:100%;max-width:240px}
.risk .rs{font-size:.74rem;letter-spacing:2px;color:var(--mute)}
.risk .big{font-size:3.4rem;font-weight:800;line-height:1}
.risk .big small{font-size:1.1rem;color:var(--dim);font-weight:500}
.band{display:inline-block;margin-top:10px;padding:6px 18px;border-radius:999px;font-weight:800;letter-spacing:1.5px;border:1px solid}
.band.LOW{color:var(--low);border-color:rgba(45,212,191,.5);background:rgba(45,212,191,.1)}
.band.MODERATE{color:var(--mod);border-color:rgba(251,191,36,.5);background:rgba(251,191,36,.1)}
.band.HIGH{color:var(--high);border-color:rgba(251,113,133,.5);background:rgba(251,113,133,.1)}
.disc{margin-top:14px;font-size:.84rem;color:var(--mute);line-height:1.55;border-left:2px solid var(--violet);padding-left:12px}

/* ---------- attribution ---------- */
.attr{display:grid;grid-template-columns:130px 1fr 150px;gap:12px;align-items:center;padding:10px 0;border-bottom:1px solid var(--line)}
.attr:last-child{border:0}
.attr .n{font-weight:700;text-transform:capitalize}
.attr .c{font-size:.76rem;color:var(--dim)}
.attr .d{font-size:.8rem;color:var(--mute);text-align:right}
.tag{display:inline-block;font-size:.72rem;padding:3px 9px;margin:4px 4px 0 0;border-radius:999px;background:rgba(139,92,246,.12);border:1px solid rgba(196,181,253,.22);color:var(--lav)}
.hist{display:flex;align-items:flex-end;gap:6px;height:90px;margin-top:10px}
.hist i{flex:1;background:var(--grad);border-radius:6px 6px 2px 2px;min-height:3px;opacity:.9;animation:ov-up .9s ease-out both;transform-origin:bottom}
@keyframes ov-up{from{transform:scaleY(0)}}
.emb{display:grid;grid-template-columns:repeat(16,1fr);gap:3px;margin-top:10px}
.emb i{height:26px;border-radius:5px}

/* ---------- checklist & compare ---------- */
.ck{display:flex;gap:12px;align-items:flex-start}
.ck .ic{flex:0 0 30px;height:30px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-weight:800;border:1px solid var(--line)}
.ck.ok .ic{color:var(--low);background:rgba(45,212,191,.1)}.ck.warn .ic{color:var(--mod);background:rgba(251,191,36,.1)}.ck.unk .ic{color:var(--mute)}
.ck div b{display:block}.ck div small{color:var(--mute)}
.visit .t{font-size:.72rem;letter-spacing:2px;color:var(--cyan);font-weight:700}
.visit .u{color:var(--mute);margin-bottom:8px}
.trend{display:inline-block;padding:6px 16px;border-radius:999px;font-weight:800;letter-spacing:1px;border:1px solid var(--line)}
.trend.IMPROVING{color:var(--low)}.trend.WORSENING{color:var(--high)}.trend.STABLE{color:var(--cyan)}.trend.INSUFFICIENT_DATA{color:var(--mute)}
.link{display:flex;align-items:center;justify-content:center;gap:10px;margin:14px 0;color:var(--mute)}
.link svg{width:min(420px,100%)}

/* ---------- transparency & footer ---------- */
.tr{display:grid;grid-template-columns:repeat(5,1fr);gap:12px}
.tr .glass{padding:16px;font-size:.88rem;color:var(--mute);line-height:1.45}
.tr .glass b{display:block;color:var(--text);margin-bottom:4px}
.foot{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--dim);font-size:.8rem;display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px}

/* streamlit widget polish */
.stButton>button,.stDownloadButton>button,[data-testid="stFormSubmitButton"]>button{background:var(--grad)!important;color:#041022!important;border:0!important;font-weight:700!important;border-radius:12px!important;transition:.25s}
.stButton>button:hover,.stDownloadButton>button:hover,[data-testid="stFormSubmitButton"]>button:hover{transform:translateY(-2px);box-shadow:0 10px 28px rgba(79,124,255,.4)}
[data-testid="stCheckbox"] label,[data-testid="stNumberInput"] label{color:var(--text)!important}
[data-testid="stForm"]{border:0!important;padding:0!important}

/* ---------- responsive ---------- */
@media(max-width:900px){
  .ov-hero{grid-template-columns:1fr}.g4,.g5,.tr{grid-template-columns:repeat(2,1fr)}.g3{grid-template-columns:repeat(2,1fr)}
  .tl{grid-template-columns:repeat(3,1fr)}.ov-links{display:none}.ov-menu{display:block}
}
@media(max-width:640px){
  .block-container{padding:5.4rem .75rem 2rem!important}
  .ov-proto{display:none}.ov-cta{padding:7px 12px}
  .g2,.g3,.g4,.g5,.tr{grid-template-columns:1fr}.strip{grid-template-columns:repeat(2,1fr)}
  .risk{grid-template-columns:1fr;justify-items:center;text-align:center}
  .attr{grid-template-columns:1fr}.attr .d{text-align:left}
  .tl{grid-template-columns:repeat(2,1fr)}.ov-herovis{border-radius:18px}.stApp::after{display:none}
  div[class*="st-key-glass"]{padding:14px}
}
@media(prefers-reduced-motion:reduce){
  *,*::before,*::after{animation:none!important;transition:none!important;scroll-behavior:auto!important}
}
</style>
"""