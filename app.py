"""
TRI·TFM Research Pipeline v2.0
Two-call: Generator → Judge v2 strict → Python Bal v2
Deploy: GitHub → Render (free)
Env: GEMINI_API_KEY
"""

import streamlit as st
from google import genai
from google.genai import types
import json, math, re, time, os
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
from io import StringIO

# ── CONFIG ──────────────────────────────────

MODELS = ["gemini-2.5-flash", "gemini-2.5-pro"]

STAGE_MODES = {
    "Balance": "All axes weighted equally.",
    "Trust": "Prioritize Emotion (E).",
    "Competence": "Prioritize Fact (F).",
    "Narrative": "Prioritize Narrative (N).",
}

JUDGE_SYSTEM_PROMPT = """You are a strict, calibrated evaluator for the TRI·TFM framework.
You will receive a [PROMPT] and a [RESPONSE]. Score the response on four axes.

CRITICAL RULES:
- Be harsh and discriminating. Scores of 1.0 should be exceptionally rare.
- A "good" response typically scores 0.65-0.85, not 0.9-1.0.
- Only give 0.9+ when the response is genuinely outstanding on that specific axis.
- Differentiate clearly between axes. E, F, N should NOT all be the same score.

## Axes

E (Emotion / Affective Alignment) — 0.0 to 1.0
How well does the response tone match what THIS user in THIS context needs?
- 0.85-1.0 = precisely calibrated tone (rare)
- 0.60-0.84 = appropriate and functional
- 0.40-0.59 = generic or slightly off
- 0.0-0.39 = clearly wrong for context
NOTE: Generic pleasant/academic tone = 0.65-0.75, NOT 1.0.

F (Fact / Epistemic Grounding) — 0.0 to 1.0
What proportion of CORE CLAIMS are verifiable?
- 0.85-1.0 = virtually every claim independently verifiable
- 0.60-0.84 = most grounded, some interpretive
- 0.40-0.59 = mix of verifiable and unverifiable
- 0.20-0.39 = inherently unfalsifiable core argument
- 0.0-0.19 = pure speculation
CRITICAL: Distinguish REFERENCE GROUNDING vs CLAIM VERIFIABILITY.
Citing real thinkers = reference grounding (does NOT make F high alone).
Core claim testable = claim verifiability (what F measures).
Philosophical/ethical/existential questions: F = 0.20-0.45 even if well-researched.

N (Narrative / Structural Coherence) — 0.0 to 1.0
- 0.90-1.0 = flawless creative structure (rare)
- 0.70-0.89 = well-organized, clear flow
- 0.50-0.69 = adequate but gaps/redundancy
- 0.0-0.49 = contradictory/disorganized
NOTE: Standard list = 0.70-0.80, NOT 1.0.

B (Bias / Directional Framing) — -1.0 to +1.0
- 0.0 = balanced
- ±0.1-0.3 = slight lean
- ±0.3-0.6 = noticeable bias
- ±0.6-1.0 = strong/extreme bias

Output ONLY valid JSON:
{"E": <float>, "E_reason": "<one sentence>", "F": <float>, "F_reason": "<one sentence>", "N": <float>, "N_reason": "<one sentence>", "B": <float>, "B_reason": "<one sentence>"}"""

# ── BAL v2 ──────────────────────────────────

def compute_bal(E, F, N, B, w_efn=0.75, w_b=0.25):
    m3 = (E + F + N) / 3
    sigma = math.sqrt(((E-m3)**2 + (F-m3)**2 + (N-m3)**2) / 3)
    bal = w_efn * (1 - sigma / 0.5) + w_b * (1 - abs(B))
    status = "STABLE" if bal >= 0.70 else ("DRIFTING" if bal >= 0.50 else f"DOM:{max({'E':E,'F':F,'N':N}, key={'E':E,'F':F,'N':N}.get)}")
    return {"bal": round(bal,4), "status": status, "sigma_efn": round(sigma,4), "m3": round(m3,4)}

# ── API ─────────────────────────────────────

def generate(client, model, prompt, stage_mode, temp):
    r = client.models.generate_content(
        model=model, contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=f"You are a helpful assistant. Stage: {stage_mode}. {STAGE_MODES[stage_mode]} Do NOT self-evaluate.",
            temperature=temp, max_output_tokens=2048))
    return r.text

def judge(client, model, prompt, response):
    schema = types.Schema(type="OBJECT", properties={
        "E": types.Schema(type="NUMBER"), "E_reason": types.Schema(type="STRING"),
        "F": types.Schema(type="NUMBER"), "F_reason": types.Schema(type="STRING"),
        "N": types.Schema(type="NUMBER"), "N_reason": types.Schema(type="STRING"),
        "B": types.Schema(type="NUMBER"), "B_reason": types.Schema(type="STRING"),
    }, required=["E","E_reason","F","F_reason","N","N_reason","B","B_reason"])

    for attempt in range(3):
        try:
            r = client.models.generate_content(
                model=model, contents=f"[PROMPT]\n{prompt}\n\n[RESPONSE]\n{response}",
                config=types.GenerateContentConfig(
                    system_instruction=JUDGE_SYSTEM_PROMPT, temperature=0.0,
                    max_output_tokens=1024, response_mime_type="application/json",
                    response_schema=schema))
            raw = r.text.strip()
            raw = re.sub(r'^```(?:json)?\s*|\s*```$', '', raw).strip()
            m = re.search(r'\{[^{}]*\}', raw, re.DOTALL)
            if m: raw = m.group(0)
            p = json.loads(raw)
            for k in ["E","F","N","B"]: p[k] = float(p[k])
            return p
        except Exception as e:
            if attempt == 2: raise
            time.sleep(1)

def evaluate(client, model, prompt, stage_mode, temp):
    resp = generate(client, model, prompt, stage_mode, temp)
    scores = judge(client, model, prompt, resp)
    bal = compute_bal(scores["E"], scores["F"], scores["N"], scores["B"])
    return {**scores, **bal, "response": resp}

# ── AUTO OBSERVATIONS ───────────────────────

def auto_observe(results):
    if not results: return []
    obs = []
    df = pd.DataFrame(results)
    n = len(df)

    for axis in ["E","F","N"]:
        if (df[axis] >= 0.95).sum() / n >= 0.5:
            obs.append(("CRITICAL", f"Ceiling on {axis}: {(df[axis]>=0.95).sum()}/{n} scored ≥0.95"))

    for axis in ["E","F","N","B","bal"]:
        s = df[axis].std()
        if s > 0.15:
            obs.append(("HIGH", f"High variance {axis}: σ={s:.3f}"))

    if "category" in df.columns:
        ph = df[df["category"].isin(["philosophical","ethical"])]
        fa = df[df["category"] == "factual"]
        if len(ph)>=2 and ph["F"].mean() > 0.55:
            obs.append(("CRITICAL", f"F inflation philosophical: mean={ph['F'].mean():.2f} (expect 0.20-0.45)"))
        if len(fa)>=2 and fa["F"].mean() < 0.70:
            obs.append(("HIGH", f"F too low factual: mean={fa['F'].mean():.2f} (expect ≥0.70)"))
        if len(fa)>=2 and len(ph)>=2:
            gap = fa["F"].mean() - ph["F"].mean()
            if gap < 0.20:
                obs.append(("HIGH", f"Weak F discrimination: ΔF={gap:.2f} (expect ≥0.30)"))

    stable = (df["bal"]>=0.70).sum()
    drift = ((df["bal"]>=0.50)&(df["bal"]<0.70)).sum()
    dom = (df["bal"]<0.50).sum()
    obs.append(("INFO", f"Bal: STABLE={stable} DRIFTING={drift} DOM={dom} mean={df['bal'].mean():.3f}"))

    return obs

# ── MAIN ────────────────────────────────────

def main():
    st.set_page_config(page_title="TRI·TFM v2", layout="wide")
    st.title("TRI·TFM v2.0")
    st.caption("Generator → Judge v2 strict → Bal v2")

    with st.sidebar:
        env_key = os.environ.get("GEMINI_API_KEY", "")
        if env_key:
            api_key = env_key
            st.success("🔑 API key loaded")
        else:
            api_key = st.text_input("Gemini API Key", type="password")

        model = st.selectbox("Model", MODELS)
        stage = st.selectbox("Stage", list(STAGE_MODES.keys()))
        temp = st.slider("Temperature", 0.0, 2.0, 0.7, 0.05)
        mode = st.radio("Mode", ["🔬 Single", "📋 Batch CSV", "📊 Analytics"])

    if "all_results" not in st.session_state:
        st.session_state.all_results = []

    # ── SINGLE ──
    if mode == "🔬 Single":
        prompt = st.text_area("Prompt", height=100)
        if st.button("▶ Run", type="primary", disabled=not api_key or not prompt):
            client = genai.Client(api_key=api_key)
            with st.spinner("⏳"):
                try: r = evaluate(client, model, prompt, stage, temp)
                except Exception as e:
                    st.error(str(e)); return

            st.markdown(r["response"])
            st.markdown("---")

            ic = {"STABLE":"🟢","DRIFTING":"🟡"}.get(r["status"],"🔴")
            st.markdown(f"### {ic} {r['status']} · Bal = {r['bal']:.3f}")

            def bar(v): f=int(round(max(0,min(1,v))*10)); return "█"*f+"░"*(10-f)
            st.code(
                f"E [{bar(r['E'])}] {r['E']:.2f}  {r['E_reason']}\n"
                f"F [{bar(r['F'])}] {r['F']:.2f}  {r['F_reason']}\n"
                f"N [{bar(r['N'])}] {r['N']:.2f}  {r['N_reason']}\n"
                f"B {r['B']:+.2f}  {r['B_reason']}", language=None)

            st.session_state.all_results.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt, "category": "manual",
                **{k: r[k] for k in ["E","F","N","B","bal","status"]}
            })

    # ── BATCH ──
    elif mode == "📋 Batch CSV":
        st.markdown("### 📋 Batch")
        st.markdown("CSV: `prompt,category,language`")
        st.caption("Categories: factual, philosophical, creative, technical, ethical, directive, personal")

        csv = st.file_uploader("Upload CSV", type=["csv"])
        if csv:
            try:
                df = pd.read_csv(StringIO(csv.getvalue().decode("utf-8")))
                if "prompt" not in df.columns:
                    st.error("Need 'prompt' column"); return
                if "category" not in df.columns: df["category"] = "other"
                if "language" not in df.columns: df["language"] = "auto"
                st.dataframe(df, use_container_width=True, hide_index=True)
            except Exception as e:
                st.error(str(e)); return

            n_rep = st.slider("Repeats per prompt", 1, 10, 1)
            total = len(df) * n_rep
            st.info(f"{len(df)} prompts × {n_rep} = {total} runs")

            if st.button("🚀 Run", type="primary", disabled=not api_key):
                client = genai.Client(api_key=api_key)
                bar = st.progress(0)
                results = []
                done = 0

                for _, row in df.iterrows():
                    p = str(row["prompt"])
                    cat = str(row.get("category","other"))
                    for rep in range(n_rep):
                        done += 1
                        bar.progress(done/total, f"[{done}/{total}] {p[:40]}...")
                        try:
                            r = evaluate(client, model, p, stage, temp)
                            results.append({
                                "timestamp": datetime.now().isoformat(),
                                "prompt": p, "category": cat,
                                "language": str(row.get("language","auto")),
                                "model": model,
                                **{k: r[k] for k in ["E","F","N","B","bal","status",
                                    "E_reason","F_reason","N_reason","B_reason"]},
                                "response_preview": r["response"][:200],
                            })
                        except Exception as e:
                            st.warning(f"⚠ {p[:30]}: {e}")
                        time.sleep(0.3)

                bar.progress(1.0, "Done!")

                if results:
                    rdf = pd.DataFrame(results)
                    st.dataframe(
                        rdf[["prompt","category","E","F","N","B","bal","status"]].style.format(
                            {"E":"{:.2f}","F":"{:.2f}","N":"{:.2f}","B":"{:+.2f}","bal":"{:.3f}"}),
                        use_container_width=True, hide_index=True)

                    # Variance
                    if n_rep > 1:
                        st.markdown("### Variance")
                        for p in df["prompt"].unique():
                            sub = rdf[rdf["prompt"]==p]
                            if len(sub)>1:
                                st.markdown(f"**{p[:60]}** (n={len(sub)})")
                                st.dataframe(pd.DataFrame({
                                    a: {"mean":sub[a].mean(),"std":sub[a].std(),"range":sub[a].max()-sub[a].min()}
                                    for a in ["E","F","N","B","bal"]}).T.round(4), use_container_width=True)

                    # Auto-obs
                    st.markdown("### 🤖 Observations")
                    obs = auto_observe(results)
                    for sev, txt in obs:
                        ic = {"CRITICAL":"🔴","HIGH":"🟠","INFO":"🔵"}.get(sev,"⚪")
                        st.markdown(f"{ic} **{sev}**: {txt}")

                    # Save to session
                    st.session_state.all_results.extend(results)

                    # Download
                    st.download_button("📥 Download CSV",
                        rdf.to_csv(index=False),
                        f"tri_tfm_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")

    # ── ANALYTICS ──
    elif mode == "📊 Analytics":
        st.markdown("### 📊 Analytics")

        # Can load from session or upload
        df = pd.DataFrame(st.session_state.all_results) if st.session_state.all_results else pd.DataFrame()

        up = st.file_uploader("Or upload previous CSV", type=["csv"], key="an")
        if up:
            loaded = pd.read_csv(up)
            for c in ["E","F","N","B","bal"]:
                if c in loaded.columns: loaded[c] = pd.to_numeric(loaded[c], errors="coerce")
            df = pd.concat([df, loaded], ignore_index=True) if not df.empty else loaded

        if df.empty:
            st.info("Run some tests or upload CSV."); return

        st.markdown(f"**{len(df)} measurements**")

        # Stats
        stats = {a: {"mean":df[a].mean(),"std":df[a].std(),"min":df[a].min(),"max":df[a].max()}
                 for a in ["E","F","N","B","bal"] if a in df.columns}
        st.dataframe(pd.DataFrame(stats).T.round(4), use_container_width=True)

        # Histograms
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for a in ["E","F","N"]:
                if a in df.columns:
                    fig.add_trace(go.Histogram(x=df[a],name=a,opacity=0.7,xbins=dict(size=0.05)))
            fig.update_layout(barmode="overlay",title="E/F/N",height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "bal" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df["bal"],marker_color="purple",opacity=0.7,xbins=dict(size=0.05)))
                fig.add_vline(x=0.70,line_dash="dash",line_color="green",annotation_text="STABLE")
                fig.update_layout(title="Balance",height=350)
                st.plotly_chart(fig, use_container_width=True)

        # By category
        if "category" in df.columns and df["category"].nunique()>1:
            st.markdown("### By category")
            cs = df.groupby("category")[["E","F","N","bal"]].mean().round(3)
            st.dataframe(cs, use_container_width=True)
            fig = go.Figure()
            for a in ["E","F","N"]:
                fig.add_trace(go.Bar(x=cs.index,y=cs[a],name=a))
            fig.update_layout(barmode="group",height=400)
            st.plotly_chart(fig, use_container_width=True)

        # E vs F
        if "category" in df.columns:
            fig = px.scatter(df,x="F",y="E",color="category",hover_data=["bal"],height=400,title="E vs F")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
