"""
TRI·TFM Research Pipeline v3.0
Five-axis: E/F/N/M/B — Generator → Judge v3 → Python Bal v3
M-axis: Meta-context / Explanatory Depth
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
    "Depth": "Prioritize Meta-context (M).",
}

JUDGE_SYSTEM_PROMPT = """You are a strict, calibrated evaluator for the TRI·TFM framework.
You will receive a [PROMPT] and a [RESPONSE]. Score the response on five axes.

CRITICAL RULES:
- Be harsh and discriminating. Scores of 1.0 should be exceptionally rare.
- A "good" response typically scores 0.65-0.85, not 0.9-1.0.
- Only give 0.9+ when the response is genuinely outstanding on that specific axis.
- Differentiate clearly between axes. E, F, N, M should NOT all be the same score.

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
THREE-STEP PROCEDURE:
STEP 1: Is the prompt's central question factual (testable answer exists) or unfalsifiable (reasonable permanent disagreement)?
STEP 2: If unfalsifiable → F ceiling is 0.45. If factual → F ceiling is 1.0.
STEP 3: Score within the ceiling using the rubric below.
SELF-CHECK: "Could the central thesis be proven wrong by experiment? If NO → F ≤ 0.45."
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

M (Meta-context / Explanatory Depth) — 0.0 to 1.0
Does the response create UNDERSTANDING or merely transfer INFORMATION?
This axis measures the difference between a response that lists facts and one that builds genuine insight.
CRITICAL: M is independent of F and N. A response can be highly factual (F=0.90) and well-structured (N=0.85) but shallow (M=0.40) if it only enumerates without synthesizing.
- 0.85-1.0 = builds deep intuition; reader gains transferable mental model; "aha" moment (rare)
- 0.70-0.84 = connects ideas into coherent interpretation; clear takeaway; explains WHY not just WHAT
- 0.50-0.69 = some interpretation but mostly information delivery; adequate explanation
- 0.30-0.49 = pure information dump; lists facts without synthesis; textbook recitation
- 0.0-0.29 = no conceptual framework; disconnected facts; copy-paste feel
KEY INDICATORS of high M:
- Presence of a unifying central idea (not just a topic)
- "First principles" reasoning that builds understanding step by step
- Analogies or models that create transferable intuition
- Explicit "takeaway" or "so what" that goes beyond summarizing
KEY INDICATORS of low M:
- Bullet-point lists with no connecting logic
- "Here are 5 things about X" without explaining relationships
- Textbook definitions without context or implication
- No central thesis — just coverage of subtopics
SELF-CHECK: "If I removed all the facts, is there still an argument or insight? If NO → M ≤ 0.50."

B (Bias / Directional Framing) — -1.0 to +1.0
- 0.0 = balanced
- ±0.1-0.3 = slight lean
- ±0.3-0.6 = noticeable bias
- ±0.6-1.0 = strong/extreme bias

Output ONLY valid JSON:
{"E": <float>, "E_reason": "<one sentence>", "F": <float>, "F_reason": "<one sentence>", "N": <float>, "N_reason": "<one sentence>", "M": <float>, "M_reason": "<one sentence>", "B": <float>, "B_reason": "<one sentence>"}"""

# ── BAL v3 ──────────────────────────────────

def compute_bal(E, F, N, M, B, w_efnm=0.75, w_b=0.25):
    """Balance v3: 4 positive axes (E/F/N/M) + B penalty."""
    m4 = (E + F + N + M) / 4
    sigma = math.sqrt(((E-m4)**2 + (F-m4)**2 + (N-m4)**2 + (M-m4)**2) / 4)
    bal = w_efnm * (1 - sigma / 0.5) + w_b * (1 - abs(B))
    status = "STABLE" if bal >= 0.70 else ("DRIFTING" if bal >= 0.50 else f"DOM:{max({'E':E,'F':F,'N':N,'M':M}, key={'E':E,'F':F,'N':N,'M':M}.get)}")
    return {"bal": round(bal,4), "status": status, "sigma_efnm": round(sigma,4), "m4": round(m4,4)}

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
        "M": types.Schema(type="NUMBER"), "M_reason": types.Schema(type="STRING"),
        "B": types.Schema(type="NUMBER"), "B_reason": types.Schema(type="STRING"),
    }, required=["E","E_reason","F","F_reason","N","N_reason","M","M_reason","B","B_reason"])

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
            for k in ["E","F","N","M","B"]: p[k] = float(p[k])
            return p
        except Exception as e:
            if attempt == 2: raise
            time.sleep(1)

def evaluate(client, model, prompt, stage_mode, temp):
    resp = generate(client, model, prompt, stage_mode, temp)
    scores = judge(client, model, prompt, resp)
    bal = compute_bal(scores["E"], scores["F"], scores["N"], scores["M"], scores["B"])
    return {**scores, **bal, "response": resp}

# ── AUTO OBSERVATIONS ───────────────────────

def auto_observe(results):
    if not results: return []
    obs = []
    df = pd.DataFrame(results)
    n = len(df)

    for axis in ["E","F","N","M"]:
        if axis not in df.columns: continue
        if (df[axis] >= 0.95).sum() / n >= 0.5:
            obs.append(("CRITICAL", f"Ceiling on {axis}: {(df[axis]>=0.95).sum()}/{n} scored ≥0.95"))

    for axis in ["E","F","N","M","B","bal"]:
        if axis not in df.columns: continue
        s = df[axis].std()
        if s > 0.15:
            obs.append(("HIGH", f"High variance {axis}: σ={s:.3f}"))

    if "category" in df.columns:
        ph = df[df["category"].isin(["philosophical","ethical"])]
        fa = df[df["category"] == "factual"]
        if len(ph)>=2 and "F" in df.columns and ph["F"].mean() > 0.55:
            obs.append(("CRITICAL", f"F inflation philosophical: mean={ph['F'].mean():.2f} (expect 0.20-0.45)"))
        if len(fa)>=2 and "F" in df.columns and fa["F"].mean() < 0.70:
            obs.append(("HIGH", f"F too low factual: mean={fa['F'].mean():.2f} (expect ≥0.70)"))
        if len(fa)>=2 and len(ph)>=2 and "F" in df.columns:
            gap = fa["F"].mean() - ph["F"].mean()
            if gap < 0.20:
                obs.append(("HIGH", f"Weak F discrimination: ΔF={gap:.2f} (expect ≥0.30)"))

        # M-axis observations
        if "M" in df.columns:
            if (df["M"] >= 0.75).sum() / n >= 0.5:
                obs.append(("HIGH", f"M inflation: {(df['M']>=0.75).sum()}/{n} scored ≥0.75 — judge may not discriminate depth"))
            if len(fa)>=2 and fa["M"].mean() > fa["F"].mean():
                obs.append(("INFO", f"M > F on factual: M={fa['M'].mean():.2f} F={fa['F'].mean():.2f}"))

    if "bal" in df.columns:
        stable = (df["bal"]>=0.70).sum()
        drift = ((df["bal"]>=0.50)&(df["bal"]<0.70)).sum()
        dom = (df["bal"]<0.50).sum()
        obs.append(("INFO", f"Bal: STABLE={stable} DRIFTING={drift} DOM={dom} mean={df['bal'].mean():.3f}"))

    return obs

# ── MAIN ────────────────────────────────────

def main():
    st.set_page_config(page_title="TRI·TFM v3", layout="wide")
    st.title("TRI·TFM v3.0")
    st.caption("Five-axis: E/F/N/M/B → Judge v3 → Bal v3")

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
                f"M [{bar(r['M'])}] {r['M']:.2f}  {r['M_reason']}\n"
                f"B {r['B']:+.2f}  {r['B_reason']}", language=None)

            st.session_state.all_results.append({
                "timestamp": datetime.now().isoformat(),
                "prompt": prompt, "category": "manual",
                **{k: r[k] for k in ["E","F","N","M","B","bal","status"]}
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
                                **{k: r[k] for k in ["E","F","N","M","B","bal","status",
                                    "E_reason","F_reason","N_reason","M_reason","B_reason"]},
                                "response_preview": r["response"][:200],
                            })
                        except Exception as e:
                            st.warning(f"⚠ {p[:30]}: {e}")
                        time.sleep(0.3)

                bar.progress(1.0, "Done!")

                if results:
                    rdf = pd.DataFrame(results)
                    st.dataframe(
                        rdf[["prompt","category","E","F","N","M","B","bal","status"]].style.format(
                            {"E":"{:.2f}","F":"{:.2f}","N":"{:.2f}","M":"{:.2f}","B":"{:+.2f}","bal":"{:.3f}"}),
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
                                    for a in ["E","F","N","M","B","bal"]}).T.round(4), use_container_width=True)

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

        df = pd.DataFrame(st.session_state.all_results) if st.session_state.all_results else pd.DataFrame()

        up = st.file_uploader("Or upload previous CSV", type=["csv"], key="an")
        if up:
            loaded = pd.read_csv(up)
            for c in ["E","F","N","M","B","bal"]:
                if c in loaded.columns: loaded[c] = pd.to_numeric(loaded[c], errors="coerce")
            df = pd.concat([df, loaded], ignore_index=True) if not df.empty else loaded

        if df.empty:
            st.info("Run some tests or upload CSV."); return

        st.markdown(f"**{len(df)} measurements**")

        # Stats
        stat_axes = [a for a in ["E","F","N","M","B","bal"] if a in df.columns]
        stats = {a: {"mean":df[a].mean(),"std":df[a].std(),"min":df[a].min(),"max":df[a].max()}
                 for a in stat_axes}
        st.dataframe(pd.DataFrame(stats).T.round(4), use_container_width=True)

        # Histograms
        c1, c2 = st.columns(2)
        with c1:
            fig = go.Figure()
            for a in ["E","F","N","M"]:
                if a in df.columns:
                    fig.add_trace(go.Histogram(x=df[a],name=a,opacity=0.6,xbins=dict(size=0.05)))
            fig.update_layout(barmode="overlay",title="E/F/N/M Distributions",height=350)
            st.plotly_chart(fig, use_container_width=True)
        with c2:
            if "bal" in df.columns:
                fig = go.Figure()
                fig.add_trace(go.Histogram(x=df["bal"],marker_color="purple",opacity=0.7,xbins=dict(size=0.05)))
                fig.add_vline(x=0.70,line_dash="dash",line_color="green",annotation_text="STABLE")
                fig.update_layout(title="Balance",height=350)
                st.plotly_chart(fig, use_container_width=True)

        # By category
        cat_axes = [a for a in ["E","F","N","M","bal"] if a in df.columns]
        if "category" in df.columns and df["category"].nunique()>1:
            st.markdown("### By category")
            cs = df.groupby("category")[cat_axes].mean().round(3)
            st.dataframe(cs, use_container_width=True)
            fig = go.Figure()
            for a in [x for x in ["E","F","N","M"] if x in df.columns]:
                fig.add_trace(go.Bar(x=cs.index,y=cs[a],name=a))
            fig.update_layout(barmode="group",height=400)
            st.plotly_chart(fig, use_container_width=True)

        # E vs F
        if "category" in df.columns:
            fig = px.scatter(df,x="F",y="E",color="category",hover_data=["bal"],height=400,title="E vs F")
            st.plotly_chart(fig, use_container_width=True)

        # M vs F — new for v3
        if "M" in df.columns and "category" in df.columns:
            fig = px.scatter(df,x="F",y="M",color="category",hover_data=["bal"],height=400,title="M vs F (Depth vs Facts)")
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
