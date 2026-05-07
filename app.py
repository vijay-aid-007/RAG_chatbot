"""
app.py
======
Gradio-based chatbot UI — fully compatible with Gradio 5.x
and huggingface-hub >= 1.5.0.

Install (clean slate):
    pip install "gradio>=5.0,<6.0" "huggingface-hub>=1.5.0,<2.0" --force-reinstall

Run:
    python app.py
    # Opens at http://localhost:7860
"""

from __future__ import annotations
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import gradio as gr
from core.chatbot import answer
from core.persona_extractor import load_persona
from config import (
    PERSONA_JSON,
    TOPIC_CHECKPOINTS,
    MSG_CHECKPOINTS,
    FAISS_INDEX_PATH,
)


# ── Pipeline / Persona helpers ─────────────────────────────────────────────────

def _pipeline_status() -> str:
    checks = {
        "Persona JSON"        : PERSONA_JSON,
        "Topic Checkpoints"   : TOPIC_CHECKPOINTS,
        "100-msg Checkpoints" : MSG_CHECKPOINTS,
        "FAISS Index"         : FAISS_INDEX_PATH,
    }
    lines: list[str] = []
    all_ok = True
    for label, path in checks.items():
        ok = Path(path).exists()
        lines.append(f"{'✅' if ok else '❌'}  {label}")
        if not ok:
            all_ok = False

    header = (
        "**✅ System ready.**"
        if all_ok
        else "**⚠️  Run `python build_pipeline.py` first.**"
    )
    return header + "\n\n" + "\n\n".join(lines)


def _persona_summary() -> str:
    try:
        p = load_persona()
    except FileNotFoundError:
        return "_Persona not built yet. Run `python build_pipeline.py`._"
    except Exception as exc:
        return f"_Error loading persona: {exc}_"

    lines: list[str] = ["### 🧠 User Persona"]

    def _bullet_list(items: list, limit: int = 6) -> str:
        return "  \n".join(f"• {i}" for i in items[:limit])

    if p.get("habits"):
        lines.append("**Habits**  \n" + _bullet_list(p["habits"]))
    if p.get("personal_facts"):
        lines.append("**Personal Facts**  \n" + _bullet_list(p["personal_facts"]))
    if p.get("personality_traits"):
        lines.append("**Personality Traits**  \n" + _bullet_list(p["personality_traits"]))
    if p.get("hobbies"):
        lines.append("**Hobbies**  \n" + _bullet_list(p["hobbies"]))

    cs = p.get("communication_style", {})
    if cs:
        lines.append(
            f"**Communication Style**  \n"
            f"• Tone: {cs.get('tone', '?')}  \n"
            f"• Message length: {cs.get('message_length', '?')}  \n"
            f"• Emojis: {cs.get('emoji_usage', '?')}"
        )

    return "\n\n".join(lines)


# ── Chat function ──────────────────────────────────────────────────────────────

def chat(
    user_message: str,
    history: list[dict],
) -> tuple[str, list[dict]]:
    """
    Gradio 5.x messages format:
        history is a list of {"role": "user"|"assistant", "content": str}
    """
    user_message = user_message.strip()
    if not user_message:
        return "", history

    try:
        bot_reply = answer(user_message)
    except Exception as exc:
        bot_reply = f"⚠️  Error generating answer: {exc}"

    history = history + [
        {"role": "user",      "content": user_message},
        {"role": "assistant", "content": bot_reply},
    ]
    return "", history


def clear_chat() -> tuple[str, list]:
    return "", []


# ── Example questions ──────────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    ["What kind of person is this user?"],
    ["What are their daily habits?"],
    ["How do they communicate — tone, style, emoji usage?"],
    ["What are their hobbies and interests?"],
    ["What personal facts do you know about this person?"],
    ["Which topics appear most often in the conversations?"],
    ["Is this user more serious or humorous?"],
    ["What does this user typically talk about with their contacts?"],
]


# ── CSS ────────────────────────────────────────────────────────────────────────

css = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Bricolage+Grotesque:opsz,wght@12..96,400;12..96,600;12..96,700;12..96,800&display=swap');

/* ═══════════════ DESIGN TOKENS ═══════════════ */
:root {
  --bg-base:       #070B12;
  --bg-surface:    #0D1220;
  --bg-elevated:   #111827;

  --border-subtle:  rgba(255,255,255,0.055);
  --border-default: rgba(255,255,255,0.09);
  --border-focus:   rgba(99,102,241,0.55);

  --accent:        #6366F1;
  --accent-hi:     #818CF8;
  --accent-dim:    rgba(99,102,241,0.12);
  --accent-glow:   rgba(99,102,241,0.22);

  --green:         #10B981;
  --green-dim:     rgba(16,185,129,0.1);
  --amber:         #F59E0B;
  --red:           #F87171;

  --tx-1:  #F1F5F9;
  --tx-2:  #94A3B8;
  --tx-3:  #475569;
  --tx-4:  #1E293B;

  --r-sm: 8px;
  --r-md: 12px;
  --r-lg: 16px;

  --font-d: 'Bricolage Grotesque', sans-serif;
  --font-b: 'Plus Jakarta Sans', sans-serif;

  --shadow: 0 1px 3px rgba(0,0,0,0.45), 0 6px 24px rgba(0,0,0,0.2);
}

/* ═══════════════ RESET ═══════════════ */
*, *::before, *::after { box-sizing: border-box; }

body {
  font-family: var(--font-b) !important;
  background: var(--bg-base) !important;
  color: var(--tx-1) !important;
  -webkit-font-smoothing: antialiased;
}

/* Ambient glow */
body::before {
  content:'';
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background:
    radial-gradient(ellipse 65% 50% at 12% 0%,  rgba(99,102,241,0.08) 0%, transparent 60%),
    radial-gradient(ellipse 45% 40% at 88% 100%, rgba(16,185,129,0.05) 0%, transparent 58%);
  animation: glow 22s ease-in-out infinite alternate;
}
@keyframes glow {
  from { opacity:1; transform:scale(1); }
  to   { opacity:0.55; transform:scale(1.07); }
}

.gradio-container {
  background: var(--bg-base) !important;
  max-width: 100% !important;
  padding: 0 !important;
  font-family: var(--font-b) !important;
  position: relative; z-index: 1;
}

/* ═══════════════ NAV ═══════════════ */
#rag-nav {
  display: flex; align-items: center; gap: 14px;
  padding: 16px 28px 14px;
  border-bottom: 1px solid var(--border-subtle);
  animation: slideDown .45s cubic-bezier(.22,1,.36,1) both;
}
@keyframes slideDown {
  from { opacity:0; transform:translateY(-10px); }
  to   { opacity:1; transform:translateY(0); }
}

.n-logo {
  width:38px; height:38px;
  background: linear-gradient(135deg,#6366F1,#8B5CF6);
  border-radius:10px;
  display:flex; align-items:center; justify-content:center;
  font-size:18px; flex-shrink:0;
  box-shadow: 0 0 0 1px rgba(99,102,241,.3), 0 4px 16px rgba(99,102,241,.22);
}
.n-title {
  font-family:var(--font-d); font-weight:700; font-size:1.1rem;
  letter-spacing:-.025em; color:var(--tx-1); line-height:1;
}
.n-sub { font-size:.73rem; color:var(--tx-3); margin-top:2px; font-weight:300; }
.n-space { flex:1; }
.n-badge {
  display:flex; align-items:center; gap:7px;
  background:var(--green-dim); border:1px solid rgba(16,185,129,.2);
  color:var(--green); font-size:.68rem; font-weight:600;
  letter-spacing:.06em; text-transform:uppercase;
  padding:5px 13px; border-radius:999px;
}
.n-dot {
  width:6px; height:6px; background:var(--green);
  border-radius:50%; box-shadow:0 0 6px var(--green);
  animation:blink 2s ease-in-out infinite;
}
@keyframes blink {
  0%,100%{opacity:1;} 50%{opacity:.3;}
}

/* ═══════════════ MAIN PADDING ═══════════════ */
.main-wrap { padding: 18px 28px 24px; }

/* ═══════════════ CHATBOT ═══════════════ */
#chatbox {
  border: 1px solid var(--border-subtle) !important;
  border-radius: var(--r-lg) !important;
  background: var(--bg-surface) !important;
  box-shadow: var(--shadow) !important;
  animation: fadeUp .5s cubic-bezier(.22,1,.36,1) .05s both;
}
@keyframes fadeUp {
  from { opacity:0; transform:translateY(12px); }
  to   { opacity:1; transform:translateY(0); }
}

/* Gradio 5 internal bubble selectors */
#chatbox .message.user {
  background: linear-gradient(135deg,rgba(99,102,241,.18),rgba(139,92,246,.12)) !important;
  border: 1px solid rgba(99,102,241,.22) !important;
  border-radius: 14px 14px 3px 14px !important;
  color: var(--tx-1) !important;
  font-size:.9rem !important; line-height:1.65 !important;
  box-shadow: 0 2px 10px rgba(99,102,241,.08) !important;
}
#chatbox .message.bot {
  background: var(--bg-elevated) !important;
  border: 1px solid var(--border-subtle) !important;
  border-radius: 14px 14px 14px 3px !important;
  color: var(--tx-2) !important;
  font-size:.88rem !important; line-height:1.7 !important;
}

/* ═══════════════ INPUT BAR ═══════════════ */
.input-bar {
  display:flex; gap:8px; align-items:center;
  background: var(--bg-surface);
  border: 1px solid var(--border-subtle);
  border-radius: var(--r-lg);
  padding: 7px 7px 7px 4px;
  box-shadow: var(--shadow);
  transition: border-color .2s ease, box-shadow .2s ease;
  margin-top: 10px;
}
.input-bar:focus-within {
  border-color: var(--border-focus);
  box-shadow: 0 0 0 1px var(--border-focus), 0 0 20px var(--accent-glow);
}

#msg-box .wrap,
#msg-box .wrap-inner,
#msg-box label,
#msg-box > div { border:none !important; background:transparent !important; box-shadow:none !important; padding:0 !important; }

#msg-box textarea,
#msg-box input[type=text] {
  background:transparent !important; border:none !important;
  box-shadow:none !important; outline:none !important;
  color:var(--tx-1) !important; font-family:var(--font-b) !important;
  font-size:.91rem !important; font-weight:400 !important;
  padding:9px 14px !important; resize:none !important;
  caret-color:var(--accent-hi);
}
#msg-box textarea::placeholder { color:var(--tx-3) !important; font-weight:300 !important; }

#send-btn {
  background:var(--accent) !important; border:none !important;
  border-radius:var(--r-md) !important; color:#fff !important;
  font-family:var(--font-d) !important; font-weight:600 !important;
  font-size:.84rem !important; letter-spacing:.02em !important;
  padding:10px 22px !important; min-width:86px !important; height:40px !important;
  cursor:pointer !important; flex-shrink:0;
  transition: background .18s, transform .15s, box-shadow .18s !important;
  box-shadow: 0 1px 2px rgba(0,0,0,.3), 0 4px 14px rgba(99,102,241,.28) !important;
}
#send-btn:hover {
  background:var(--accent-hi) !important;
  transform:translateY(-1px) !important;
  box-shadow:0 2px 4px rgba(0,0,0,.3), 0 8px 22px rgba(99,102,241,.38) !important;
}
#send-btn:active { transform:translateY(0) !important; }

#clear-btn {
  background:transparent !important;
  border:1px solid var(--border-default) !important;
  border-radius:var(--r-md) !important; color:var(--tx-3) !important;
  font-family:var(--font-b) !important; font-size:.82rem !important;
  padding:10px 16px !important; height:40px !important;
  cursor:pointer !important; flex-shrink:0;
  transition: all .18s ease !important;
}
#clear-btn:hover {
  border-color:rgba(248,113,113,.35) !important;
  color:var(--red) !important; background:rgba(248,113,113,.05) !important;
}

/* ═══════════════ EXAMPLES ═══════════════ */
.gr-examples .label-wrap span,
.examples-wrap label {
  font-size:.7rem !important; font-weight:600 !important;
  letter-spacing:.07em !important; text-transform:uppercase !important;
  color:var(--tx-3) !important;
}
.gr-examples table td {
  background:var(--bg-surface) !important;
  border:1px solid var(--border-subtle) !important;
  border-radius:var(--r-sm) !important;
  color:var(--tx-2) !important; font-family:var(--font-b) !important;
  font-size:.79rem !important; padding:6px 13px !important;
  cursor:pointer !important; white-space:nowrap;
  transition:all .15s ease !important;
}
.gr-examples table td:hover {
  background:var(--accent-dim) !important;
  border-color:rgba(99,102,241,.28) !important;
  color:var(--accent-hi) !important;
}

/* ═══════════════ SIDEBAR ═══════════════ */
.rag-aside { animation: fadeUp .55s cubic-bezier(.22,1,.36,1) .12s both; }

/* Accordion */
.gr-accordion, details.gr-accordion {
  background:var(--bg-surface) !important;
  border:1px solid var(--border-subtle) !important;
  border-radius:var(--r-lg) !important;
  box-shadow:var(--shadow) !important;
  overflow:hidden !important; margin-bottom:0 !important;
  transition:border-color .2s !important;
}
.gr-accordion:hover { border-color:var(--border-default) !important; }

.gr-accordion > .label-wrap,
details.gr-accordion > summary {
  background:var(--bg-surface) !important;
  padding:12px 16px !important;
  border-bottom:1px solid var(--border-subtle) !important;
}
.gr-accordion > .label-wrap span,
details.gr-accordion > summary span {
  font-family:var(--font-d) !important;
  font-size:.75rem !important; font-weight:600 !important;
  letter-spacing:.05em !important; text-transform:uppercase !important;
  color:var(--tx-2) !important;
}
.gr-accordion .gr-accordion-content,
details.gr-accordion > div {
  background:var(--bg-surface) !important;
  padding:14px 16px !important;
}

/* Sidebar prose */
.sidebar-card p, .sidebar-card li {
  font-size:.81rem !important; color:var(--tx-2) !important;
  line-height:1.72 !important; font-weight:400 !important;
}
.sidebar-card strong {
  display:block; color:var(--tx-1) !important; font-weight:600 !important;
  font-size:.72rem !important; letter-spacing:.04em !important;
  text-transform:uppercase !important; margin-top:10px; margin-bottom:1px;
}
.sidebar-card h3 {
  font-family:var(--font-d) !important; font-size:.86rem !important;
  font-weight:700 !important; color:var(--accent-hi) !important;
  margin-bottom:10px !important; letter-spacing:-.01em !important;
}

/* Refresh btn */
#refresh-btn {
  background:var(--accent-dim) !important;
  border:1px solid rgba(99,102,241,.2) !important;
  border-radius:var(--r-sm) !important; color:var(--accent-hi) !important;
  font-family:var(--font-b) !important; font-size:.77rem !important;
  font-weight:500 !important; padding:7px 0 !important;
  width:100% !important; cursor:pointer !important; margin-top:10px !important;
  transition:all .18s !important;
}
#refresh-btn:hover {
  background:rgba(99,102,241,.2) !important;
  border-color:rgba(99,102,241,.4) !important;
}

/* ═══════════════ FOOTER HIDE ═══════════════ */
footer { display:none !important; }

/* ═══════════════ SCROLLBAR ═══════════════ */
::-webkit-scrollbar { width:4px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:rgba(99,102,241,.2); border-radius:99px; }
::-webkit-scrollbar-thumb:hover { background:rgba(99,102,241,.4); }
"""

# ── UI ─────────────────────────────────────────────────────────────────────────

with gr.Blocks(
    title="RAG Conversation Chatbot",
    css=css,
    theme=gr.themes.Base(
        primary_hue="indigo",
        secondary_hue="violet",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif"],
    ),
) as demo:

    # ── Top nav (pure HTML — fully controlled) ────────────────────────────────
    gr.HTML("""
    <div id="rag-nav">
      <div class="n-logo">🤖</div>
      <div>
        <div class="n-title">RAG Chatbot</div>
        <div class="n-sub">Conversation Intelligence Platform</div>
      </div>
      <div class="n-space"></div>
      <div class="n-badge"><span class="n-dot"></span>Live</div>
    </div>
    """)

    # ── Body ──────────────────────────────────────────────────────────────────
    with gr.Row(equal_height=False, elem_classes=["main-wrap"]):

        # ── Chat panel ────────────────────────────────────────────────────────
        with gr.Column(scale=3, min_width=420):

            chatbot_ui = gr.Chatbot(
                elem_id="chatbox",
                label="",
                height=490,
                type="messages",
                bubble_full_width=False,
                show_label=False,
                avatar_images=(None, None),
            )

            with gr.Row(elem_classes=["input-bar"]):
                msg_box = gr.Textbox(
                    placeholder="Ask anything about the user…",
                    label="",
                    show_label=False,
                    lines=1,
                    scale=5,
                    autofocus=True,
                    elem_id="msg-box",
                    container=False,
                )
                send_btn = gr.Button(
                    "Send ➤",
                    variant="primary",
                    scale=1,
                    elem_id="send-btn",
                )
                clear_btn = gr.Button(
                    "🗑 Clear",
                    variant="secondary",
                    scale=1,
                    elem_id="clear-btn",
                )

            gr.Examples(
                examples=EXAMPLE_QUESTIONS,
                inputs=[msg_box],
                label="💡 Example questions — click to load",
                examples_per_page=8,
            )

        # ── Sidebar ───────────────────────────────────────────────────────────
        with gr.Column(scale=1, min_width=268, elem_classes=["rag-aside"]):

            with gr.Accordion("📊 System Status", open=True):
                status_md = gr.Markdown(
                    _pipeline_status(),
                    elem_classes=["sidebar-card"],
                )
                refresh_btn = gr.Button(
                    "🔄  Refresh status",
                    size="sm",
                    elem_id="refresh-btn",
                )

            with gr.Accordion("🧠 Persona Summary", open=True):
                gr.Markdown(
                    _persona_summary(),
                    elem_classes=["sidebar-card"],
                )

            with gr.Accordion("ℹ️ How it works", open=False):
                gr.Markdown(
                    """
1. **Pipeline** — conversations are chunked, embedded, and indexed in FAISS.  
2. **Persona** — a structured persona is extracted from topic/message checkpoints.  
3. **RAG** — your question retrieves the most relevant chunks; the LLM answers using them plus the persona.
                    """,
                    elem_classes=["sidebar-card"],
                )

    # ── Events ────────────────────────────────────────────────────────────────

    send_btn.click(
        fn=chat,
        inputs=[msg_box, chatbot_ui],
        outputs=[msg_box, chatbot_ui],
        queue=True,
    )

    msg_box.submit(
        fn=chat,
        inputs=[msg_box, chatbot_ui],
        outputs=[msg_box, chatbot_ui],
        queue=True,
    )

    clear_btn.click(
        fn=clear_chat,
        inputs=[],
        outputs=[msg_box, chatbot_ui],
        queue=False,
    )

    refresh_btn.click(
        fn=_pipeline_status,
        inputs=[],
        outputs=[status_md],
        queue=False,
    )


# ── Launch ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
    )