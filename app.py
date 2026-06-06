import streamlit as st
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch

st.set_page_config(page_title="Mental Health Support",
                   page_icon="🌿", layout="centered")

st.markdown("""
<style>
.chat-user {
    background:#d8f3dc; border-radius:18px 18px 4px 18px;
    padding:10px 16px; margin:6px 0 6px auto;
    max-width:72%; color:#1b4332; font-size:0.95rem;
}
.chat-bot {
    background:#ffffff; border-radius:18px 18px 18px 4px;
    padding:10px 16px; margin:6px auto 6px 0;
    max-width:72%; color:#2d3748; font-size:0.95rem;
    border:1px solid #e2e8f0;
}
.disclaimer {
    background:#fff3cd; border-left:4px solid #f6a623;
    padding:10px 14px; border-radius:6px;
    font-size:0.82rem; color:#856404; margin-bottom:1rem;
}
</style>
""", unsafe_allow_html=True)

MODEL_PATH = "./mental_health_chatbot/model/final"

@st.cache_resource
def load_model():
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    mod = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    tok.pad_token = tok.eos_token
    gen = pipeline('text-generation', model=mod, tokenizer=tok,
                   device=0 if torch.cuda.is_available() else -1)
    return gen, tok

def get_response(gen, tok, user_input):
    prompt = f"User: {user_input}\nBot:"
    result = gen(prompt, max_new_tokens=60, temperature=0.75,
                 top_p=0.92, do_sample=True,
                 pad_token_id=tok.eos_token_id)
    full  = result[0]['generated_text']
    reply = full.split("Bot:")[-1].strip()
    reply = reply.split("\nUser:")[0].strip()
    return reply if reply else "I'm here for you. Can you tell me more?"

st.markdown("## 🌿 Mental Health Support Chat")
st.markdown("*A safe space to share how you feel.*")
st.markdown("""<div class="disclaimer">
⚠️ <b>Disclaimer:</b> This chatbot is not a substitute for professional
mental health care. If you are in crisis please contact a helpline.
</div>""", unsafe_allow_html=True)

if "messages" not in st.session_state:
    st.session_state.messages = [{"role":"bot",
        "text":"Hello 👋 I'm here to listen. How are you feeling today?"}]

for msg in st.session_state.messages:
    css = "chat-user" if msg["role"] == "user" else "chat-bot"
    prefix = "" if msg["role"] == "user" else "🌿 "
    st.markdown(f'<div class="{css}">{prefix}{msg["text"]}</div>',
                unsafe_allow_html=True)

with st.form("chat_form", clear_on_submit=True):
    col1, col2 = st.columns([5, 1])
    with col1:
        user_input = st.text_input("",
            placeholder="Share what's on your mind...",
            label_visibility="collapsed")
    with col2:
        submitted = st.form_submit_button("Send")

if submitted and user_input.strip():
    st.session_state.messages.append({"role":"user","text":user_input})
    with st.spinner("..."):
        try:
            gen, tok = load_model()
            response = get_response(gen, tok, user_input)
        except Exception:
            response = "I hear you. I'm here with you."
    st.session_state.messages.append({"role":"bot","text":response})
    st.rerun()

with st.sidebar:
    st.markdown("### 💚 Crisis Helplines")
    st.markdown("""
- **Umang (Pakistan):** 0317-4288665
- **Rozan counselling:** 051-2890505
- **iCall:** 9152987821
- **Crisis Text:** Text HOME to 741741
- **International:** findahelpline.com
    """)
    st.markdown("---")
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = [{"role":"bot",
            "text":"Hello 👋 How are you feeling today?"}]
        st.rerun()
    st.caption("Built with 🤗 Transformers + Streamlit")
