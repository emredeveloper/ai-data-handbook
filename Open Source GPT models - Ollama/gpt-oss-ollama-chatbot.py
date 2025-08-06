import streamlit as st
import json
import ollama
import time
from typing import Dict, Any, List
import re
from datetime import datetime

# Configure page
st.set_page_config(
    page_title="GPT-OSS Chat Demo", 
    layout="wide",
    page_icon="🤖",
    initial_sidebar_state="expanded"
)
# Custom CSS for UI
st.markdown("""
<style>
/* Main app background */
.stApp {
    background: linear-gradient(135deg, #f1f5f9 0%, #e2e8f0 100%);
    min-height: 100vh;
}

/* MathJax support */
.MathJax {
    font-size: 1.1em !important;
}

.MathJax_Display {
    margin: 1em 0 !important;
}

/* Mathematical content styling */
.math-content {
    font-family: 'Computer Modern', 'Latin Modern Math', 'Times New Roman', serif;
    line-height: 1.8;
    font-size: 16px;
}

.math-content code {
    background: #f1f5f9;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
    font-size: 14px;
    color: #1f2937;
}

.math-content .math-block {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 15px;
    margin: 15px 0;
    font-family: 'Computer Modern', serif;
    text-align: center;
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
}

/* Mathematical expressions */
.math-inline {
    font-family: 'Computer Modern', serif;
    font-style: italic;
    color: #374151;
    background: #f9fafb;
    padding: 2px 4px;
    border-radius: 3px;
}

/* Main container styling */
.main .block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
    max-width: 1200px;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    margin-top: 2rem;
    margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.08);
    backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.8);
}

/* Sidebar styling */
.css-1d391kg, .css-1cypcdb, .css-17eq0hr {
    background: linear-gradient(180deg, #f8fafc 0%, #e2e8f0 100%) !important;
    border-right: 1px solid #e2e8f0;
}

.css-1d391kg .css-1v0mbdj {
    color: #374151 !important;
}

.css-1d391kg .stMarkdown {
    color: #374151 !important;
}

/* Box styling */
.reasoning-box {
    background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    padding: 25px;
    margin: 20px 0;
    border-radius: 15px;
    border-left: 5px solid #22c55e;
    box-shadow: 0 8px 25px rgba(34, 197, 94, 0.1);
    border: 1px solid rgba(34, 197, 94, 0.2);
    animation: slideInUp 0.6s ease-out;
}

.answer-box {
    background: linear-gradient(135deg, #fefbff 0%, #f3f4f6 100%);
    padding: 25px;
    margin: 20px 0;
    border-radius: 15px;
    border-left: 5px solid #6366f1;
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.2);
    animation: slideInUp 0.8s ease-out;
}

.metric-card {
    background: linear-gradient(135deg, #ffffff 0%, #f9fafb 100%);
    padding: 25px;
    margin: 20px 0;
    border-radius: 15px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.05);
    text-align: center;
    border: 2px solid #e5e7eb;
    transition: all 0.3s ease;
}

.metric-card:hover {
    transform: translateY(-5px);
    box-shadow: 0 15px 40px rgba(0, 0, 0, 0.1);
    border-color: #d1d5db;
}

/* Chat message styling */
.chat-message {
    padding: 20px;
    margin: 15px 0;
    border-radius: 15px;
    animation: slideInLeft 0.5s ease-out;
}

.user-message {
    background: linear-gradient(135deg, #f0f9ff 0%, #e0f2fe 100%);
    border-left: 5px solid #0ea5e9;
    box-shadow: 0 5px 15px rgba(14, 165, 233, 0.1);
    border: 1px solid rgba(14, 165, 233, 0.2);
}

.assistant-message {
    background: linear-gradient(135deg, #fefce8 0%, #fef3c7 100%);
    border-left: 5px solid #f59e0b;
    box-shadow: 0 5px 15px rgba(245, 158, 11, 0.1);
    border: 1px solid rgba(245, 158, 11, 0.2);
}

/* Animations */
@keyframes slideInUp {
    from {
        opacity: 0;
        transform: translateY(30px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

@keyframes slideInLeft {
    from {
        opacity: 0;
        transform: translateX(-30px);
    }
    to {
        opacity: 1;
        transform: translateX(0);
    }
}

/* Button styling */
.stButton > button {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 15px 30px;
    font-weight: 600;
    font-size: 16px;
    transition: all 0.3s ease;
    box-shadow: 0 5px 20px rgba(99, 102, 241, 0.2);
    width: 100%;
}

.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(99, 102, 241, 0.3);
    background: linear-gradient(135deg, #5b21b6 0%, #7c3aed 100%);
}

.stButton > button:active {
    transform: translateY(-1px);
    box-shadow: 0 5px 15px rgba(99, 102, 241, 0.4);
}

/* Input styling */
.stTextArea > div > div > textarea {
    border-radius: 12px;
    border: 2px solid #e5e7eb;
    transition: all 0.3s ease;
    font-size: 16px;
    padding: 15px;
    background: rgba(255, 255, 255, 0.95);
}

.stTextArea > div > div > textarea:focus {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Selectbox styling */
.stSelectbox > div > div {
    border-radius: 12px;
    border: 2px solid #e5e7eb;
    transition: all 0.3s ease;
    background: rgba(255, 255, 255, 0.95);
}

.stSelectbox > div > div:focus-within {
    border-color: #6366f1;
    box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.1);
}

/* Slider styling */
.stSlider > div > div > div > div {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
}

/* Title styling */
h1 {
    background: linear-gradient(135deg, #374151 0%, #6b7280 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    text-align: center;
    font-size: 3.5rem;
    font-weight: 800;
    margin-bottom: 1rem;
}

/* Section headers */
h3 {
    color: #374151;
    font-weight: 700;
    margin-bottom: 1rem;
    font-size: 1.5rem;
}

/* Metrics card content */
.metric-card h4 {
    color: #6366f1;
    margin-bottom: 20px;
    font-size: 1.3rem;
    font-weight: 700;
}

.metric-card p {
    margin: 12px 0;
    color: #4b5563;
    line-height: 1.6;
    font-size: 14px;
}

/* Expander styling */
.streamlit-expanderHeader {
    background: linear-gradient(135deg, #f9fafb 0%, #f3f4f6 100%);
    border-radius: 12px;
    border: 2px solid #e5e7eb;
    font-weight: 600;
    color: #374151;
}

/* Spinner styling */
.stSpinner > div {
    border-top-color: #6366f1 !important;
    border-right-color: #8b5cf6 !important;
}

/* Checkbox styling */
.stCheckbox > label {
    color: #374151 !important;
    font-weight: 500;
}

/* Success/Error message styling */
.stSuccess {
    background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%);
    border: 1px solid #22c55e;
    border-radius: 12px;
}

.stError {
    background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
    border: 1px solid #ef4444;
    border-radius: 12px;
}

/* Responsive design */
@media (max-width: 768px) {
    .main .block-container {
        padding: 1rem;
        margin: 1rem;
    }
    
    h1 {
        font-size: 2.5rem;
    }
    
    .metric-card {
        padding: 15px;
    }
}
</style>

<!-- MathJax Configuration -->
<script type="text/javascript" async
  src="https://cdnjs.cloudflare.com/ajax/libs/mathjax/3.2.0/es5/tex-mml-chtml.js">
</script>

<script type="text/x-mathjax-config">
MathJax.Hub.Config({
  tex2jax: {
    inlineMath: [['$', '$'], ['\\(', '\\)']],
    displayMath: [['$$', '$$'], ['\\[', '\\]']],
    processEscapes: true,
    processEnvironments: true
  },
  "HTML-CSS": {
    scale: 110,
    linebreaks: { automatic: true },
    fonts: ["STIX-Web"]
  },
  CommonHTML: { 
    scale: 110,
    linebreaks: { automatic: true }
  },
  SVG: {
    scale: 110,
    linebreaks: { automatic: true }
  }
});
</script>

""", unsafe_allow_html=True)


def call_model(messages: List[Dict], model_name: str = "gpt-oss:20b", temperature: float = 1.0) -> Dict[str, Any]:
    try:
        start_time = time.time()        
        # Prepare options for Ollama
        options = {
            'temperature': temperature,
            'top_p': 1.0,
        }        
        response = ollama.chat(
            model=model_name, 
            messages=messages,
            options=options
        )
        end_time = time.time()        
        if isinstance(response, dict) and 'message' in response:
            content = response['message'].get('content', '')
        elif hasattr(response, 'message'):
            content = getattr(response.message, 'content', '')
        else:
            content = str(response)         
        return {
            'content': content, 
            'response_time': end_time - start_time, 
            'success': True
        }
    except Exception as e:
        return {
            'content': f"Error: {e}", 
            'response_time': 0, 
            'success': False
        }
        
        
def format_mathematical_content(content: str) -> str:
    """
    Format mathematical content for better display
    """
    # Replace common mathematical symbols and expressions
    content = re.sub(r'\b(\d+/\d+)\b', r'<span class="math-inline">$\1$</span>', content)
    content = re.sub(r'\b(sqrt\(([^)]+)\))', r'<span class="math-inline">$\\sqrt{\2}$</span>', content)
    content = re.sub(r'\b(x\^(\d+))\b', r'<span class="math-inline">$x^{\2}$</span>', content)
    content = re.sub(r'\b((\d+)\^(\d+))\b', r'<span class="math-inline">$\2^{\3}$</span>', content)
    
    # Format equations on separate lines
    content = re.sub(r'(\w+)\s*=\s*([^.\n]+)', r'<div class="math-block">$\1 = \2$</div>', content)
    
    # Add proper spacing around mathematical operators
    content = re.sub(r'(\d+)\s*\+\s*(\d+)', r'\1 + \2', content)
    content = re.sub(r'(\d+)\s*-\s*(\d+)', r'\1 - \2', content)
    content = re.sub(r'(\d+)\s*\*\s*(\d+)', r'\1 × \2', content)
    content = re.sub(r'(\d+)\s*/\s*(\d+)', r'\1 ÷ \2', content)
    
    # Wrap the entire content in math-content class
    return f'<div class="math-content">{content}</div>'


def parse_reasoning_response(content: str) -> Dict[str, str]:
    patterns = [
        r"<thinking>(.*?)</thinking>",
        r"Let me think.*?:(.*?)(?=\n\n|\nFinal|Answer:)",
        r"Reasoning:(.*?)(?=\n\n|\nAnswer:|\nConclusion:)",
    ]
    reasoning = ""
    answer = content    
    for pat in patterns:
        m = re.search(pat, content, re.DOTALL | re.IGNORECASE)
        if m:
            reasoning = m.group(1).strip()
            answer = content.replace(m.group(0), "").strip()
            break    
    if not reasoning and len(content.split('\n')) > 3:
        lines = content.split('\n')
        for i, l in enumerate(lines):
            if any(k in l.lower() for k in ['therefore', 'in conclusion', 'final answer', 'answer:']):
                reasoning = '\n'.join(lines[:i]).strip()
                answer = '\n'.join(lines[i:]).strip()
                break
    return {
        'reasoning': reasoning or "No explicit reasoning detected.", 
        'answer': answer or content
    }
    
# Initialize history
if 'history' not in st.session_state:
    st.session_state.history = []
# Sidebar for settings
with st.sidebar:
    st.markdown("### ⚙️ Configuration")
    
    st.markdown("#### 🤖 Model Selection")
    model_choice = st.selectbox(
        "Model Selection", 
        ["gpt-oss:20b", "gpt-oss:120b"],
        help="Choose between 20B (faster) or 120B (more capable)",
        label_visibility="hidden"
    )    
    
    st.markdown("#### 🧠 Reasoning Effort")
    effort = st.selectbox(
        "Reasoning Effort", 
        ["low", "medium", "high"], 
        index=1,
        help="Controls depth of reasoning shown",
        label_visibility="hidden"
    )    
    
    st.markdown("#### 🌡️ Temperature")
    temperature = st.slider(
        "Temperature", 
        0.0, 2.0, 1.0, 0.1,
        help="Controls response randomness",
        label_visibility="hidden"
    )    
    
    st.markdown("#### 🔍 Display Options")
    show_reasoning = st.checkbox(
        "🧠 Show Chain-of-Thought", 
        True,
        help="Display model's thinking process"
    )    
    show_metrics = st.checkbox(
        "📊 Show Performance Metrics",
        True,
        help="Display response time and model info"
    )    
    
    st.markdown("---")        
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.history = []
        st.rerun()  
# Main Chat Interface
st.markdown("""
<div style='text-align: center; margin-bottom: 2rem;'>
    <h1 style='margin-bottom: 0.5rem;'>🤖 GPT-OSS Interactive Chat</h1>
    <p style='font-size: 1.3rem; color: #6b7280; font-weight: 500; margin: 0;'>
        🚀 Advanced AI Assistant powered by Open Source GPT models
    </p>
    <div style='width: 100px; height: 3px; background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%); margin: 1rem auto; border-radius: 2px;'></div>
</div>
""", unsafe_allow_html=True)
examples = [
    "",
    "📊 120 km mesafeyi 1.5 saatte, sonra 80 km'yi 45 dakikada giden bir trenin ortalama hızı nedir?",
    "🧮 √2'nin irrasyonel olduğunu ispatlayın.",
    "🔢 x² + 5x + 6 = 0 denklemini çözün.",
    "� Bir dik üçgenin hipotenüsü 13 cm, bir dik kenarı 5 cm ise diğer dik kenar kaç cm'dir?",
    "💻 En uzun palindromik alt dizeyi bulan bir fonksiyon yazın.",
    "🔬 Kuantum dolanıklığı basit terimlerle açıklayın.",
    "🎯 Bir öneri sistemi nasıl tasarlarsınız?",
]

st.markdown("### 💡 Choose from Example Questions")
selected_from_dropdown = st.selectbox("Example Questions", examples, label_visibility="hidden")
# Get the question text
question_value = ""
if hasattr(st.session_state, 'selected_example'):
    question_value = st.session_state.selected_example
    del st.session_state.selected_example
elif selected_from_dropdown:
    question_value = selected_from_dropdown
st.markdown("### ✍️ Or Enter Your Custom Question")
question = st.text_area(
    "Custom Question", 
    value=question_value, 
    height=120,
    placeholder="💭 Ask anything! Try different reasoning effort levels to see how the model's thinking changes...",
    label_visibility="hidden"
)
# Submit button
st.markdown("### 🚀 Get AI Response")
col1, col2, col3 = st.columns([2, 1, 2])
with col2:
    submit_button = st.button("🤖 Ask GPT-OSS", type="primary", use_container_width=True)
if submit_button and question.strip():
    # System prompts 
    system_prompts = {
        'low': 'Sen yardımcı bir asistansın. Kısa ve net cevaplar ver. Matematiksel ifadeleri düzgün LaTeX formatında yaz.',
        'medium': f'Sen yardımcı bir asistansın. Cevabından önce kısa bir mantık yürütme göster. Matematiksel formülleri LaTeX formatında ($...$ veya $$...$$) düzgün bir şekilde yaz. Akıl yürütme seviyesi: {effort}',
        'high': f'Sen yardımcı bir asistansın. Adım adım tam bir düşünce zinciri mantığı göster. Problemi dikkatli bir şekilde düşün ve son cevabını vermeden önce tüm matematiksel işlemleri LaTeX formatında ($...$ veya $$...$$) düzgün yaz. Akıl yürütme seviyesi: {effort}'
    }    
    # Message history
    msgs = [{'role': 'system', 'content': system_prompts[effort]}]    
    msgs.extend(st.session_state.history[-6:])
    msgs.append({'role': 'user', 'content': question})    
    with st.spinner(f"🤖 GPT-OSS is thinking with {effort} effort... 🧠"):
        res = call_model(msgs, model_choice, temperature)
    if res['success']:
        parsed = parse_reasoning_response(res['content'])        
        st.session_state.history.append({'role': 'user', 'content': question})
        st.session_state.history.append({'role': 'assistant', 'content': res['content']})        
        col1, col2 = st.columns([3, 1] if show_metrics else [1])        
        with col1:
            if show_reasoning and parsed['reasoning'] != 'No explicit reasoning detected.':
                st.markdown("### 🧠 Chain-of-Thought Reasoning")
                formatted_reasoning = format_mathematical_content(parsed['reasoning'])
                st.markdown(f"""
                <div class='reasoning-box'>
                    <strong style='color: #22c55e; font-size: 1.1rem;'>💭 AI Thinking Process:</strong>
                    <br><br>
                    {formatted_reasoning}
                </div>
                """, unsafe_allow_html=True)            
            st.markdown("### ✨ Final Answer")
            formatted_answer = format_mathematical_content(parsed['answer'])
            st.markdown(f"""
            <div class='answer-box'>
                <strong style='color: #6366f1; font-size: 1.1rem;'>🎯 AI Response:</strong>
                <br><br>
                {formatted_answer}
            </div>
            """, unsafe_allow_html=True)       
        if show_metrics:
            with col2:
                st.markdown(f"""
                <div class="metric-card">
                    <h4>📊 Performance Metrics</h4>
                    <p><strong>⏱️ Time:</strong><br><span style="color: #6366f1; font-size: 1.1rem; font-weight: 600;">{res['response_time']:.2f}s</span></p>
                    <p><strong>🤖 Model:</strong><br><span style="color: #22c55e; font-weight: 600;">{model_choice}</span></p>
                    <p><strong>🎯 Effort:</strong><br><span style="color: #f59e0b; font-weight: 600;">{effort.title()}</span></p>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.error(f"{res['content']}")
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 📜 Conversation History")
    st.markdown("<p style='color: #64748b; margin-bottom: 1rem;'>Recent exchanges (Last 4 conversations)</p>", unsafe_allow_html=True)
    
    recent_history = st.session_state.history[-8:]  # Last 8 messages (4 exchanges)    
    for i, msg in enumerate(recent_history):
        if msg['role'] == 'user':
            st.markdown(f"""
            <div class="chat-message user-message">
                <strong>👤 You:</strong> {msg['content']}
            </div>
            """, unsafe_allow_html=True)
        else:
            parsed_hist = parse_reasoning_response(msg['content'])
            with st.expander(f"🤖 GPT-OSS Response #{(i//2) + 1}", expanded=False):
                if parsed_hist['reasoning'] != 'No explicit reasoning detected.':
                    st.markdown("**🧠 Reasoning Process:**")
                    st.markdown(format_mathematical_content(parsed_hist['reasoning']), unsafe_allow_html=True)
                st.markdown("**✨ Final Answer:**")
                st.markdown(format_mathematical_content(parsed_hist['answer']), unsafe_allow_html=True)
# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 2rem 0;'>
    <div style='background: linear-gradient(135deg, #f8fafc 0%, #e2e8f0 100%); 
                padding: 25px; 
                border-radius: 20px; 
                margin: 1rem 0;
                box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
                border: 2px solid #e5e7eb;'>
        <p style='color: #374151; font-size: 1.2rem; font-weight: 700; margin: 0;'>
            🚀 <strong>Powered by GPT-OSS-20B via Ollama</strong>
        </p>
        <p style='color: #6b7280; font-size: 1rem; margin: 8px 0 0 0; font-weight: 500;'>
            🔬 Advanced Open Source AI • 🧠 Chain-of-Thought Reasoning • ⚡ Real-time Performance
        </p>
        <div style='width: 80px; height: 2px; background: #d1d5db; margin: 15px auto; border-radius: 1px;'></div>
        <p style='color: #9ca3af; font-size: 0.9rem; margin: 0; font-style: italic;'>
            Experience the future of open-source AI conversation
        </p>
    </div>
</div>
""", unsafe_allow_html=True)