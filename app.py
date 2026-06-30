import streamlit as st
import pandas as pd
import sys
import io
from transformers import pipeline

# Set page config for a premium and clean layout
st.set_page_config(
    page_title="Malayalam Hate Speech & Offensive Language Classifier",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium styling using CSS
st.markdown("""
<style>
    /* Custom fonts & colors */
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    /* Main banner styling */
    .hero-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
        padding: 2.5rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .hero-title {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    .hero-subtitle {
        font-size: 1.1rem;
        font-weight: 300;
        opacity: 0.9;
    }
    
    /* Glassmorphic cards */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 1.5rem;
        backdrop-filter: blur(10px);
        margin-bottom: 1rem;
    }
    
    /* Classification badges */
    .badge {
        padding: 0.4rem 0.8rem;
        border-radius: 50px;
        font-weight: 600;
        font-size: 0.9rem;
        display: inline-block;
    }
    .badge-offensive {
        background-color: rgba(239, 68, 68, 0.2);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.4);
    }
    .badge-clean {
        background-color: rgba(16, 185, 129, 0.2);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }
    .badge-mixed {
        background-color: rgba(245, 158, 11, 0.2);
        color: #fbbf24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
</style>
""", unsafe_allow_html=True)

# Cache model loader to ensure it downloads only once and is reused across app runs
@st.cache_resource
def load_classifier(model_source):
    try:
        import torch
        device = 0 if torch.cuda.is_available() else -1
    except ImportError:
        device = -1

    if model_source == "Local Fine-Tuned Model":
        model_name = "./finetuned_model"
        # Return local pipeline
        return pipeline(
            "text-classification",
            model=model_name,
            device=device
        )
    else:
        token = ""
        model_name = "Hate-speech-CNERG/malayalam-codemixed-abusive-MuRIL"
        # Return Hugging Face pipeline
        return pipeline(
            "text-classification",
            model=model_name,
            token=token,
            device=device
        )

# Label formatting and descriptions based on model config
# Labels: Not_offensive, Not_in_intended_language, Off_target_group, Profanity, Off_target_ind
LABEL_MAP = {
    'Not_offensive': {'clean': 'Not Offensive', 'color': 'green', 'badge_class': 'badge-clean', 'desc': 'Neutral, respectful, or positive content.'},
    'Not_in_intended_language': {'clean': 'Not Malayalam / Code-Mixed', 'color': 'orange', 'badge_class': 'badge-mixed', 'desc': 'Text belongs to another language or system.'},
    'Off_target_group': {'clean': 'Offensive Targeting Group', 'color': 'red', 'badge_class': 'badge-offensive', 'desc': 'Offensive language targeted at a group of people.'},
    'Profanity': {'clean': 'Profanity / Vulgarity', 'color': 'red', 'badge_class': 'badge-offensive', 'desc': 'Contains abusive, vulgar, or obscene words.'},
    'Off_target_ind': {'clean': 'Offensive Targeting Individual', 'color': 'red', 'badge_class': 'badge-offensive', 'desc': 'Offensive language targeting a specific person.'}
}

def get_label_info(raw_label):
    return LABEL_MAP.get(raw_label, {
        'clean': raw_label.replace('_', ' ').title(),
        'color': 'gray',
        'badge_class': 'badge-mixed',
        'desc': 'Unclassified class'
    })

# Main Page Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">🛡️ Malayalam Hate Speech & Offensive Content Classifier</div>
    <div class="hero-subtitle">Analyze Malayalam script and Malayalam-English code-mixed comments/posts for offensive language and profanity using XLM-RoBERTa model.</div>
</div>
""", unsafe_allow_html=True)

# Sidebar with model details & info
with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=60)
    
    st.markdown("### Model Settings")
    model_source = st.selectbox(
        "Select Model Source:",
        ["Local Fine-Tuned Model", "Hugging Face Model (CNERG)"],
        index=0
    )
    
    st.markdown("---")
    st.markdown("### Model Properties")
    if model_source == "Local Fine-Tuned Model":
        st.markdown("""
        - **Model Source:** Local Directory (`./finetuned_model`)
        - **Base Architecture:** XLM-RoBERTa-Base
        - **Status:** Locally Trained / Loaded
        """)
    else:
        st.markdown("""
        - **Model ID:** [Hate-speech-CNERG/malayalam-codemixed-abusive-MuRIL](https://huggingface.co/Hate-speech-CNERG/malayalam-codemixed-abusive-MuRIL)
        - **Base Architecture:** XLM-RoBERTa-Base
        - **Languages:** Malayalam, Malayalam-English Code-Mixed
        - **Authors:** Complex Network Research Group (CNERG), IIT Kharagpur
        - **Shared Task:** DravidianLangTech-EACL2021
        """)
        
    st.markdown("---")
    st.markdown("### Supported Classes")
    for key, val in LABEL_MAP.items():
        st.markdown(f"**{val['clean']}**\n*{val['desc']}*")

# Load model pipeline
try:
    with st.spinner(f"Initializing model '{model_source}' and loading weights..."):
        pipe = load_classifier(model_source)
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# Tabs for Single Text Classification, Batch Upload & Live Moderation
tab1, tab2, tab3 = st.tabs([
    "💬 Single Text Classification", 
    "📁 Batch Processing (CSV/Excel)",
    "🛡️ Live Moderation Simulation"
])

with tab1:
    st.subheader("Enter Malayalam/Code-mixed text to analyze:")
    text_input = st.text_area(
        label="Input text",
        value="നല്ലൊരു ദിവസം ആശംസിക്കുന്നു",
        placeholder="Type or paste Malayalam text here (e.g. 'നീ ഒരു മണ്ടൻ ആണ്' or 'poda maire')",
        height=120,
        label_visibility="collapsed"
    )
    
    if st.button("Classify Text", type="primary"):
        if text_input.strip() == "":
            st.warning("Please enter some text to classify.")
        else:
            with st.spinner("Analyzing text..."):
                raw_results = pipe(text_input, top_k=None)
                if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], list):
                    results = raw_results[0]
                else:
                    results = raw_results
                
                # Sum up probability of being offensive vs safe
                offensive_score = sum(r['score'] for r in results if "not" not in r['label'].lower())
                safe_score = sum(r['score'] for r in results if "not" in r['label'].lower())
                
                is_offensive = offensive_score > safe_score
                
                # Select the highest-scoring label within the winning category group
                if is_offensive:
                    group_results = [r for r in results if "not" not in r['label'].lower()]
                    score = offensive_score
                else:
                    group_results = [r for r in results if "not" in r['label'].lower()]
                    score = safe_score
                
                top_result = max(group_results, key=lambda x: x['score'])
                raw_label = top_result['label']
                
                label_info = get_label_info(raw_label)
                
                # Visual output cards
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.markdown("### Classification Result")
                    st.markdown(f"""
                    <div style="padding: 1rem; border-radius: 8px; border-left: 5px solid {label_info['color']}; background-color: rgba(255,255,255,0.02);">
                        <span class="badge {label_info['badge_class']}">{label_info['clean']}</span>
                        <h4 style="margin-top: 10px; margin-bottom: 5px;">Confidence: {score:.2%}</h4>
                        <small style="color: #888;">{label_info['desc']}</small>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("### Confidence Distribution")
                    # Show progress bar visualization
                    st.progress(int(score * 100))
                    st.write(f"The model is {score:.1%} confident that this text is classified as: **{label_info['clean']}**")

with tab2:
    st.subheader("Batch File Processing")
    st.markdown("Upload a CSV or Excel file containing a column with the texts you want to classify.")
    
    uploaded_file = st.file_uploader("Choose CSV or Excel file", type=["csv", "xlsx"])
    
    if uploaded_file is not None:
        try:
            # Read file
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
                
            st.write("File loaded successfully. Data preview:")
            st.dataframe(df.head(), use_container_width=True)
            
            # Select column
            columns = df.columns.tolist()
            text_column = st.selectbox("Select the column containing texts to classify:", columns)
            
            if st.button("Run Batch Classification", type="primary"):
                with st.spinner(f"Classifying {len(df)} rows..."):
                    # Create progress bar
                    progress_bar = st.progress(0)
                    
                    labels = []
                    human_labels = []
                    scores = []
                    
                    # Process in batch or iteratively
                    for idx, row in df.iterrows():
                        text = str(row[text_column])
                        if text.strip() == "":
                            labels.append("None")
                            human_labels.append("None")
                            scores.append(0.0)
                        else:
                            raw_results = pipe(text, top_k=None)
                            if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], list):
                                results = raw_results[0]
                            else:
                                results = raw_results
                            
                            offensive_score = sum(r['score'] for r in results if "not" not in r['label'].lower())
                            safe_score = sum(r['score'] for r in results if "not" in r['label'].lower())
                            
                            is_offensive = offensive_score > safe_score
                            if is_offensive:
                                group_results = [r for r in results if "not" not in r['label'].lower()]
                                score = offensive_score
                            else:
                                group_results = [r for r in results if "not" in r['label'].lower()]
                                score = safe_score
                            
                            top_result = max(group_results, key=lambda x: x['score'])
                            labels.append(top_result['label'])
                            human_labels.append(get_label_info(top_result['label'])['clean'])
                            scores.append(score)
                            
                        # Update progress
                        progress_bar.progress((idx + 1) / len(df))
                        
                    df['raw_label'] = labels
                    df['classified_category'] = human_labels
                    df['confidence_score'] = scores
                    
                    st.success("Batch classification complete!")
                    st.dataframe(df, use_container_width=True)
                    
                    # Prepare export
                    csv_buffer = io.StringIO()
                    df.to_csv(csv_buffer, index=False)
                    csv_data = csv_buffer.getvalue().encode('utf-8')
                    
                    st.download_button(
                        label="📥 Download Results as CSV",
                        data=csv_data,
                        file_name="classified_results.csv",
                        mime="text/csv"
                    )
                    
        except Exception as e:
            st.error(f"Error processing file: {e}")

with tab3:
    st.subheader("🛡️ Live Moderation Simulation")
    st.markdown(
        "Simulate a live comment section or forum. As messages are posted, they are processed in real-time. "
        "If a message exceeds the offensive threshold, it is automatically hidden to protect users."
    )
    
    # 1. Moderation Settings inside the tab
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        mod_active = st.toggle("Enable Live Auto-Moderation Filter", value=True, key="mod_active")
    with col_settings2:
        mod_threshold = st.slider(
            "Auto-Hide Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.50,
            step=0.05,
            help="Hide comments with an offensive probability higher than this threshold."
        )
    
    # Initialize session state for comments if not present
    if "moderation_feed" not in st.session_state:
        st.session_state.moderation_feed = [
            {
                "id": 1,
                "user": "Midhun Kumar",
                "text": "എല്ലാവർക്കും എൻ്റെ ഹൃദയം നിറഞ്ഞ ഓണാശംസകൾ! 🌸🌼",
                "is_offensive": False,
                "label": "Not_offensive",
                "confidence": 0.985,
                "revealed": False
            },
            {
                "id": 2,
                "user": "Cyber_Warrior",
                "text": "നീ വെറും ഒരു മണ്ടൻ ആണ്, ഒന്നും അറിയില്ല എങ്കിൽ മിണ്ടാതിരിക്കണം. 😡",
                "is_offensive": True,
                "label": "Off_target_ind",
                "confidence": 0.892,
                "revealed": False
            },
            {
                "id": 3,
                "user": "Anjali S.",
                "text": "This movie was absolutely amazing! highly recommended to everyone.",
                "is_offensive": False,
                "label": "Not_offensive",
                "confidence": 0.991,
                "revealed": False
            },
            {
                "id": 4,
                "user": "Malayali_Chekkan",
                "text": "poda maire, ninakk entha karyam",
                "is_offensive": True,
                "label": "Profanity",
                "confidence": 0.943,
                "revealed": False
            }
        ]
        st.session_state.next_id = 5

    # Preset Sample Inputs for easy testing
    st.markdown("### ⚡ Quick-Test Presets")
    st.caption("Click any sample comment below to immediately post it into the live feed:")
    
    presets = [
        {"text": "നല്ലൊരു ദിവസം ആശംസിക്കുന്നു", "desc": "Have a nice day (Safe)", "color": "green"},
        {"text": "നീ എന്ത് കോപ്പാണ് ഈ പറയുന്നത്?", "desc": "What nonsense are you saying? (Individual offensive)", "color": "red"},
        {"text": "സുഖമാണോ കൂട്ടുകാരെ?", "desc": "How are you friends? (Safe)", "color": "green"},
        {"text": "poda maire", "desc": "Abusive slang (Profanity)", "color": "red"}
    ]
    
    # Render presets as buttons side by side
    preset_cols = st.columns(len(presets))
    clicked_preset_text = None
    for i, p in enumerate(presets):
        with preset_cols[i]:
            btn_label = f"📝 {p['text'][:15]}..."
            if st.button(btn_label, key=f"preset_btn_{i}", help=f"Click to post: '{p['text']}' ({p['desc']})"):
                clicked_preset_text = p['text']

    # 2. Add New Comment Input Form
    with st.form("new_comment_form", clear_on_submit=True):
        new_comment_text = st.text_input("Post a new comment:", placeholder="Write a comment in Malayalam/Code-mixed...")
        submit_btn = st.form_submit_button("Post Comment", type="primary")

    # Handle posting (either from submit button or preset button)
    comment_to_post = None
    if submit_btn and new_comment_text.strip() != "":
        comment_to_post = new_comment_text.strip()
    elif clicked_preset_text:
        comment_to_post = clicked_preset_text

    if comment_to_post:
        # Perform classification on-the-fly
        with st.spinner("Moderating comment..."):
            raw_results = pipe(comment_to_post, top_k=None)
            if isinstance(raw_results, list) and len(raw_results) > 0 and isinstance(raw_results[0], list):
                results = raw_results[0]
            else:
                results = raw_results
            
            offensive_score = sum(r['score'] for r in results if "not" not in r['label'].lower())
            safe_score = sum(r['score'] for r in results if "not" in r['label'].lower())
            
            is_offensive = offensive_score > safe_score
            if is_offensive:
                group_results = [r for r in results if "not" not in r['label'].lower()]
                score = offensive_score
            else:
                group_results = [r for r in results if "not" in r['label'].lower()]
                score = safe_score
            
            top_result = max(group_results, key=lambda x: x['score'])
            raw_label = top_result['label']
            
            # Store in session state
            import random
            users = ["Rohan P.", "Maya Nair", "Gautham Z.", "Aparna B.", "Kiran K.", "User_404"]
            st.session_state.moderation_feed.insert(0, {
                "id": st.session_state.next_id,
                "user": random.choice(users),
                "text": comment_to_post,
                "is_offensive": is_offensive,
                "label": raw_label,
                "confidence": score,
                "revealed": False
            })
            st.session_state.next_id += 1
            st.rerun()

    # 3. Render Dashboard Metrics
    total_comments = len(st.session_state.moderation_feed)
    flagged_comments = sum(1 for c in st.session_state.moderation_feed if c["is_offensive"] and c["confidence"] >= mod_threshold)
    flagged_pct = (flagged_comments / total_comments * 100) if total_comments > 0 else 0
    
    col_stat1, col_stat2, col_stat3 = st.columns(3)
    with col_stat1:
        st.metric("Total Comments", total_comments)
    with col_stat2:
        st.metric("Flagged Comments", flagged_comments)
    with col_stat3:
        st.metric("Flagged Percentage", f"{flagged_pct:.1f}%")
        
    st.markdown("---")
    
    # 4. Display Feed
    st.subheader("💬 Comment Feed")
    if not st.session_state.moderation_feed:
        st.info("No comments in the feed yet. Write one above!")
    else:
        for idx, comment in enumerate(st.session_state.moderation_feed):
            cid = comment["id"]
            # Check if this comment is flagged as offensive based on current threshold
            comment_is_flagged = comment["is_offensive"] and comment["confidence"] >= mod_threshold
            
            # Create a card design
            if mod_active and comment_is_flagged and not comment["revealed"]:
                # Hidden offensive card
                label_info = get_label_info(comment["label"])
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid rgba(239, 68, 68, 0.4); border-left: 5px solid #ef4444; border-radius: 8px; padding: 1rem; background-color: rgba(239, 68, 68, 0.05); margin-bottom: 12px;">
                        <span style="font-weight: 700; color: #ef4444;">🛡️ Content Filtered</span>
                        <span style="font-size: 0.85rem; color: #888; margin-left: 10px;">posted by <b>{comment['user']}</b></span>
                        <p style="font-style: italic; color: #f87171; margin-top: 5px; margin-bottom: 5px;">
                            This comment has been hidden because it was flagged as offensive ({label_info['clean']} - {comment['confidence']:.1%})
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    # Button to reveal this specific comment
                    if st.button("👁️ Show Comment anyway", key=f"reveal_btn_{cid}_{idx}"):
                        comment["revealed"] = True
                        st.rerun()
            else:
                # Normal comment card
                is_flagged_but_shown = comment["is_offensive"] and comment["confidence"] >= mod_threshold
                badge_html = ""
                border_color = "rgba(255, 255, 255, 0.1)"
                border_left = "rgba(255, 255, 255, 0.1)"
                bg_color = "rgba(255, 255, 255, 0.02)"
                
                if is_flagged_but_shown:
                    label_info = get_label_info(comment["label"])
                    badge_html = f'<span class="badge {label_info["badge_class"]}" style="margin-left: 10px;">{label_info["clean"]} ({comment["confidence"]:.1%})</span>'
                    border_color = "rgba(245, 158, 11, 0.4)"
                    border_left = "#f59e0b"
                    bg_color = "rgba(245, 158, 11, 0.05)"
                else:
                    badge_html = '<span class="badge badge-clean" style="margin-left: 10px;">Approved</span>'
                    border_color = "rgba(16, 185, 129, 0.2)"
                    border_left = "#10b981"
                    bg_color = "rgba(16, 185, 129, 0.02)"
                
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid {border_color}; border-left: 5px solid {border_left}; border-radius: 8px; padding: 1rem; background-color: {bg_color}; margin-bottom: 12px;">
                        <div style="display: flex; align-items: center; justify-content: space-between; margin-bottom: 5px;">
                            <span style="font-weight: 700; font-size: 1rem;">👤 {comment['user']}</span>
                            {badge_html}
                        </div>
                        <p style="margin-top: 5px; margin-bottom: 5px; font-size: 1.1rem; line-height: 1.5;">{comment['text']}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    # If flagged and shown, give option to hide it again
                    if is_flagged_but_shown:
                        if st.button("🔒 Hide Comment", key=f"hide_btn_{cid}_{idx}"):
                            comment["revealed"] = False
                            st.rerun()
                            
    # Reset feed button
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Reset Moderation Feed", type="secondary"):
        if "moderation_feed" in st.session_state:
            del st.session_state.moderation_feed
        if "next_id" in st.session_state:
            del st.session_state.next_id
        st.rerun()
