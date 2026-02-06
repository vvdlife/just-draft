import os
import json
import time
from typing import Optional, Dict, Any, List
import pandas as pd

import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# ---------------------------------------------------------
# 1. Configuration & Setup (Architect View)
# ---------------------------------------------------------

def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if st.session_state["password"] == st.secrets["APP_PASSWORD"]:
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # don't store password
        else:
            st.session_state["password_correct"] = False

    # 1. If password is not set in secrets, allow access (or warn)
    # For security, we assume if secrets are used, password must be there.
    if "APP_PASSWORD" not in st.secrets:
        st.error("⚠️ 설정 오류: APP_PASSWORD가 Secret에 설정되지 않았습니다.")
        return False

    # 2. Return True if the user has already authenticated
    if st.session_state.get("password_correct", False):
        return True

    # 3. Show input for password
    st.title("🔒 보호된 페이지")
    st.text_input(
        "비밀번호를 입력하세요", type="password", on_change=password_entered, key="password"
    )
    
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 비밀번호가 틀렸습니다.")

    return False

def configure_page():
    """Setup Streamlit page metadata."""
    st.set_page_config(
        page_title="Just Draft",
        page_icon="📝",
        layout="wide",
        initial_sidebar_state="collapsed"  # Mobile optimization: Auto-collapse sidebar
    )

def init_session_state():
    """Initialize session state for history."""
    if "history" not in st.session_state:
        st.session_state.history = []
    if "reset_key" not in st.session_state:
        st.session_state.reset_key = 0

def reset_app():
    """Reset the application state to start a new task."""
    st.session_state.pop('current_result', None)
    # Increment reset_key to force-recreate widgets with new keys
    st.session_state.reset_key += 1

# ---------------------------------------------------------
# 2. Key Prompts (Planner View)
# ---------------------------------------------------------
SYSTEM_PROMPT = """
### Role
You are 'Just Draft', an AI agent that converts unstructured user text into structured JSON data.

### Goal
Analyze the input text and extract 'Tasks' and 'Memos'. Return the result strictly in the defined JSON format.

### Processing Rules
1. Analysis: Identify actionable items (Tasks) and reference information (Memos/Ideas).
2. Refinement: Convert tasks into clear, action-oriented sentences (ending with verbs like -하기). Remove filler words.
3. Categorization: Assign a category (Work, Personal, Health, Shopping, Other).
4. Priority & Date: Detect urgency for priority ("High"/"Normal") and extract dates if present.
5. Language: Output content must be in Korean.

### Output Schema (JSON Only)
{
  "tasks": [
    {
      "category": "String (Work/Personal/Shopping/Health/Other)",
      "action": "String (Refined action item)",
      "priority": "String (High/Normal)",
      "deadline": "String (YYYY-MM-DD, Time, or text description / null if none)"
    }
  ],
  "memos": [
    {
      "content": "String (Non-actionable notes or ideas)"
    }
  ]
}
"""

# ---------------------------------------------------------
# 3. Core Logic (Developer View)
# ---------------------------------------------------------
def process_input(api_key: str, user_text: str, image_file=None, audio_file=None) -> Dict[str, Any]:
    """
    Call Gemini API to process text, image, or audio.
    Supports multi-modal inputs.
    """
    if not (user_text.strip() or image_file or audio_file):
        return {}

    genai.configure(api_key=api_key)
    
    # Priority list of models
    # Note: 1.5-flash and above support multi-modal efficiently
    candidate_models = [
        "gemini-3-flash-preview", # Correct ID from docs
        "gemini-1.5-flash"        # Fallback
    ]
    
    # Prepare Content parts
    content_parts = []
    
    # 1. Text
    if user_text:
        content_parts.append(user_text)
    else:
        # If no text provided, add a prompt to guide the model for image/audio
        content_parts.append("Analyze this content and extract tasks/memos.")

    # 2. Image (Bytes)
    if image_file:
        # Use raw bytes for robust retry (avoid PIL file pointer issues)
        image_file.seek(0)
        image_bytes = image_file.read()
        content_parts.append({
            "mime_type": image_file.type,
            "data": image_bytes
        })

    # 3. Audio (Bytes + MimeType)
    if audio_file:
        # Streamlit audio_input returns a file-like object (WAV)
        # We need to read bytes and specify mime_type
        audio_bytes = audio_file.read()
        content_parts.append({
            "mime_type": "audio/wav",
            "data": audio_bytes
        })

    last_error = None

    for model_name in candidate_models:
        try:
            # Skip text-only models if media is present
            # 1.5, 2.0, 2.5, 3 support multimodal
            # Check for "gemini-3" to cover "gemini-3-flash-preview"
            is_multimodal = "1.5" in model_name or "2.0" in model_name or "2.5" in model_name or "gemini-3" in model_name
            if (image_file or audio_file) and not is_multimodal:
                continue

            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=SYSTEM_PROMPT,
                generation_config={"response_mime_type": "application/json"}
            )
            
            # Generate
            response = model.generate_content(content_parts)
            
            # Parse JSON
            data = json.loads(response.text)
            return data
            
        except Exception as e:
            last_error = e
            if "404" in str(e) or "not found" in str(e).lower():
                continue
            continue

    raise RuntimeError(
        f"All models failed. Last error: {str(last_error)}"
    )

def convert_to_csv(data: List[Dict]) -> str:
    """Convert list of dicts to CSV string."""
    if not data:
        return ""
    df = pd.DataFrame(data)
    return df.to_csv(index=False).encode('utf-8-sig')

def convert_to_markdown(tasks: List[Dict], memos: List[Dict]) -> str:
    """Convert data to Markdown format."""
    md = "# Brain Cleaner Results\n\n"
    
    md += "## ✅ Tasks\n"
    for t in tasks:
        p_icon = "🔥" if t.get('priority') == 'High' else "🔹"
        md += f"- [{t.get('category')}] {t.get('action')} {p_icon}"
        if t.get('deadline'):
            md += f" (📅 {t['deadline']})"
        md += "\n"
        
    md += "\n## 💡 Memos\n"
    for m in memos:
        md += f"- {m.get('content')}\n"
        
    return md

# ---------------------------------------------------------
# 4. User Interface (Frontend)
# ---------------------------------------------------------
def main():
    configure_page()
    init_session_state()
    
    # 0. Security Check
    if not check_password():
        st.stop()

    # 1. Compact Header (Mobile First)
    st.title("📝 Just Draft")
    # Removed verbose description to save screen space
    
    # 2. Configuration (Collapsible for Mobile)
    api_key = None
    
    with st.expander("⚙️ 설정 (API Key)", expanded=True):
        api_key_input = st.text_input(
            "Google API Key",
            type="password",
            placeholder="AI Studio 키 입력 (필수)",
            help="저장되지 않음. 1회성 사용."
        )
        if api_key_input:
            api_key = api_key_input
            st.success("Custom Key 사용 중")
        else:
            st.warning("API Key가 필요합니다.")
            st.markdown("[키 발급받기](https://aistudio.google.com/)")

    # History Drawer (Sidebar)
    with st.sidebar:
        st.header("🕒 히스토리")
        if st.session_state.history:
            for item in reversed(st.session_state.history):
                st.text(f"• {item.get('summary', 'Input')}")
        else:
            st.caption("기록 없음")
        
        st.divider()
        st.caption("v1.4.0 (Mobile First)")

    # 3. Main Input (Full Width & Touch Friendly)
    if not api_key:
        st.error("👆 위 설정에서 API Key를 먼저 입력해주세요.")
        return

    # Tabs for simple switching
    tab_text, tab_image, tab_audio = st.tabs(["📝 텍스트", "📸 이미지", "🎙️ 음성"])
    
    user_text = ""
    image_file = None
    audio_file = None
    submit = False

    with tab_text:
        user_text = st.text_area(
            "Quick Input",
            height=120,
            placeholder="생각나는 대로 적으세요...",
            label_visibility="collapsed",
            key=f"user_text_input_{st.session_state.reset_key}"
        )
        # Big Button for Touch
        if st.button("🚀 텍스트로 정리하기", type="primary", use_container_width=True):
            submit = True

    with tab_image:
        image_file = st.file_uploader("이미지 업로드", type=['png', 'jpg', 'jpeg'], label_visibility="collapsed", key=f"image_input_{st.session_state.reset_key}")
        if image_file:
            st.image(image_file, use_container_width=True)
            if st.button("🚀 이미지 분석하기", type="primary", use_container_width=True):
                submit = True
    
    with tab_audio:
        audio_file = st.audio_input("음성 녹음", key=f"audio_input_{st.session_state.reset_key}")
        if audio_file:
            if st.button("🚀 음성 정리하기", type="primary", use_container_width=True):
                submit = True

    # Processing
    if submit:
        source_summary = "Text"
        if image_file: source_summary = "Image"
        if audio_file: source_summary = "Audio"
        if user_text: source_summary = user_text[:15] + "..."

        with st.spinner("분석 중..."):
            try:
                result_data = process_input(api_key, user_text, image_file, audio_file)
                if result_data:
                    st.session_state.history.append({
                        "summary": source_summary,
                        "result": result_data,
                        "timestamp": time.time()
                    })
                    st.session_state['current_result'] = result_data
            except Exception as e:
                st.error(f"Error: {str(e)}")

    # Rendering (Mobile Optimized View)
    if 'current_result' in st.session_state:
        result_data = st.session_state['current_result']
        st.divider()
        
        # Tasks
        st.subheader("✅ 할 일")
        tasks = result_data.get("tasks", [])
        updated_tasks = []
        
        if tasks:
            # Simple Data Editor for Mobile? st.data_editor might be cramped on mobile.
            # But user wanted "Interactive". Let's keep it but minimize config.
            df_tasks = pd.DataFrame(tasks)
            cols = [c for c in ['priority', 'action'] if c in df_tasks.columns] # Show less cols on mobile
            if 'category' in df_tasks.columns: cols.insert(0, 'category')
            
            df_display = df_tasks[cols] if cols else pd.DataFrame(tasks)
            
            edited_df = st.data_editor(
                df_display,
                num_rows="dynamic",
                use_container_width=True,
                key="mobile_editor"
            )
            updated_tasks = edited_df.to_dict('records')
        else:
            st.info("No tasks.")

        # Memos
        memos = result_data.get("memos", [])
        if memos:
            st.subheader("💡 메모")
            for memo in memos:
                st.info(f"{memo['content']}") # Use info box for card-like feel

        # Export (Full Width Buttons)
        with st.expander("📥 결과 내보내기"):
            json_str = json.dumps({"tasks": updated_tasks, "memos": memos}, indent=2, ensure_ascii=False)
            st.download_button("JSON 저장", json_str, "brain.json", "application/json", use_container_width=True)
            
            csv_data = convert_to_csv(updated_tasks)
            if csv_data: st.download_button("CSV 저장 (할 일)", csv_data, "tasks.csv", "text/csv", use_container_width=True)

            csv_memos = convert_to_csv(memos)
            if csv_memos: st.download_button("CSV 저장 (메모)", csv_memos, "memos.csv", "text/csv", use_container_width=True)
            
            md_data = convert_to_markdown(updated_tasks, memos)
            st.download_button("Markdown 저장", md_data, "brain.md", "text/markdown", use_container_width=True)

        st.button("🔄 새로 시작하기", on_click=reset_app, type="secondary", use_container_width=True)

if __name__ == "__main__":
    main()
