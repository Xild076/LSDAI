import streamlit as st
import sqlite3
from hashlib import sha256
import os
from datetime import datetime
from speech import SpeechAnalysis
from streamlit_echarts import st_echarts
import librosa
import soundfile as sf
import time

st.set_page_config(page_title="LSDAI - Your personal AI Speech and Debate Coach", layout="wide")

os.makedirs("web_management", exist_ok=True)  # Ensure the directory exists

@st.cache_resource
def get_connection():
    db_path = os.path.join("web_management", "user_data.db")
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return conn

conn = get_connection()
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY, 
                password TEXT)''')
c.execute('''CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT,
                content_feedback TEXT,
                emphasis_feedback TEXT,
                tone_feedback TEXT,
                speed_feedback TEXT,
                cumulative_score REAL,
                analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (username) REFERENCES users(username))''')
conn.commit()

def hash_password(password):
    return sha256(password.encode()).hexdigest()

def create_user(username, password):
    try:
        c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, hash_password(password)))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False

def authenticate(username, password):
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", (username, hash_password(password)))
    return c.fetchone()

def sign_in():
    with st.sidebar.form(key="sign_in_form"):
        st.markdown("### Sign In")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        sign_in_clicked = st.form_submit_button("Sign In")
    if sign_in_clicked:
        if authenticate(username, password):
            st.session_state['logged_in'] = True
            st.session_state['username'] = username
            load_previous_feedback(username)
            st.sidebar.success(f"Signed in as {username}")
            st.experimental_rerun()
        else:
            st.sidebar.error("Invalid username or password")

def create_account():
    with st.sidebar.form(key="create_account_form"):
        st.markdown("### Create Account")
        username = st.text_input("New Username")
        password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        create_account_clicked = st.form_submit_button("Create Account")
    if create_account_clicked:
        if password != confirm_password:
            st.sidebar.error("Passwords do not match")
        elif c.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
            st.sidebar.error("Username already exists")
        elif create_user(username, password):
            st.sidebar.success("Account created successfully")
            st.sidebar.info("Please sign in")
        else:
            st.sidebar.error("Error creating account")

def init_page():
    st.title("LSDAI - Your personal AI Speech and Debate Coach")

def save_feedback_to_db(username, feedback, cumulative_score):
    c.execute('''INSERT INTO feedback (username, content_feedback, emphasis_feedback, 
                tone_feedback, speed_feedback, cumulative_score) 
                VALUES (?, ?, ?, ?, ?, ?)''', 
              (username, 
               feedback.get('content', ["", 0])[0], 
               feedback.get('emphasis', ["", "", 0])[1], 
               feedback.get('tone', ["", "", 0])[1], 
               feedback.get('speed', ["", 0])[0],
               cumulative_score))
    conn.commit()
    if 'previous_feedback' in st.session_state:
        del st.session_state['previous_feedback']
    load_previous_feedback(username)

def load_previous_feedback(username):
    if 'previous_feedback' not in st.session_state:
        st.session_state['previous_feedback'] = []
    c.execute('''SELECT id, content_feedback, emphasis_feedback, tone_feedback, 
                        speed_feedback, cumulative_score, analysis_date 
                 FROM feedback 
                 WHERE username = ? 
                 ORDER BY analysis_date DESC''', 
              (username,))
    st.session_state['previous_feedback'] = c.fetchall()

def previous_feedback_page():
    st.title("Previous Feedback")
    if 'previous_feedback' in st.session_state and st.session_state['previous_feedback']:
        feedback_entries = st.session_state['previous_feedback']
        for entry in feedback_entries:
            st.markdown(f"### Analysis on {entry[-1]}")
            st.markdown(f"#### Cumulative Score: {entry[5]}/100")
            if entry[1]:
                st.markdown(f"**Content Feedback**\n\n{entry[1]}\n\n")
            if entry[2]:
                st.markdown(f"**Emphasis Feedback**\n\n{entry[2]}\n\n")
            if entry[3]:
                st.markdown(f"**Tone Feedback**\n\n{entry[3]}\n\n")
            if entry[4]:
                st.markdown(f"**Speed Feedback**\n\nWPM: {round(float(entry[4]))}\n\n")
            st.write("---")
    else:
        st.info("No previous feedback found.")

def plot_cumulative_score(scores, labels):
    total_score = sum(scores) / len(scores)
    shades_of_green = ["#00e676", "#00c853", "#00a152", "#007b33"]
    data = [
        {"value": score / len(scores), "name": label, "itemStyle": {"color": shades_of_green[i]}} 
        for i, (score, label) in enumerate(zip(scores, labels))
    ]
    red_part = 100 - total_score
    data.append({"value": red_part, "name": "Remaining", "itemStyle": {"color": "#d32f2f"}})
    options = {
        "tooltip": {
            "trigger": "item",
            "formatter": "{a} <br/>{b}: {c} ({d}%)"
        },
        "series": [
            {
                "name": "Scores",
                "type": "pie",
                "radius": ["40%", "70%"],
                "avoidLabelOverlap": False,
                "itemStyle": {
                    "borderRadius": 10,
                    "borderColor": "#fff",
                    "borderWidth": 2
                },
                "label": {
                    "show": False,
                    "position": "center"
                },
                "emphasis": {
                    "label": {
                        "show": True,
                        "fontSize": "20",
                        "fontWeight": "bold"
                    }
                },
                "labelLine": {
                    "show": False
                },
                "data": data
            }
        ]
    }
    st_echarts(options=options, width="100%", height="400px")

def home_page():
    st.title("Home")
    st.write("Welcome to the home page. Please upload a file and proceed with the analysis")
    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])
    if uploaded_file is not None:
        base_name = uploaded_file.name.split('.')[0]
        file_path = os.path.join("web_management", f"{base_name}.wav")
        if "wav" not in uploaded_file.type:
            def convert_mp3_to_wav(input_file, output_file):
                try:
                    y, sr = librosa.load(input_file, sr=None)
                    sf.write(output_file, y, sr)
                except Exception as e:
                    st.error(f"Error during conversion: {str(e)}. Please convert the file to WAV manually.")
            file_path = os.path.join("web_management", f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            with st.spinner('Converting file to WAV format... Please wait.'):
                convert_mp3_to_wav(uploaded_file, file_path)
            st.success(f"File uploaded and converted to WAV format. Saved as {file_path}")
        else:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded and saved as {file_path}")
        with st.expander("Analysis Options", expanded=True):
            content_analysis = st.checkbox("Content Analysis")
            emphasis_analysis = st.checkbox("Emphasis Analysis")
            tone_analysis = st.checkbox("Tone Analysis")
            speed_analysis = st.checkbox("Speed Analysis")
            preset = st.selectbox("Choose a preset", ["N/A", "Impromptu", "Extempt", "Oratory"])
            analyze_button = st.button("Analyze")
        if analyze_button:
            st.session_state['analysis_config'] = {
                "content_analysis": content_analysis,
                "emphasis_analysis": emphasis_analysis,
                "tone_analysis": tone_analysis,
                "speed_analysis": speed_analysis,
                "file_path": file_path,
                "preset": preset,
                "progress": 0,
                "feedback_saved": False
            }
    if 'analysis_config' in st.session_state:
        config = st.session_state['analysis_config']
        progress_bar = st.progress(config['progress'])
        if config['progress'] < 100:
            if config['progress'] == 0:
                time.sleep(1)
                config['progress'] = 20
                progress_bar.progress(config['progress'])
                analysis = SpeechAnalysis(config['file_path'], config['preset'].lower())
                feedback, analysis_config = analysis.analyse([
                    1 if config['content_analysis'] else 0,
                    1 if config['emphasis_analysis'] else 0,
                    1 if config['tone_analysis'] else 0,
                    1 if config['speed_analysis'] else 0,
                ])
                config.update(feedback=feedback, analysis_config=analysis_config)
                config['progress'] = 70
                progress_bar.progress(config['progress'])
            scores = []
            labels = []
            feedback_text = []
            if config.get('analysis_config', [])[0] == 1:
                labels.append("Content")
                scores.append(config['feedback']['content'][1])
                feedback_text.append(f"### Content Feedback\n {config['feedback']['content'][0]}\n\n #### Score: {round(config['feedback']['content'][1])}/100")
            if config.get('analysis_config', [])[1] == 1:
                labels.append("Emphasis")
                scores.append(config['feedback']['emphasis'][2])
                feedback_text.append(f"### Emphasis Feedback\n\n #### Emphasized Text\n {config['feedback']['emphasis'][0]}\n\n #### Feedback\n {config['feedback']['emphasis'][1]}\n\n #### Score: {round(config['feedback']['emphasis'][2])}/100")
            if config.get('analysis_config', [])[2] == 1:
                labels.append("Tone")
                scores.append(config['feedback']['tone'][2])
                feedback_text.append(f"### Tone Feedback\n\n #### Determined Tonal Usage\n {config['feedback']['tone'][0]}\n\n #### Feedback\n {config['feedback']['tone'][1]}\n\n #### Score: {round(config['feedback']['tone'][2])}/100")
            if config.get('analysis_config', [])[3] == 1:
                labels.append("Speed")
                scores.append(config['feedback']['speed'][1])
                feedback_text.append(f"### Speed Feedback\n\n #### Words per minute: {round(config['feedback']['speed'][0])} \n\n #### Score: {round(config['feedback']['speed'][1])}/100")
            config['progress'] = 90
            progress_bar.progress(config['progress'])
            col1, col2 = st.columns(2)
            with col1:
                st.write("### Cumulative Score")
                plot_cumulative_score(scores, labels)
            with col2:
                st.write("### Numerical Scores")
                st.write(f"#### Total Score: {round(sum(scores) / len(scores))}/100")
                for label, score in zip(labels, scores):
                    st.write(f"#### {label}: {round(score)}/100")
            st.write("### Detailed Feedback")
            for feedback in feedback_text:
                st.markdown(feedback)
            cumulative_score = round(sum(scores) / len(scores))
            if not config['feedback_saved']:
                save_feedback_to_db(st.session_state['username'], config['feedback'], cumulative_score)
                config['feedback_saved'] = True
            config['progress'] = 100
            progress_bar.progress(config['progress'])
            st.success("Analysis complete!")
            progress_bar.empty()

def doc_page():
    st.title("Documentation")
    st.write("Documentation content has been removed.")

def sign_out():
    st.session_state.clear()
    st.sidebar.success("You have signed out")
    st.experimental_rerun()

page_names_to_funcs = {
    "Home": home_page,
    "Documentation": doc_page,
    "Previous Feedback": previous_feedback_page,
    "Sign Out": sign_out
}

if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    init_page()
if st.session_state['logged_in']:
    st.sidebar.markdown(f"### Welcome, {st.session_state['username']}")
    demo_name = st.sidebar.radio("Navigation", list(page_names_to_funcs.keys()))
    page_names_to_funcs[demo_name]()
else:
    option = st.sidebar.selectbox("Select an option", ["Sign In", "Create Account"])
    if option == "Sign In":
        sign_in()
    else:
        create_account()