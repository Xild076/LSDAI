import streamlit as st
import sqlite3
from hashlib import sha256
import os
from pydub import AudioSegment
from datetime import datetime
from speech import SpeechAnalysis
from streamlit_echarts import st_echarts

st.set_page_config(page_title="LSDAI - Your personal AI Speech and Debate Coach")

documentation = """
## Overview

The LSDAI serves as the main entry point for a Streamlit application designed to perform advanced audio analysis tasks, including emphasis detection, tone analysis, and speech speed evaluation. This application leverages multiple Python modules and files to deliver a comprehensive analysis of audio files.

## Dependencies

The application requires several Python libraries and external modules, which should be installed before running the app:

- `streamlit`: For building the web interface.
- `emphasis.py`: Handles the detection of emphasized words or phrases in the audio.
- `tone.py`: Manages the tone analysis functionality, providing insights into the emotional content of speech.
- `util.py`: Contains utility functions that support the main analysis, such as file handling and formatting.
- `content.py`: Facilitates content extraction and organization for analysis.
- `speech.py`: Implements speech analysis, particularly focusing on the quality and clarity of spoken words.
- `speed.py`: Provides functionality for calculating and analyzing the speed of speech.
- `api_key.txt`: A text file containing the necessary API keys for accessing external services [oai_citation:1,api_key.txt](file-service://file-5E4uxQPy7XwCOPpBqKHWbJpl).

## Key Features

- **Audio File Upload**: Users can upload audio files of various formats. The app processes these files and conducts multiple forms of analysis.
  
- **Emphasis Detection**: The `emphasis.py` module is used to detect and highlight emphasized words or phrases within the audio, making it easier to identify key points or stressed content.
  
- **Tone Analysis**: The `tone.py` module provides a detailed analysis of the tone used in the speech, helping to discern emotional undertones or intentions.
  
- **Speech Speed Analysis**: The `speed.py` module measures the speed of speech, which can be an indicator of various factors such as nervousness or excitement.
  
- **Speech Clarity**: The `speech.py` module assesses the clarity of the speech, ensuring that the spoken words are easily understood.
  
- **User Authentication**: The app includes user authentication functionality, which uses the API key from `api_key.txt` to manage access and usage.
  
- **Persistent Data**: Data and analysis results are saved and can be revisited or updated as needed. This includes maintaining previous analysis for each user.

## Usage

1. **Running the App**: To run the app, use the Streamlit command:
   ```bash
   streamlit run app.py
   ```

2. **Uploading Audio Files**: Users can upload audio files directly through the app's interface. The app will automatically process these files and display results.

3. **Analysis**: Once an audio file is uploaded, the app will conduct emphasis detection, tone analysis, and speed analysis. Results are displayed in an organized manner, with the option to save or revisit previous analyses.

4. **API Key Management**: Ensure that the `api_key.txt` file contains a valid API key. This key is crucial for accessing the external services used in the analysis.

#### Notes

- **File Size Limitations**: There may be size limitations on the audio files that can be uploaded, depending on the configuration of the Streamlit server and the underlying infrastructure.
  
- **Customizations**: The app is modular, allowing for easy customization. Developers can add additional analysis modules or modify existing ones by editing the corresponding `.py` files.

- **Error Handling**: Basic error handling is implemented, but users should ensure that the audio files are properly formatted and that all dependencies are installed correctly.

## Future Enhancements

- **Improved GUI**: Plans to enhance the graphical interface, including animations and dynamic updates, are under consideration.
  
- **Expanded Analysis**: Additional forms of analysis, such as sentiment analysis or linguistic complexity, may be integrated into future versions.

- **Multi-User Support**: Expanding the app's capabilities to support multiple users with separate accounts and analysis histories.

"""

intro = """

The `LSDAI` script is the heart of a cutting-edge audio analysis application, designed to offer deep insights into spoken content. By integrating multiple specialized Python modules, this application provides users with the ability to scrutinize audio files in ways that go far beyond basic transcription. Whether you're a linguist, a speech coach, a content creator, or anyone interested in audio analysis, this tool equips you with the ability to understand the nuances of speech with precision and clarity.

The application supports a variety of analysis types, including emphasis detection, which identifies the most stressed words or phrases in speech, tone analysis, which deciphers the emotional undertones, and speech speed analysis, which evaluates the pacing of the spoken content. This comprehensive suite of tools ensures that you can get the most out of your audio data, allowing for detailed feedback and insights that can be critical in both professional and personal contexts.

In addition, the application is designed with user convenience in mind. It allows for easy file uploads, intuitive interaction, and ensures that your data is saved and easily accessible for future reference. The modular design also means that the application can be easily expanded or customized to meet specific needs, making it a versatile choice for any audio analysis task.

### Unleash the Power of Speech Analysis with Our Revolutionary Audio Analysis App!

Are you ready to take your audio analysis to the next level? Our app is not just another speech-to-text tool—it's a full-fledged audio analysis powerhouse that gives you unprecedented control and insight into your spoken content. Whether you're analyzing interviews, podcasts, speeches, or any other form of verbal communication, this app empowers you to dive deeper into the details that matter most.

**Why settle for less when you can have it all?** Our app's advanced features make it an indispensable tool for anyone looking to understand and refine speech. From detecting emphasized words that carry the most weight to analyzing the emotional tone of speech and assessing the speed and clarity of delivery, our app has you covered.

With a sleek and user-friendly interface, it's never been easier to upload your audio files, perform in-depth analyses, and revisit your results whenever you need them. Plus, with our state-of-the-art technology, you can trust that your insights are both accurate and actionable.

### Key Features

- **Emphasis Detection**: Automatically identify and highlight the most stressed words or phrases in your audio, helping you pinpoint key messages and emotions.

- **Tone Analysis**: Understand the emotional undertones of speech with detailed tone analysis, giving you a clearer picture of the speaker's intent.

- **Speech Speed Analysis**: Measure and analyze the speed of speech to evaluate pacing, an essential factor in communication effectiveness.

- **Speech Clarity Evaluation**: Assess the clarity of spoken words, ensuring that your message is both clear and comprehensible.

- **User Authentication and Data Persistence**: Securely manage your analysis with user authentication and save your data for easy access and review.

- **Modular and Customizable Design**: Tailor the app to your specific needs by easily adding or modifying analysis modules.

- **Streamlined User Interface**: Enjoy a seamless experience with a sleek, intuitive interface designed for easy navigation and use.

Don't just analyze—**dominate** the world of audio with our all-in-one solution. Your speech deserves the best, and with our app, that's exactly what you'll get."""

os.makedirs("to_analyse", exist_ok=True)

conn = sqlite3.connect('web_management/user_data.db')
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
            st.session_state.clear() 
            st.session_state['logged_in'], st.session_state['username'] = True, username
            load_previous_feedback(username)
            st.sidebar.success(f"Signed in as {username}")
            st.rerun()
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
    st.title("LSD AI - Your personal AI Speech and Debate Coach")
    st.write(intro)

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

    # Clear previous feedback from session state before reloading
    if 'previous_feedback' in st.session_state:
        del st.session_state['previous_feedback']

    # Load all previous feedback into session state
    load_previous_feedback(username)

def load_previous_feedback(username):
    # Ensure session state is cleared before loading
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
                st.markdown(f"**Content Feedback**\n\n {entry[1]}\n\n")
            if entry[2]:
                st.markdown(f"**Emphasis Feedback**\n\n {entry[2]}\n\n")
            if entry[3]:
                st.markdown(f"**Tone Feedback**\n\n {entry[3]}\n\n")
            if entry[4]:
                st.markdown(f"**Speed Feedback**:\n\n WPM: {round(float(entry[4]))}\n\n")
            st.write("---")
    else:
        st.info("No previous feedback found.")

def plot_cumulative_score(scores, labels):
    total_score = (sum(scores) / len(scores))

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
    st.write("Welcome to the home page.")

    uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

    if uploaded_file is not None:
        base_name = uploaded_file.name.split('.')[0]

        file_path = os.path.join("to_analyse", f"{base_name}.wav")

        if "wav" not in uploaded_file.type:
            audio = AudioSegment.from_mp3(uploaded_file)
            file_path = os.path.join("to_analyse", f"{base_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav")
            audio.export(file_path, format="wav")
            st.success(f"File uploaded and converted to WAV format. Saved as {file_path}")
        else:
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            st.success(f"File uploaded and saved as {file_path}")

        st.session_state['analysis_config'] = {
            "content_analysis": st.checkbox("Content Analysis"),
            "emphasis_analysis": st.checkbox("Emphasis Analysis"),
            "tone_analysis": st.checkbox("Tone Analysis"),
            "speed_analysis": st.checkbox("Speed Analysis"),
            "file_path": file_path,
            "preset": st.selectbox("Choose a preset", ["N/A", "Impromptu", "Extempt", "Oratory"]),
            "progress": 0
        }

    if 'analysis_config' in st.session_state:
        config = st.session_state['analysis_config']
        
        if st.button("Analyze") or config['progress'] > 0:
            progress_bar = st.progress(config['progress'])

            if config['progress'] == 0:
                config['progress'] = 20
                progress_bar.progress(config['progress'])
                analysis = SpeechAnalysis(config['file_path'], config['preset'].lower())

                feedback, analysis_config = analysis.analyse(
                    [
                        1 if config['content_analysis'] else 0,
                        1 if config['emphasis_analysis'] else 0,
                        1 if config['tone_analysis'] else 0,
                        1 if config['speed_analysis'] else 0,
                    ]
                )
                config.update(feedback=feedback, analysis_config=analysis_config)
                config['progress'] = 60

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
                feedback_text.append(f"### Tone Feedback\n\n #### Determined Tonal Usage\n {config['feedback']['tone'][0]}\n\n #### Feedback\n {config['feedback']['tone'][1]}\n\n #### Score: {round(config['feedback']['tone'][2])}/100".replace("$", "S"))
            
            if config.get('analysis_config', [])[3] == 1:
                labels.append("Speed")
                scores.append(config['feedback']['speed'][1])
                feedback_text.append(f"### Speed Feedback\n\n #### Words per minute: {round(config['feedback']['speed'][0])} \n\n #### Score: {round(config['feedback']['speed'][1])}/100")

            config['progress'] = 80
            progress_bar.progress(config['progress'])

            col1, col2 = st.columns([1, 1])

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
            save_feedback_to_db(st.session_state['username'], config['feedback'], cumulative_score)

            config['progress'] = 100
            progress_bar.progress(config['progress'])
            st.success("Analysis complete!")
            progress_bar.empty()

def doc_page():
    st.title("Documentation")
    st.write(documentation)

def sign_out():
    st.session_state.clear()
    st.session_state['logged_in'] = False
    st.sidebar.success("You have signed out")
    st.rerun()

page_names_to_funcs = {
    "Home": home_page,
    "Doc": doc_page,
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