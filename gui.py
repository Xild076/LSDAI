import customtkinter as ctk
from tkinter import filedialog, ttk
from emphasis import generate_emphasis_feedback
from tone import get_full_tonal_feedback
from wpm import word_per_minute
from utility import transcibe_audio
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import numpy as np
import sounddevice as sd
import wave
import matplotlib.animation as animation
from customtkinter import DISABLED, NORMAL
import openai
from utility import get_api_key
openai.api_key = get_api_key('api_key.txt')


class SAA:
    def __init__(self, root):
        self.root = root
        self.root.title('S&D Analysis App')
        self.root.geometry('1200x800')

        self.file_path = None
        self.feedback_data = {}
        self.is_recording = False
        self.analysis_thread = None

        self.frame_left = ctk.CTkFrame(root, width=300)
        self.frame_left.pack(side='left', fill='y', padx=10, pady=10)

        self.frame_right = ctk.CTkFrame(root)
        self.frame_right.pack(side="right", fill="both", expand=True, padx=10, pady=10)

        self.frame_controls = ctk.CTkFrame(self.frame_left)
        self.frame_controls.pack(padx=10, pady=20, fill="x")

        self.frame_score = ctk.CTkFrame(self.frame_left)
        self.frame_score.pack(padx=10, pady=20, fill="x")

        self.upload_button = ctk.CTkButton(self.frame_controls, text='Upload File', command=self.upload_file, font=("Helvetica", 16))
        self.upload_button.pack(pady=10)

        self.record_button = ctk.CTkButton(self.frame_controls, text='Record Audio', command=self.toggle_record_audio, font=("Helvetica", 16))
        self.record_button.pack(pady=10)

        self.file_select = ctk.CTkLabel(self.frame_controls, text='Selected File: None', font=("Helvetica", 16))
        self.file_select.pack(pady=10)

        self.fbck_var = ctk.IntVar()
        self.emph_var = ctk.IntVar()
        self.tonal_var = ctk.IntVar()
        self.wpm_var = ctk.IntVar()

        self.fbck_check = ctk.CTkCheckBox(self.frame_controls, text='General', variable=self.fbck_var, font=("Helvetica", 16))
        self.fbck_check.pack(pady=5)

        self.emph_check = ctk.CTkCheckBox(self.frame_controls, text="Emphasis", variable=self.emph_var, font=("Helvetica", 16))
        self.emph_check.pack(pady=5)

        self.tonal_check = ctk.CTkCheckBox(self.frame_controls, text="Tonal", variable=self.tonal_var, font=("Helvetica", 16))
        self.tonal_check.pack(pady=5)

        self.wpm_check = ctk.CTkCheckBox(self.frame_controls, text="Words Per Minute", variable=self.wpm_var, font=("Helvetica", 16))
        self.wpm_check.pack(pady=5)

        self.start_button = ctk.CTkButton(self.frame_controls, text="Start Analysis", command=self.start_analysis_thread, font=("Helvetica", 16))
        self.start_button.pack(pady=20)

        self.progress_bar = ctk.CTkProgressBar(self.frame_controls)
        self.progress_bar.pack(pady=10)
        self.progress_bar.set(0)

        self.score_label = ctk.CTkLabel(self.frame_score, text="Score: ", font=("Helvetica", 24))
        self.score_label.pack(pady=10)
        self.canvas = None

        self.feedback_label = ctk.CTkLabel(self.frame_right, text="Feedback", anchor="w", font=("Helvetica", 24))
        self.feedback_label.pack(pady=10, padx=10, anchor="w")

        self.notebook = ttk.Notebook(self.frame_right)
        self.notebook.pack(pady=20, padx=20, fill="both", expand=True)

        self.general_feedback_frame = ctk.CTkFrame(self.notebook)
        self.emph_feedback_frame = ctk.CTkFrame(self.notebook)
        self.tonal_feedback_frame = ctk.CTkFrame(self.notebook)
        self.wpm_feedback_frame = ctk.CTkFrame(self.notebook)

        self.notebook.add(self.general_feedback_frame, text="General Feedback")
        self.notebook.add(self.emph_feedback_frame, text="Emphasis")
        self.notebook.add(self.tonal_feedback_frame, text="Tonal")
        self.notebook.add(self.wpm_feedback_frame, text="WPM")

        self.general_feedback_text = ctk.CTkTextbox(self.general_feedback_frame, font=("Helvetica", 16), state='disabled')
        self.general_feedback_text.pack(fill="both", padx=10, pady=10, expand=True)
        self.general_feedback_score = ctk.CTkLabel(self.general_feedback_frame, text="", font=("Helvetica", 24))
        self.general_feedback_score.pack(pady=10)

        self.emph_text_frame = ctk.CTkFrame(self.emph_feedback_frame)
        self.emph_text_frame.pack(fill="both", padx=10, pady=10, expand=True)
        self.emph_text_textbox = ctk.CTkTextbox(self.emph_text_frame, font=("Helvetica", 16), state='disabled')
        self.emph_text_textbox.pack(fill="both", padx=10, pady=10, expand=True)
        self.emph_feedback_text = ctk.CTkTextbox(self.emph_feedback_frame, font=("Helvetica", 16), state='disabled')
        self.emph_feedback_text.pack(fill="both", padx=10, pady=10, expand=True)
        self.emph_feedback_score = ctk.CTkLabel(self.emph_feedback_frame, text="", font=("Helvetica", 24))
        self.emph_feedback_score.pack(pady=10)

        self.tonal_feedback_text = ctk.CTkTextbox(self.tonal_feedback_frame, font=("Helvetica", 16), state='disabled')
        self.tonal_feedback_text.pack(fill="both", padx=10, pady=10, expand=True)
        self.tonal_feedback_score = ctk.CTkLabel(self.tonal_feedback_frame, text="", font=("Helvetica", 24))
        self.tonal_feedback_score.pack(pady=10)

        self.wpm_feedback_text = ctk.CTkTextbox(self.wpm_feedback_frame, font=("Helvetica", 16), state='disabled')
        self.wpm_feedback_text.pack(fill="both", padx=10, pady=10, expand=True)
        self.wpm_graph_frame = ctk.CTkFrame(self.wpm_feedback_frame)
        self.wpm_graph_frame.pack(fill="both", padx=10, pady=10, expand=True)
        self.wpm_feedback_score = ctk.CTkLabel(self.wpm_feedback_frame, text="", font=("Helvetica", 24))
        self.wpm_feedback_score.pack(pady=10)

        self.lock_notebook_tabs()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def lock_notebook_tabs(self):
        for i in range(4):
            self.notebook.tab(i, state='disabled')

    def unlock_notebook_tab(self, index):
        self.notebook.tab(index, state=NORMAL)
        text_widgets = [self.general_feedback_text, self.emph_text_textbox, self.emph_feedback_text, self.tonal_feedback_text, self.wpm_feedback_text]
        text_widgets[index].configure(state=NORMAL)

    def upload_file(self):
        self.file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav *.mp3")])
        if self.file_path:
            self.file_select.configure(text=f'Selected File: {self.file_path}')

    def toggle_record_audio(self):
        if self.is_recording:
            self.is_recording = False
            self.record_button.configure(text="Record Audio")
            sd.stop()
        else:
            self.is_recording = True
            self.record_button.configure(text="Stop Recording")
            threading.Thread(target=self.record_audio).start()

    def record_audio(self):
        self.file_path = "output.wav"
        with wave.open(self.file_path, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(44100)
            with sd.InputStream(samplerate=44100, channels=1, dtype='int16') as stream:
                while self.is_recording:
                    wf.writeframes(stream.read(1024)[0])
        self.file_select.configure(text=f'Selected File: {self.file_path}')

    def write_wave_file(self, filename, data, fs):
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(1)  # Mono channel
            wf.setsampwidth(2)  # 2 bytes per sample
            wf.setframerate(fs)
            wf.writeframes(data.tobytes())

    def start_analysis_thread(self):
        self.upload_button.configure(state=DISABLED)
        self.start_button.configure(state=DISABLED)
        self.lock_notebook_tabs()
        self.clear_feedback()
        self.analysis_thread = threading.Thread(target=self.start_analysis)
        self.analysis_thread.start()

    def update_progress(self, value):
        self.root.after(0, lambda: self.progress_bar.set(value / 100))

    def start_analysis(self):
        if not self.file_path:
            self.root.after(0, lambda: self.general_feedback_text.configure(state=NORMAL))
            self.root.after(0, lambda: self.general_feedback_text.insert('1.0', 'Selected File: Please Insert'))
            self.root.after(0, lambda: self.general_feedback_text.configure(state=DISABLED))
            self.upload_button.configure(state=NORMAL)
            self.start_button.configure(state=NORMAL)
            return

        scores = {}
        self.root.after(0, lambda: self.progress_bar.set(0))

        checked = [self.fbck_var.get(), self.emph_var.get(), self.tonal_var.get(), self.wpm_var.get()]
        if self.fbck_var.get() or self.tonal_var.get():
            tt = transcibe_audio(self.file_path, 'sentence')
            full_text = ' '.join(tt)

        denom = sum(checked)
        numer = 0

        self.clear_feedback()

        if self.fbck_var.get():
            numer += 1
            prompt = full_text + "\nGive general feedback for the content of this speech. Then give a score out of 100. Put it into the following format: {feedback} | {score}. The score must be just its value, not x/100, nor with a period at the end."
        
            def chat_with_gpt(prompt):
                response = openai.chat.completions.create(
                    model="gpt-4",
                    messages=[
                        {"role": "system", "content": "You are a speech coach who gives feedback for a student's content in their speeches."},
                        {"role": "user", "content": prompt},
                    ]
                )
                return response.choices[0].message.content
            
            result = chat_with_gpt(prompt).split(' | ')
            proper_response = result[0]
            score_fbck = int(result[1])
            self.root.after(0, lambda: self.general_feedback_text.configure(state=NORMAL))
            self.root.after(0, lambda: self.general_feedback_text.insert("end", f"General Feedback:\n{proper_response}\n\n", ("bold", "color")))
            self.root.after(0, lambda: self.general_feedback_text.configure(state=DISABLED))
            self.root.after(0, lambda: self.general_feedback_score.configure(text=f"Score: {round(score_fbck)}/100"))
            scores['General'] = score_fbck
            self.root.after(0, lambda: self.update_progress(numer / denom * 100))
            self.root.after(0, lambda: self.unlock_notebook_tab(0))

        if self.emph_var.get():
            numer += 1
            self.root.after(0, lambda: self.update_progress(25))
            emph_text, proper_response, score_emph = generate_emphasis_feedback(self.file_path)
            self.root.after(0, lambda: self.emph_text_textbox.configure(state=NORMAL))
            self.root.after(0, lambda: self.emph_text_textbox.insert("end", f"Emphasized Texts:\n{emph_text}\n\n", ("bold", "color")))
            self.root.after(0, lambda: self.emph_text_textbox.configure(state=DISABLED))
            self.root.after(0, lambda: self.emph_feedback_text.configure(state=NORMAL))
            self.root.after(0, lambda: self.emph_feedback_text.insert("end", f"Emphasis Feedback:\n{proper_response}\n\n", ("italic", "color")))
            self.root.after(0, lambda: self.emph_feedback_text.configure(state=DISABLED))
            self.root.after(0, lambda: self.emph_feedback_score.configure(text=f"Score: {round(score_emph)}/100"))
            scores['Emphasis'] = score_emph
            self.root.after(0, lambda: self.update_progress(numer / denom * 100))
            self.root.after(0, lambda: self.unlock_notebook_tab(1))

        if self.tonal_var.get():
            numer += 1
            self.root.after(0, lambda: self.update_progress(50))
            score_tonal, importance, sentiment_t, sentiment_a = get_full_tonal_feedback(self.file_path)
            interleaved_string = ""
            for sente, impor, senti_t, senti_a in zip(tt, importance, sentiment_t, sentiment_a):
                st, sa = 'neutral', 'neutral'
                if int(senti_t) == 1:
                    st = 'positive'
                if int(senti_t) == -1:
                    st = 'negative'
                if int(senti_a) == 1:
                    sa = 'positive'
                if int(senti_a) == -1:
                    sa = 'negative'
                interleaved_string += f"Sentence: {sente}\nImportance: {impor}\nSentiment Values: Text emotion is {st}, Audio emotion is {sa}\n\n"
            self.root.after(0, lambda: self.tonal_feedback_text.configure(state=NORMAL))
            self.root.after(0, lambda: self.tonal_feedback_text.insert("end", f"Tonal Feedback:\n\n{interleaved_string}\n\n", ("italic", "color")))
            self.root.after(0, lambda: self.tonal_feedback_text.configure(state=DISABLED))
            self.root.after(0, lambda: self.tonal_feedback_score.configure(text=f"Score: {round(score_tonal)}/100"))
            scores['Tonal'] = score_tonal
            self.root.after(0, lambda: self.update_progress(numer / denom * 100))
            self.root.after(0, lambda: self.unlock_notebook_tab(2))

        if self.wpm_var.get():
            numer += 1
            self.root.after(0, lambda: self.update_progress(75))
            wpm, score_wpm = word_per_minute(self.file_path)
            self.root.after(0, lambda: self.wpm_feedback_text.configure(state=NORMAL))
            self.root.after(0, lambda: self.wpm_feedback_text.insert("end", f"Words Per Minute: {wpm}\n\n", ("bold", "color")))
            self.root.after(0, lambda: self.wpm_feedback_text.configure(state=DISABLED))
            self.root.after(0, lambda: self.plot_wpm_graph(wpm))
            self.root.after(0, lambda: self.wpm_feedback_score.configure(text=f"Score: {round(score_wpm)}/100"))
            scores['WPM'] = score_wpm
            self.root.after(0, lambda: self.update_progress(numer / denom * 100))
            self.root.after(0, lambda: self.unlock_notebook_tab(3))

        if scores:
            # Define the weights for each feedback type
            weights = {'General': 0.4, 'Emphasis': 0.3, 'Tonal': 0.2, 'WPM': 0.1}
            total_weight = sum(weights[feedback] for feedback in scores.keys())
            normalized_weights = {feedback: weights[feedback] / total_weight for feedback in scores.keys()}
            total_score = sum(scores[feedback] * normalized_weights[feedback] for feedback in scores)
            self.root.after(0, lambda: self.display_score_circle(total_score, scores, normalized_weights))
            self.root.after(0, lambda: self.score_label.configure(text=f"Score: {round(total_score)}/100"))

        self.save_feedback(scores)

        self.root.after(0, lambda: self.upload_button.configure(state=NORMAL))
        self.root.after(0, lambda: self.start_button.configure(state=NORMAL))

    def plot_wpm_graph(self, wpm):
        # Clear any previous plots
        for widget in self.wpm_graph_frame.winfo_children():
            widget.destroy()

        fig, ax = plt.subplots(figsize=(5, 3))
        x = np.linspace(50, 200, 500)
        y = 2 * np.exp((x - 125) / 20) / ((np.exp(2 * (x - 125) / 20)) + 1)
        ax.plot(x, y, label="WPM Score Function", color='blue')
        ax.axvline(x=wpm, color='red', linestyle='--', label=f'Your WPM: {wpm}')
        ax.legend()

        ax.set_xlabel("Words Per Minute")
        ax.set_ylabel("Score")
        ax.set_title("WPM Score Function")

        def animate(i):
            ax.axvline(x=wpm, color='red', linestyle='--', label=f'Your WPM: {wpm}')

        ani = animation.FuncAnimation(fig, animate, frames=np.arange(0, 200), interval=20)

        wpm_canvas = FigureCanvasTkAgg(fig, master=self.wpm_graph_frame)
        wpm_canvas.draw()
        wpm_canvas.get_tk_widget().pack(pady=20, fill='both', expand=True)

    def display_score_circle(self, total_score, scores, weights):
        # Clear previous plot if it exists
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()

        fig, ax = plt.subplots(figsize=(5, 5))

        # Calculate total scores and prepare for pie chart
        total_scores = [scores[feedback] * weights[feedback] for feedback in scores]
        left_score = 100 - total_score

        labels = list(scores.keys())
        colors = ['#4CAF50', '#388E3C', '#66BB6A', '#2E7D32'][:len(labels)]

        labels.append('Remaining')
        colors.append('#D32F2F')
        total_scores.append(left_score)

        wedges, texts = ax.pie(
            total_scores,
            labels=[f'{label} ({round(score, 1)})' for label, score in zip(labels, total_scores)],
            colors=colors,
            startangle=90,
            counterclock=False,
            textprops={'color': "w"}
        )

        # Create a circle in the middle to make it a donut chart
        center_circle = plt.Circle((0, 0), 0.70, fc='#212121')
        fig.gca().add_artist(center_circle)

        fig.patch.set_facecolor('#212121')  # Match frame background
        ax.set_facecolor('#212121')

        self.canvas = FigureCanvasTkAgg(fig, master=self.frame_score)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(pady=20, fill='both', expand=True)

    def clear_feedback(self):
        self.general_feedback_text.configure(state=NORMAL)
        self.general_feedback_text.delete('1.0', 'end')
        self.general_feedback_text.configure(state=DISABLED)

        self.emph_text_textbox.configure(state=NORMAL)
        self.emph_text_textbox.delete('1.0', 'end')
        self.emph_text_textbox.configure(state=DISABLED)

        self.emph_feedback_text.configure(state=NORMAL)
        self.emph_feedback_text.delete('1.0', 'end')
        self.emph_feedback_text.configure(state=DISABLED)

        self.tonal_feedback_text.configure(state=NORMAL)
        self.tonal_feedback_text.delete('1.0', 'end')
        self.tonal_feedback_text.configure(state=DISABLED)

        self.wpm_feedback_text.configure(state=NORMAL)
        self.wpm_feedback_text.delete('1.0', 'end')
        self.wpm_feedback_text.configure(state=DISABLED)

        for frame in self.notebook.winfo_children():
            for widget in frame.winfo_children():
                if isinstance(widget, FigureCanvasTkAgg):
                    widget.get_tk_widget().pack_forget()

        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
            self.canvas = None

    def save_feedback(self, scores):
        feedback_entry = {
            'file_path': self.file_path,
            'scores': scores,
            'general_feedback': self.general_feedback_text.get('1.0', 'end').strip(),
            'emphasis_feedback': self.emph_feedback_text.get('1.0', 'end').strip(),
            'tonal_feedback': self.tonal_feedback_text.get('1.0', 'end').strip(),
            'wpm_feedback': self.wpm_feedback_text.get('1.0', 'end').strip()
        }
        self.feedback_data[self.file_path] = feedback_entry

    def load_feedback(self, file_path):
        feedback_entry = self.feedback_data.get(file_path)
        if feedback_entry:
            self.general_feedback_text.configure(state=NORMAL)
            self.general_feedback_text.delete('1.0', 'end')
            self.general_feedback_text.insert('1.0', feedback_entry['general_feedback'])
            self.general_feedback_text.configure(state=DISABLED)

            self.emph_feedback_text.configure(state=NORMAL)
            self.emph_feedback_text.delete('1.0', 'end')
            self.emph_feedback_text.insert('1.0', feedback_entry['emphasis_feedback'])
            self.emph_feedback_text.configure(state=DISABLED)

            self.tonal_feedback_text.configure(state=NORMAL)
            self.tonal_feedback_text.delete('1.0', 'end')
            self.tonal_feedback_text.insert('1.0', feedback_entry['tonal_feedback'])
            self.tonal_feedback_text.configure(state=DISABLED)

            self.wpm_feedback_text.configure(state=NORMAL)
            self.wpm_feedback_text.delete('1.0', 'end')
            self.wpm_feedback_text.insert('1.0', feedback_entry['wpm_feedback'])
            self.wpm_feedback_text.configure(state=DISABLED)

            scores = feedback_entry['scores']
            weights = {'General': 0.4, 'Emphasis': 0.3, 'Tonal': 0.2, 'WPM': 0.1}
            total_weight = sum(weights[feedback] for feedback in scores.keys())
            normalized_weights = {feedback: weights[feedback] / total_weight for feedback in scores.keys()}
            self.display_score_circle(sum(scores[feedback] * normalized_weights[feedback] for feedback in scores), scores, normalized_weights)

    def on_closing(self):
        try:
            self.root.destroy()
        except:
            pass
        if self.analysis_thread and self.analysis_thread.is_alive():
            self.analysis_thread.join()
        exit()


if __name__ == "__main__":
    app = ctk.CTk()
    SAA(app)
    app.mainloop()