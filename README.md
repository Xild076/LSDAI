## Overview

The LSDAI serves as the main entry point for a Streamlit application designed to perform advanced audio analysis tasks, including emphasis detection, tone analysis, and speech speed evaluation. This application leverages multiple Python modules and files to deliver a comprehensive analysis of audio files.

## Warnings

Currently, the application is running off of a personal API key. The usage of external API keys will be implemented in the future.

## Dependencies

The application requires several Python libraries and external modules, which should be installed before running the app:

- `streamlit`: For building the web interface.
- `emphasis.py`: Handles the detection of emphasized words or phrases in the audio.
- `tone.py`: Manages the tone analysis functionality, providing insights into the emotional content of speech.
- `util.py`: Contains utility functions that support the main analysis, such as file handling and formatting.
- `content.py`: Facilitates content extraction and organization for analysis.
- `speech.py`: Implements speech analysis, particularly focusing on the quality and clarity of spoken words.
- `speed.py`: Provides functionality for calculating and analyzing the speed of speech.
- `api_key.encrypted`: A text file containing the necessary API keys for accessing external services (Paid OpenAI Key)

## Key Features

- **Audio File Upload**: Users can upload audio files of various formats. The app processes these files and conducts multiple forms of analysis.
  
- **Emphasis Detection**: The `emphasis.py` module is used to detect and highlight emphasized words or phrases within the audio, making it easier to identify key points or stressed content.
  
- **Tone Analysis**: The `tone.py` module provides a detailed analysis of the tone used in the speech, helping to discern emotional undertones or intentions.
  
- **Speech Speed Analysis**: The `speed.py` module measures the speed of speech, which can be an indicator of various factors such as nervousness or excitement.
  
- **Speech Clarity**: The `speech.py` module assesses the clarity of the speech, ensuring that the spoken words are easily understood.
  
- **User Authentication**: The app includes user authentication functionality, which uses the API key from `api_key.encrypted` to manage access and usage.
  
- **Persistent Data**: Data and analysis results are saved and can be revisited or updated as needed. This includes maintaining previous analysis for each user.

## Usage

1. **Running the App**: To run the app, use the Streamlit command:
   ```bash
   streamlit run app.py
   ```

2. **Uploading Audio Files**: Users can upload audio files directly through the app's interface. The app will automatically process these files and display results.

3. **Analysis**: Once an audio file is uploaded, the app will conduct emphasis detection, tone analysis, and speed analysis. Results are displayed in an organized manner, with the option to save or revisit previous analyses.

4. **API Key Management**: Ensure that the `api_key.encrypted` file contains a valid API key. This key is crucial for accessing the external services used in the analysis.

#### Notes

- **File Size Limitations**: There may be size limitations on the audio files that can be uploaded, depending on the configuration of the Streamlit server and the underlying infrastructure.
  
- **Customizations**: The app is modular, allowing for easy customization. Developers can add additional analysis modules or modify existing ones by editing the corresponding `.py` files.

- **Error Handling**: Basic error handling is implemented, but users should ensure that the audio files are properly formatted and that all dependencies are installed correctly.

## Future Enhancements

- **Improved GUI**: Plans to enhance the graphical interface, including animations and dynamic updates, are under consideration.
  
- **Expanded Analysis**: Additional forms of analysis, such as sentiment analysis or linguistic complexity, may be integrated into future versions.

- **Multi-User Support**: Expanding the app's capabilities to support multiple users with separate accounts and analysis histories.
