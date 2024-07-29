# TalkTracker

**TalkTracker** is an application designed to identify speakers based on their voice. The project is my first attempt in speech recognition and I appreciate your contribution to it! This project leverages speech recognition and machine learning techniques to build a speaker identification system. It's a great starting point for exploring audio processing and classification with Python.

## Features

-   **Add MP3 Files**: Upload audio files with corresponding labels to train the model.
-   **Train Model**: Train a machine learning model using Support Vector Classification (SVC) on the provided audio data.
-   **Predict from File**: Predict the speaker of a given audio file.
-   **Predict from Microphone**: Record audio from your microphone and predict the speaker in real-time.

## Installation

To run TalkTracker, you'll need Python and several dependencies. Follow these steps to set up your environment:

1. **Clone the Repository**

    ```bash
    git clone https://github.com/semajavoye/TalkTracker.git
    cd TalkTracker
    ```

2. **Create a Virtual Environment (Optional but Recommended)**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3. **Install Dependencies**

    Make sure you have `pip` installed. Then run:

    ```bash
    pip install -r requirements.txt
    ```

    The `requirements.txt` file should include the following libraries:

    ```
    tkinter
    librosa
    numpy
    soundfile
    scikit-learn
    pyaudio
    wave
    ```

## Usage

1. **Run the Application**

    After installing the dependencies, you can run the application with:

    ```bash
    python main.py
    ```

    This will launch a GUI window.

2. **Add MP3 Files**

    - Enter a label for the speaker in the "Label" field.
    - Click the "Add MP3 File" button to select and add an MP3 file. Ensure that each file is labeled correctly.

3. **Train the Model**

    - Click the "Train Model" button. The application will process the added files, convert them to WAV format, extract features, and train the model.
    - If the training is successful, you'll receive a message with the model's accuracy.

4. **Predict from File**

    - Click the "Predict from File" button to select an audio file (MP3 or WAV) and predict the speaker.
    - The application will display the predicted speaker label.

5. **Predict from Microphone**

    - Click the "Predict from Microphone" button to record audio from your microphone.
    - The recording will be processed, and the application will display the predicted speaker label.

## Contributing

Contributions are welcome! To contribute to TalkTracker, follow these steps:

1. **Fork the Repository**

    Click the "Fork" button at the top right of the repository page on GitHub.

2. **Clone Your Fork**

    ```bash
    git clone https://github.com/semajavoye/TalkTracker.git
    cd TalkTracker
    ```

3. **Create a Branch**

    ```bash
    git checkout -b your-feature-branch
    ```

4. **Make Your Changes**

    Modify the code, fix bugs, or add new features. Ensure your changes are well-tested.

5. **Commit Your Changes**

    ```bash
    git add .
    git commit -m "Describe your changes"
    ```

6. **Push Your Changes**

    ```bash
    git push origin your-feature-branch
    ```

7. **Create a Pull Request**

    Go to the GitHub repository page and create a new pull request from your branch. Provide a clear description of your changes.

## License

TalkTracker is licensed under the [MIT License](LICENSE).

## Acknowledgements

-   **Librosa**: For audio processing.
-   **Scikit-Learn**: For machine learning.
-   **PyAudio**: For audio recording.

Feel free to reach out with any questions or suggestions. Happy coding!
