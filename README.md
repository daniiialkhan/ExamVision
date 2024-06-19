**ExamVision**📝👀 

**Deep Learning Project for Real-Time Exam Cheating Detection**

This project implements a real-time object detection system using Streamlit and YOLOv8 to monitor potential cheating activities during exams.

**About**

Cheating in exams can pose a significant challenge to educational institutions. This project aims to maintain academic integrity by providing a tool for real-time exam monitoring.

**System Architecture**

The system leverages the following components:

- **Streamlit:** A Python framework for creating user-friendly web applications. It facilitates the creation of a real-time video stream interface for monitoring exams.
- **YOLOv8:** A state-of-the-art object detection model known for its speed and accuracy. It's trained to detect specific objects or activities indicative of cheating, such as mobile phones, textbooks, or students looking around excessively.

**Features**

- **Real-time Video Stream:** The application displays a real-time video feed of the exam environment, allowing for continuous monitoring.
- **Face Detection:** YOLOv8 identifies and highlights activities potentially associated with cheating.

**Setup and Usage**

**Prerequisites:**

- Python ([https://www.python.org/](https://www.python.org/))
- pip ([https://pip.pypa.io/en/stable/](https://pip.pypa.io/en/stable/))
- Streamlit ([https://streamlit.io/](https://streamlit.io/))
- YOLOv8 ([https://roboflow.com/model/yolov8](https://roboflow.com/model/yolov8)) 

**Installation:**

1. Clone this repository:

   ```bash
   git clone https://github.com/daniiialkhan/ExamVision.git
   ```

2. Navigate to the project directory:

   ```bash
   cd ExamVision
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

**Running the Application:**

1. Start the Streamlit app:

   ```bash
   streamlit run streamlit_app.py
   ```

2. A web app will launch in your default browser, displaying the real-time video stream and object detection results.

**Disclaimer**

This project is intended for educational purposes and to promote academic integrity. It's essential to comply with all legal and ethical considerations regarding exam monitoring practices in your jurisdiction. Consult relevant authorities before deploying this system in a real-world exam setting.

**Further Development**

- Explore advanced features like bounding box overlays, notification triggers, and integration with exam management systems.
- Enhance the system's accuracy by fine-tuning the YOLOv8 model with your specific exam environment data.

**Contributing**

We welcome contributions to improve this project. Feel free to submit pull requests for bug fixes, enhancements, or new features 🌟 

