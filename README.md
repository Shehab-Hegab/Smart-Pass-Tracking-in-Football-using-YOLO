# ⚽ Smart Pass Tracking in Football using YOLO

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://www.python.org/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-brightgreen?logo=ultralytics)](https://github.com/ultralytics/ultralytics)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8%2B-blue?logo=opencv)](https://opencv.org/)

---

## 📌 Overview

**Smart Pass Tracking in Football** is an end-to-end computer vision pipeline for analyzing passes in football (soccer) matches using state-of-the-art deep learning. Built on **YOLOv8**, this project detects and tracks players and the ball, analyzes pass events, and visualizes passing patterns in match videos.

**Core goals:**
- Detect players, referees, and the ball in real time.
- Assign players to teams using clustering based on jersey colors.
- Track ball possession and pass events.
- Visualize passes and player movement to support match analysis.

---

## 🎥 Demo Video

_A full simulation video will be added here._  
👉 **[Upload your final output video here!]**

<p align="center">
  <img src="https://github.com/user-attachments/assets/placeholder.gif" width="700"/>
</p>

---

## 🚀 Features

- 🎯 **Object Detection:** Detects players, referees, and the ball using YOLOv8.
- 🏳️ **Team Clustering:** Groups players into teams using KMeans on jersey colors.
- ⚽ **Pass Detection:** Tracks who passes to whom and when.
- 🏃 **Player & Ball Tracking:** Maintains IDs and tracks motion across frames.
- 📊 **Pass Statistics:** Generates pass maps and passing percentages.
- 📏 **Real-World Metrics:** Converts pixel coordinates to real-world distances.
- 🎥 **Annotated Videos:** Outputs videos with bounding boxes, player IDs, teams, and pass arrows.

---

## ⚙️ Architecture

```mermaid
graph TD;
    A[Input Video] --> B[YOLOv8 Detection];
    B --> C[Player/Referee/Ball Extraction];
    C --> D[Team Assignment (KMeans)];
    D --> E[Player & Ball Tracking];
    E --> F[Pass Detection];
    F --> G[Visualization & Stats];
    G --> H[Annotated Video Output];
```

---

## ⚡ Quick Start

### 1️⃣ Clone & Install
```bash
git clone https://github.com/Shehab-Hegab/Smart-Pass-Tracking-in-Football-using-YOLO.git
cd Smart-Pass-Tracking-in-Football-using-YOLO/Pass\ Analysis
pip install -r requirements.txt
```

### 2️⃣ Prepare Your Video
- Place your match video in `local_exec/input_video/`
- Edit `local_exec/config/config.py` to set `VIDEO_SRC` and `OUT_VIDEO`

### 3️⃣ Run the Pipeline
```bash
cd local_exec
python main_test.py
```
- The annotated output video will appear in `local_exec/output_video/`

---

## ✅ Example Output

<p align="center">
  <img src="https://github.com/user-attachments/assets/sample-pass.gif" width="700"/>
</p>

- Player IDs, team colors, ball possession, and pass visualization are overlaid in the output.

---

## 🔗 Tech Stack

- **Python 3.8+**
- [YOLOv8](https://github.com/ultralytics/ultralytics)
- [OpenCV](https://opencv.org/)
- [scikit-learn](https://scikit-learn.org/)
- [supervision](https://github.com/roboflow/supervision)
- [PyTorch](https://pytorch.org/)
- [NumPy](https://numpy.org/)
- [matplotlib](https://matplotlib.org/)

---

## 📂 Project Structure

```
Pass Analysis/
├── local_exec/
│   ├── main_test.py         # Main entry point
│   ├── config/              # Configurations
│   ├── utils/               # Helper functions
│   ├── team_assigner/       # Team color clustering
│   ├── input_video/         # Raw videos
│   └── output_video/        # Processed videos
├── kaggle_exec/             # Training scripts/notebooks
├── requirements.txt         # Dependencies
└── README.md
```

---

## 🔬 Model Training (Optional)

- Train custom YOLO models with your own data via notebooks in `kaggle_exec/`
- Use [Roboflow](https://roboflow.com/) or custom datasets for fine-tuning

---

## ⚠️ Challenges & Future Work

**Current Challenges:**
- Handling overlapping players
- Accurate pass detection in crowded scenes
- Jersey color overlap

**Future Improvements:**
- Integrate pose estimation for finer pass detection
- Use more robust clustering for teams/roles
- Add event classification: tackles, shots, goals

---

## 🤝 Contributing

Contributions are welcome! Please open issues or submit PRs for bugs, features, or enhancements.

1. Fork this repo.
2. Create a new branch: `git checkout -b feature/YourFeature`.
3. Commit changes: `git commit -m 'Add new feature'`.
4. Push: `git push origin feature/YourFeature`.
5. Submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 📎 Repository

**[GitHub Repo](https://github.com/Shehab-Hegab/Smart-Pass-Tracking-in-Football-using-YOLO)**

---

**Need help or want to show your results?**  
Feel free to open an issue or reach out!
