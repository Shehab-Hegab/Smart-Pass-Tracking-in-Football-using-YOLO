VIDEO_SRC = "F:\\projects-hol\\Yolo\\football-analysis-CV-main\\football-analysis-CV-main\\output3.mp4"
OUT_VIDEO = ".\\output_video\\08fd33_4_output.mp4" 
MODEL_SRC = "yolov8n.pt"  # Using standard YOLOv8 nano model
MODEL_CLASSES = {
     "ball":0,
     "goalkepper":1,
     "player":2,
     "referee":3,
     "team1":4,
     "team2":5,
     "active_player":6
}
