# Traffic Vehicle Counter

A Python-based traffic vehicle detection and counting system using YOLOv8 object detection model.

## Features

- Real-time vehicle detection using YOLOv8
- Video processing with object tracking
- Vehicle counting and lane detection
- Output annotated video with detection results

## Project Structure

```
traffic-vehicle-counter/
├── models/
│   └── yolov8n.pt          # Pre-trained YOLOv8 nano model
├── data/
│   └── video/
│       ├── raw/            # Input video files
│       │   └── traffic_video_demo.mp4
│       └── processed/      # Output video files
│           └── output_lane_count.mp4
├── main.py                 # Main script
├── requirements.txt        # Project dependencies
└── README.md              # This file
```

## Installation

1. **Clone or download the project:**
   ```bash
   cd traffic-vehicle-counter
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download YOLOv8 model:**
   - Place the pre-trained `yolov8n.pt` model in the `models/` directory
   - Or let the script auto-download it on first run

## Usage

1. **Place your video file:**
   - Add your traffic video to `data/video/raw/` directory
   - Update the video path in `main.py` if needed

2. **Run the detection:**
   ```bash
   python main.py
   ```

3. **View results:**
   - Output video will be saved to `data/video/processed/output_lane_count.mp4`

## Requirements

- Python 3.8+
- OpenCV
- YOLOv8 (Ultralytics)
- PyTorch
- NumPy

## Model Information

- **Model:** YOLOv8 Nano (yolov8n)
- **Input Size:** 640x640
- **Detects:** Various object classes including vehicles

## Future Enhancements

- [ ] Multi-lane vehicle counting
- [ ] Speed estimation
- [ ] Traffic flow analysis
- [ ] Real-time camera feed support
- [ ] Database logging
- [ ] Web interface

## License

This project is open source and available for educational purposes.

## Support

For issues or questions, please refer to the [YOLOv8 Documentation](https://docs.ultralytics.com/)
