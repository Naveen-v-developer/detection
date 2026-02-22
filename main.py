from detection import process_video

if __name__ == "__main__":
    input_path = "data/raw/Traffic_video_demo.mp4"
    output_path = "data/processed/output.mp4"

    # Change to "left" or "right"
    lane_side = "left"

    process_video(input_path, output_path, lane_side)