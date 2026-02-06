import cv2
import os
import argparse

def visualize_crop_guides(frame, x_start, x_end, y_start, y_end, window_name="Crop Preview"):
    h, w = frame.shape[:2]
    x1, x2 = int(x_start * w), int(x_end * w)
    y1, y2 = int(y_start * h), int(y_end * h)

    preview = frame.copy()

    # Shade outside the crop
    overlay = preview.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, preview, 0.4, 0, preview)

    # Crop rectangle + guide lines
    cv2.rectangle(preview, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.line(preview, (x1, 0), (x1, h), (0, 255, 255), 1)
    cv2.line(preview, (x2, 0), (x2, h), (0, 255, 255), 1)
    cv2.line(preview, (0, y1), (w, y1), (0, 255, 255), 1)
    cv2.line(preview, (0, y2), (w, y2), (0, 255, 255), 1)

    cv2.imshow(window_name, preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def visualize_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"Failed to read: {video_path}")
        return None, None, None

    h, w = frame.shape[:2]
    grid = frame.copy()

    # major grid lines (0.1, 0.2, ...)
    for f in [i / 10 for i in range(1, 10)]:
        x = int(w * f)
        y = int(h * f)
        # vertical
        cv2.line(grid, (x, 0), (x, h), (255, 255, 255), 2)
        cv2.putText(grid, f"{f:.1f}", (x + 5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # horizontal
        cv2.line(grid, (0, y), (w, y), (255, 255, 255), 2)
        cv2.putText(grid, f"{f:.1f}", (5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # minor grid lines (0.05, 0.15, ...)
    overlay = grid.copy()
    for f in [i / 20 for i in range(1, 20, 2)]:
        x = int(w * f)
        y = int(h * f)
        cv2.line(overlay, (x, 0), (x, h), (255, 255, 255), 1)
        cv2.line(overlay, (0, y), (w, y), (255, 255, 255), 1)

    # make minor lines semi-transparent
    cv2.addWeighted(overlay, 0.3, grid, 0.7, 0, grid)

    cv2.imshow("Sample Frame (with grid)", grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return frame, w, h




def crop_video(video_path, output_path, x_start, x_end, y_start, y_end):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    x1, x2 = int(x_start * width), int(x_end * width)
    y1, y2 = int(y_start * height), int(y_end * height)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (x2 - x1, y2 - y1))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cropped = frame[y1:y2, x1:x2]
        out.write(cropped)

    cap.release()
    out.release()
    print(f"Saved cropped video: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Crop all videos in a folder based on scaled coordinates.")
    parser.add_argument("--input_folder", type=str, default="../../data/", help="Path to folder containing videos")
    parser.add_argument("--output_folder", type=str, default="../../cropped/", help="Path to save cropped videos")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)


    for fname in os.listdir(args.input_folder):

        video_path = os.path.join(args.input_folder, fname)
        print(f"\nVisualizing: {fname}")

        sample_frame, _, _ = visualize_frame(video_path)
        if sample_frame is None:
            continue

        while True:
            print("Enter cropping coordinates (scaled 0–1):")
            x_start = float(input("x_start: "))
            x_end   = float(input("x_end: "))
            y_start = float(input("y_start: "))
            y_end   = float(input("y_end: "))

            visualize_crop_guides(sample_frame, x_start, x_end, y_start, y_end)

            confirm = input("Adjust crop lines? (y/n): ").strip().lower()
            if confirm == "n":
                break

        visualize_crop_guides(sample_frame, x_start, x_end, y_start, y_end)

        base = os.path.splitext(fname)[0]
        out_path = os.path.join(args.output_folder, f"{base}_cropped.mp4")
        crop_video(video_path, out_path, x_start, x_end, y_start, y_end)

if __name__ == "__main__":
    main()
