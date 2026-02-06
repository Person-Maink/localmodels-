import cv2
import os
from natsort import natsorted

def images_to_video(input_folder, output_path, fps=30):
    images = [f for f in os.listdir(input_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    images = natsorted(images, key=lambda x: os.path.splitext(x)[0])

    if not images:
        raise ValueError(f"No images found in folder: {input_folder}")

    first_frame = cv2.imread(os.path.join(input_folder, images[0]))
    height, width, _ = first_frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # or 'XVID'
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_name in images:
        img_path = os.path.join(input_folder, img_name)
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"Warning: Skipping unreadable image {img_name}")
            continue
        out.write(frame)

    out.release()


def process_all_folders(base_folder, output_base_folder, fps=30):
    # Iterate through all subfolders in the base folder
    for folder_name in os.listdir(base_folder):
        folder_path = os.path.join(base_folder, folder_name)

        # Check if it's a directory
        if os.path.isdir(folder_path):
            visualizations_folder = os.path.join(folder_path, "visualizations")

            # Check if the "visualizations" subfolder exists
            if os.path.exists(visualizations_folder):
                output_video_path = os.path.join(output_base_folder, f"{folder_name}_combined.mp4")

                # Check if there are any images in the "visualizations" folder
                if any(f.lower().endswith(('.png', '.jpg', '.jpeg')) for f in os.listdir(visualizations_folder)):
                    print(f"Processing folder: {folder_name}")
                    images_to_video(visualizations_folder, output_video_path, fps)
                else:
                    print(f"No images found in {visualizations_folder}, skipping.")
            else:
                print(f"Visualizations folder not found in {folder_name}, skipping.")


if __name__ == "__main__":
    base_folder = "/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor"
    output_base_folder = "/home/mayank/Documents/Uni/TUD/Thesis Extra/comparative study/outputs/wilor/videos"
    process_all_folders(base_folder, output_base_folder)
