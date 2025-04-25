import cv2
import os
import argparse
import numpy as np


def manual_crop(image_path, output_dir, target_size=(512, 512), mode='scale_and_pad'):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Failed to load {image_path}")
        return

    orig_h, orig_w = image.shape[:2]
    display_max_dim = 1200
    scale = min(display_max_dim / orig_w, display_max_dim / orig_h, 1.0)
    display_img = cv2.resize(image, (int(orig_w * scale), int(orig_h * scale)))

    print(f"Cropping: {image_path}")
    cv2.namedWindow("Select Flag Region", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Select Flag Region", int(orig_w * scale), int(orig_h * scale))
    roi = cv2.selectROI("Select Flag Region", display_img, fromCenter=False, showCrosshair=True)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("Skipped")
        return

    # Rescale ROI to original image size
    x, y, w, h = [int(val / scale) for val in roi]
    cropped = image[y:y+h, x:x+w]

    target_w, target_h = target_size
    if mode == 'scale_and_pad':
        # Resize maintaining aspect ratio
        ratio = min(target_w / w, target_h / h)
        new_w, new_h = int(w * ratio), int(h * ratio)
        resized = cv2.resize(cropped, (new_w, new_h))
        resized_bgra = cv2.cvtColor(resized, cv2.COLOR_BGR2BGRA)

        output_img = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        pad_x = (target_w - new_w) // 2
        pad_y = (target_h - new_h) // 2
        output_img[pad_y:pad_y+new_h, pad_x:pad_x+new_w] = resized_bgra

    elif mode == 'pad_only':
        # No scaling, center with padding
        cropped_bgra = cv2.cvtColor(cropped, cv2.COLOR_BGR2BGRA)
        ch, cw = cropped_bgra.shape[:2]

        output_img = np.zeros((target_h, target_w, 4), dtype=np.uint8)
        if ch > target_h or cw > target_w:
            print("Cropped image larger than target — skipping.")
            return

        pad_x = (target_w - cw) // 2
        pad_y = (target_h - ch) // 2
        output_img[pad_y:pad_y+ch, pad_x:pad_x+cw] = cropped_bgra
    else:
        print(f"Unknown mode: {mode}")
        return

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
    out_path = os.path.join(output_dir, base_name)
    cv2.imwrite(out_path, output_img)
    print(f"Saved: {out_path}")

def batch_crop_images(input_folder, output_folder, target_size=(512, 512), mode='scale_and_pad'):
    for filename in sorted(os.listdir(input_folder)):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(input_folder, filename)
            manual_crop(path, output_folder, target_size, mode)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input folder of flag images")
    parser.add_argument("output", help="Output folder for cropped PNGs")
    parser.add_argument("--size", type=int, default=512, help="Target square size (default 512)")
    parser.add_argument("--mode", choices=["scale_and_pad", "pad_only"], default="scale_and_pad",
                        help="Padding mode: scale and center, or just center")
    args = parser.parse_args()

    batch_crop_images(args.input, args.output, (args.size, args.size), args.mode)
