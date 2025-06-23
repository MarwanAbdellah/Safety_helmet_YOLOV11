from PIL import Image

from ultralytics import YOLO

def inference(img_path: str):
    # Loading the weights for the YOLO model
    model = YOLO("runs/trial_3/weights/best.pt")
    model.to('cpu')

    # Run inference
    results = model(img_path)

    # Visualize the results
    for i, r in enumerate(results):
        # Plot results image
        im_bgr = r.plot()  # BGR-order numpy array
        im_rgb = Image.fromarray(im_bgr[..., ::-1])  # RGB-order PIL image

        return im_rgb

if __name__ == "__main__":
    inference(r"safety-helmet-1\test\images\european-engineer-handsome-man-architect-600w-2217615241_jpg.rf.6e3c6825629e7735dd58408378a412b6.jpg")