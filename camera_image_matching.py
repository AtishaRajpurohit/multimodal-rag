import cv2
import logging

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

def capture_images_from_webcam(window_name="Python Webcam Screenshot"):
    """
    Opens the webcam, displays live video, and captures an image
    each time the SPACE key is pressed.
    Press ESC to close the window safely.
    """
    cam = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # Use CAP_DSHOW on Windows if needed
    if not cam.isOpened():
        logger.error("Could not open webcam.")
        return

    cv2.namedWindow(window_name)
    img_counter = 0

    try:
        while True:
            ret, frame = cam.read()
            if not ret:
                logger.error("Failed to grab frame.")
                break

            cv2.imshow(window_name, frame)
            k = cv2.waitKey(1)

            if k % 256 == 27:
                # ESC pressed
                logger.info("Escape key pressed. Closing...")
                break

            elif k % 256 == 32:
                # SPACE pressed
                img_name = f"opencv_frame_{img_counter}.png"
                cv2.imwrite(img_name, frame)
                logger.info(f"Screenshot saved: {img_name}")
                img_counter += 1

    except KeyboardInterrupt:
        logger.warning("KeyboardInterrupt detected. Stopping capture loop...")

    finally:
        # Ensure proper cleanup
        cam.release()
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        logger.info("Camera released and windows closed.")

if __name__ == "__main__":
    capture_images_from_webcam()
