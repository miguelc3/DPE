# Import stuff
import cv2
import PySpin


# Main function
def main():

    print('Press "q" if you want to exit')

    # initial setup - define a video capture object
    capture = cv2.VideoCapture(1, cv2.CAP_DSHOW)

    if not capture.isOpened():
        raise IOError("Cannot open webcam")

    while True:
        # Capture video frame
        _, frame = capture.read()
        color_cv_image = cv2.cvtColor(frame[..., 0], cv2.COLOR_BAYER_BG2BGR)

        # Display frame
        window_name = 'camera'
        cv2.imshow(window_name, color_cv_image)
        # cv.imwrite('frame.png', frame)

        # Press any key to exit
        key = cv2.waitKey(10)
        if key == ord('q'):
            print('You typed "q" to exit')
            break

    capture.release()


if __name__ == '__main__':
    main()
