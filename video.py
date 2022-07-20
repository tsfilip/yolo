import cv2
import tensorflow as tf

from absl import app, flags, logging
from model import yolo_v3

FLAGS = flags.FLAGS
flags.DEFINE_string("video_file", None, "Path to video file.")
flags.DEFINE_string("class_names", "./coco.names", "Path to file with class names.")
flags.DEFINE_string("weights_path", "./weights/yolov3.tf", "Path to saved Yolo weights.")
flags.DEFINE_integer("width", 416, "Reshaped video width.")
flags.DEFINE_integer("height", 416, "Reshaped video height.")


def process_frame(frame, width, height):
    frame = cv2.resize(frame, (width, height))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = tf.image.convert_image_dtype(frame, dtype=tf.float32)
    frame = tf.expand_dims(frame, axis=0)
    return frame


def main(args):
    n_class = 80
    color = (255, 0, 0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    coco_names = [label.strip() for label in open(FLAGS.class_names)]
    input_shape = [416, 416, 3]

    yolo = yolo_v3(input_shape, n_class)
    yolo.load_weights(FLAGS.weights_path)

    cap = cv2.VideoCapture(FLAGS.video_file)

    if not cap.isOpened():
        logging.error("Error opening video file.")

    while cap.isOpened():
        ret, frame = cap.read()

        if ret:
            processed_frame = process_frame(frame, FLAGS.width, FLAGS.height)
            pred = yolo(processed_frame)

            recs = pred[0] * tf.cast(tf.tile([frame.shape[1], frame.shape[0]], [2]), tf.float32)
            recs = tf.cast(recs, dtype=tf.int32)
            for i in range(pred[3][0]):  # n_rec for first image in batch
                title = f"{coco_names[int(pred[2][0, i])]}: {pred[1][0, i]:.2f}"
                frame = cv2.putText(frame, title, (recs[0, i, 0].numpy(), (recs[0, i, 1] - 3).numpy()), font, 0.6, color)
                frame = cv2.rectangle(frame, recs[0, i, :2].numpy(), recs[0, i, 2:4].numpy(), color, 2)
            cv2.imshow("Frame", frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        else:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    app.run(main)
