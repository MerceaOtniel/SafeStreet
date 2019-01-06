import cv2
import numpy as np
from keras.models import load_model
import logging
import collections
import copy
import _thread
import tensorflow as tf

global graph, model
graph = tf.get_default_graph()

MAX_FAILED_FRAMES = 200
MAX_FRAMES_SENT = 50
logging.basicConfig(filename='vidclient.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
failed_frames = 0
model = load_model('cnn_lstm.hdf5')
primaval = np.float32(0.0)
douaval = np.float32(0.0)
treiaval = np.float32(0.0)

primaloss = np.float32(0.0)
doualoss = np.float32(0.0)
treialoss = np.float32(0.0)


def thread1(u):
    v1 = np.array(u[0][0])
    v2 = np.array(u[0][10])
    v3 = np.array(u[0][20])
    v = np.array(u[0][20:23])
    nn = np.array([[v1, v2, v3]])
    with graph.as_default():
        p = model.predict(nn)

    global primaval
    global primaloss
    primaval = p[0][0]
    primaloss = p[0][1]


def thread2(u):
    v1 = np.array(u[0][25])
    v2 = np.array(u[0][30])
    v3 = np.array(u[0][35])
    v = np.array(u[0][50:53])
    nn = np.array([[v1, v2, v3]])
    with graph.as_default():
        p = model.predict(nn)

    global douaval
    global doualoss
    douaval = p[0][0]
    doualoss = p[0][1]


def thread3(u):
    v1 = np.array(u[0][40])
    v2 = np.array(u[0][45])
    v3 = np.array(u[0][49])
    v = np.array(u[0][80:83])
    nn = np.array([[v1, v2, v3]])
    with graph.as_default():
        p = model.predict(nn)

    global treiaval
    global treialoss
    treiaval = p[0][0]
    treialoss = p[0][1]


def main(url=0):
    '''Start video client for raspberry'''
    logging.info("Starting video capture at url %s" % url)
    vcap = cv2.VideoCapture(url)
    i = 0
    q = collections.deque(maxlen=MAX_FRAMES_SENT)
    while (True):
        # Capture frame-by-frame
        ret, frame = vcap.read()
        if frame is not None:
            # Display the resulting frame
            failed_frames = 0
            frame = cv2.resize(frame, (224, 224))
            # frame = cv2.transpose(frame)
            # frame = cv2.flip(frame, 0)
            q.append(frame)
            cv2.imshow('frame', frame)
            i = i + 1
            # prediction
            np_array = np.array([np.array(q)])
            if i % 50 == 0:
                thread.start_new_thread(thread1, (np_array,))
                thread.start_new_thread(thread2, (np_array,))
                thread.start_new_thread(thread3, (np_array,))
                q = collections.deque(maxlen=MAX_FRAMES_SENT)
                medie = (primaval + douaval + treiaval) / 3
                print("medie =")
                print(primaval)
                print(douaval)
                print(treiaval)
                print(primaloss)
                print(doualoss)
                print(treialoss)
                if medie > 0.5:
                    print("FIGHT %s thread2" % medie)
                else:
                    print("NOT FIGHT %s thread2" % medie)

            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
            yield frames
        else:
            logging.warning("failed frame no %s" % failed_frames)
            # handle the loss of connectivity scenario
            failed_frames += 1
            if failed_frames < MAX_FAILED_FRAMES:
                logging.error("Maximum failed frames reached, quitting")
                break
            else:
                logging.warn("Failed frame, maximum not reached, continuing")
            continue

    # When everything done, release the capture
    vcap.release()
    cv2.destroyAllWindows()
    logging.info("Video stop")


if __name__ == '__main__':
    for frame in main(url=0):
        pass