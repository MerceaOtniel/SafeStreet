import cv2
import numpy as np
from keras.models import load_model
import logging
import collections
import copy
import thread
import tensorflow as tf
from threading import Thread
global graph,model
graph = tf.get_default_graph()

MAX_FAILED_FRAMES = 200
MAX_FRAMES_SENT = 50
logging.basicConfig(filename='vidclient.log', level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler())
failed_frames = 0
model = load_model('cnn_lstm.hdf5')
primaval=np.float32(0.0)
douaval=np.float32(0.0)
treiaval=np.float32(0.0)


def thread1(u,firstframe,secondframe,thirdframe):
	v1=np.array(u[0][firstframe])
	v2=np.array(u[0][secondframe])
	v3=np.array(u[0][thirdframe])
	v=np.array(u[0][20:23])
	nn=np.array([[v1,v2,v3]])
	with graph.as_default():
		p = model.predict(nn)
		
	primaval=p[0][0]
		


def main(url=0):
    '''Start video client for raspberry'''
    logging.info("Starting video capture at url %s" % url)
    vcap = cv2.VideoCapture(url)
    i=0
    q = collections.deque(maxlen=MAX_FRAMES_SENT)
    while(True):
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
            i=i+1
            # prediction
            np_array = np.array([np.array(q)])
	    if i % 50 == 0:
		threads = []
		t=Thread( target=thread1,args=(np_array,5,10,15 ) )
		t.start()
		threads.append(t)
		t1=Thread( target=thread1,args=(np_array,25,30,35 ) )
		threads.append(t1)
		t1.start()
		t2=Thread( target=thread1,args=(np_array,40,45,49 ) )
		threads.append(t2)
		t2.start()
		q = collections.deque(maxlen=MAX_FRAMES_SENT)
		for t in threads:
			t.join()
		medie=(primaval+douaval+treiaval)/3
		print("medie =")
		print(primaval)
		print(douaval)
		print(treiaval)
		if medie>0.5:
			print("FIGHT %s thread2" % medie)
		else:
			print("NOT FIGHT %s thread2" % medie)
		
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
            yield frame
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
