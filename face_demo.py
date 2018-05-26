from training_functions.training_utils import *
import_check()

detector = MTCNN()

dir_path = "C:/Users/Tanmay Patil/Downloads/Face Recognition/images/"

with open('classifier.pkl', 'rb') as fid:
    clf = cPickle.load(fid)

def take_picture():
    camera_port = 0
    ramp_frames = 15
    camera = cv2.VideoCapture(camera_port)
    for i in range(ramp_frames):
        is_captured,image = camera.read()
    is_captured,image = camera.read()
    del(camera)
    return image

names = []
for name in tqdm((sorted(os.listdir(dir_path)))):
    names.append(name)
print(names)
i = 0
while i < 5:
    i += 1
    predictions = []
    image = take_picture()
    detected_faces = detector.detect_faces(image)
    if(len(detected_faces) == 0):
        cropped = image
        continue
    else:
        for face in detected_faces:
            for z in range(2):
                if(face['box'][z]<0):
                    face['box'][z] = 0
            [x,y,w,h] = face['box']  
            left = x
            right = x+w
            top = y
            bottom = y+h
            cropped = image[y:y+h,x:x+w]
            cropped = cv2.resize(cropped,(160,160))
            cropped = cv2.cvtColor(cropped,cv2.COLOR_BGR2RGB)
            predictions.append(names[clf.predict(get_encodings(cropped))[0]])
            plt.title(sorted(predictions))
            plt.imshow(cropped)
            plt.show()

