from training_functions.training_utils import *

import_check()

dir_path = "C:/Users/Tanmay Patil/Downloads/Face Recognition/images/"

detector = MTCNN()

face_encodings = []
labels = []
names = []
cropped=  []
for k,i in tqdm(enumerate(sorted(os.listdir(dir_path)))):
    if(i=='.DS_Store'):
        continue
    
    names.append(i)

    for o,j in enumerate(os.listdir(dir_path+i)):
        if(j=='.DS_Store'):
            continue
        image_path = dir_path+i+"/"+j
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        detected_faces = detector.detect_faces(image)
        if(len(detected_faces) == 0):
            cropped = image
            print("No Face Detected")
            continue
        else:
          face = detected_faces[0]
          
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

        
        face_encodings.append(get_encodings(cropped))
        labels.append(k)

face_encodings = np.concatenate(face_encodings)
labels = np.array(labels)



clf = train(face_encodings,labels)


with open('classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    
