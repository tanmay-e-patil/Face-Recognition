# For Unstructured data

from training_functions.training_utils import *
import_check()


def show_image(image,title="None"):
    plt.title(title)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(image)
    plt.show()

o = 1
dir_path = "C:/Users/Tanmay Patil/Downloads/Face Recognition/Unstructured Data/"
dir_path_structured = "C:/Users/Tanmay Patil/Downloads/Face Recognition/Structured Data/"

if not os.path.exists(dir_path_structured):
    os.makedirs(dir_path_structured)

detector = MTCNN()

face_encodings = []
labels = []
names = []

def manual_train_image(image):
    global o
    clf = None
    detected_faces = detector.detect_faces(image)
    if(len(detected_faces) == 0):
        cropped = image
        print("No Face Detected")
        return False 
    else:
        for face in detected_faces:
            for z in range(2):
                if(face['box'][z]<0):
                    face['box'][z] = 0
            [x,y,w,h] = face['box']

            if  len(os.listdir(dir_path_structured)) == 2:
                clf = train(np.concatenate(face_encodings),np.array(labels),classifier="SVC")

            left = x
            right = x+w
            top = y
            bottom = y+h
            cropped = image[y:y+h,x:x+w]
            cropped = cv2.resize(cropped,(160,160))
            encoding = get_encodings(cropped)

            if len(os.listdir(dir_path_structured)) < 2:
                show_image(cropped)
                name = input("Enter the name of the person ")
                names.append(name)
                face_encodings.append(encoding)
                labels.append(get_name_index(names,name))
                if not os.path.exists(dir_path_structured + name):
                            os.makedirs(dir_path_structured + name)
                cv2.imwrite(dir_path_structured + name + "/" + str(o) + ".jpg",cv2.cvtColor(cropped,cv2.COLOR_RGB2BGR))
                o += 1
                continue 
            
            else:
                prediction = clf.predict(encoding)[0]
                show_image(cropped,names[prediction])
                same_person = input("Does the name match. Enter Y/N/DC ") # DC -> Dont Care
                if(same_person == "Y" or same_person == "y"):
                    cv2.imwrite(dir_path_structured + names[prediction] + "/" + str(o) + ".jpg",cv2.cvtColor(cropped,cv2.COLOR_RGB2BGR))
                    face_encodings.append(encoding)
                    labels.append(get_name_index(names,names[prediction]))


                elif (same_person == "N" or same_person == "n"):
                    print(names)
                    name = input("Enter the name of the person ")
                    if (name not in names):     
                        if not os.path.exists(dir_path_structured + name):
                            os.makedirs(dir_path_structured + name)
                        names.append(name)
                    face_encodings.append(encoding)
                    labels.append(get_name_index(names,name))
                    cv2.imwrite(dir_path_structured + name +"/" + str(o) + ".jpg",cv2.cvtColor(cropped,cv2.COLOR_RGB2BGR))
                    clf = train(np.concatenate(face_encodings),np.array(labels),classifier="SVC")
                
                else:
                    print("Image Skipped")
                o += 1
            
for k,i in tqdm(enumerate(sorted(os.listdir(dir_path)))):
    if(i=='.DS_Store'):
        continue
    
    elif str(i).endswith(".JPG") or str(i).endswith(".jpg"):
        image_path = dir_path + i
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        manual_train_image(image)
    
    elif str(i).endswith("mp4"):
        vidcap = cv2.VideoCapture(dir_path + i)
        success,image = vidcap.read()
        count = 0
        success = True
        while success:
            success,image = vidcap.read()
            if(success):
                manual_train_image(image)
            else:
                print("Failed to read "+ str(count+1) + " frame")
            count += 1
    
    else:
        print("Unsupported File Type")

    
    

clf = train(np.concatenate(face_encodings),np.array(labels),classifier="SVC")

with open('manual_classifier.pkl', 'wb') as fid:
    cPickle.dump(clf, fid)    

