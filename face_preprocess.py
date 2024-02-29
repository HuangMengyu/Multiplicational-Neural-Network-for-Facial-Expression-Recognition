import math
import cv2
import dlib
import numpy as np
import os
import pickle


RESCALE_SIZE = 128

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./dlibmodel/shape_predictor_68_face_landmarks.dat")

def getCoordinates(shape):
    coordinates = np.empty([68, 2])
    for i in range(68):
        coordinates[i, 0] = shape.part(i).x
        coordinates[i, 1] = shape.part(i).y
    return coordinates

def rotateFace(image, left_eye, right_eye):
    # rotation center
    center = ((left_eye[0] + right_eye[0])/2, (left_eye[1] + right_eye[1])/2)
    # rotation angle
    eye_direction = (right_eye[0] - left_eye[0], right_eye[1] - left_eye[1])
    angle = - math.atan2(float(eye_direction[1]),float(eye_direction[0]))

    rotate_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    print(image.shape)
    rotate_image = cv2.warpAffine(image, rotate_matrix, dsize=(image.shape[0] * 2, image.shape[1] * 2))

    return rotate_image

def cropFace(image, coordinates):
    cor_1 = np.asarray(( min(coordinates[17:67, 0]), min(coordinates[17:67, 1]) ) , dtype=int)
    cor_2 = np.asarray( (max(coordinates[17:67,0]), (np.round((coordinates[8, 1] + coordinates[57, 1]) / 2))) , dtype=int)
    print('cor_1', cor_1, 'cor_2', cor_2)
    if cor_1.all() > 0 and cor_2.all() > 0:
        face_image = image[cor_1[1]: cor_2[1], cor_1[0]:cor_2[0]]
        face_image = cv2.resize(face_image, (RESCALE_SIZE, RESCALE_SIZE))
    else:
        print("cropFace: coordinates for cropped image < 0")
        exit(-1)

    return face_image



def calibrateImage(imgpath):
    '''Calibrate the image of the face'''
    imgcv_gray = cv2.imread(imgpath, cv2.IMREAD_GRAYSCALE)

    if imgcv_gray is None:
        print('Unexpected ERROR: The value read from the imagepath is None. No image was loaded')
        exit(-1)
    dets = detector(imgcv_gray,1)
    print("number of faces detected : %d" % len(dets))
    if len(dets) == 0:
        print("No face was detected^^^^^^^^^^^^^^")
        return False, imgcv_gray

    coordinates = np.empty([68, 2])
    for id, det in enumerate(dets):
        if id > 0:
            print("ONLY process the first face>>>>>>>>>")
            break
        shape = predictor(imgcv_gray, det)

        # retrieve the coordinates of the 68 landmarks
        coordinates = getCoordinates(shape)

    #rotate the face image based on eye coordinates
    print(np.mean(coordinates[36:42, 0], axis = 0),np.mean(coordinates[36:42, 1], axis = 0))
    rotate_image = rotateFace(imgcv_gray,
                             (np.mean(coordinates[36:42, 0], axis = 0),np.mean(coordinates[36:42, 1], axis = 0)),
                             (np.mean(coordinates[42:48, 0], axis = 0),np.mean(coordinates[42:48, 1], axis = 0)))

    #crop and resize face
    new_dets = detector(rotate_image, 1)
    if len(new_dets) == 0:
        print("No new face was detected^^^^^^^^^^^^^^")
        return False, rotate_image

    # crop face area and resize
    cropped_image = np.empty([128, 128])
    for id, det in enumerate(new_dets):
        if id > 0:
            print("ONLY process the first face>>>>>>>>>")
            break
        new_shape = predictor(rotate_image, det)
        new_coordinates = getCoordinates(new_shape)
        cropped_image = cropFace(rotate_image, new_coordinates)

    return True, cropped_image

def CKPDatasetPreprocess(dataset_dir, label_dir, output_pkl_dir):
    '''dataset preprocessing and store as pkl file'''
    if not os.path.exists(dataset_dir):
        raise NotADirectoryError(f'{dataset_dir} not found!')
    if not os.path.exists(label_dir):
        raise NotADirectoryError(f'{label_dir} not found!')

    images = []
    labels = []

    dirs_level_one = os.listdir(dataset_dir)
    for i in dirs_level_one:
        subject_dir = os.path.join(dataset_dir, i)
        if os.path.isdir(subject_dir):
            dirs_level_two = os.listdir(subject_dir)
            for j in dirs_level_two:
                sample_dir = os.path.join(subject_dir, j)
                if os.path.isdir(sample_dir):
                    image_seq = os.listdir(sample_dir)
                    image_seq = sorted(image_seq)
                    target_dir = os.path.join(sample_dir, image_seq[-1])
                    _, target_image = calibrateImage(target_dir)
                    # cv2.imshow("Image", target_image)
                    # cv2.waitKey(0)
                    # cv2.destroyAllWindows()
                    #read corresponding label
                    label_file_name = image_seq[-1].split('.')[0] + '_emotion' + '.txt'
                    label_dir_name = os.path.join(label_dir, i, j, label_file_name)
                    if os.path.isfile(label_dir_name):
                        with open(label_dir_name, mode="r", encoding='utf-8') as f:
                            target_label = f.read()
                            target_label = target_label.strip().split('.')[0]
                            target_label = int(target_label)
                            #print(target_label)
                        if target_label != 0 and target_label != 2:
                            images.append(target_image)
                            labels.append(target_label)

    data = {'images': images, 'labels': labels}

    with open(output_pkl_dir, 'wb') as f:
        pickle.dump(data, f)

    return images, labels



dataset_dir = "D:\ck+_dataset\extended-cohn-kanade-images\cohn-kanade-images"
label_dir = "D:\ck+_dataset\Emotion_labels\Emotion"
CKPDatasetPreprocess(dataset_dir, label_dir, 'preprocess_pkl/ckp_6.pkl')



