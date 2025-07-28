import os
import face_recognition
import cv2
import numpy as np

def init():
    face_folder = os.listdir('known_face/')
    known_face_names = []
    known_face_encodings = []

    for index, img_file in enumerate(face_folder):
        file_path = os.path.join('known_face/', img_file)
        image  = face_recognition.load_image_file(file_path)
        known_face_encodings.append(face_recognition.face_encodings(image)[0])
        known_face_names.append(os.path.splitext(img_file)[0]) 
    
    return known_face_encodings, known_face_names

def detect_face(image):
    # Ensure image has 3 channels
    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(f"[ERROR] Expected 3-channel color image but got shape: {image.shape}")
    rgb_image = image[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_image)
    return face_locations

def recognize(image, known_face_encodings, known_face_names):
    face_names = []
    face_locations = detect_face(image)
    face_encodings = face_recognition.face_encodings(image, face_locations)
    
    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Unknown"
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
        face_names.append(name)
    
    return face_names, face_locations

def label_image(image, face_locations, face_names):
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(image, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(image, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
    return image

def main():
    known_face_encodings, known_face_names = init()
    image_file = "arnab_knife.png"
    image_path = os.path.join('assets/',image_file)
    image = cv2.imread(image_path)
    face_names, face_locations = recognize(image, known_face_encodings, known_face_names)
    image = label_image(image, face_locations, face_names)
    output_dir = "processed"
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir,'recognized_'+image_file)
    cv2.imwrite(save_path, image)

if __name__ == "__main__":
    main()
