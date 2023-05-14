import streamlit as st
import cv2
from Recognize import *

st.set_page_config(page_title=" Image Processing", page_icon="📸", layout="wide",initial_sidebar_state="collapsed")

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)

with open("style.css") as source_des:
    st.markdown(f"""<style>{source_des.read()}</style>""", unsafe_allow_html=True)

#...................................................... Face Detection ........................................................................
FacesImages = []
def face_detect(image,scaleFactor,minNeighbors,k):
    face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    faces = face_classifier.detectMultiScale(
        image, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=(60, 60))
    
    for i, (x, y, w, h)  in enumerate(faces):
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 4)

        face = image[y:y+h, x:x+w]
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = cv2.resize(face,(128,128))
        cv2.imwrite("Faces\Face"+str(i)+".jpg",face)

        filename = r'C:\Users\power\Desktop\cv-task5\Faces\Face' +str(i) + '.jpg'
        detected_face=Image.open(filename).convert('L')
        detected_face= np.asarray(detected_face,dtype=float)/255.0
        FacesImages.append(detected_face)

        Mean, Zero_mean_matrix, u_list = eigen()
        t = FacesImages[i]
        test = t.flatten()
        zero_mean_test = test - np.transpose(Mean)
        name = Project(k,zero_mean_test,80)  #threshold =80
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, name, (x-70,y-8), font, 2, (255,0,0), 2)


    return image
#.................................................................................................................................................

side = st.sidebar
uploaded_img =side.file_uploader("Upload Image",type={"png", "jpg", "jfif" , "jpeg"})
scaleFactor = side.number_input('Scale Factor',min_value=1.00,max_value=50.0, value=1.1,step=0.1)
minNeighbors = side.number_input('Minimum Neighbors',min_value=1,max_value=50, value=8,step=1)

col1,col2 = st.columns(2)
col1.title("Input Image")
col2.title("Output Image")

if uploaded_img is not None:
    file_path = 'Images/'  +str(uploaded_img.name)
    input_img = cv2.imread(file_path)
    input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
    col1.image(input_img)
    detected_image = face_detect(input_img, scaleFactor, minNeighbors,103)
    col2.image(detected_image)




