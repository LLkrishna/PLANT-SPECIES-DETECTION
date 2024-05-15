import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input as mobilenet_v2_preprocess_input,decode_predictions

model = tf.keras.models.load_model('/content/drive/My Drive/Colab Notebooks/plant_1/Model_weights.h5')
model_2 = MobileNetV2(weights="imagenet")
preprocess_input = mobilenet_v2_preprocess_input
### load file
uploaded_file = st.file_uploader("Choose a image file", type="jpg")

map_dict = {0: 'Alstonia Scholaris',
            1: 'Arjun',
            2: 'Basil',
            3: 'Chinar',
            4: 'Gauva',
            5: 'Jamun',
            6: 'Jatropha',
            7: 'Lemon',
            8: 'Mango',
            9: 'Pomegranate',
            10:'PongamiaPinnata'
            }
map_b = {   0: 'Alstonia scholaris, commonly called blackboard tree, scholar tree, milkwood or devils tree. This is a toxic plant. '
        'Cancer:- Alstonia scholaris has been shown to have anti-cancer properties in laboratory studies. However, more research is needed to confirm its effectiveness in humans. ',
        
            1: 'arjun grows to about 20–25 metres tall; usually has a buttressed trunk, and forms a wide canopy at the crown, from which branches drop downwards.  '
            'SILK PRODUCTION :- The arjuna is one of the species whose leaves are fed on by the Antheraea paphia moth which produces the tassar silk, a wild silk of commercial importance',
           
            2: 'Basil is an annual, or sometimes perennial, herb used for its leaves. Depending on the variety, plants can reach heights of between 30 and 150 cm (1 and 5 ft).   '
            'MEDICINAL :- Basil has been used in traditional medicine for centuries to treat a variety of conditions, including indigestion, nausea, vomiting, diarrhea, headaches, and respiratory problems. Basil is also known to have anti-inflammatory and antibacterial properties.',
            
            3: 'Platanus orientalis, the Old World sycamore or Oriental plane,[2] is a large, deciduous tree of the Platanaceae family   '
            'MEDICINAL :- Platanus orientalis has been used in traditional medicine for centuries to treat a variety of conditions, including diarrhea, dysentery, coughs, colds, and skin diseases.',
            
            4: 'Guava is a common tropical fruit cultivated in many tropical and subtropical regions.    '
            'NUTRITION :- Guavas are rich in dietary fiber and vitamin C, with moderate levels of folic acid (nutrition table).',
            
            5: 'Syzygium cumini, commonly known as Malabar plum, Java plum, black plum, jamun.   '
            'It is an evergreen tropical tree in the flowering plant family Myrtaceae, and favored for its fruit, timber, and ornamental value  ',
            
            6: 'Jatropha curcas is a species of flowering plant in the spurge family, Euphorbiaceae, that is native to the American tropics, most likely Mexico and Central America.    '
            'Jatropha is commonly used as Aviation fuels may be more widely replaced by biofuels such as jatropha oil than fuels for other forms of transportation',
            
            7: 'The lemon is a species of small evergreen tree in the flowering plant family Rutaceae, native to Asia, primarily Northeast India, Northern Myanmar, or China      '
            'USE CASES :- They are a rich source of vitamin C and consuming them may reduce the risk of heart disease and cancer.',
            
            8: 'A mango is an edible stone fruit produced by the tropical tree Mangifera indica. It is believed to have originated in southern Asia, particularly in eastern India, Bangladesh, and the Andaman Islands'      
            'MEDICINAL :- Mango leaves have hypotensive properties that mean it helps reduce blood pressure.',
            
            
            9: 'The pomegranate is a fruit-bearing deciduous shrub in the family Lythraceae, subfamily Punicoideae, that grows between 5 and 10 m tall     '
            ' USE CASES :- They are rich in antioxidants,are a good source of fiber,provide vitamin C',
            
            10:'Pongamia pinnata is a species of tree in the pea family, Fabaceae, native to eastern and tropical Asia, Australia, and the Pacific islands.      '
            'MEDICINAL USE :- Pongamia pinnata has been applied as crude drug for the treatment of tumors, piles, skin diseases, and ulcers'
            }

map_c = {
            0:  'https://en.wikipedia.org/wiki/Alstonia_scholaris',
            1: 'https://en.wikipedia.org/wiki/Terminalia_arjuna',
            2: 'https://en.wikipedia.org/wiki/Basil',
            3: 'https://en.wikipedia.org/wiki/Platanus_orientalis',
            4: 'https://en.wikipedia.org/wiki/Guava',
            5: 'https://en.wikipedia.org/wiki/Syzygium_cumini',
            6: 'https://en.wikipedia.org/wiki/Jatropha',
            7: 'https://en.wikipedia.org/wiki/Lemon',
            8: 'https://en.wikipedia.org/wiki/Mango_(retailer)',
            9: 'https://en.wikipedia.org/wiki/Pomegranate',
            10:'https://en.wikipedia.org/wiki/Pongamia'
}

if uploaded_file is not None:
    st.success("Photo uploaded successfully")
    # Convert the file to an opencv image.
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(opencv_image,(224,224))
    # Now do something with the image! For example, let's display it:
    st.image(opencv_image, channels="RGB")

    resized = mobilenet_v2_preprocess_input(resized)
    img_reshape = resized[np.newaxis,...]
    
    Genrate_pred = st.button("Generate Prediction")    
    if Genrate_pred:
        predictions = model.predict(img_reshape)
        prediction = predictions.argmax()
        confidence = predictions[0][prediction]

        st.title(" ")
        st.title(" Our model prediction")
        st.write("Predicted Label for the image is {} with confidence {:.2%}".format(map_dict[prediction], confidence),font = "small")

        
        try:
            predictions_2 = model_2.predict(img_reshape)
            top_predictions = decode_predictions(predictions_2, top=1)[0]
            label, confidence = top_predictions[0][1], top_predictions[0][2]

            st.title(" Another pre-trained prediction")
            st.write("Predicted Label for the image is {} with confidence {:.2%}".format(label, confidence),font = "small")
       

        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

        
        
        with st.expander("Click to read More"):
            st.title(" ")
            st.write("DESCRIPTION:", map_b[prediction], font="small")
            st.title(" ")
            st.write("FOR MORE INFO:", map_c[prediction], font="small")
    
    
        
   