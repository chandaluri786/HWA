# Loading necessary packages and models
import streamlit as st
@st.experimental_singleton
def initial():
    import cv2
    import numpy as np
    import pandas as pd
    from sklearn import svm
    import plotly.express as px
    from feature_extractor import Image_fe
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from skimage.filters import threshold_otsu
    from tensorflow.keras.models import model_from_json

    with open('model.json', 'r') as f:
        dl_model = model_from_json(f.read())
    dl_model.load_weights('model.h5')

    return cv2, np, pd, svm, px, Image_fe, StandardScaler, train_test_split, threshold_otsu, dl_model
cv2, np, pd, svm, px, Image_fe, StandardScaler, train_test_split, threshold_otsu, dl_model = initial()

def tiler(image):
    thresh = threshold_otsu(image)
    image = (image > thresh)*255
    tiles = [
      cv2.resize(cv2.merge([image[i:i+600, j:j+600],image[i:i+600, j:j+600],image[i:i+600, j:j+600]]).astype(float),(224,224))
        for i in range(0, image.shape[0], 600)
        for j in range(0, image.shape[1], 600)
        if (image[i:i+600, j:j+600]).sum() <= 89964000
    ]
    return np.array(tiles)

# WebApp
st.subheader('Welcome to HWA')
with st.expander('Additional Information 📝'):
    st.text('1.Please make sure the image is either scanned or captured by a steady hand.')
    st.text('2.Take the picture in good lighting setup')
#     st.text('3.Avoid using flash while taking the picture')
st.session_state['features'] = False
st.select_slider(label = 'Photo Options', options = ['Upload', 'Camera'], key = 'upload')
if st.session_state['upload'] == 'Upload':
    img_file_buffer = st.file_uploader('Upload a picture')
else:
    img_file_buffer = st.camera_input('Take a picture')
labels = ['good', 'bad']
if img_file_buffer:
    if st.session_state['upload'] == 'Upload':
        st.image(img_file_buffer)
    img = cv2.imdecode(np.frombuffer(img_file_buffer.getvalue(), np.uint8), cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_features = Image_fe(img_name = gray, local = False)
    if not st.session_state['features']:
        st.session_state['features'] = img_features.word_fe()
    display_options = ['Slant Variance', 'Spacing', 'Height Uniformity', 'Area Uniformity']
    imgs = [img_features.slant_img, img_features.space_img, img_features.height_img, img_features.area_img]
    tabs = st.tabs(display_options)
    for i, tab in enumerate(tabs):
        feature = display_options[i]
        tab.subheader(feature)
        size = tab.slider('Adjust Plot Size ', min_value = 100, max_value = 1500, value = 750, step = 50, key = feature.split(' ')[0] + '_plotsize')
        tab.plotly_chart(px.imshow(imgs[i], color_continuous_scale = 'magma' if i != 1 else 'gray', height = size, width = size))
    model = svm.SVC()
    df = pd.read_csv('df.csv', index_col = 0)
    scaler = StandardScaler()
    scaler.fit(df.iloc[:, :-1].to_numpy())
    X_train, X_test, y_train, y_test = train_test_split(scaler.transform(df.iloc[:, :-1]), df['target'], test_size = 0.3, random_state = 20)
    model.fit(X_train, y_train)
    prediction = model.predict(scaler.transform(st.session_state['features'].to_numpy().reshape(1, -1)))

    # DL prediction 
    dl_prediction = [ np.argmax(i) for i in dl_model.predict(tiler(gray))]
    dl_prediction_doc_level = max(dl_prediction,key=dl_prediction.count)
    # st.write(prediction)
    tab1, tab2 = st.tabs(['Machine Learning', 'Deep Learning'])
    
    if prediction[0]:
        tab1.success('Your Handwriting is quite good. 👌')
    else:
        tab1.error('Your Handwriting is bad. 🙅‍♂️')
    
    if dl_prediction_doc_level:
        tab2.success('Your Handwriting is quite good. 👌')
    else:
        tab2.error('Your Handwriting is bad. 🙅‍♂️')
        
else:
    st.session_state['features'] = False
