# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

from  PIL import Image, ImageEnhance
import streamlit as st
import numpy as np
import pickle

# loading the saved model
loaded_model = pickle.load(open('knn.pkl', 'rb'))

def prdict_ven(input_data):
    input_data = (400277,4500016200,20,5120000000000,5120000000000,1680)

# changing the input_data to numpy array
    input_data_as_numpy_array = np.asarray(input_data)

# reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)
    print(prediction)

    if (prediction[0] == 0):
        print('The person is not diabetic')
    else:
        print('The person is diabetic')

  
def main():
    image = Image.open('nupco.png') #Brand logo image (optional)
    st.image(image, width=150)
    st.title("Prediction Vender ")
    html_temp = """
    <div style="background-color:tomato;padding:6px">
    <h2 style="color:white;text-align:center;"> Prediction Vender ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    Vender_code=st.text_input('Vender code')
    Purchasing_Document=st.text_input('Purchasing Documen')
    PO_Item=st.text_input('PO Item')
    Mat_Code=st.text_input('Generic Mat Code')
    Nupco_Trade_Code=st.text_input('Nupco Trade Code')
    Ordered_Quantity=st.text_input('Ordered Quantity')
    
    diagnocses=''
    
    
    if st.button('Predict'):
        diagnocses=prdict_ven([Vender_code,Purchasing_Document,PO_Item,Mat_Code,Nupco_Trade_Code,Ordered_Quantity])
        st.success(diagnocses)
    
    
if __name__=='__main__':
    main()