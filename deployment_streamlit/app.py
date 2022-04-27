import pickle
import streamlit as st
import numpy as np

model = pickle.load(open('rf_regressor.pkl','rb'))


def prediction(Avg_session_length,Time_on_app,Time_on_website,Length_of_membership):
    prediction = model.predict([[np.log(Avg_session_length),np.log(Time_on_app),np.log(Time_on_website),np.log(Length_of_membership)]])
    print(prediction)
    return prediction
    
    
    
def main():
    st.title('E-Com Sale Prediction')
    html_temp = """
    <div style = "background-color:yellow;padding:13px">
    <h1 style ="color:black;text-align:center;"> E-Commerce sales Prediction ML App </h1>
    </div>
    """
    
    
    
    #display frontend aspect
    st.markdown(html_temp,unsafe_allow_html=True)
    
    #creat box to enter value
    
    Avg_session_length=st.number_input('Average Session Length')
    Time_on_app       =st.number_input('Time on appication')
    Time_on_website   =st.number_input('Time on website')
    Length_of_membership = st.number_input('Length of membership')
    result = ""
    
    #after predict is clicked
    
    if st.button('Predict'):
        result = prediction(Avg_session_length,Time_on_app,Time_on_website,Length_of_membership)
        st.success('E-Com sale is approximately {}'.format(result))
        
    
if __name__=='__main__' :
    main()
    