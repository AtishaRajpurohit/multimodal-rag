import streamlit as st
#from src.camera_image_matching import process_single_image

st.set_page_config(page_title="Face Detection & Matching", page_icon="🎥")
st.title("Face Detection & Matching")

img_file = st.camera_input("Take a picture")

if img_file is not None:
    st.write("Image uploaded successfully!")
    if st.button("Process Image"):
        try:
            with open("temp_image.png", "wb") as f:
                f.write(img_file.getbuffer())
            st.success("Image captured and saved!")
        except Exception as e:
            st.error(f"Error saving image: {e}")