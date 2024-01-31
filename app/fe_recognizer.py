import streamlit as st
import functions
st.set_option('deprecation.showPyplotGlobalUse', False)

def main():
    st.title('Why the long face?\u00A9')

    # Layout with two columns
    col1, col2 = st.columns([3, 1])

    with col1:
        st.markdown('### Try our state of the art facial expression recognizer!')
        st.markdown('- Click on "Browse Files" or drag and drop an image\n- Wait a few seconds while the model processes de image\n- Enjoy your facial expression predictions!')

    with col2:
        st.image('juan-donkey.png', width=150)

    # Back to single column layout
    st.markdown("---")

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with st.spinner('Recognizing facial expression...'):
            plot = functions.process_image_and_predict(uploaded_file.name, 'ResNet50_model.h5')
            st.pyplot(plot)

        st.markdown("---")

        st.markdown('## A special thank you to my collaborators')

        # Local GIF below the Markdown content
        st.image("ih.gif", use_column_width=True)

        st.markdown("---")

        # Layout with two columns
        col1, col2 = st.columns([3, 1])

        with col1:
            st.markdown('#### Disclaimer:')
            st.markdown("No picture has been used without explicit consent.")

        # with col2:
        #     st.image('juan-donkey.png', width=150)

if __name__ == "__main__":
    main()