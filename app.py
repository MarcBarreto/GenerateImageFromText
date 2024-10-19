import torch
import diffusers
import streamlit as st
from PIL import Image
from io import BytesIO
from diffusers import StableDiffusionPipeline

# Title of the App
st.title('Generate Image from Text')

@st.cache_resource
def load_model():
    pipeline = StableDiffusionPipeline.from_pretrained('CompVis/stable-diffusion-v1-4', torch_dtype = torch.float16)

    if torch.cuda.is_available():
        pipeline = pipeline.to('cuda')
    
    return pipeline

pipeline = load_model()

#input
prompt = st.text_input('Type your prompt to generate image', value="Type prompt")
steps = st.slider('Select the number of Inference Steps', min_value=20, max_value = 150, value = 50)

if st.button('Generate Image'):
    with st.spinner('Generating Image...'):
        try:
            generator = torch.Generator(device = 'cuda' if torch.cuda.is_available() else 'cpu')

            image = pipeline(prompt, num_inference_steps = steps, generator = generator).images[0]

            # Show Image
            st.image(image, caption='Generated Image')

            #Convert Image
            img_buffer = BytesIO()
            image.save(img_buffer, format='PNG')
            img_buffer.seek(0)

            st.download_button(label='Download',
                               data=img_buffer,
                               file_name='generated_image.png',
                               mime='image/png')
        
        except Exception as e:
            st.error(f'Error: {e}')


