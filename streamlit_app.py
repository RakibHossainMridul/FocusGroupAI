import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    model_name = "meta-llama/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
    return tokenizer, model

tokenizer, model = load_model()

st.title("LLaMA Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("What is your question?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        for output in model.generate(**inputs, max_length=1000, num_return_sequences=1, do_sample=True, temperature=0.7, top_k=50, top_p=0.95):
            full_response = tokenizer.decode(output, skip_special_tokens=True)
            message_placeholder.markdown(full_response + "▌")
        
        message_placeholder.markdown(full_response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": full_response})
