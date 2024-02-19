## Integrating code with openai Api
import os
from constants import openai_key
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate 
import streamlit as st
from langchain.chains import LLMChain
from langchain.chains import SequentialChain

os.environ["OPENAI_API_KEY"]=openai_key

# Streamlit framework

st.title("Langchain demo with openai api")
input_text = st.text_input("Search the topic you want")

#Prompt Template
llm=OpenAI(temperature = 0.8)
first_input_prompt = PromptTemplate(
    input_variables = ['name'],
    template = "Tell me about {name}"
)
chain = LLMChain(llm=llm,prompt=first_input_prompt,verbose=True,output_key="person")

second_input_prompt = PromptTemplate(
    input_variables = ['person'],
    template = "When was {person} born"
)
chain2 = LLMChain(llm=llm,prompt=second_input_prompt,verbose=True,output_key="dob")

third_input_prompt = PromptTemplate(
    input_variables = ['dob'],
    template = "Mention 5 major events happened around {dob} in the world"
)
chain3 = LLMChain(llm=llm,prompt=third_input_prompt,verbose=True,output_key="description")

parent_chain = SequentialChain(chains=[chain,chain2,chain3],
                               input_variables=['name'],output_variables=["person","dob","description"],verbose=True)
if input_text:
    st.write(parent_chain({'name':input_text}))