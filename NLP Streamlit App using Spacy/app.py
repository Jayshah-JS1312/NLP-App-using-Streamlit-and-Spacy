import streamlit as st
import spacy as sp
import textblob
from textblob import TextBlob, Word, Blobber
from gensim.summarization import summarize
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lex_rank import LexRankSummarizer

def text_analysis(text):
    nlp = sp.load('en_core_web_sm')
    docs = nlp(text)

    tokens = [token.text for token in docs]
    finaldata = [('"Tokens":{},\n"Lemma:{}"'.format(token.text,token.lemma_)) for token in docs]
    return finaldata

@st.cache
def entity_extractor(text):
    nlp = sp.load('en_core_web_sm')
    docs = nlp(text)
    tokens = [token.text for token in docs]
    entities = [(ent.text, ent.label_) for ent in docs.ents]
    finaldata = ['"Tokens":{},\n"Entities":{}'.format(tokens,entities)]
    return finaldata

def sumy_method(text):
    parser = PlaintextParser.from_string(text,Tokenizer("english"))
    lex_summarizer = LexRankSummarizer()
    summary = lex_summarizer(parser.document,3)
    summary_list = [str(sent) for sent in summary]
    finalres = ' '.join(summary_list)
    return finalres

def main():
    st.title("NLP Application using Streamlit")
    st.subheader("Natural Language Processing")

    #Tokenization
    if st.checkbox("Show Tokens and Lemma"):
        st.subheader("Tokenize your text...")
        msg = st.text_area("Enter text here...","Please type here....")
        if st.button("Analyze"):
            res1 = text_analysis(msg)
            st.json(res1)

    #Named Entity
    if st.checkbox("Show Named Entities"):
        st.subheader("Extract entities from your text...")
        msg = st.text_area("Enter text here...","Please type here....")
        if st.button("Extract Now"):
            res1 = entity_extractor(msg)
            st.json(res1)

    #Sentiment Analysis
    if st.checkbox("Show Sentiment Analysis"):
        st.subheader("Sentiment Analysis of your text...")
        msg = st.text_area("Enter text here...","Please type here....")
        if st.button("Analyze"):
            blob = TextBlob(msg)
            res_sent = blob.sentiment
            st.success(res_sent)

    #Text Summarization
    if st.checkbox("Show Text Summarization"):
        st.subheader("Summarize your text...")
        msg = st.text_area("Enter text here...","Please type here....")
        options = st.selectbox("Choose summarizer:",("Gensim","Sumy"))
        if st.button("Summarize"):
            if options == 'Gensim':
                st.text("Using Gensim Summarizer...")
                summary = summarize(msg)
            elif options == 'Sumy':
                st.text("Using Sumy Summarizer...")
                summary = sumy_method(msg)
            else:
                st.warning("Using Default Summarizer of Gensim...")
                st.text("Using Gensim..")
                summary = summarize(msg)
            
            st.success(summary)
    
    st.sidebar.subheader("About the Application")
    st.sidebar.text("It is web application for Natural Language Processing.Here on this website you can tokenize your text, extract named entities as well as you can summarize the text and also use the function of sentiment analysis.")
    st.sidebar.subheader("Thanks for visiting the Website..")

if __name__ == '__main__':
    main()