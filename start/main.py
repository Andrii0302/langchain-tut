import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()


def main():
    information = """То́мас А́лва Е́дісон (англ. Thomas Alva Edison; 11 лютого 1847 Майлен, Огайо, США — 18 жовтня 1931, Вест-Орандж, Нью-Джерсі, США)  — американський науковець і винахідник, автор винаходів, на які видано 1093 патенти США та 1239 патентів інших країн[10]. Більшість своїх винаходів зробив у лабораторії в Менло-Парк у штаті Нью-Джерсі, протягом 1876—1887 років, включаючи створення лампи розжарення у 1879 році. Створив систему розподілу електроенергії споживачам, мікрофон для телефону, диктофон і фонограф. Він запропонував використовувати на початку телефонної розмови слово «алло». """
    summary_template = """Given the given information {information} about a person I want you to create:
    1. A short summary
    2. two interesting facts about them"""
    summary_prompt_template = PromptTemplate(
        input_variables=["information"], template=summary_template
    )
    llm = ChatGoogleGenerativeAI(temperature=0, model="gemini-2.5-flash")
    chain = summary_prompt_template | llm
    response = chain.invoke(input={"information": information})
    print(response.content)


if __name__ == "__main__":
    main()
