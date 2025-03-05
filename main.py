import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import streamlit as st

data = [
    (101, "Dr. Alice", "Computer Vision, AI, Image Processing", "MIT", 523),
    (102, "Dr. Bob", "Quantum Computing, Cryptography", "Stanford", 412),
    (103, "Dr. Carol", "Deep Learning, NLP, Reinforcement Learning", "Harvard", 390),
    (104, "Dr. Dave", "AI, Computer Vision, Medical Imaging", "MIT", 315),
    (105, "Dr. Eve", "Machine Learning, Data Science", "Oxford", 295),
    (106, "Dr. Frank", "Robotics, Autonomous Systems, Control Systems", "Cambridge", 487),
    (107, "Dr. Grace", "Natural Language Processing, Speech Recognition, Linguistics", "UC Berkeley", 368),
    (108, "Dr. Hank", "Bioinformatics, Computational Biology, Genomics", "Yale", 276),
    (109, "Dr. Ivy", "Quantum Physics, Optics, Photonics", "Princeton", 444),
    (110, "Dr. Jack", "Neural Networks, Computer Vision, Pattern Recognition", "Caltech", 389),
    (111, "Dr. Karen", "Signal Processing, Wireless Communications, Image Compression", "Columbia", 321),
    (112, "Dr. Leo", "Human-Computer Interaction, UX Design, User Modeling", "University of Washington", 498),
    (113, "Dr. Mia", "Computer Graphics, Virtual Reality, Augmented Reality", "Georgia Tech", 452),
    (114, "Dr. Nate", "Embedded Systems, IoT, Cyber-Physical Systems", "ETH Zurich", 310),
    (115, "Dr. Olivia", "Medical AI, Health Informatics, Bioinformatics", "Johns Hopkins", 399),
    (116, "Dr. Paul", "Cybersecurity, Network Security, Cryptanalysis", "Carnegie Mellon", 275),
    (117, "Dr. Quinn", "Theoretical Physics, Quantum Mechanics, String Theory", "University of Chicago", 362),
    (118, "Dr. Rachel", "Computer Vision, Remote Sensing, Geospatial Analysis", "UCLA", 334),
    (119, "Dr. Sam", "Reinforcement Learning, Game Theory, Multi-Agent Systems", "University of Toronto", 421),
    (120, "Dr. Tina", "Climate Modeling, Environmental Science, Atmospheric Physics", "Imperial College London", 289),
    (121, "Dr. Uma", "Artificial Intelligence, Explainable AI, Fairness in AI", "University of Edinburgh", 376),
    (122, "Dr. Victor", "Distributed Systems, Cloud Computing, Edge Computing", "University of Melbourne", 412),
    (123, "Dr. Wendy", "Speech Processing, Natural Language Understanding, Conversational AI", "Tsinghua University", 328),
    (124, "Dr. Xavier", "Astrophysics, Cosmology, Space Science", "University of Tokyo", 401),
    (125, "Dr. Yolanda", "Quantum Chemistry, Molecular Modeling, Computational Chemistry", "University of Zurich", 354),
    (126, "Dr. Zach", "Data Mining, Big Data Analytics, Knowledge Discovery", "University of Sydney", 433),
    (127, "Dr. Anwar", "Artificial Intelligence, Machine Learning, Robotics", "Bangladesh University of Engineering and Technology (BUET)", 390),
    (128, "Dr. Rina", "Computer Vision, Image Processing, AI in Healthcare", "Dhaka University", 325),
    (129, "Dr. Sajjad", "Data Science, Big Data Analytics, Cloud Computing", "BRAC University", 411),
    (130, "Dr. Meem", "Medical Imaging, Deep Learning, Computational Biology", "North South University", 355),
    (131, "Dr. Karim", "Neural Networks, AI Ethics, Natural Language Processing", "Independent University, Bangladesh", 372),
    (132, "Dr. Shama", "Cybersecurity, IoT, Smart Cities", "Jahangirnagar University", 290),
    (133, "Dr. Badrun", "Quantum Computing, Quantum Algorithms", "Rajshahi University of Engineering & Technology (RUET)", 410),
    (134, "Dr. Amina", "Robotics, Artificial Intelligence, Cognitive Computing", "University of Dhaka", 402),
    (135, "Dr. Faiz", "Speech Recognition, AI, Signal Processing", "BRAC University", 318),
    (136, "Dr. Tahmina", "Machine Learning, Data Mining, Social Media Analytics", "Bangladesh University of Professionals", 450),
    (137, "Dr. Junaid", "Geospatial Information Systems, Remote Sensing, Big Data", "University of Chittagong", 367),
    (138, "Dr. Rifat", "Medical Informatics, Healthcare AI, Data Science", "Jahangirnagar University", 385),
    (139, "Dr. Laila", "Natural Language Processing, Text Analytics, Sentiment Analysis", "University of Dhaka", 433),
    (140, "Dr. Asif", "Computer Vision, Deep Learning, Augmented Reality", "North South University", 379),
    (141, "Dr. Shamim", "Cloud Computing, AI, Cyber-Physical Systems", "BRAC University", 412),
    (142, "Dr. Tanjin", "Bioinformatics, Computational Biology, Genetic Algorithms", "Rajshahi University", 355),
    (143, "Dr. Arif", "AI in Education, Learning Analytics, Smart Classrooms", "University of Dhaka", 400),
    (144, "Dr. Dina", "Artificial Intelligence, Human-Robot Interaction, Automation", "University of Chittagong", 380),
    (145, "Dr. Mizan", "Cloud Security, Network Security, AI in Cybersecurity", "Bangladesh University of Engineering and Technology (BUET)", 409),
    (146, "Dr. Tabassum", "Natural Language Understanding, Speech Processing, AI", "Jahangirnagar University", 390),
    (147, "Dr. Tanvir", "Mathematical Modeling, AI in Health, Predictive Analytics", "Independent University, Bangladesh", 400),
    (148, "Dr. Motiur", "Image Processing, Video Analytics, Autonomous Systems", "BRAC University", 348),
    (149, "Dr. Nazma", "AI for Healthcare, Computer Vision, Diagnostic Systems", "North South University", 394),
    (150, "Dr. Nayeem", "AI for Business, Data Science, Decision Support Systems", "University of Dhaka", 378),
    (151, "Dr. Shafique", "Data Privacy, Cybersecurity, Cryptography", "Rajshahi University of Engineering & Technology (RUET)", 370),
    (152, "Dr. Tanima", "AI, Natural Language Processing, Text Mining", "Bangladesh University of Professionals", 385),
    (153, "Dr. Shahed", "Robotics, Human-Robot Interaction, Cognitive Robotics", "University of Chittagong", 360),
    (154, "Dr. Sani", "Healthcare AI, Computer Vision, Clinical Data", "Jahangirnagar University", 430),
    (155, "Dr. Samia", "AI, Cyber-Physical Systems, Smart Healthcare", "University of Dhaka", 399),
    (156, "Dr. Farah", "Embedded Systems, Internet of Things, Machine Learning", "BRAC University", 415),
    (157, "Dr. Shibbir", "Big Data, AI for Climate Change, Environmental Science", "University of Chittagong", 367),
    (158, "Dr. Arifur", "AI in Manufacturing, Industry 4.0, Automation", "North South University", 410),
    (159, "Dr. Saad", "Wireless Communication, Signal Processing, Machine Learning", "Bangladesh University of Engineering and Technology (BUET)", 375),
    (160, "Dr. Munir", "Artificial Intelligence, Data Science, Robotics", "Jahangirnagar University", 387)
]

df = pd.DataFrame(data, columns=["ID", "Name", "Interests", "University", "Citations"])

df["Profile"] = df["Name"] + " " + df["Interests"] + " " + df["University"]

model = SentenceTransformer('all-MiniLM-L6-v2')

df["Embedding"] = list(model.encode(df["Profile"]))

user_name = "Fahim"
user_interest = "AI and Data Science, Machine Learning, Robotics", "Bio informatics"
user_university = "MIT"

def get_recommendations(user_name, user_interest, user_university):
    user_profile = f"{user_name} {user_interest} {user_university}"
    user_embedding = model.encode([user_profile])
    user_embedding = user_embedding.reshape(1, -1)
    df_embeddings = list(df["Embedding"])
    similarities= cosine_similarity(user_embedding, df_embeddings)
    df["Similarity"] = similarities.flatten()
    recommended = df.sort_values(by="Similarity", ascending=False)
    top_3_recommendations = recommended[["ID", "Name", "Interests", "University", "Citations"]].head(3)
    return top_3_recommendations


st.title("Researcher Recommendation System")

user_name = st.text_input("Enter your name")
user_interest = st.text_area("Enter your research interests")
user_university = st.text_input("Enter your university")

st.markdown("""
    <style>
        .dataframe {
            width: 100% !important;
            max-width: 100% !important;
            overflow-x: auto;
        }
        .stTable th, .stTable td {
            text-align: center;
            padding: 10px 20px;
        }
    </style>
""", unsafe_allow_html=True)

if st.button("Get Recommedation"):
    if user_name and user_interest and user_university:
        recommendations = get_recommendations(user_name, user_interest, user_university)
        st.table(recommendations)
    else:
        st.warning("Please fill in all the fields")


