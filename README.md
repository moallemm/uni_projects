this read me contains a description for each project:

1)data science project:

the project is about the crime data from an open source goverment data for some city in the U.S that i can not remember its been a couple months since i finished this.

anyways the project is 5 tasks :1) loading the data set(just loading the data set a presenting the first 5 rows,and giving a breef description of what the data is about)
                                2) cleaning the data set(this included filling missing values,removing duplicates,check for outliers that may skew our data affecting the results)
                                3) Transofroming the data set(standarizing columns that contain large numbers, dropping columns with unstructred data that the model may find a hard time reading,and much more the project is full of comments...)
                                4) Data visulisaton and summarization(we used different types of plots to visualize the data e.g:barplot,boxplots,scatterplots) 
                                5) Modeling (we used the data above to create a model i was an amatur i did not do it right i brought another data set from kaggle and made a linear regrission model that predicts the humidty in some area given a set of a data since i did not know what to do with the data set that i worked on above [crime_data])

libraries used:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import seaborn as sns 

2)python project:

this is a small project about gesture rock-paper-scissors that uses libraries such as cv2 and cvzone for handtracking all credits to (Murtazas workshop his youtube channel) i learned alot and was a huge help in this project (the link for his vidio: https://www.youtube.com/watch?v=k2EahPgl0ho)
thats all for this project

3)web project:

the project is a DIY Website using HTML/CSS/JavaScript/Restful APIs with Python and Flask and database integration
the project consists of 9 templates:1)login (for logging in)
                                    2)resgister (if already have account you can immediately)
                                    3)intro (the page you first see when entering to the website takes you to login,register and about)                             
                                    4)about (a page with info on the website owners or devs)                                  
                                    5)Homepage(the page you are redirected into after register or login you can view a number of the tutorials we offer and search for somthing specific)           
                                    6)searcg(after searching in the homepage you redirected into here were you can view you search or serach for some thing else)
                                    7)infopage(once you fined what your searching for you press info that takes you to this page where you can read the tutorial and fined a vidio)   
                                    8)Admin(only admin can login into this page where he can do alot of operations such as adding,removing,editing crafts banning,unbanning users, or even giving or revoking admin previliges for new admins also you can view users feedback that the users submit from the feedback page)        
                                    9)feedback(where the users can submit feedbacks about there experince or if they want more crafts)

libraries used:
from flask import Flask, render_template, request, redirect, url_for, flash, session, abort
from flask_sqlalchemy import SQLAlchemy
import urllib.parse
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
and some more you can find in the project it is well documented with comments 



4)project(NLP):
This project trains a next-word prediction model using BERT (Bidirectional Encoder Representations from Transformers). The model takes a sequence of words and predicts the most likely next word based on the context. The dataset is a plain text file(about a story for sherlok holmes the dataset is chosen poorly a dataset for search engines searchs would be much better but i was to lazy to search + i had no time to search slowely for a proper dataset anyways), tokenized using BERT’s tokenizer, and processed into training sequences. A pre-trained BERT model with a Masked Language Model (MLM) head is fine-tuned for this specific next-word prediction task.

Libraries Used & Their Purpose
NumPy (numpy) → Used for numerical operations (though not explicitly needed here).
Pandas (pandas) → Typically used for handling structured data (not essential here but often useful in NLP tasks).
PyTorch (torch, torch.nn) → Deep learning framework used for defining, training, and evaluating the model.
Scikit-Learn (sklearn.model_selection, sklearn.metrics) → Used for splitting data into training and validation sets (train_test_split) and evaluating performance (classification_report).
Transformers (transformers) → Hugging Face’s library for loading pre-trained Transformer models like BERT.
AutoModelForMaskedLM → Pre-trained BERT with a masked language modeling head, used for next-word prediction.
BertTokenizerFast → Tokenizer for converting text into BERT-compatible token IDs.
CUDA (torch.device("cuda")) → Enables GPU acceleration if available, speeding up model training.
PyTorch Dataloader (torch.utils.data) → Helps in efficiently loading training data in batches for better performance.
This project fine-tunes BERT for next-word prediction by leveraging deep learning and NLP techniques using PyTorch and Hugging Face Transformers.
