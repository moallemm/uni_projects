# University Projects

A collection of projects developed during my B.Sc. in Data Science at the University of Sciences and Arts in Lebanon (USAL). The projects cover data analysis, machine learning, natural language processing, computer vision, and web development.

These projects reflect my learning at different stages of my degree. Some are exploratory coursework rather than production-ready applications; I have documented their scope and limitations accordingly.

## Projects at a glance

| Project | Focus | Main technologies |
| --- | --- | --- |
| [Crime Data Analysis](./data%20science%20project) | Crime patterns, geographic and temporal analysis; supplementary regression exercise | Python, Pandas, NumPy, scikit-learn, Matplotlib, Seaborn |
| [Gesture Rock-Paper-Scissors](./python%20project/python%20project) | Webcam-based hand gesture recognition | Python, OpenCV, cvzone |
| [DIY Website](./webproject) | Full-stack tutorial website with authentication and admin functions | Flask, SQLAlchemy, HTML, CSS, JavaScript |
| NLP / BERT Experiment | Fine-tuning a masked-language model on a literary text | PyTorch, Hugging Face Transformers, scikit-learn |
| [SafeWalk AI](./safewalk_run%2Breports%28FYP%29) | Assistive navigation desktop prototype | YOLOv8, MobileNetV3, MiDaS, Python |

> **Note:** Some links may need to be adjusted to match the exact folder names in this repository, particularly the crime analysis and NLP folders.

---

## 1. Crime Data Analysis — Exploratory Data Science

**Focus:** Data preparation, exploratory data analysis (EDA), and decision-support insights from public safety records.

This coursework project explored an open-government crime dataset from a U.S. city. The goal was to examine reported incidents, identify patterns, and communicate findings that could help inform public safety decisions. The dataset included geographic coordinates (latitude and longitude), enabling location-based analysis.

### Workflow

1. **Data loading and inspection:** Imported the dataset, reviewed its fields, and examined sample records.
2. **Data cleaning:** Investigated missing values, duplicate records, and possible outliers, applying cleaning steps where appropriate.
3. **Data transformation:** Prepared columns for analysis, including numerical scaling and handling fields that were unsuitable for the selected methods.
4. **Exploratory analysis and visualization:** Used descriptive statistics, bar charts, box plots, scatter plots, and other visualizations to examine incident distributions.
5. **Supplementary regression exercise:** Practiced supervised learning with a **separate Kaggle weather dataset**, building a linear regression model to predict humidity. This was an independent modeling exercise, **not a crime prediction model**.

### Analytical questions

The crime-data exploration examined questions such as:

- Which streets or areas had the highest numbers of reported incidents?
- At what hours were reported crimes most frequent?
- What patterns appeared in incident locations and law-enforcement activity?

These are descriptive findings, not causal conclusions. Incident counts do not automatically establish neighborhood risk, and officer-level activity counts should not be treated as performance rankings without additional context.

### Technologies

**Python · Pandas · NumPy · Matplotlib · Seaborn · scikit-learn** (`StandardScaler`, `LinearRegression`)

### Reflection and potential extension

The main value of this project was the **exploratory analysis itself**: a useful analytical project does not require a predictive model. A natural future extension would be **DBSCAN-based geographic clustering** of latitude/longitude coordinates to explore possible incident hotspots. **DBSCAN was not implemented in the original project**; it is a proposed next step. Geographic clustering would require appropriate distance calculations, parameter tuning, and validation.

> **Scope clarification:** The crime analysis and humidity regression use different datasets. The regression exercise was included to practice modeling techniques and should not be interpreted as a model of crime patterns.

---

## 2. Gesture Rock-Paper-Scissors

**Focus:** Introductory real-time computer vision and hand tracking.

A webcam-based Rock-Paper-Scissors game that uses hand tracking to interpret player gestures. The project was a learning exercise in video capture, gesture recognition, and interactive Python programming.

**Tools:** Python, OpenCV (`cv2`), cvzone.

**Learning resource and attribution:** This project was developed with guidance from [Murtaza's Workshop tutorial](https://www.youtube.com/watch?v=k2EahPgl0ho). Credit for the tutorial approach belongs to its creator.

---

## 3. DIY Tutorial Website

**Focus:** Full-stack web development, authentication, relational data, and administration.

A Flask-based website for browsing and searching DIY craft tutorials, reading instructions, and submitting feedback. It includes user-facing pages and an administrative interface.

**Main features**

- User registration, login, and authenticated sessions.
- Introductory and about pages.
- Home page with craft listings and search.
- Search results and individual tutorial pages with instructions and video content.
- Feedback submission.
- Admin tools to add, edit, and remove crafts; manage user access; grant or revoke administrator privileges; and review feedback.

**Tools:** Python, Flask, Flask-SQLAlchemy, Flask-Login, Werkzeug password hashing, HTML, CSS, JavaScript, and a relational database.

**Implementation details:** The application uses Flask templates, database-backed models, login/session management, and role-restricted administrative routes. Consult the project source for its configuration and run instructions.

---

## 4. NLP: BERT Language Modeling Experiment

**Focus:** Tokenization, transformer fine-tuning, and language-model experimentation.

An experimental NLP project that fine-tunes a pretrained BERT model with a masked-language-modeling (MLM) head using a Sherlock Holmes text corpus. The project explores tokenization, training-data preparation, batching, model training, and evaluation.

**Technical clarification:** BERT is designed for **masked-token prediction**, not conventional left-to-right next-word generation. Although the original goal was next-word prediction, the use of `AutoModelForMaskedLM` makes this best described as a BERT MLM experiment. Its suitability for next-word prediction depends on the exact masking and inference procedure in the code.

**Tools:** Python, PyTorch, Hugging Face Transformers (`BertTokenizerFast`, `AutoModelForMaskedLM`), scikit-learn, and CUDA when available.

**Limitations and possible improvements:** The literary corpus was chosen for experimentation rather than as an optimal dataset for predictive text. Future work could use a more representative corpus, evaluate on a held-out test set, and compare the approach with a causal language model designed for next-token prediction.

---

## 5. SafeWalk AI — Final Year Project

**Focus:** Assistive computer vision and spoken navigation guidance.

SafeWalk AI is a desktop prototype designed to help visually impaired users interpret their surroundings. Its pipeline combines indoor/outdoor scene classification, YOLOv8 object detection, MiDaS depth estimation, rule-based guidance, and text-to-speech output.

As part of the outdoor detection work, I collected and manually annotated **224 images containing 2,209 bounding boxes across 15 classes** using Roboflow. This custom dataset supplemented existing outdoor detection data.

**Current status:** Desktop prototype. Mobile deployment is a planned next step, not a completed feature.

See the [SafeWalk AI project folder](./safewalk_run%2Breports%28FYP%29) for its dedicated documentation.

---

## Notes

These are educational projects. Their README descriptions document what was implemented, rather than presenting all experiments as production-ready systems. For individual setup steps, dependencies, datasets, and results, refer to the source code and any project-specific documentation.

## Contact

- [GitHub](https://github.com/moallemm)
- [LinkedIn](https://www.linkedin.com/in/mohamad-moallem-ds/)
- [Portfolio](https://moallemm.github.io/Portfolio/)
