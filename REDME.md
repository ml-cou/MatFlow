# MatFlow
To clone the project

```
git clone  https://github.com/ml-cou/MatFlow.git

```
Install all dependencies of this project

```
pip install -r requirements.txt

```
Berofe working on this project (every time) you should pull the code from MatFlow (default branch)

```
git pull origin MatFlow

```
Set up virtual Environement

```
pipenv shell

```
Install streamlit

```
pip install streamlit

```
Run the Project

```
streamlit run home.py

```
### Deploy streamlit on Render
1. Login on render 
2. Click new project from dashboard
3. Select Web Service
4. Conntect to project repository
5. Fill up the required fields
6. Select Environment  to "Python 3"
7. Provide the build commands
```
pip install -r requirements.txt

```

9. Provide the start  commands
```
streamlit run home.py
```
### Deploy streamlit on (https://share.streamlit.io)
1. Login on (https://share.streamlit.io) 
2. Click new app from dashboard
3. Fill up the required fields
- Select Repository and branch (MatFlow)
- Main file path : home.py
4. Click the Deploy button
<br> 
This project is available here:  (https://matflow.streamlit.app)
