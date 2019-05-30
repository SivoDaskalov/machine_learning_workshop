# machine_learning_workshop

Create a GitHub account
Open GitHub Desktop and log in
Create local repository
Shift-right click while in the repository root and open a PowerShell window
Create a virtual environment with: virtualenv venv
Open created repository as a PyCharm project
Set the path to the git executable: C:\Users\student\AppData\Local\GitHubDesktop\app-1.6.6\resources\app\git\cmd\git.exe
Set the path to the Python interpreter: C:\Users\student\Documents\GitHub\machine_learning_workshop\venv\Scripts\python.exe
Install the following libraries: numpy, pandas, scikit-learn, opencv-python, matplotlib
Create the freeze batch script: .\venv\Scripts\activate && pip freeze > requirements.txt
Create the install batch script: virtualenv venv && .\venv\Scripts\activate && pip install -r requirements.txt
Edit the .gitignore and add venv and .idea to the ignored directories