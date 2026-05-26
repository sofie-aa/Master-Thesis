# Master Thesis

## How to Run the Web Application on Your Own Computer

First, download or clone the code from GitHub.

### Prerequisites

Make sure Python is installed on your computer.

You may also need to install the required Python libraries. Open a terminal (MacOS) or Command Prompt/PowerShell (Windows) and run:

```bash
pip install flask waitress numpy pandas scikit-learn
```

---

### For Windows

1. Download or clone the project from GitHub.

2. Open **Command Prompt** or **PowerShell**.

3. Navigate to the project folder using:

```bash
cd path\to\your\project
```

Example:

```bash
cd C:\Users\YourName\Downloads\Master-Thesis
```

4. Run the application:

```bash
python app.py
```

5. Open your browser and go to:

```text
http://127.0.0.1:8080/
```

The web application should now be running locally.

---

### For MacOS

1. Download or clone the project from GitHub.

2. Open **Terminal**.

3. Navigate to the project folder:

```bash
cd /path/to/your/project
```

Example:

```bash
cd ~/Downloads/Master-Thesis
```

4. Run the application:

```bash
python app.py
```

If this does not work, try:

```bash
python3 app.py
```

5. Open Safari (or another browser) and go to:

```text
http://127.0.0.1:8080/
```

The web application should now be running locally.


## AI use declaration: 
Generative artificial intelligence tools, specifically ChatGPT, were used as assistance during parts of the development process in this thesis. ChatGPT was primarily used to assist in generating and refining code related to the production of the synthetic dataset. 

All AI-generated code used throughout this thesis was reviewed, verified, and, where necessary, modified by the author to ensure correctness and consistency with the statistical findings reported by O. Brus et al. \cite{brus2017self}. In particular, the synthetic dataset generation process was manually validated to ensure that the resulting dataset reflected the summary statistics presented in Table 1 of the referenced study. This included verifying that demographic and clinical distributions, such as the proportions of unipolar and bipolar depression, corresponded as closely as possible to the reported percentages in order to reproduce the statistical characteristics of the original dataset.

ChatGPT were also used to assist in troubleshooting programming issues, including interpreting error messages, identifying potential causes of program crashes, and suggesting possible solutions to implementation problems encountered during development. For example, AI assistance was used to understand runtime errors, debug code behavior, and identify syntax or implementation mistakes during programming.



