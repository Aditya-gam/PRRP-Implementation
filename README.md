# PRRP-Implementation

## 📌 Overview
This project implements **P-Regionalization through Recursive Partitioning (PRRP)**, an algorithm designed for **spatial regionalization**, and extends it to support **graph partitioning** while ensuring spatial contiguity.

### **Why PRRP?**
Regionalization is an NP-hard problem where spatially contiguous areas are clustered based on user-defined constraints. **PRRP** enhances existing approaches by:
- Ensuring statistical independence of generated partitions.
- Using a recursive three-phase process: **Region Growing, Region Merging, and Region Splitting**.
- Extending the method to partition **graph-based data** while maintaining connectivity constraints.

This implementation follows the methodology from the research paper:
> **"Statistical Inference for Spatial Regionalization"**  
> _Authors: Hussah Alrashid, Amr Magdy, Sergio Rey_  
> **ACM SIGSPATIAL 2023**  
> [Read the Paper](https://doi.org/10.1145/3589132.3625608)

---

## 📌 **Installation & Setup**
This guide walks you through setting up PRRP-Implementation on **Ubuntu (via WSL on Windows 10)**.

### **1️⃣ Fork & Clone the Repository**
If you want to contribute or modify this project, first fork it on GitHub. Then, clone it:

```bash
git clone https://github.com/YOUR_USERNAME/PRRP-Implementation.git
cd PRRP-Implementation
```

---

### **2️⃣ Set Up WSL (Windows Users Only)**
**WSL (Windows Subsystem for Linux)** is required for this project because **METIS** works best on Unix-based systems.

- Ensure WSL is installed:  
  ```bash
  wsl --list --verbose
  ```
  If WSL is not installed, [follow this guide](https://learn.microsoft.com/en-us/windows/wsl/install) to install it.

- Update WSL and install required tools:  
  ```bash
  sudo apt update && sudo apt install -y build-essential software-properties-common
  ```

---

### **3️⃣ Create a Virtual Environment**
Inside **WSL**, create an isolated environment to manage dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
```

To deactivate the environment later, run:

```bash
deactivate
```

---

### **4️⃣ Install Dependencies**
All required libraries are listed in `requirements.txt`. Install them by running:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### **5️⃣ Install METIS (Required for Graph Partitioning)**
To use **graph-based PRRP**, install METIS:

```bash
sudo apt install -y metis libmetis-dev
```

Verify METIS is installed by running:

```bash
gpmetis --help
```

---

### **6️⃣ Verify Your Setup**
Check if all dependencies are installed correctly:

```bash
python -c "import networkx, scipy, numpy, pymetis; print('All libraries installed successfully!')"
```

If you see `"All libraries installed successfully!"`, your environment is ready!

---

## 📌 **Project Structure**
```
PRRP-Implementation/
│── src/                # Core source code
│   ├── spatial_prrp.py # PRRP for spatial regionalization
│   ├── graph_prrp.py   # PRRP for graphs
│   ├── utils.py        # Utility functions
│   ├── metis_comparison.py # Compares PRRP results with METIS
│── data/               # Stores datasets
│── notebooks/          # Jupyter notebooks for testing
│── tests/              # Unit tests for all modules
│── README.md           # Documentation
│── requirements.txt    # Python dependencies
│── setup.py            # Installation script
│── .gitignore          # Git ignore file
```

---

## 📌 **Contributing**
Contributions are welcome! To contribute:
1. Fork this repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -m "Added new feature"`).
4. Push to your fork (`git push origin feature-branch`).
5. Submit a pull request.

---

## 📌 **Citation**
If you use this implementation, please cite the original paper:

```
@inproceedings{10.1145/3589132.3625608,
author = {Alrashid, Hussah and Magdy, Amr and Rey, Sergio},
title = {Statistical Inference for Spatial Regionalization},
year = {2023},
isbn = {9798400701689},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3589132.3625608},
doi = {10.1145/3589132.3625608},
booktitle = {Proceedings of the 31st ACM International Conference on Advances in Geographic Information Systems},
articleno = {65},
numpages = {12},
keywords = {statistical inference, regionalization, spatial clustering},
location = {Hamburg, Germany},
series = {SIGSPATIAL '23}
}
```
