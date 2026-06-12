# File Anomaly Detector

A host-based intrusion detection system (HIDS) that uses a layered combination of unsupervised machine learning and YARA rule-based signature matching to identify suspicious files on a filesystem. Evaluated against real malware samples from [theZoo](https://github.com/ytisf/theZoo) repository.

## Installation

```bash
git clone https://github.com/dwoz045/file-anomaly-detector
cd file-anomaly-detector
python3 -m venv venv
source venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install pandas scikit-learn rich yara-python psutil pyarrow
```

## Usage

**Scan a directory:**
```bash
python3 main.py scan /path/to/directory
```

**Scan with custom contamination rate and export results:**
```bash
python3 main.py scan /path/to/directory --contamination 0.05 --top 30 --export results.csv
```

**Save model and rescan later:**
```bash
python3 main.py scan /path/to/directory --save-model
python3 main.py rescan /path/to/directory
```

**Monitor a directory continuously:**
```bash
python3 main.py monitor /path/to/directory --interval 60
```

**Run evaluation against labelled test data:**
```bash
python3 evaluate.py
```

## Tech Stack

Python, PyTorch, scikit-learn, YARA, pandas, rich, pyarrow

## Safe Malware Handling

Evaluation against theZoo malware samples was conducted inside an isolated Ubuntu VM following industry-standard safe malware handling practices. No samples are included in this repo.
