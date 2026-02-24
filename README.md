# EV Charging Station Anomaly Detection  

## Overview

This project implements a hybrid anomaly detection system for EV charging station event logs.

The system detects abnormal charger behavior using:

- **Isolation Forest** (statistical anomaly detection)
- **Domain-informed rule layer**
  - Firmware error codes
  - Negative power readings
  - Station-level z-score thresholds
  - Power discrepancy tolerance checks

The final anomaly label is the union of model-based detection and rule-based detection.

---
## Project Structure
```
├── eda.py               # Exploratory Data Analysis (plots + summary)
├── features.py          # Full feature engineering pipeline
├── model.py             # Isolation Forest + rule logic
├── train.py             # Training script
├── evaluate.py          # Evaluation & diagnostics
├── predict.py           # Inference script
├── requirements.txt
├── REPORT.md
├── AI_USAGE.md
└── charging_logs.csv
```

##  InstallationCreate a virtual environment 
```   
python -m venv venv
source venv/bin/activate      # Mac/Linux
venv\Scripts\activate         # Windows   
```

Install dependencies:

```  
 pip install -r requirements.txt   
```

 Run EDA
--------------

Performs exploratory analysis and saves plots.

```   
python eda.py  --input data/charging_logs.csv
```

This generates:

*   Feature distributions
    
*   Correlation heatmap
    
*   Temporal patterns
    
*   Error code analysis
    
*   Station-level error rates
    
*   Power vs temperature relationship
    
*   Power discrepancy distribution
    

Plots are saved to outputs/eda/.

Train the Model
----------------------

Trains the Isolation Forest model and applies rule-based detection.

```   
python train.py  --input charging_logs.csv
```

Outputs:

*   model.joblib
    
*   charging\_logs\_with\_predictions.csv
    
*   anomaly\_summary.csv
    

Evaluate the Model
-------------------------

Generates evaluation metrics and saves diagnostic plots.

```   
python evaluate.py --input charging_logs.csv
```

Evaluation includes:

*   Anomaly score distribution
    
*   Station-level anomaly rates
    
*   Hourly anomaly trends
    
*   Contamination sensitivity analysis
    
*   Feature correlation with anomaly score
    
*   Model vs rule breakdown
    
*   Detection rate on known fault patterns
    

Plots are saved to outputs/evaluation/.

Predict on New Data
--------------------------

Run inference on a new CSV file:

```
python predict.py --input charging_logs.csv --output predictions.csv 
```

The output CSV will include:

*   is\_anomaly → Final binary label (0 = normal, 1 = anomaly)
    
*   anomaly\_score → Continuous severity score (lower = more anomalous)
    

Modeling Approach
--------------------

### Isolation Forest

Used to detect multivariate statistical anomalies in engineered feature space.

*   Handles high-dimensional tabular data
    
*   Scales well to large log datasets
    
*   Produces interpretable anomaly scores
    

### Domain-Informed Rule Layer

Ensures detection of known fault signatures:

*   Non-zero firmware error codes
    
*   Negative power readings (physical constraint)
    
*   Extreme station-level z-scores (±3.5)
    
*   Large power discrepancy (> 2 kW)
    

Final anomaly decision:

```
Anomaly = IsolationForest_prediction OR Rule_trigger   
```

Key Design Decisions
-----------------------

*   Per-station rolling statistics capture local deviations.
    
*   Session-level aggregates capture contextual anomalies.
    
*   Station-level z-scores normalize inter-station differences.
    
*   Median imputation ensures robustness.
    
*   Contamination set to 3% based on observed anomaly prevalence.
    
