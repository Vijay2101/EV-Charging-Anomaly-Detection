AI Tool Usage Documentation
===========================


Overview
--------

AI tools were used as productivity accelerators during development. All final design decisions, modeling choices, evaluation strategies, and debugging were validated and understood before integration.

AI assistance was used selectively for:

*   Structuring modular code
    
*   Reviewing modeling reasoning
    
*   Refining documentation
    
*   Sanity-checking evaluation logic
    

All modeling logic, feature engineering strategy, and anomaly detection design decisions were made intentionally and validated manually.

Tools Used
----------

*   ChatGPT (GPT-5)
    
*   Claude
    

How AI Was Used
---------------

### 1\. Code Structuring and Modularization

AI assisted in:

*   Organizing feature engineering into features.py
    
*   Separating model logic into model.py
    
*   Structuring train.py, predict.py, evaluate.py
    
*   Drafting CLI boilerplate with argparse
    

These were productivity improvements rather than conceptual contributions.

All generated code was reviewed and modified as necessary.

### 2\. Clarifying Isolation Forest Behavior

AI was used to validate understanding of:

*   Contamination parameter behavior
    
*   Thresholding logic during inference
    
*   Difference between training-time thresholding and per-batch fitting
    
*   Interpretation of anomaly scores
    

All explanations were verified against scikit-learn documentation.

### 3\. Report Drafting and Refinement

AI helped refine:

*   Wording clarity
    
*   Technical justification structure
    
*   Tradeoff explanations
    
*   Production-level reasoning articulation
    

However, all reasoning content was based on actual experimentation and analysis.

### 4\. Evaluation Strategy Review

AI was consulted to:

*   Validate contamination sensitivity analysis
    
*   Ensure anomaly breakdown logic was sound
    
*   Confirm interpretation of model-only vs rule-only detection
    

All metrics were computed and interpreted manually.

Where AI Was Helpful
--------------------

*   Rapid scaffolding of boilerplate code
    
*   Improving documentation clarity
    
*   Structuring technical writing
    
*   Ensuring completeness of evaluation coverage
    

It significantly accelerated writing and organization tasks.

Where AI Struggled
------------------

*   Maintaining exact alignment with my implemented logic
    
*   Avoiding over-generalization of anomaly detection techniques
    
*   Over-suggesting unnecessary complexity (e.g., advanced explainability methods)
    

I rejected suggestions that deviated from the implemented design.