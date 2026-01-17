Charge Variant Distribution Estimation Tool
===========================================

A Python-based offline application for estimating and visualizing
charge-variant distributions in biopharmaceutical proteins by
combining probabilistic effects of post-translational modifications (PTMs).

--------------------------------------------------
Overview
--------------------------------------------------
Proteins often contain multiple PTM sites, each contributing small
random charge shifts. Combining these effects leads to a large
combinatorial space of possible total charges.

This tool efficiently computes the overall charge distribution using
multiple algorithmic strategies with different accuracy–performance
trade-offs.

--------------------------------------------------
Key Features
--------------------------------------------------
• Exact and approximate charge-variant computation

• Supports proteins with repeated PTMs

• Fast and scalable algorithms

• Fully offline, self-contained GUI

• Visualization and CSV export

--------------------------------------------------
Core Concept
--------------------------------------------------
• Each PTM copy is modeled as an independent discrete random variable

• Total protein charge is the sum of all PTM charge contributions

• Repeated PTMs are handled using optimized mathematical techniques

--------------------------------------------------
Algorithms
--------------------------------------------------
1. Exact Convolution
   - Exact PMF convolution
   - Accurate but slow for large PTM counts
   - Time: O(K² log n)

2. FFT-Based Convolution
   - Uses FFT to accelerate convolution
   - Exact and faster than naive convolution
   - Time: O(N log N)

3. Moment Matching (Approximation) - Bonus algorithm mainly for **Stress Testing**
   - Gaussian approximation using CLT
   - Propagates only mean and variance
   - Very fast, ideal for exploration
   - Time: O(N + W)

--------------------------------------------------
Data Validation
--------------------------------------------------
• Missing field checks

• PMF row-sum enforcement (= 1)

• Negative probability detection

--------------------------------------------------
Execution Model
--------------------------------------------------
• Fully offline (no cloud dependency)

• Local computation only

• GUI built using Tkinter

• All assets bundled with the application

--------------------------------------------------
Performance Summary
--------------------------------------------------
Convolution        : Exact       | Slow

FFT Convolution    : Exact       | Fast

Moment Matching    : Approximate | Very Fast

--------------------------------------------------
Use Cases
--------------------------------------------------
• Charge-envelope estimation

• Formulation and stability analysis

• PTM-heavy protein stress testing

• Academic and research demonstrations

--------------------------------------------------
Offline Loading
--------------------------------------------------

Offline UI Loading: 

• The entire user interface is rendered locally, with no online assets, ensuring instant, reliable UI startup.

Offline Algorithm Loading:

• All computational algorithms run fully on-device, requiring no external services or network connectivity.

Offline Resource Loading: 

• All data, presets, and supporting files are loaded from local storage, eliminating any cloud dependencies.

Offline Execution Model: 

• The tool operates as a fully self-contained executable packaged with all dependencies, enabling true offline functionality.


**Tkinter UI is embedded inside the executable, so interface assets (themes, layouts, widgets) load instantly without web resources**.

--------------------------------------------------
## Usage
--------------------------------------------------
```bash
python appfinal.py
```

--------------------------------------------------
Contributors
--------------------------------------------------
Pranav Senthilkumaran

Glory E

Mushira S







