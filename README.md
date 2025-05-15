Stock Portfolio Heatmap Analyzer
Overview
The Stock Portfolio Heatmap Analyzer is an AI-powered tool that leverages deep learning and financial data visualization to help investors, analysts, and researchers better understand portfolio relationships and make data-driven investment decisions. By transforming stock correlation matrices into heatmap images and applying a Convolutional Neural Network (CNN), the analyzer predicts investment recommendations based on visual patterns in market data.

Features
Automated Data Fetching: Downloads historical stock price data using yfinance.

Correlation Analysis: Computes daily returns and generates correlation matrices for selected stock tickers.

Heatmap Visualization: Converts correlation matrices into intuitive heatmap images using Seaborn and Matplotlib.

Deep Learning Prediction: Trains a CNN to classify heatmaps and predict binary investment recommendations.

Modular Pipeline: Easily extensible for new data sources, more complex models, or integration into larger analytics systems.

Synthetic Labeling: Supports experimentation with synthetic labels for proof-of-concept; ready for integration with real investment outcomes.
Example
![Sample Heatmap](heatmaps/sample_heat

Portfolio optimization and risk management

Sector trend and anomaly detection

Automated investment recommendation systems

Educational and research purposes in financial analytics

Contributing
Contributions are welcome! Please open an issue or submit a pull request for improvements, bug fixes, or new features.

License
This project is licensed under the MIT License. See LICENSE for more details.

References
yfinance Documentation

Seaborn Heatmap Documentation

TensorFlow Keras Documentation

See the research paper and project documentation for further reading.
