# **Customer Segmentation Analysis Project**  

![Customer Segmentation](https://github.com/kthalkari/Customer-Segmentation-Analysis/blob/main/Customer-Segmentation.png)

## **📌 Overview**  
This project analyzes customer purchasing behavior and demographics to segment customers into meaningful groups. By leveraging machine learning models, businesses can better understand customer preferences, optimize marketing strategies, and improve product recommendations.  

The analysis includes:  
✅ **Data Cleaning & Preprocessing**  
✅ **Exploratory Data Analysis (EDA)**  
✅ **Machine Learning Modeling**  
✅ **Model Evaluation & Hyperparameter Tuning**  

---

## **📋 Features**  
### **🔹 Data Processing**  
- **Date Conversion**: Standardize `Order Date` and `Delivery Date` into datetime format.  
- **Monetary Value Cleaning**: Remove currency symbols and convert price-related columns (`Unit Price`, `Total`, `Shipping Cost`, etc.) to numerical values.  
- **Missing Value Handling**: Check and address null values.  

### **🔹 Exploratory Data Analysis (EDA)**  
📊 **Numerical Features Visualization**:  
- Histograms for `Order Quantity`, `Unit Price`, `Total After Discount`, `Shipping Price`, and `Revenue`.  

📊 **Categorical Features Visualization**:  
- Count plots for `Order Priority`, `Product Category`, `Customer Segment`, and `Region`.  

📊 **Correlation Analysis**:  
- Heatmap to identify relationships between numerical features.  

📊 **Profit & Revenue Analysis**:  
- Boxplot of `Profit Margin` by `Product Category`.  
- Bar plots of `Total Revenue` by `Customer Segment` and `Region`.  

📊 **Time-Series Analysis**:  
- Monthly order volume trends.  
- Monthly revenue trends.  

### **🔹 Machine Learning Models**  
Three models were trained and evaluated for **customer segment prediction**:  
1. **Logistic Regression** (Baseline model)  
2. **Decision Tree Classifier** (Non-linear model)  
3. **Support Vector Classifier (SVC)** (Kernel-based model)  

### **🔹 Model Evaluation**  
- **Accuracy Scores**: Comparison of model performance.  
- **Confusion Matrices**: Visualizing true vs. predicted classifications.  
- **Classification Reports**: Precision, recall, and F1-score for each segment.  

### **🔹 Hyperparameter Tuning**  
- Optimized **Logistic Regression** by testing different `C` (regularization) values.  
- Identified the best-performing model with the highest accuracy.  

---

## **⚙️ Requirements**  
- **Python 3.x**  
- **Libraries**:  
  ```bash
  pandas
  numpy
  matplotlib
  seaborn
  scikit-learn
  openpyxl (for Excel file reading)
  ```
  
Install dependencies with:  
```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

## **📂 Data Source**  
The dataset is available at:  
🔗 [Data Project Store.xlsx](https://data.world/jerrys/sql-project/workspace/file?filename=Data+Project+Store.xlsx)  

**Columns Included**:  
- `Order Date`, `Delivery Date`, `Order Quantity`, `Unit Price`, `Total`, `Total After Discount`, `Shipping Cost`, `Box Cost`, `Revenue`, `Order Priority`, `Product Category`, `Customer Segment`, `Region`  

---

## **🚀 Workflow**  
1. **Data Loading** → Read Excel file using `pd.read_excel()`.  
2. **Data Cleaning** → Convert dates, clean monetary values, handle missing data.  
3. **EDA & Visualization** → Generate insights via histograms, count plots, heatmaps, and time-series plots.  
4. **Feature Engineering** → One-hot encoding for categorical variables.  
5. **Model Training** → Split data into train/test sets, train three ML models.  
6. **Model Evaluation** → Compare accuracy, confusion matrices, and classification reports.  
7. **Hyperparameter Tuning** → Optimize Logistic Regression using different `C` values.  

---

## **📊 Models Used**  
| Model | Description | Best Accuracy |
|--------|------------|--------------|
| **Logistic Regression** | Linear classifier predicting based on weighted inputs. | ~85% (after tuning) |
| **Decision Tree** | Splits data into segments based on feature values. | ~82% |
| **Support Vector Classifier (SVC)** | Finds optimal hyperplane to separate classes. | ~80% |

---

## **📈 Results & Evaluation**  
### **🔹 Best Model: Logistic Regression (Tuned)**  
- **Optimal C Value**: `10.0`  
- **Accuracy**: ~85%  
- **Confusion Matrix**:  


### **🔹 Key Insights**  
- **High-Value Customers**: Identified by `Total After Discount` and `Region`.  
- **Most Profitable Categories**: Some product categories yield higher profit margins.  
- **Seasonal Trends**: Order volume peaks in certain months.  

---

## **🛠️ How to Use**  
### **Running the Analysis**  
1. **Download the dataset** from the provided link.  
2. **Run the Jupyter Notebook / Python Script**:
     
   ```bash
   jupyter SourceCode.ipynb
   ```
  
3. **Review Outputs**:  
   - Visualizations (saved as `.png` files).  
   - Model evaluation metrics in the console.  

### **Customizing the Analysis**  
- Modify `test_size` in `train_test_split()` for different train-test ratios.  
- Experiment with other ML models (e.g., Random Forest, XGBoost).  
- Adjust EDA plots by changing `figsize` or color schemes.  

---

## **📜 License**  
MIT License  

---
