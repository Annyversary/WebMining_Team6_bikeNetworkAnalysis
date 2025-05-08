## WebMining_Team6_bikeNetworkAnalysis

The `data.zip` archive contains GML files for the years 2020 through 2024, split into quartals.  
**Note:** The files from the year 2020 are not entirely consistent, as they contain data points from months outside of their respective quarters. Therefore, the 2020 files require additional preprocessing.

In total, the `data` folder contains approximately **2 GB** of data, which exceeds the file size limit for uploading to GitHub.  
To access and transform the data, please follow these steps:

1. **Install the requirements**  
   ```bash
   pip install -r requirements.txt
   ```
   
2. **Unzip the `data.zip` archive**  
   ```bash
   tar -xf data.zip
   ```

3. **Convert GML files to GraphML format**
    ```bash
    python src\utils\data_splits\wrapper\transform_GML_into_graphML.py
    ```

**Optional: If you do not already have access to the feature-populated data (Google Drive link shared with project members), you can populate it yourself here (see Optional functionalities)**

4. **Generate training and validation splits using 2021-2023 for training and validation**
    ```bash
    python src\utils\data_splits\create_training_and_validation_data.py
    ```

5. **Generate training and validation splits using 2021-2022 for training and 2023 for validation**
    ```bash
    python src\utils\data_splits\create_timed_data_splits.py
    ```

6. **Generate test set**
    ```bash
    python src\utils\data_splits\create_test_data.py
    ```

Optional functionalities:

**Adding in-degree, out-degree, k-core, closeness centrality and distance between nodes to the GraphML files. This should be done between steps 3 and 4. (WARNING! Due to the size of the data, this can take up to 2 hours depending on your machine!)**  
```bash
python src\utils\data_featuring\add_features_to_data.py
```

**Outlier detection on the 'tracks' edge attribute. Works on the GraphML files, featured or not.**
```bash
python src\feature_engineering\outlier_detection.py
```