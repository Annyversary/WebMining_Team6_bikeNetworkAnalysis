## WebMining_Team6_bikeNetworkAnalysis

The `data.zip` archive contains GML files for the years 2020 through 2024, split into quartals.  
**Note:** The files from the year 2020 are not entirely consistent, as they contain data points from months outside of their respective quarters. Therefore, the 2020 files require additional preprocessing.

In total, the `data` folder contains approximately **2 GB** of data, which exceeds the file size limit for uploading to GitHub.  
To access and transform the data, please follow these steps:

1. **Unzip the `data.zip` archive**  
   ```bash
   tar -xf data.zip

2. **Convert GML files to GraphML format**
    ```bash
    jupyter nbconvert --to notebook --execute src\utils\wrapper\transform_GML_into_graphML.ipynb --inplace

3. **Generate training and validation splits**
    ```bash
    jupyter nbconvert --to notebook --execute src\utils\data_splits\create_training_and_validation_data.ipynb--inplace


