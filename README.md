## WebMining_Team6_bikeNetworkAnalysis

The `data.zip` archive contains GML files for the years 2020 through 2024, split into quartals.  
**Note:** The files from the year 2020 are not entirely consistent, as they contain data points from months outside of their respective quarters. Therefore, the 2020 files require additional preprocessing.

In total, the `data` folder contains approximately **2 GB** of data, which exceeds the file size limit for uploading to GitHub.  
To access and transform the data, please follow these steps:

1. **Unzip the `data.zip` file**  
   Open PowerShell or CMD and run the following command to extract the contents of `data.zip`:
   ```bash
   tar -xf data.zip

2. **Navigate to the script directory**
    Change the directory to where the transformation script is located:
    ```bash
    cd src\utils\wrapper

3. **Run the transformation script in Jupyter Notebook**
    Launch Jupyter Notebook and open the transform_GML_into_graphML.ipynb file:
    ```bash
    jupyter notebook transform_GML_into_graphML.ipynb

Once the notebook is open, execute all cells to transform the GML files into the appropriate GraphML format.




