import pandas as pd 
import numpy as np
from bs4 import BeautifulSoup
import json
from typing import Union

class TableProcessor:
    def __init__(self, html_table:str):
        """
        Args:
            html_table[str]: a html table.
        """
        self.html_table = html_table
        self.methods = None
        self.df = None
        self.nparray = None

    def table_processor(self) -> Union[pd.Series, pd.DataFrame]:
        """Takes a html table and returns the method and the estimates ± se dataframe.

        Returns:
            method[pd.Series]: sketching method.
            estimates ± se dataframe [pd.DataFrame]: each row corresponds to a method. Print column name to see more details about the column.
        """
        rename_method = {'proposal1':'RRR', 
                             'proposal1(adaptive)': 'RRR(adaptive)', 
                             'proposal2':'RRS', 
                             'proposal2(adaptive)':'RRS(adaptive)'}
        df = pd.read_html(self.html_table)[0]
        methods = df.iloc[:, 0]
        df = df.drop(df.columns[0], axis=1)
        self.methods = [rename_method[method] if method in rename_method else method for method in methods]
        self.df = df
        return self.methods, self.df

    def _extract_estimate_and_ci(self, value:str) -> Union[float, float]:
        """Extract estimate and standard errors. Split estimates ± se [str] and convert them to float.

        Returns:
            estimate[float]
            se[float]
        """
        estimate, ci = value.split(' ± ')
        return float(estimate), float(ci)

    def toarray(self)->np.ndarray:
        """Extract estimates and standard errors and returns a 3-dimensional np.array where the [0,:,:] contains estimates and [1,:,:] contains confidence intervals.

        Returns:
            estimates and se[np.ndarray]
        """
        # Create lists to store estimates and standard errors
        estimates = []
        se = []

        # Apply the extraction to each column and store in the lists
        for col in self.df.columns:
            estimates_col = []
            se_col = []
            for value in self.df[col]:
                estimate, ci = self._extract_estimate_and_ci(value)
                estimates_col.append(estimate)
                se_col.append(ci)
            
            estimates.append(estimates_col)
            se.append(se_col)

        # Convert lists to numpy arrays and stack them along a new axis (3rd dimension)
        estimates_np = np.array(estimates).T
        se_np = np.array(se).T

        # Stack the arrays along a new axis to create the 3D array
        result_np = np.stack([estimates_np, se_np], axis=0)
        self.nparray = result_np

        return result_np

class HTMLTable:
    def __init__(self):
        pass

    def extract_tables(self, html_content:str, custom_table_names:str=None):
        soup = BeautifulSoup(html_content, 'html.parser')
        # Find all tables in the HTML
        tables = soup.find_all('table')
        tables_data = {} 

        for idx, table in enumerate(tables):
            # Get the raw HTML of the table
            table_html = str(table)
            
            # Add the table HTML to the dictionary with a custom name
            table_name = custom_table_names[idx] if custom_table_names != None and idx < len(custom_table_names) else f"table_{idx+1}"
            tables_data[table_name] = table_html

        # Save to a JSON file
        with open('./RandNLA_Regression/sim_result.json', 'w') as f:
            json.dump(tables_data, f, indent=4)

        return tables_data
    
if __name__ == "__main__":
    with open('./RandNLA_Regression/analysis/vis.html', 'r') as file:
        html_content = file.read()
    table_names = ['delta_0_1_normal', 'delta_0_1_laplace', 'delta_0_1_cauchy', 'delta_0_1_point_mass', 
                   'delta_0_2_cauchy', 'delta_0_2_point_mass',
                   'k_40_normal', 'k_40_cauchy', 'k_40_point_mass',
                   'k_160_normal', 'k_160_cauchy', 'k_160_point_mass',
                   'contamination_delta_0_1_k_40', 'contamination_delta_0_2_k_40', 'contamination_delta_0_1_k_160', 'contamination_delta_0_2_k_160',
                   'time_delta_0_2_k_40_point_mass']
    html_tables = HTMLTable().extract_tables(html_content, table_names)