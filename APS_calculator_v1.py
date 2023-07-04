# Developed by Hikmet Can Çubukçu

import streamlit as st
st.set_page_config(layout="wide")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from math import ceil
from decimal import Decimal
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

with st.sidebar:
    with open('./template/template.xlsx', "rb") as template_file:
        template_byte = template_file.read()
    # download template excel file
    st.download_button(label="Click to Download Template File",
                        data=template_byte,
                        file_name="template.xlsx",
                        mime='application/octet-stream')
    
    # DATA UPLOAD AND COLUMN SELECTION
    @st.cache_data(experimental_allow_widgets=True)
    def process_file(file):
        # data of analyte selection
        try:
            uploaded_file = pd.read_excel(file)
        except:
            uploaded_file = pd.read_csv(file, sep=None)
        analyte_name_box = st.selectbox("**Select Measurand Name**", tuple(uploaded_file.columns))
        analyte_data = uploaded_file[analyte_name_box]
        analyte_data = analyte_data.dropna(axis=0).reset_index()
        analyte_data = analyte_data[analyte_name_box]
        return analyte_data, analyte_name_box

    # upload file
    uploaded_file = st.file_uploader('#### **Upload your .xlsx (Excel) or .csv file:**', type=['csv','xlsx'], accept_multiple_files=False)

    # column name (data) selection
    if uploaded_file is not None:
        # data of analyte selection
        analyte_data, analyte_name_box = process_file(uploaded_file)

    #-----------------------------------------------------------------------

    # enter number of decimal places of data
    number_of_decimals = st.number_input('**Enter Number of Decimal Places of The Selected Data**', min_value=0, max_value=12)

    #st.subheader('Enter Number of Clinical Decision Limit(s) Below')
    number_CDL = st.number_input('**Enter Number of Clinical Decision Limit(s) Below**', min_value=1, max_value=7)
    
    #st.subheader('Enter Clinical Decision Limit(s) Below')
    if number_CDL == 1:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f", key=1)
    elif number_CDL == 2:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=2)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=3)
    elif number_CDL == 3:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=4)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=5)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=6)
    elif number_CDL == 4:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=7)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=8)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=9)
        cdl_4 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=10)
    elif number_CDL == 5:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=11)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=12)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=13)
        cdl_4 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=14)
        cdl_5 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=15)
    elif number_CDL == 6:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=16)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=17)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=18)
        cdl_4 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=19)
        cdl_5 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=20)
        cdl_6 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=21)
    elif number_CDL == 7:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below:** Please check final category intervals on "Distribution of data" page',min_value=0.00000 ,format="%.f",key=22)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=23)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=24)
        cdl_4 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=25)
        cdl_5 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=26)
        cdl_6 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=27)
        cdl_7 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=28)   
    else:
        st.warning('Enter Clinical Decision Limit(s)')
    
    # Aggreement thresholds
    st.subheader('Enter aggreement thresholds below')
    c1, c2, c3 = st.columns(3)
    with c1:
        min_agg_threshold = st.number_input('Minimum', value = 90)
    with c2:
        des_agg_threshold = st.number_input('Desirable', value = 95)
    with c3:
        opt_agg_threshold = st.number_input('Optimal', value = 99) 
        
    # Action button "Simulate & Calculate"
    analyze_button = st.button('**:green[Simulate & Calculate]**')
    
    #st.markdown('---')
    st.info('*Developed by Hikmet Can Çubukçu, MD, EuSpLM* <hikmetcancubukcu@gmail.com>')
    
 
st.image('./images/APS_Calculator_v2.png')
st.markdown('#### **:blue[A Data-Driven Tool for Setting Outcome-Based Analytical Performance Specifications for Measurement Uncertainty Using Specific Clinical Requirements and Population Data]**')
st.markdown('---')
# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📖 **Instructions**", "📊 **:green[Distribution of data]**", 
                                "🚦 **:blue[APS based on overall agreement]**", "🚥 **:violet[APS based on sublevel aggreement]**"],)
with tab1:
    st.markdown("""
                This web application is designed to help laboratory professinals to determine 
                their analytical performance specifications for measurement uncertainty based on their intended clinical setting and population of concern.
                
                #### Instructions
                
                1. Upload your data.
                Make sure that the first row of the Excel file you upload has measurand names and the other rows have analyte values, like the following example:
                
                | Glucose | LDL-cholesterol |
                | ----------- | ----------- |
                | 100 | 90 |
                | 120 | 120 |
                | 90 | 100 |
                | 170 | 110 |        

                2. Select the measurand name.
                2. Enter number of decimal places of the selected data (e.g., for 126, number of decimal places is 0; for 10.95, number of decimal places is 2).
                3. Enter the number of clinical decision limits you want to include in the APS determination processs
                4. Enter the value(s) of clinical decision limit(s)
                5. Enter the aggreement thresholds that will be used to determine minimum, desirable and optimal analytical performance specifications.
                6. Click on "Simulate & Analyze" button
                
                #### Simulation & Calculation Process
                
                During the simulation process, laboratory results uploaded into the application are assumed to represent the "true" values. 
                The application then simulates "measured" values by introducing measurement uncertainty into the actual concentration of the analyte, using the following formula: 
                """)
    formula = """
                Result_M = Result_O*[(1+ n(0,1))*MU]
                """
    st.latex(formula)       
    definition_caption = """
                ResultO: Original concentration of the measurand
                ResultM: Measured (Simulated) concentration of an analyte
                n(0,1): A random number generated with normal distribution (mean = 0, standard deviaiton = 1)\n
                MU: Relative standard measurement uncertainty
                """
    st.caption(definition_caption)
    st.markdown("""        
                
                ##### **The simulation process was performed in three steps as follows:**
            
                - Step 1- Categorization of original concentration of the analyte according to its clinical decision limits.
                - Step 2- Measured (simulated) result generation
                - Step 3- Recategorization of measured (simulated) concentration (as mentioned in step 1)
                
                The simulation is repeated for 331 different measurement uncertainty rates ranging from 0% to 33.1% with intervals of 0.1%. 
                
                ##### **Calculation Process**
                
                Aggreement level between original concentration and measured (simulated) concentration of the analyte were calculated. 
                Minimum, desirable, and optimal analytical performance specifications were determined according to the aggreement thresholds that entered previously.
                
                ##### **Contour Plots**
                The web application produces a series of contour plots that visually represent agreement rates as percentages on the x-axis and their corresponding measurement uncertainty values on the y-axis. 
                Horizontal lines are added to indicate the minimum, desirable, and optimal APSs for measurement uncertainty, corresponding to different agreement levels. 

                """)
    st.markdown('---')
    content=""" Disclaimer:

    The APS Calculator is provided solely as a decision support tool for assisting users in 
    setting analytical performance specifications based on indirect outcome. 
    It is important to note that while the tool aims to provide valuable guidance, 
    the ultimate responsibility for setting appropriate analytical performance specifications lies with the user.
    The user should exercise caution and consider relevant regulations, scientific literature, 
    and any other applicable guidelines when determining their analytical performance specifications. 
    It is crucial to ensure compliance with privacy, responsibility, and safety standards specific to the user's jurisdiction and intended application.
    While every effort has been made to ensure the accuracy and reliability of the APS Calculator, 
    it is important to understand that no tool can guarantee absolute accuracy or address all possible scenarios. 
    Therefore, users should use their professional judgment and seek additional expert advice when necessary.
    The APS Calculator does not collect any personally identifiable information (PII) or sensitive data from users. 
    However, it is recommended to exercise general caution when using any online application and take necessary 
    precautions to protect your privacy and data security. By using the APS Calculator, users acknowledge and 
    agree that they assume all risks and responsibilities associated with its use. The continuous availability 
    of the calculator cannot be guaranteed, as it may experience periods of unavailability due to various factors, 
    including technical issues, maintenance, or unforeseen circumstances.The creators, developers, and maintainers 
    of the APS Calculator shall not be held liable for any direct or indirect damages or losses arising from its use
    or resulting from the APS calculator's temporary or permanent unavailability. It is advised to regularly 
    review this disclaimer, as it may be updated to reflect any changes in regulations, best practices, or other relevant factors.
            """
    st.warning(content, icon="⚠️")
    st.markdown('---')
     
# table maker from list 
def create_table(category_intervals):
    # Create the table header
    table = f"|{':blue[Category Intervals]':<12}|\n"
    table += f"|{'-'*12}|\n"
    
    # Add each category to the table
    for category in category_intervals:
        table += f"|{category:<12}|\n"
    
    return table

# bins format checker function
def check_number(num):
    if isinstance(num, int):
        return num
    elif isinstance(num, float):
        if num.is_integer():
            return int(num)
        else:
            num_str = str(num)
            num_decimals = len(num_str.split('.')[1])
            if num_decimals == 1 and num_str.endswith('0'):
                return int(num)
            else:
                return num
    else:
        return "Input is not a number."
    
# converter function for cutting bins
def add_one(num):
    if isinstance(num, int):
        return num + 1
    elif isinstance(num, float):
        if num.is_integer():
            return int(num) + 1
        else:
            num_str = str(num)
            num_decimals = len(num_str.split('.')[1])
            if num_decimals == 1 and num_str.endswith('0'):
                return int(num) + 1
            else:
                return round(num + 0.1**num_decimals, num_decimals)
    else:
        return "Input is not a number."
          
### Subclass accuracy, specificity, sensitivity functions
# subclass specificity calculator
def subclass_specificity(confusion_matrix, class_id):
    """
    confusion matrix of multi-class classification
    class_id: id of a particular class 
    """
    confusion_matrix = np.float64(confusion_matrix)
    TP = confusion_matrix[class_id,class_id]
    FN = np.sum(confusion_matrix[class_id]) - TP
    FP = np.sum(confusion_matrix[:,class_id]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    # specificity = 0 if TN == 0
    if TN != 0:
        specificity = TN/(TN+FP)
    else:
        specificity = 0.
    return specificity

# subclass sensitivity calculator
def subclass_sensitivity(confusion_matrix, class_id):
    """
    confusion matrix of multi-class classification
    class_id: id of a particular class 
    """
    confusion_matrix = np.float64(confusion_matrix)
    TP = confusion_matrix[class_id,class_id]
    FN = np.sum(confusion_matrix[class_id]) - TP
    FP = np.sum(confusion_matrix[:,class_id]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    # sensitivity = 0 if TP == 0
    if TP != 0:
        sensitivity = TP/(TP+FN)
    else:
        sensitivity = 0.
    return sensitivity

# subclass accuracy calculator
def subclass_accuracy(confusion_matrix, class_id):
    """
    confusion matrix of multi-class classification
    class_id: id of a particular class 
    """
    confusion_matrix = np.float64(confusion_matrix)
    TP = confusion_matrix[class_id,class_id]
    FN = np.sum(confusion_matrix[class_id]) - TP
    FP = np.sum(confusion_matrix[:,class_id]) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    # accuracy = 0 if TP+TN == 0
    if TP+TN != 0:
        accuracy = (TP+TN)/(TP+FP+FN+TN)
    else:
        accuracy = 0
    return accuracy

### Overall (micro) accuracy, sensitivity, and specificity functions
# micro overall sensitivity
def calculate_micro_overall_sensitivity(confusion_matrix):
    confusion_matrix = np.float64(confusion_matrix)
    TP = np.sum(np.diag(confusion_matrix))
    FN = np.sum(confusion_matrix) - TP
    micro_overall_sensitivity = TP / (TP + FN)
    return micro_overall_sensitivity

# micro_overall_accuracy
def calculate_micro_overall_accuracy(confusion_matrix):
    confusion_matrix = np.float64(confusion_matrix)
    TP = np.diag(confusion_matrix)
    total = np.sum(confusion_matrix) 
    overall_accuracy = np.sum(TP) / np.sum(total)
    return overall_accuracy

# micro_overall_specificity
def calculate_micro_overall_specificity(confusion_matrix):
    confusion_matrix = np.float64(confusion_matrix)
    TP = np.diag(confusion_matrix)
    FN = np.sum(confusion_matrix, axis=1) - TP
    FP = np.sum(confusion_matrix, axis=0) - TP
    TN = np.sum(confusion_matrix) - TP - FN - FP
    overall_specificity = np.sum(TN) / np.sum(TN + FP)
    return overall_specificity
   
# action after clicking the button "simulate & analyze" 
if analyze_button:
    try:
        with st.spinner('**Please wait... Status:**'):
            placeholder = st.empty()
            placeholder.success('**Data preprocessing**', icon = "📂")
            
            column_name = analyte_name_box
            # Numeric data to categorical data conversion
            if number_CDL == 1:
                bins = [0, cdl_1-0.000001,np.inf]
                names = [f'<{check_number(cdl_1)}', f'≥{check_number(cdl_1)}']
                value = [1, 2]

            elif number_CDL == 2: # MODIFED
                bins = [0, cdl_1-0.000001, cdl_2, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}' ,f'≥{check_number(add_one(cdl_2))}']
                value = [1, 2, 3]

            elif number_CDL == 3:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'≥{check_number(add_one(cdl_3))}']
                value = [1, 2, 3, 4]

            elif number_CDL == 4:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'≥{check_number(add_one(cdl_4))}']
                value = [1, 2, 3, 4, 5]

            elif number_CDL == 5:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'{check_number(add_one(cdl_4))}-{check_number(cdl_5)}',f'≥{check_number(add_one(cdl_5))}']
                value = [1, 2, 3, 4, 5, 6]

            elif number_CDL == 6:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, cdl_6, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'{check_number(add_one(cdl_4))}-{check_number(cdl_5)}',f'{check_number(add_one(cdl_5))}-{check_number(cdl_6)}',f'≥{check_number(add_one(cdl_6))}']
                value = [1, 2, 3, 4, 5, 6, 7]

            elif number_CDL == 7:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, cdl_6, cdl_7, np.inf]
                names = [f'<{check_number(cdl_1)}', f'{check_number(cdl_1)}-{check_number(cdl_2)}', f'{check_number(add_one(cdl_2))}-{check_number(cdl_3)}' ,f'{check_number(add_one(cdl_3))}-{check_number(cdl_4)}',f'{check_number(add_one(cdl_4))}-{check_number(cdl_5)}',f'{check_number(add_one(cdl_5))}-{check_number(cdl_6)}',f'{check_number(add_one(cdl_6))}-{check_number(cdl_7)}',f'≥{check_number(add_one(cdl_7))}']
                value = [1, 2, 3, 4, 5, 6, 7, 8]
                
            else:
                st.warning('Enter Clinical Decision Limit(s)')
            
            # Numeric data to categorical data conversion
            cat_analyte= pd.cut(analyte_data, bins, labels=names)
            cat_analyte_df= pd.DataFrame(cat_analyte)

            # concat dfs
            cat_analyte_df = cat_analyte_df.rename(columns={column_name: "Analyte_category"})
            analyte_last_df = pd.concat([analyte_data, cat_analyte_df],axis = 1)

            # Category to number conversion
            analyte_last_df['cat_n'] = analyte_last_df['Analyte_category'].replace(
                to_replace=names,
                value=value, inplace=False)

            with tab2:
                # histogram of original data
                col1, col2 = st.columns([4,1.2])
                col1.write(" ")
                col1.info('Revise your clinical decision limits if the category intervals are not appropriate', icon = "ℹ️")
                col1.info('Make sure you have entered the number of decimal places of the selected data correctly', icon = "ℹ️")
                col2.write(" ")
                col2.write(create_table(names))
                
                st.write(" ")
                st.markdown('##### **:green[Histogram of the original data]**')

                # Get unique categories
                categories_h = analyte_last_df['Analyte_category'].unique()
                
                # Define sorting function to prioritize '<' and '≥' categories and sort others based on numbers before '-'
                def custom_sort_key(cat):
                    if cat.startswith("<"):
                        return (0, cat)  # Sort '<' categories first
                    elif cat.startswith("≥"):
                        return (2, cat)  # Sort '≥' categories last
                    else:
                        num_part = cat.split('-')[0]  # Extract the numbers before '-'
                        try:
                            num = float(num_part)
                            return (1, num)  # Sort other categories based on the extracted number
                        except ValueError:
                            return (1, cat)  # Sort categories without numbers as the original string
                
                # Sort the categories using the custom_sort_key function
                categories_h = sorted(categories_h, key=custom_sort_key)
                
                # Create the histogram figure
                fig_h = go.Figure()             
                # Iterate over categories and add histogram traces
                for i, category in enumerate(categories_h):
                    data = analyte_last_df[analyte_last_df['Analyte_category'] == category][column_name]
                    fig_h.add_trace(go.Histogram(
                        x=data,
                        name=category,
                        opacity=0.7,
                        autobinx=True
                    ))
                # Customize the layout of the figure
                fig_h.update_layout(
                    xaxis=dict(
                        title=column_name,
                        title_font=dict(
                            size=12
                        ),
                        tickfont=dict(
                            size=11
                        ),
                    ),
                    yaxis=dict(
                        title='Count',
                        title_font=dict(
                            size=12
                        ),
                        tickfont=dict(
                            size=11
                        )
                    ),
                    legend=dict(
                        title='Category Intervals',
                        xanchor='right',  # Position the legend on the right
                        yanchor='top',  # Position the legend on the top
                        x=0.98,  # Adjust the x position of the legend
                        y=0.98,  # Adjust the y position of the legend
                        traceorder='normal',
                        font=dict(
                            size=11
                        )
                    ),
                    margin=dict(
                        t=10,
                        r=10,
                        b=10,
                        l=10
                    ),
                    height=500,
                    width=800, barmode='stack'
                )
                # Show the figure using Streamlit
                st.plotly_chart(fig_h, theme="streamlit", use_container_width=True)
            
            # ------------------------------------------------------------------------------------
            placeholder.success('**Simulation**', icon ="🔄")
            ## SIMULATION CODES
            
            ### Selection of original data and original data category
            od = analyte_last_df[column_name] # original data
            o_cat_n = analyte_last_df['cat_n'] # original data category
                        
            error_l = [] # MU list

            accuracy_overall_l = [] # accuracy overall list , agreement
            sub_accuracy_score_l = [] # sub_accuracy score (sub group accuracy)

            sensitivity_overall_l = [] # sensitivity overall list
            sub_sensitivity_l = [] # sub_sensitivity score (sub group sensitivity)

            specificity_overall_l = [] # specificity overall list
            sub_specificity_l = [] # sub_specificity score (sub group specificity)


            sensitivity_recall_l = [] # sensitivity score list , agreement
            specificity_l = [] # specificity score list , agreement

            n_data = len(od) # sample size
            np.random.seed(2) # seed for reproducible results
            imprec_data_raw = np.random.normal(0, 1, n_data)
            imprec_data_raw = pd.Series(imprec_data_raw)
   
            # MU simulation
            ### MU simulation
            for e in np.arange(0,0.332,0.001): # CV constant
                                
                n_cat_n = []
                o_cat_n = list(o_cat_n)   
                e_CVA=e # MU error
                y_od = od + od*imprec_data_raw*e_CVA  # MU applied
                nd = round(y_od, number_of_decimals) # round generated values according to number of decimals of the selected data entered by user
                nd_cat= pd.cut(nd, bins, labels=names) # Categorization of the new data
                nd_cat_n = nd_cat.replace(to_replace=names,
                value=value, inplace=False)
                n_cat_n.append(nd_cat_n)
                n_cat_n = [item for sublist in n_cat_n for item in sublist]
                n_cat_n = pd.Series(n_cat_n)
                n_cat_n = n_cat_n.fillna(1)
                error_l.append(e) # MU rate save
                
                matrix = confusion_matrix(o_cat_n, n_cat_n) # establishing confusion matrix
                
                accuracy_overall_l.append(calculate_micro_overall_accuracy(matrix)) # overall accuracy save to list
                sensitivity_overall_l.append(calculate_micro_overall_sensitivity(matrix)) # overall sensitivity save to list
                specificity_overall_l.append(calculate_micro_overall_specificity(matrix)) # overall specificity save to list
                
                sub_accuracy_score_ll = []
                sub_sensitivity_ll = []
                sub_specificity_ll = []
                for i in range (0, len(names)):
                    sub_accuracy_score_ll.append(subclass_accuracy(matrix, i)) # subclass accuracy save to list
                    sub_sensitivity_ll.append(subclass_sensitivity(matrix, i)) # subclass sensitivity save to list
                    sub_specificity_ll.append(subclass_specificity(matrix, i)) # subclass specificity save to list
                sub_accuracy_score_l.append(sub_accuracy_score_ll)
                sub_sensitivity_l.append(sub_sensitivity_ll)
                sub_specificity_l.append(sub_specificity_ll)

            ### Accuracy table
            # sub_accuracy score data frame
            sub_accuracy_df = pd.DataFrame(sub_accuracy_score_l)
            sub_accuracy_df.columns = names
            # list to series conversion
            error_l = pd.Series(error_l) 
            accuracy_overall_l = pd.Series(accuracy_overall_l)
            # tables comprised of all data
            error_kappa_table = pd.concat([error_l, accuracy_overall_l], axis = 1)
            error_kappa_table.columns = ["Measurement Uncertainty", "Agreement"]
            error_kappa_table_2 = pd.concat([error_kappa_table, sub_accuracy_df], axis = 1)     
            # reset index
            error_kappa_table_2 = error_kappa_table_2.reset_index() # reset index
            
            ### Sensitivity table
            # sub_sensitivity data frame
            sub_sensitivity_df = pd.DataFrame(sub_sensitivity_l)
            sub_sensitivity_df.columns = names
            # list to series conversion
            error_l = pd.Series(error_l) 
            sensitivity_overall_l = pd.Series(sensitivity_overall_l)
            # tables comprised of all sensitivity and MU data
            error_sensitivity_table = pd.concat([error_l, sensitivity_overall_l], axis = 1)
            error_sensitivity_table.columns = ["Measurement Uncertainty", "Sensitivity"]
            error_sensitivity_table_2 = pd.concat([error_sensitivity_table, sub_sensitivity_df], axis = 1)
            # reset index
            error_sensitivity_table_2 = error_sensitivity_table_2.reset_index() # reset index
            
            ### Specificity table
            # sub_specificity data frame
            sub_specificity_df = pd.DataFrame(sub_specificity_l)
            sub_specificity_df.columns = names
            # list to series conversion
            error_l = pd.Series(error_l) 
            specificity_overall_l = pd.Series(specificity_overall_l)
            # tables comprised of all specificity and MU data
            error_specificity_table = pd.concat([error_l, specificity_overall_l], axis = 1)
            error_specificity_table.columns = ["Measurement Uncertainty", "Specificity"]
            error_specificity_table_2 = pd.concat([error_specificity_table, sub_specificity_df], axis = 1)  
            # reset index
            error_specificity_table_2 = error_specificity_table_2.reset_index() # reset index
            
            # ------------------------------------------------------------------------------------
            
            ### Disagreement based calculation for accuracy
            bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
            names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
            cat_Agree= pd.cut(error_kappa_table_2['Agreement'], bins_k, labels=names_k)
            cat_Agree_df= pd.DataFrame(cat_Agree)                        
            # Concat dfs
            cat_Agree_df = cat_Agree_df.rename(columns={"Agreement": "Agreement Category"})
            # Outcome full
            error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)
            error_kappa_table_3 = error_kappa_table_2_v2
            
            ### Disagreement based calculation for sensitivity
            bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
            names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
            cat_Agree= pd.cut(error_sensitivity_table_2['Sensitivity'], bins_k, labels=names_k)
            cat_Agree_df= pd.DataFrame(cat_Agree)
            # Concat dfs
            cat_Agree_df = cat_Agree_df.rename(columns={"Sensitivity": "Sensitivity Category"})
            # Outcome full
            error_sensitivity_table_2_v2 = pd.concat([error_sensitivity_table_2, cat_Agree_df],axis = 1)
            error_sensitivity_table_3 = error_sensitivity_table_2_v2
            
            ### Disagreement based calculation for specificity
            bins_k = [0, (min_agg_threshold/100)-0.0000000001, (des_agg_threshold/100)-0.0000000001, (opt_agg_threshold/100)-0.0000000001, np.inf]
            names_k = [f'<{min_agg_threshold}%',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
            cat_Agree= pd.cut(error_specificity_table_2['Specificity'], bins_k, labels=names_k)
            cat_Agree_df= pd.DataFrame(cat_Agree)        
            # Concat dfs
            cat_Agree_df = cat_Agree_df.rename(columns={"Specificity": "Specificity Category"})
            # Outcome full
            error_specificity_table_2_v2 = pd.concat([error_specificity_table_2, cat_Agree_df],axis = 1)
            error_specificity_table_3 = error_specificity_table_2_v2
            
            #-------------------------------------------------------------------------------------------
            placeholder.success('**Visualization and Calculation**',icon="📉")
            ## Visiualisation of Data
            ### Contour Plot
            # Accuracy percentage unit conversion by multiplying 100
            error_kappa_table_2_v3 = error_kappa_table_2_v2[['Measurement Uncertainty','Agreement']].apply(lambda x: x*100) # percentage unit conversion
            error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
            ### Specificity percentage unit conversion by multiplying 100
            error_specificity_table_3_v3 = error_specificity_table_3[['Measurement Uncertainty','Specificity']].apply(lambda x: x*100) # percentage unit conversion
            error_specificity_table_3_v3 = pd.concat([error_specificity_table_3_v3, error_specificity_table_3[['Specificity Category']]], axis = 1) # concat 
            ### Specificity percentage unit conversion by multiplying 100
            error_sensitivity_table_3_v3 = error_sensitivity_table_3[['Measurement Uncertainty','Sensitivity']].apply(lambda x: x*100) # percentage unit conversion
            error_sensitivity_table_3_v3 = pd.concat([error_sensitivity_table_3_v3, error_sensitivity_table_3[['Sensitivity Category']]], axis = 1) # concat 

            with tab3:
                ### interactive contour plot
                st.write("##### **:blue[APS for MU based on overall agreement]**")
                ###### interactive plot

                limit_xn = min_agg_threshold - 5
                limit_xp = 100
                ylim_v1 = error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= min_agg_threshold - 5]['Measurement Uncertainty'].max()

                # min, desirable, optimal MU limits
                ylim_v2 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= min_agg_threshold]['Measurement Uncertainty'].max(), 1)
                ylim_v3 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= des_agg_threshold]['Measurement Uncertainty'].max(), 1)
                ylim_v4 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement'] >= opt_agg_threshold]['Measurement Uncertainty'].max(), 1)

                # Define the colors based on the thresholds
                category_colors = {
                    f'≥{min_agg_threshold}%': 'rgb(166, 0, 0)',    # Dark Red
                    f'≥{des_agg_threshold}%': 'rgb(0, 176, 0)',    # Green
                    f'≥{opt_agg_threshold}%': 'rgb(0, 0, 255)',    # Blue
                }

                # Convert Agreement categories to colors
                error_kappa_table_2_v3['Color'] = error_kappa_table_2_v3['Agreement Category'].map(category_colors)

                # Create the scatter plot
                fig = go.Figure()

                # Add the scatter plot
                fig.add_trace(go.Scatter(
                    x=error_kappa_table_2_v3['Agreement'],
                    y=error_kappa_table_2_v3['Measurement Uncertainty'],
                    mode='markers',
                    marker=dict(
                        color=error_kappa_table_2_v3['Color'],
                        size=10,
                        opacity=0.7
                    ),
                    name='Agreement',
                    hovertemplate='Agreement: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                ))


                # Add the horizontal lines for percentile agreement thresholds
                fig.add_shape(
                    type='line',
                    x0=min_agg_threshold,
                    y0=0,
                    x1=min_agg_threshold,
                    y1=ylim_v1,
                    line=dict(
                        color='rgba(255, 0, 0, 0.5)',  # Transparent Dark Red
                        width=1.5,
                        dash='dash'
                    ),
                    xref='x',
                    yref='y'
                )
                fig.add_annotation(
                    x=min_agg_threshold + 0.1,
                    y=ylim_v1- 1,
                    text=f'{min_agg_threshold}% Agreement',
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    font=dict(
                        color='rgba(255, 0, 0, 0.8)'  # Dark Red
                    ),
                    textangle=270  # Rotate the text by 270 degrees
                )

                fig.add_shape(
                    type='line',
                    x0=des_agg_threshold,
                    y0=0,
                    x1=des_agg_threshold,
                    y1=ylim_v1,
                    line=dict(
                        color='rgba(0, 176, 0, 0.5)',  # Transparent Green
                        width=1.5,
                        dash='dash'
                    ),
                    xref='x',
                    yref='y'
                )
                fig.add_annotation(
                    x=des_agg_threshold + 0.2,  # Adjusted x-coordinate
                    y=ylim_v1 - 1,              # Adjusted y-coordinate
                    text=f'{des_agg_threshold}% Agreement',
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    font=dict(
                        color='rgba(0, 176, 0, 0.8)'  # Green
                    ),
                    textangle=270  # Rotate the text by 270 degrees
                )

                fig.add_shape(
                    type='line',
                    x0=opt_agg_threshold,
                    y0=0,
                    x1=opt_agg_threshold,
                    y1=ylim_v1,
                    line=dict(
                        color='rgba(0, 0, 255, 0.5)',  # Transparent Blue
                        width=1.5,
                        dash='dash'
                    ),
                    xref='x',
                    yref='y'
                )
                fig.add_annotation(
                    x=opt_agg_threshold + 0.2,  # Adjusted x-coordinate
                    y=ylim_v1 - 1,              # Adjusted y-coordinate
                    text=f'{opt_agg_threshold}% Agreement',
                    showarrow=False,
                    xanchor='left',
                    yanchor='middle',
                    font=dict(
                        color='rgba(0, 0, 255, 0.8)'  # Blue
                    ),
                    textangle=270  # Rotate the text by 270 degrees
                )

                # Set the x and y axis limits
                fig.update_xaxes(range=[limit_xn, limit_xp])
                fig.update_yaxes(range=[0, ylim_v1 + 1])

                # Create the legend
                fig.update_layout(
                    legend=dict(
                        title='Metrics',
                        x=0.01,
                        y=0.01,
                        traceorder='normal',
                        font=dict(
                            size=11
                        )
                    )
                )

                # Set the axis labels and font styles
                fig.update_layout(
                    xaxis=dict(
                        title='Agreement (%)',
                        title_font=dict(
                            size=14,
                        ),
                        tickfont=dict(
                            size=12,
                        )
                    ),
                    yaxis=dict(
                        title='Relative Standard Measurement Uncertainty (%)',
                        title_font=dict(
                            size=14,
                        ),
                        tickfont=dict(
                            size=12,
                        )
                    ),
                    margin=dict(
                        t=10,
                        r=10,
                        b=10,
                        l=10
                    ),
                    height=500,
                    width=800,
                    plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Light background color
                    paper_bgcolor='rgba(255, 255, 255, 0.9)'  # Light background color
                )

                # Text info of APS
                col22, col33 = st.columns([4, 1])
                na_quote_1 = ' '
                na_quote_2 = ' '
                # >33% MU is unaccceptable
                if ylim_v2 > 33:
                    ylim_v2 = "NA"
                    na_quote_1 = "NA: Not available"
                if ylim_v3 > 33:
                    ylim_v3 = "NA"
                    na_quote_1 = "NA: Not available"
                if ylim_v4 > 33:
                    ylim_v4 = "NA"
                    na_quote_1 = "NA: Not available"
                 
                # =0% MU is notobtainable
                if ylim_v2 == 0:
                    ylim_v2 = "NO"
                    na_quote_2 = "NO: Not obtainable"
                if ylim_v3 == 0:
                    ylim_v3 = "NO"
                    na_quote_2 = "NO: Not obtainable"
                if ylim_v4 == 0:
                    ylim_v4 = "NO"
                    na_quote_2 = "NO: Not obtainable"
                col33.write(" ")
                col33.markdown(f"""
                            | APS level | MU |
                            | ----------- | ----------- |
                            | :red[Minimum] | {ylim_v2} |
                            | :green[Desirable] | {ylim_v3} |
                            | :blue[Optimal] | {ylim_v4} |
                            """)
                col33.write(na_quote_1) # >33% MU is unaccceptable
                col33.write(na_quote_2) # =0% MU is notobtainable
                col22.plotly_chart(fig, theme="streamlit", use_container_width=True) # show figure 
                st.markdown("---")
            
            with tab4:
                st.write('##### **:blue[APS for based on sublevel aggreement]**')
                st.write('  ')
                for i in names:     
                    st.markdown(f'###### **:blue[APS based on sublevel: {i}]**')
                    level_name = i
                    
                    ### Agreement over-sub
                    cat_Agree= pd.cut(error_kappa_table_2[level_name], bins_k, labels=names_k)
                    cat_Agree_df= pd.DataFrame(cat_Agree)
                    # concat dfs
                    cat_Agree_df = cat_Agree_df.rename(columns={level_name: "Agreement Category"})
                    # Outcome full
                    error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)                
                    # percentage unit conversion by multiplying 100
                    error_kappa_table_2_v3 = error_kappa_table_2_v2[['Measurement Uncertainty',level_name]].apply(lambda x: x*100) # percentage unit conversion
                    error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
                                        
                    ### Sensitivity over-sub
                    cat_Sens= pd.cut(error_sensitivity_table_2[level_name], bins_k, labels=names_k)
                    cat_Sens_df= pd.DataFrame(cat_Sens)
                    # concat dfs
                    cat_Sens_df = cat_Sens_df.rename(columns={level_name: "Sensitivity Category"})
                    # Outcome full
                    error_sensitivity_table_2_v2 = pd.concat([error_sensitivity_table_2, cat_Sens_df],axis = 1)                
                    # percentage unit conversion by multiplying 100
                    error_sensitivity_table_2_v3 = error_sensitivity_table_2_v2[['Measurement Uncertainty',level_name]].apply(lambda x: x*100) # percentage unit conversion
                    error_sensitivity_table_2_v3 = pd.concat([error_sensitivity_table_2_v3, error_sensitivity_table_2_v2[['Sensitivity Category']]], axis = 1) # concat 

                    ### Specificity over-sub
                    cat_Spec= pd.cut(error_specificity_table_2[level_name], bins_k, labels=names_k)
                    cat_Spec_df= pd.DataFrame(cat_Spec)
                    # concat dfs
                    cat_Spec_df = cat_Spec_df.rename(columns={level_name: "Specificity Category"})
                    # Outcome full
                    error_specificity_table_2_v2 = pd.concat([error_specificity_table_2, cat_Spec_df],axis = 1)                
                    # percentage unit conversion by multiplying 100
                    error_specificity_table_2_v3 = error_specificity_table_2_v2[['Measurement Uncertainty',level_name]].apply(lambda x: x*100) # percentage unit conversion
                    error_specificity_table_2_v3 = pd.concat([error_specificity_table_2_v3, error_specificity_table_2_v2[['Specificity Category']]], axis = 1) # concat 

                    # axis limits
                    limit_xn = min_agg_threshold - 5
                    limit_xp = 100
                    ylim_v1 = error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=min_agg_threshold-5]['Measurement Uncertainty'].max()

                    # min, desirable, optimal MU limits
                    ylim_v2 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=min_agg_threshold]['Measurement Uncertainty'].max(), 1)
                    ylim_v3 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=des_agg_threshold]['Measurement Uncertainty'].max(), 1)
                    ylim_v4 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=opt_agg_threshold]['Measurement Uncertainty'].max(), 1)

                    # Define the colors based on the thresholds
                    category_colors = {
                        f'≥{min_agg_threshold}%': 'rgb(166, 0, 0)',    # Dark Red
                        f'≥{des_agg_threshold}%': 'rgb(0, 176, 0)',    # Green
                        f'≥{opt_agg_threshold}%': 'rgb(0, 0, 255)',    # Blue
                    }

                    # Convert Agreement categories to colors
                    error_kappa_table_2_v3['Color'] = error_kappa_table_2_v3['Agreement Category'].map(category_colors)
                    
                    # Create the scatter plot
                    fig = go.Figure()

                    # Add the scatter plot of Agreement (accuracy)
                    fig.add_trace(go.Scatter(
                        x=error_kappa_table_2_v3[level_name],
                        y=error_kappa_table_2_v3['Measurement Uncertainty'],
                        mode='markers',
                        marker=dict(
                            color=error_kappa_table_2_v3['Color'],
                            size=10,
                            opacity=0.7
                        ),
                        name='Agreement',
                        hovertemplate='Agreement: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                    ))

                    # Add the Specificity scatter plot
                    fig.add_trace(go.Scatter(
                        x=error_specificity_table_2_v3[level_name],
                        y=error_specificity_table_2_v3['Measurement Uncertainty'],
                        mode='markers',
                        marker=dict(
                            symbol='pentagon',
                            color= 'rgb(204, 85, 119)' , 
                            size=7,
                            opacity=0.6
                        ),
                        name='Specificity',
                        hovertemplate='Specificity: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                    ))

                    # Add the Sensitivity scatter plot 
                    fig.add_trace(go.Scatter(
                        x=error_sensitivity_table_2_v3[level_name],
                        y=error_sensitivity_table_2_v3['Measurement Uncertainty'],
                        mode='markers',
                        marker=dict(
                            symbol='hexagon2',
                            color='rgb(128, 0, 128)',
                            size=7,
                            opacity=0.6
                        ),
                        name='Sensitivity',
                        hovertemplate='Sensitivity: %{x}%<br>Measurement Uncertainty: %{y}%<br>'
                    ))

                    # Add the horizontal lines for percentile agreement thresholds
                    fig.add_shape(
                        type='line',
                        x0=min_agg_threshold,
                        y0=0,
                        x1=min_agg_threshold,
                        y1=ylim_v1,
                        line=dict(
                            color='rgba(255, 0, 0, 0.5)',  # Transparent Dark Red
                            width=1.5,
                            dash='dash'
                        ),
                        xref='x',
                        yref='y'
                    )
                    fig.add_annotation(
                        x=min_agg_threshold + 0.1,
                        y=ylim_v1- 1,
                        text=f'{min_agg_threshold}% Agreement',
                        showarrow=False,
                        xanchor='left',
                        yanchor='middle',
                        font=dict(
                            color='rgba(255, 0, 0, 0.8)'  # Dark Red
                        ),
                        textangle=270  # Rotate the text by 270 degrees
                    )

                    fig.add_shape(
                        type='line',
                        x0=des_agg_threshold,
                        y0=0,
                        x1=des_agg_threshold,
                        y1=ylim_v1,
                        line=dict(
                            color='rgba(0, 176, 0, 0.5)',  # Transparent Green
                            width=1.5,
                            dash='dash'
                        ),
                        xref='x',
                        yref='y'
                    )
                    fig.add_annotation(
                        x=des_agg_threshold + 0.2,  # Adjusted x-coordinate
                        y=ylim_v1 - 1,              # Adjusted y-coordinate
                        text=f'{des_agg_threshold}% Agreement',
                        showarrow=False,
                        xanchor='left',
                        yanchor='middle',
                        font=dict(
                            color='rgba(0, 176, 0, 0.8)'  # Green
                        ),
                        textangle=270  # Rotate the text by 270 degrees
                    )

                    fig.add_shape(
                        type='line',
                        x0=opt_agg_threshold,
                        y0=0,
                        x1=opt_agg_threshold,
                        y1=ylim_v1,
                        line=dict(
                            color='rgba(0, 0, 255, 0.5)',  # Transparent Blue
                            width=1.5,
                            dash='dash'
                        ),
                        xref='x',
                        yref='y'
                    )
                    fig.add_annotation(
                        x=opt_agg_threshold + 0.2,  # Adjusted x-coordinate
                        y=ylim_v1 - 1,              # Adjusted y-coordinate
                        text=f'{opt_agg_threshold}% Agreement',
                        showarrow=False,
                        xanchor='left',
                        yanchor='middle',
                        font=dict(
                            color='rgba(0, 0, 255, 0.8)'  # Blue
                        ),
                        textangle=270  # Rotate the text by 270 degrees
                    )

                    # Set the x and y axis limits
                    fig.update_xaxes(range=[limit_xn, limit_xp])
                    fig.update_yaxes(range=[0, ylim_v1 + 1])

                    # Create the legend
                    fig.update_layout(
                        legend=dict(
                            title='Metrics',
                            x=0.01,
                            y=0.01,
                            traceorder='normal',
                            font=dict(
                                size=11
                            )
                        )
                    )

                    # Set the axis labels and font styles
                    fig.update_layout(
                        xaxis=dict(
                            title='Metrics (%)',
                            title_font=dict(
                                size=14,
                            ),
                            tickfont=dict(
                                size=12,
                            )
                        ),
                        yaxis=dict(
                            title='Relative Standard Measurement Uncertainty (%)',
                            title_font=dict(
                                size=14,
                            ),
                            tickfont=dict(
                                size=12,
                            )
                        ),
                        margin=dict(
                            t=10,
                            r=10,
                            b=10,
                            l=10
                        ),
                        height=500,
                        width=800,
                        plot_bgcolor='rgba(255, 255, 255, 0.9)',  # Light background color
                        paper_bgcolor='rgba(255, 255, 255, 0.9)'  # Light background color
                    )

                    # Text info of APS
                    col22, col33 = st.columns([4, 1])
                    na_quote_3 = ' '
                    na_quote_4 = ' '
                    # >33% MU is unaccceptable
                    if ylim_v2 > 33:
                        ylim_v2 = "NA"
                        na_quote_3 = "NA: Not available"
                    if ylim_v3 > 33:
                        ylim_v3 = "NA"
                        na_quote_3 = "NA: Not available"
                    if ylim_v4 > 33:
                        ylim_v4 = "NA"
                        na_quote_3 = "NA: Not available"
                    
                    # =0% MU is notobtainable
                    if ylim_v2 == 0:
                        ylim_v2 = "NO"
                        na_quote_4 = "NO: Not obtainable"
                    if ylim_v3 == 0:
                        ylim_v3 = "NO"
                        na_quote_4 = "NO: Not obtainable"
                    if ylim_v4 == 0:
                        ylim_v4 = "NO"
                        na_quote_4 = "NO: Not obtainable"
                    col33..write(" ")
                    col33.markdown(f"""
                                | APS level | MU |
                                | ----------- | ----------- |
                                | :red[Minimum] | {ylim_v2} |
                                | :green[Desirable] | {ylim_v3} |
                                | :blue[Optimal] | {ylim_v4} |            
                                """)
                    col33.write(na_quote_3) # >33% MU is unaccceptable 
                    col33.write(na_quote_4) # =0% MU is notobtainable
                    col22.plotly_chart(fig, theme="streamlit", use_container_width=True) # show figure 
                    st.markdown("---")
            placeholder.success('**Done**', icon="✅")
            time.sleep(2)
            placeholder.empty()
       
    except NameError:
        st.error('Please upload your file')
    except ValueError: 
        st.error('Inappropriate clinical decision limit was entered.', icon="❗")
else:
    st.info('Upload your file and follow the instructions to calculate APS', icon = "📁")
        
