# Developed by Hikmet Can Çubukçu

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import time
from decimal import Decimal
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

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
        uploaded_file = pd.read_excel(file)
        analyte_name_box = st.selectbox("**Select Analyte Name**", tuple(uploaded_file.columns))
        analyte_data = uploaded_file[analyte_name_box]
        analyte_data = analyte_data.dropna(axis=0).reset_index()
        analyte_data = analyte_data[analyte_name_box]

        return analyte_data, analyte_name_box

    # upload file
    uploaded_file = st.file_uploader('#### **Upload your .xlsx (Excel) file:**')

    # column name (data) selection
    if uploaded_file is not None:
        # data of analyte selection
        analyte_data, analyte_name_box = process_file(uploaded_file)

    #-----------------------------------------------------------------------
    

    #st.subheader('Enter Number of Clinical Decision Limit(s) Below')
    number_CDL = st.number_input('**Enter Number of Clinical Decision Limit(s) Below**', min_value=1, max_value=7)
    
    #st.subheader('Enter Clinical Decision Limit(s) Below')
    if number_CDL == 1:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below**',min_value=0.00000 ,format="%.f", key=1)
    elif number_CDL == 2:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below**',min_value=0.00000 ,format="%.f",key=2)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=3)
    elif number_CDL == 3:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below**',min_value=0.00000 ,format="%.f",key=4)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=5)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=6)
    elif number_CDL == 4:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below**',min_value=0.00000 ,format="%.f",key=7)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=8)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=9)
        cdl_4 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=10)
    elif number_CDL == 5:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below**',min_value=0.00000 ,format="%.f",key=11)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=12)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=13)
        cdl_4 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=14)
        cdl_5 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=15)
    elif number_CDL == 6:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below**',min_value=0.00000 ,format="%.f",key=16)
        cdl_2 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=17)
        cdl_3 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=18)
        cdl_4 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=19)
        cdl_5 = st.number_input(label='',min_value=0.00000 ,format="%.f",key=20)
        cdl_6 = st.number_input(label='',min_value=0.00000 ,format="%.f", key=21)
    elif number_CDL == 7:
        cdl_1 = st.number_input(label='**Enter Clinical Decision Limit(s) Below**',min_value=0.00000 ,format="%.f",key=22)
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
    with c1:
        opt_agg_threshold = st.number_input('Optimal', value = 99) 
    
        
    # Action button "Simulate & Calculate"
    analyze_button = st.button('**:green[Simulate & Calculate]**')
    
    #st.markdown('---')
    st.info('*Developed by Hikmet Can Çubukçu, MD, EuSpLM* <hikmetcancubukcu@gmail.com>')
    
 
col1, col2 = st.columns(2)
col1.title(':green[*APS Calculator*]')
col2.image('./images/eflm_icon.png')
st.markdown('#### **:blue[A Data-Driven Approach for Setting Analytical Performance Specifications Based on Intended Clinical Settings and Population]**')
st.markdown('---')
# TABS
tab1, tab2, tab3, tab4 = st.tabs(["📖 **Instructions**", "📊 **:green[Distribution of data]**", 
                                "🚦 **:blue[APS based on overall agreement]**", "🚥 **:violet[APS based on sublevel aggreement]**"],)
with tab1:
                st.markdown("""
                ### A Data-Driven Approach For Setting Analytical Performance Specifications Based on Intended Clinical Settings and Population    
                This web application is designed to help laboratory professinals to determine 
                their analytical performance specifications based on their intended clinical setting and population of concern.
                
                #### Instructions
                
                1. Kindly, upload your data belonged to intended clinical setting and population of concern.
                Make sure that the first row of the Excel file you upload has analyte names and the other rows have analyte values, as follows:
                
                | Glucose | LDL-cholesterol |
                | ----------- | ----------- |
                | 100 | 90 |
                | 120 | 120 |
                | 90 | 100 |
                | 170 | 110 |        
                
                2. Then, enter the number of clinical decision limits you want to include in the APS determination processs
                3. Enter the clinical decision limit(s)
                4. Enter the aggreement thresholds that will be used to determine minimum, desirable and optimal analytical performance specifications.
                5. Push "Simulate & Analyze" button
                
                #### Simulation & Calculation Process
                
                Firstly, APS calculator introduces different degrees of measurement uncertainty on your data by simulation, as follows: 
                """)
                formula = """
                Result_M = Result_T*[(1+ n(0,1))*MU]
                """
                st.latex(formula)       
                definition_caption = """
                ResultT: True concentration of the analyte
                ResultM: Measured (Simulated) concentration of an analyte
                n(0,1): A random number generated with normal distribution (mean = 0, standard deviaiton = 1)\n
                MU: Measurement uncertainty
                """
                st.caption("""
                ResultT: True concentration of the analyte,
                ResultM: Measured (Simulated) concentration of an analyte,
                n(0,1): A random number generated with normal distribution (mean = 0, standard deviaiton = 1),
                MU: Measurement uncertainty
                """)
                st.markdown("""        
                Then, the aggreement between original results and measured results (results with MU) were assesed.
                
                ##### **The simulation process was performed in three steps as follows:**
            
                - Step 1- Categorization of true concentration of the analyte according to its clinical decision limits.
                - Step 2- Measured (simulated) result generation
                - Step 3- Recategorization of measured (simulated) concentration (as mentioned in step 1)
                
                ##### **Calculation Process**
                
                Aggreement level between true concentration and measured (simulated) concentration of the analyte were calculated. 
                Minimum, desirable, and optimal analytical performance specifications were determined according to the aggreement thresholds that entered previously.
                
                """)
                st.markdown('---')
                
# converter function for cutting bins
def count_decimal_places(number):
    decimal_number = Decimal(str(number)) 
    if decimal_number % 1 != 0:
        num_decimal_places = len(str(decimal_number).split('.')[1]) 
    else:
        num_decimal_places = 1 
    return num_decimal_places
                
          
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
                names = [f'<{cdl_1}', f'≥{cdl_1}']
                value = [1, 2]

            elif number_CDL == 2:
                bins = [0, cdl_1-0.000001, cdl_2, np.inf]
                names = [f'<{cdl_1}', f'{cdl_1}-{cdl_2}' ,f'>{cdl_2}']
                value = [1, 2, 3]

            elif number_CDL == 3:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, np.inf]
                cdl22 =(cdl2*(10**(count_decimal_places(cdl2)))+1) / 10**(count_decimal_places(cdl2))

                names = [f'<{cdl_1}', f'{cdl_1}-{cdl_2}', f'{cdl_22}-{cdl_3}' ,f'>{cdl_3}']
                value = [1, 2, 3, 4]

            elif number_CDL == 4:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, np.inf]
                names = [f'<{cdl_1}', f'{cdl_1}-{cdl_2}', f'{cdl_2}-{cdl_3}' ,f'{cdl_3}-{cdl_4}',f'>{cdl_4}']
                value = [1, 2, 3, 4, 5]

            elif number_CDL == 5:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, np.inf]
                names = [f'<{cdl_1}', f'{cdl_1}-{cdl_2}', f'{cdl_2}-{cdl_3}' ,f'{cdl_3}-{cdl_4}',f'{cdl_4}-{cdl_5}',f'>{cdl_5}']
                value = [1, 2, 3, 4, 5, 6]

            elif number_CDL == 6:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, cdl_6, np.inf]
                names = [f'<{cdl_1}', f'{cdl_1}-{cdl_2}', f'{cdl_2}-{cdl_3}' ,f'{cdl_3}-{cdl_4}',f'{cdl_4}-{cdl_5}',f'{cdl_5}-{cdl_6}',f'>{cdl_6}']
                value = [1, 2, 3, 4, 5, 6, 7]

            elif number_CDL == 7:
                bins = [0, cdl_1-0.000001, cdl_2, cdl_3, cdl_4, cdl_5, cdl_6, cdl_7, np.inf]
                names = [f'<{cdl_1}', f'{cdl_1}-{cdl_2}', f'{cdl_2}-{cdl_3}' ,f'{cdl_3}-{cdl_4}',f'{cdl_4}-{cdl_5}',f'{cdl_5}-{cdl_6}',f'{cdl_6}-{cdl_7}',f'>{cdl_7}']
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
                st.markdown('###### **:green[Histogram of the original data]**')
                fig = plt.figure(figsize=(18, 8))
                sns.histplot(data=analyte_last_df, x=column_name, kde=False, hue="Analyte_category",
                                    discrete= False)
                plt.xlim(analyte_last_df[column_name].quantile(.001), analyte_last_df[column_name].quantile(.999))
                st.pyplot(fig)
                st.markdown('###### **:green[Density plot of the original data]**')
                fig2 = plt.figure(figsize=(18, 8))
                sns.kdeplot(data=analyte_last_df, x=column_name, hue="Analyte_category", multiple="stack", alpha=.5,common_grid=True, fill=True, linewidth=0.1)
                plt.xlim(analyte_last_df[column_name].quantile(.001), analyte_last_df[column_name].quantile(.999))
                st.pyplot(fig2)
            
            # ------------------------------------------------------------------------------------
            placeholder.success('**Simulation**', icon ="🔄")
            ## SIMULATION CODES
            
            # Selection of original data and original data category
            od = analyte_last_df[column_name] # original data
            o_cat_n = analyte_last_df['cat_n'] # original data category
            
            error_l = [] # MU list

            accuracy_score_l = [] # accuracy score list , agreement
            sub_accuracy_score_l = [] # sub_accuracy score (sub group accuracy)

            n_data = len(od) # sample size
            np.random.seed(2) # seed for reproducible results
            imprec_data_raw = np.random.normal(0, 1, n_data)
            imprec_data_raw = pd.Series(imprec_data_raw)
   
            # MU simulation
            for e in np.arange(0,0.8,0.001): # CV constant
                    
                n_cat_n = []
                o_cat_n = list(o_cat_n)   
                e_CVA=e # MU error
                y_od = od + od*imprec_data_raw*e_CVA  # MU applied
                nd = y_od 
                nd_cat= pd.cut(nd, bins, labels=names) # Categorization of the new data
                nd_cat_n = nd_cat.replace(to_replace=names,
                value=value, inplace=False)
                n_cat_n.append(nd_cat_n)
                n_cat_n = [item for sublist in n_cat_n for item in sublist]
                n_cat_n = pd.Series(n_cat_n)
                n_cat_n = n_cat_n.fillna(1)
                error_l.append(e) # MU rate save
                accuracy = accuracy_score(o_cat_n, n_cat_n) # Accuracy (Overall Agreement)
                accuracy_score_l.append(accuracy) # accuracy save
                matrix = confusion_matrix(o_cat_n, n_cat_n) # Subclass Accuracy
                matrix_2 = matrix.diagonal()/matrix.sum(axis=1)
                sub_accuracy_score_l.append(matrix_2) # Subclass Accuracy save
                    
            # sub_accuracy score data frame
            sub_accuracy_df = pd.DataFrame(sub_accuracy_score_l)
            sub_accuracy_df.columns = names

            # list to series conversion
            error_l = pd.Series(error_l) 
            accuracy_score_l = pd.Series(accuracy_score_l)

            # tables comprised of all data
            error_kappa_table = pd.concat([error_l, accuracy_score_l], axis = 1)
            error_kappa_table.columns = ["Measurement Uncertainty", "Agreement"]
            error_kappa_table_2 = pd.concat([error_kappa_table, sub_accuracy_df], axis = 1)
            
            # reset index
            error_kappa_table_2 = error_kappa_table_2.reset_index() # reset index
            
            # ------------------------------------------------------------------------------------
            
            # Disagreement based calculation
            bins_k = [0, min_agg_threshold/100-0.0000000001, des_agg_threshold/100-0.0000000001, opt_agg_threshold/100-0.0000000001, np.inf]
            names_k = [f'<{min_agg_threshold}',f'≥{min_agg_threshold}%', f'≥{des_agg_threshold}%', f'≥{opt_agg_threshold}%']
            cat_Agree= pd.cut(error_kappa_table_2['Agreement'], bins_k, labels=names_k)
            cat_Agree_df= pd.DataFrame(cat_Agree)
            
            # Concat dfs
            cat_Agree_df = cat_Agree_df.rename(columns={"Agreement": "Agreement Category"})

            # Outcome full
            error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)
            error_kappa_table_3 = error_kappa_table_2_v2
            
            #-------------------------------------------------------------------------------------------
            placeholder.success('**Visualization and Calculation**',icon="📉")
            ## Visiualisation of Data
            ### Contour Plot
            # percentage unit conversion by multiplying 100
            error_kappa_table_2_v3 = error_kappa_table_2_v2[['Measurement Uncertainty','Agreement']].apply(lambda x: x*100) # percentage unit conversion
            error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
            
            with tab3:
                # contour plot
                st.write("##### **:blue[APS for MU based on overall agreement]**")
                
                fig3 = plt.figure(figsize=(10, 5))
                palette = ['tab:red', 'xkcd:azure', 'xkcd:turquoise', 'xkcd:lime']
                palette2 = ['tab:red', 'xkcd:azure', 'xkcd:turquoise', 'xkcd:lime']
                sns.kdeplot(data=error_kappa_table_2_v3, x="Agreement",y = "Measurement Uncertainty"
                            , hue = 'Agreement Category', fill=True
                            ,palette=palette, common_norm=False,levels=30, alpha=0,warn_singular=False, bw_adjust=0.5
                            ,bw_method='scott',weights='Agreement')
                sns.scatterplot(data=error_kappa_table_2_v3, x="Agreement",y = "Measurement Uncertainty", hue = 'Agreement Category' ,palette=palette2)

                limit_xn = min_agg_threshold-5
                limit_xp = 100
                ylim_v1 = error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement']>=min_agg_threshold-5]['Measurement Uncertainty'].max()

                # min, desirable, optimal MU limits
                ylim_v2 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement']>=min_agg_threshold]['Measurement Uncertainty'].max(), 1)
                ylim_v3 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement']>=des_agg_threshold]['Measurement Uncertainty'].max(), 1)
                ylim_v4 = round(error_kappa_table_2_v3[error_kappa_table_2_v3['Agreement']>=opt_agg_threshold]['Measurement Uncertainty'].max(), 1)

                # horizontal lines percentiles 90, 95 99
                plt.axvline(x=min_agg_threshold, color= 'red', alpha = 0.4, linestyle="--",linewidth=1.5) 
                plt.text(min_agg_threshold+0.1,ylim_v1,f'{min_agg_threshold}% Agreement',rotation=0,color= 'black')
                plt.text(min_agg_threshold+0.1,ylim_v2+0.3,str(ylim_v2),rotation=0,color= 'black',fontstyle="oblique")

                plt.axvline(x=des_agg_threshold, color= 'orange', alpha = 0.8,linestyle="--",linewidth=1.5) 
                plt.text(des_agg_threshold+0.1,ylim_v1,f'{des_agg_threshold}% Agreement',rotation=0,color= 'black')
                plt.text(des_agg_threshold+0.1,ylim_v3+0.3,str(ylim_v3),rotation=0,color= 'black',fontstyle="oblique")

                plt.axvline(x=opt_agg_threshold, color= 'green', alpha = 0.4,linestyle="--",linewidth=1.5) 
                plt.text(opt_agg_threshold+0.1,ylim_v1,f'{opt_agg_threshold}% Agreement',rotation=0,color= 'black')
                plt.text(opt_agg_threshold+0.1,ylim_v4+0.3,str(ylim_v4),rotation=0,color= 'black',fontstyle="oblique")

                plt.xlim(limit_xn, limit_xp )
                plt.ylim(0, ylim_v1+1)
                plt.xlabel("Overall Agreement")

                plt.legend(loc = 'lower left', title = 'Agreement Category')
                # Text info of APS
                col22, col33 = st.columns([1.1, 4])
                col22.write(' ')
                col22.markdown(f"""
                            | APS level | MU |
                            | ----------- | ----------- |
                            | :red[Minimum] | {ylim_v2} |
                            | :orange[Desirable] | {ylim_v3} |
                            | :green[Optimal] | {ylim_v4} |
                            """)
                col33.pyplot(fig3)
            
            with tab4:
                st.write('##### **:blue[APS for based on sublevel aggreement]**')
                for i in names:
                    
                    st.markdown(f'###### **:blue[APS based on sublevel: {i}]**')
                    level_name = i
                    
                    cat_Agree= pd.cut(error_kappa_table_2[level_name], bins_k, labels=names_k)
                    cat_Agree_df= pd.DataFrame(cat_Agree)
                    
                    # concat dfs
                    cat_Agree_df = cat_Agree_df.rename(columns={level_name: "Agreement Category"})

                    # Outcome full
                    error_kappa_table_2_v2 = pd.concat([error_kappa_table_2, cat_Agree_df],axis = 1)
                    
                    # percentage unit conversion by multiplying 100
                    error_kappa_table_2_v3 = error_kappa_table_2_v2[['Measurement Uncertainty',level_name]].apply(lambda x: x*100) # percentage unit conversion
                    error_kappa_table_2_v3 = pd.concat([error_kappa_table_2_v3, error_kappa_table_2_v2[['Agreement Category']]], axis = 1) # concat 
                                                    
                    # contour plot
                    fig4 = plt.figure(figsize=(10, 5))
                    palette = ['tab:red', 'xkcd:azure', 'xkcd:turquoise', 'xkcd:lime']
                    palette2 = ['tab:red', 'xkcd:azure', 'xkcd:turquoise', 'xkcd:lime']
                    sns.kdeplot(data=error_kappa_table_2_v3, x=level_name,y = "Measurement Uncertainty"
                                , hue = 'Agreement Category', fill=True
                                ,palette=palette, common_norm=False,levels=30, alpha=0,warn_singular=False, bw_adjust=0.5
                                ,bw_method='scott',weights=level_name)
                    sns.scatterplot(data=error_kappa_table_2_v3, x=level_name ,y = "Measurement Uncertainty", hue = 'Agreement Category' ,palette=palette2)
                    limit_xn = min_agg_threshold-5
                    limit_xp = 100
                    ylim_v1 = error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=min_agg_threshold-5]['Measurement Uncertainty'].max()
                    
                    # min, desirable, optimal MU limits
                    ylim_v2 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=min_agg_threshold]['Measurement Uncertainty'].max(), 1)
                    ylim_v3 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=des_agg_threshold]['Measurement Uncertainty'].max(), 1)
                    ylim_v4 = round(error_kappa_table_2_v3[error_kappa_table_2_v3[level_name]>=opt_agg_threshold]['Measurement Uncertainty'].max(), 1)

                    # horizontal lines percentiles 90, 95 99
                    plt.axvline(x=min_agg_threshold, color= 'red', alpha = 0.4, linestyle="--",linewidth=1.5) 
                    plt.text(min_agg_threshold+0.1,ylim_v1,f'{min_agg_threshold}% Agreement',rotation=0,color= 'black')
                    plt.text(min_agg_threshold+0.1,ylim_v2+0.3,str(ylim_v2),rotation=0,color= 'black',fontstyle="oblique")

                    plt.axvline(x=des_agg_threshold, color= 'orange', alpha = 0.8,linestyle="--",linewidth=1.5) 
                    plt.text(des_agg_threshold+0.1,ylim_v1,f'{des_agg_threshold}% Agreement',rotation=0,color= 'black')
                    plt.text(des_agg_threshold+0.1,ylim_v3+0.3,str(ylim_v3),rotation=0,color= 'black',fontstyle="oblique")

                    plt.axvline(x=opt_agg_threshold, color= 'green', alpha = 0.4,linestyle="--",linewidth=1.5) 
                    plt.text(opt_agg_threshold+0.1,ylim_v1,f'{opt_agg_threshold}% Agreement',rotation=0,color= 'black')
                    plt.text(opt_agg_threshold+0.1,ylim_v4+0.3,str(ylim_v4),rotation=0,color= 'black',fontstyle="oblique")

                    plt.xlim(limit_xn, limit_xp )
                    plt.ylim(0, ylim_v1+1)
                    plt.xlabel("Agreement of the level "+level_name)
                    plt.legend(loc = 'lower left', title = 'Agreement Category')
                    # Text info of APS
                    col22, col33 = st.columns([1.1, 4])
                    col22.write(' ')
                    col22.markdown(f"""
                                | APS level | MU |
                                | ----------- | ----------- |
                                | :red[Minimum] | {ylim_v2} |
                                | :orange[Desirable] | {ylim_v3} |
                                | :green[Optimal] | {ylim_v4} |            
                                """)
                    col33.pyplot(fig4)
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
        
