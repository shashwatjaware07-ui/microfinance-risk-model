# We are bringing in a tool to build our webpage easily.
import streamlit as st

# We are bringing in a tool to handle tables of data.
import pandas as pd

# We are bringing in a tool to do math with numbers.
import numpy as np

# We are bringing in a tool to load our saved files.
import joblib

# We are bringing in a tool to draw pretty charts.
import plotly.graph_objects as go

# We are setting up how our webpage looks, making it clean and wide.
st.set_page_config(page_title="Credit Risk Assessment", layout="wide")

# We are making a rule to remember our loaded tools so we don't load them again and again.
# We are creating a little helper that loads our saved files.
@st.cache_resource
def load_saved_files():
    # We are loading the brain that guesses the risk.
    saved_model = joblib.load('credit_risk_model.pkl')
    # We are loading the tool that shrinks our numbers to the right size.
    saved_scaler = joblib.load('credit_risk_scaler.pkl')
    # We are giving back the brain and the shrinking tool.
    return saved_model, saved_scaler

# We are using our helper to get the brain and the shrinking tool.
saved_model, saved_scaler = load_saved_files()

# We are writing a professional title at the top of our page.
st.title("Credit Risk Assessment Portal")

# We are adding a small description underneath the title.
st.markdown("Evaluate loan applications by assessing the default risk probability.")

# We are adding a thin line to separate the top part from the rest.
st.markdown("---")

# We are writing a title for the side menu.
st.sidebar.header("Applicant Financial Details")

# We are asking the user for their total income.
user_income = st.sidebar.number_input("Total Income (₹)", min_value=0.0, value=50000.0, step=1000.0)

# We are asking the user for the amount they want to borrow.
user_credit = st.sidebar.number_input("Credit Amount (₹)", min_value=0.0, value=150000.0, step=1000.0)

# We are asking the user for their yearly payment amount.
user_annuity = st.sidebar.number_input("Annuity (₹)", min_value=0.0, value=7500.0, step=500.0)

# We are asking the user for the price of the goods they are buying.
user_goods_price = st.sidebar.number_input("Goods Price (₹)", min_value=0.0, value=150000.0, step=1000.0)

# We are adding some empty space in the side menu.
st.sidebar.write("")

# We are creating a big button for the user to click.
calculate_button = st.sidebar.button("Calculate Default Risk", type="primary", use_container_width=True)

# We are checking if the user clicked the big button.
if calculate_button:
    # We are creating an empty table filled with zeros for all the things the brain needs.
    user_data_table = pd.DataFrame(0, index=[0], columns=saved_model.feature_names_in_)
    
    # We are finding out which columns in the table need to be numbers.
    number_columns = saved_scaler.feature_names_in_
    
    # We are filling the empty number columns with average numbers so the brain doesn't get confused.
    user_data_table[number_columns] = saved_scaler.mean_
    
    # We are putting the user's income into our table, shrunk by a special math trick.
    user_data_table.loc[0, 'AMT_INCOME_TOTAL'] = np.log1p(user_income)
    
    # We are putting the user's credit into our table, shrunk by a special math trick.
    user_data_table.loc[0, 'AMT_CREDIT'] = np.log1p(user_credit)
    
    # We are putting the user's annuity into our table, shrunk by a special math trick.
    user_data_table.loc[0, 'AMT_ANNUITY'] = np.log1p(user_annuity)
    
    # We are putting the user's goods price into our table, shrunk by a special math trick.
    user_data_table.loc[0, 'AMT_GOODS_PRICE'] = np.log1p(user_goods_price)
    
    # We are using our shrinking tool to adjust all the numbers exactly how the brain wants them.
    user_data_table[number_columns] = saved_scaler.transform(user_data_table[number_columns])
    
    # We are asking the brain to guess the chance that the user will not pay back the money.
    guess_results = saved_model.predict_proba(user_data_table)[0]
    
    # We are grabbing just the danger chance from the results.
    danger_chance = guess_results[1]
    
    # We are turning the tiny decimal chance into a big percentage out of 100.
    danger_percentage = danger_chance * 100
    
    # We are splitting our webpage into two even columns to put our charts side by side.
    left_column, right_column = st.columns(2)
    
    # We are going to put things in the left column first.
    with left_column:
        # We are writing a small, clean title for our speedometer chart.
        st.subheader("Risk Analysis")
        
        # We are creating a new picture that looks like a speedometer.
        speedometer_chart = go.Figure(
            # We are telling the picture to be a gauge, which is a speedometer.
            go.Indicator(
                # We are choosing to show a gauge and a big number in the middle.
                mode="gauge+number",
                # We are giving the picture the exact percentage we found.
                value=danger_percentage,
                # We are adding a small piece of text under the big number to say it is a percentage.
                number={'suffix': "%"},
                # We are adding a title inside the picture.
                title={'text': "Probability of Default"},
                # We are setting up exactly how the speedometer looks.
                gauge={
                    # We are setting the lowest and highest numbers on the speedometer.
                    'axis': {'range': [0, 100]},
                    # We are creating colored sections on the speedometer.
                    'steps': [
                        # We are coloring the safe zone green from zero to thirty-five.
                        {'range': [0, 35], 'color': "rgba(144, 238, 144, 0.4)"},
                        # We are coloring the danger zone red from thirty-five to one hundred.
                        {'range': [35, 100], 'color': "rgba(255, 99, 71, 0.6)"}
                    ],
                    # We are changing the color of the thick bar that shows the score.
                    'bar': {'color': "black"}
                }
            )
        )
        
        # We are fixing the size of the chart so it fits perfectly on the screen.
        speedometer_chart.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
        
        # We are showing the speedometer picture on the webpage.
        st.plotly_chart(speedometer_chart, use_container_width=True)
        
        # We are checking if the percentage is safely under our limit of thirty-five.
        if danger_percentage < 35:
            # We are showing a green message saying the loan is approved.
            st.success("[APPROVED] The applicant's default risk is within acceptable limits.")
        # We are doing something else if the percentage is too high.
        else:
            # We are showing a red message saying the loan is denied.
            st.error("[DENIED] The applicant's default risk is too high.")
            
    # We are going to put things in the right column now.
    with right_column:
        # We are writing a small, clean title for our bar chart.
        st.subheader("Financial Overview")
        
        # We are creating a new picture that looks like a bar chart.
        bar_chart = go.Figure(
            # We are making a list of bars to draw.
            data=[
                # We are adding the first bar for the user's income and painting it blue.
                go.Bar(name='Total Income', x=['Income'], y=[user_income], marker_color='#1f77b4'),
                # We are adding the second bar for the user's credit request and painting it orange.
                go.Bar(name='Credit Amount', x=['Credit Amount'], y=[user_credit], marker_color='#ff7f0e')
            ]
        )
        
        # We are adding a title and labeling the side of the bar chart.
        bar_chart.update_layout(
            # We are changing the title of the chart.
            title="Leverage Ratio Comparison",
            # We are writing a label for the numbers on the left side.
            yaxis_title="Amount in Rupees (₹)",
            # We are fixing the size of the chart so it matches the other one.
            height=350,
            # We are making the chart look very clean and simple without extra borders.
            template="plotly_white",
            # We are making the edges look nice.
            margin=dict(l=20, r=20, t=50, b=20)
        )
        
        # We are showing the bar chart picture on the webpage.
        st.plotly_chart(bar_chart, use_container_width=True)
