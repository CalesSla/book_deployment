import pandas as pd
from supabase import create_client, Client
import streamlit as st
import matplotlib.pyplot as plt
from datetime import datetime
from statsforecast.models import CrostonOptimized
from dateutil.relativedelta import relativedelta

@st.cache_resource
def init_connection():
    url: str = st.secrets['supabase_url']
    key: str = st.secrets['supabase_key']
    client: Client = create_client(url, key)
    return client

supabase = init_connection()

def convert_dates(date):
    return datetime.strptime(date, '%m/%d/%Y')

@st.cache_data(ttl=600)
def run_query():
    return supabase.table("car_parts_monthly_sales").select('*').execute()

# @st.cache_resource(show_spinner="Making predictions...")
def generate_forecast_df(selected_parts, df, slider_value):
    unique_id = []
    extrapolated_dates = []
    forecast_values = []
    for i in selected_parts:
        filtered_df = df[df['parts_id'] == i]
        fit_data = filtered_df['volume'].values
        model = CrostonOptimized()
        model.fit(fit_data)
        forecast = list(model.forecast(fit_data, h=slider_value)['mean'])
        forecast_values += forecast
        unique_id += [i] * slider_value
        dates = [datetime.strptime(filtered_df['date'].values[-1], '%m/%d/%Y') + relativedelta(months=x+1) for x in range(slider_value)]
        extrapolated_dates += dates
    st.session_state['results_df'] = pd.DataFrame({"unique_id": unique_id, "date": extrapolated_dates, "CrostonOptimized": forecast_values})


@st.cache_data(ttl=600)
def download_csv(df):
    csv = df.to_csv(index=False)
    return csv

@st.cache_data(ttl=600)
def create_dataframe():
    rows = run_query()
    df = pd.json_normalize(rows.data)
    df['volume'] = df['volume'].astype(int)
    return df

@st.cache_data
def plot_volume(selected_parts):
    fig, ax = plt.subplots()
    
    for i in selected_parts:
        filtered_df = df[df['parts_id'] == i]
        x = filtered_df['date']
        y = filtered_df['volume']
        ax.plot(x, y)
    ax.legend(selected_parts)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    st.pyplot(fig)

df = create_dataframe()

st.title("Forecast product demand")
st.subheader("Select a product")
selected_parts = st.multiselect("Select product ID", options = sorted(df['parts_id'].unique()))

plot_volume(selected_parts)

with st.expander("Forecast"):
    if len(selected_parts) == 0:
        st.warning("Select at least one product ID to forecast")
    else:
        slider_value = st.slider("Horizon", min_value=1, max_value=12, step=1)
        forecast_button = st.button("Forecast", type="primary", on_click=generate_forecast_df, kwargs=dict(selected_parts=selected_parts, df=df, slider_value=slider_value))

    if 'results_df' in st.session_state.keys():
        csv = download_csv(st.session_state['results_df'])
        st.download_button(label="Download predictions", data=csv, file_name='data.csv', mime='text/csv')
        
