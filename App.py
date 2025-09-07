import streamlit as st
import plotly.express as px
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import plotly.graph_objs as go
import seaborn as sns

salary_data_model = ""
income_dataset_model = ""
advertising_dataset_model = ""
salary_data = ""
income_dataset = ""
advertising_dataset = ""

def load_models():
    global salary_data_model
    global income_dataset_model
    global advertising_dataset_model
    salary_data_model = joblib.load("salary_model.pkl")
    income_dataset_model = joblib.load("income_model.pkl")
    advertising_dataset_model = joblib.load("advertising_model.pkl")
    
def load_datasets():
    global salary_data
    global income_dataset
    global advertising_dataset
    salary_data = pd.read_csv("salary_data.csv")
    income_dataset = pd.read_csv("income_dataset.csv")
    advertising_dataset = pd.read_csv("advertising-dataset.csv")
    
# Fake models (replace with your trained ones)
def salary_predict(x):
    print(salary_data_model)
    result = salary_data_model.predict([[x]])[0]
    return result

def income_predict(x, y):
    result = income_dataset_model.predict([[x, y]])[0]
    return result

def advertising_predict(x, y):
    result = advertising_dataset_model.predict([[x, y]])[0]
    return result


# --- App Layout ---
st.title("Cataluña's Model Playground 🚀")


load_models()
load_datasets()

# Tabs
tab1, tab2, tab3 = st.tabs(["Salary Prediction Model", "Salary Prediction Model", "Advertising Prediction Model"])

with tab1:
    st.subheader("Salary Prediction Model")
    x = st.number_input("Enter Years of Experience", value=1.0)
    if st.button("Predict", key="m1"):
        result = salary_predict(x)
        st.success(f"Prediction: {result}")

    salary_data_X = np.array(salary_data["YearsExperience"])
    salary_data_X = salary_data_X.reshape(-1, 1)
    salary_data_y = np.array(salary_data["Salary"])
    salary_X_line = np.linspace(salary_data_X.min(), salary_data_X.max(), 100).reshape(-1, 1)
    salary_y_line = salary_data_model.predict(salary_X_line)
    fig, ax = plt.subplots()
    ax.scatter(salary_data_X, salary_data_y, color="blue", label="Data points")
    ax.plot(salary_X_line, salary_y_line, color="red", label="Regression line")
    ax.set_xlabel("Years of Experience")
    ax.set_ylabel("Salary")
    ax.set_title("Years of Experience vs Salary")
    ax.legend()
    
    st.pyplot(fig)

with tab2:
    st.subheader("Salary Prediction Model")
    x = st.number_input("Enter Age", value=1)
    y = st.number_input("Enter Experience", value=1)
    if st.button("Predict", key="m2"):
        result = income_predict(x, y)
        st.success(f"Prediction: {result}")

    x_surf, y_surf = np.meshgrid(
    np.linspace(min(income_dataset["experience"]), max(income_dataset["experience"]), 50),
    np.linspace(min(income_dataset["age"]), max(income_dataset["age"]), 50)
    )
    z_surf = income_dataset_model.predict(
        np.column_stack((y_surf.ravel(), x_surf.ravel()))
    ).reshape(x_surf.shape)

    # Plot regression plane
    surface = go.Surface(
        x=x_surf, y=y_surf, z=z_surf,
        colorscale="Reds", opacity=0.5
    )

    # Scatter data
    scatter = go.Scatter3d(
        x=income_dataset["experience"],
        y=income_dataset["age"],
        z=income_dataset["income"],
        mode="markers",
        marker=dict(size=5, color="blue")
    )
    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title="Experience",
            yaxis_title="Age",
            zaxis_title="Income"
        ),
        title="Polynomial Regression"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    st.subheader("Advertising Prediction Model")
    x = st.number_input("Enter TV", value=1.0)
    y = st.number_input("Enter Radio", value=1.0)
    if st.button("Predict", key="m3"):
        result = advertising_predict(x, y)
        st.success(f"Prediction: {result}")

    corr = advertising_dataset.corr()
    fig, ax = plt.subplots()
    # ax.set_figure(figsize=(6, 4))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap")
    st.pyplot(fig)
    
    x_surf, y_surf = np.meshgrid(
    np.linspace(min(income_dataset["experience"]), max(income_dataset["experience"]), 50),
    np.linspace(min(income_dataset["age"]), max(income_dataset["age"]), 50)
    )
    z_surf = income_dataset_model.predict(
        np.column_stack((y_surf.ravel(), x_surf.ravel()))
    ).reshape(x_surf.shape)

    # Plot regression plane
    surface = go.Surface(
        x=x_surf, y=y_surf, z=z_surf,
        colorscale="Reds", opacity=0.5
    )

    # Scatter data
    scatter = go.Scatter3d(
        x=income_dataset["experience"],
        y=income_dataset["age"],
        z=income_dataset["income"],
        mode="markers",
        marker=dict(size=5, color="blue")
    )

    fig = go.Figure(data=[surface, scatter])
    fig.update_layout(
        scene=dict(
            xaxis_title="Experience",
            yaxis_title="Age",
            zaxis_title="Income"
        ),
        title="Polynomial Regression"
    )
    
    st.plotly_chart(fig, use_container_width=True, key="chart1")