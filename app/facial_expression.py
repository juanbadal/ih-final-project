import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def main():
    st.title("CSV File Plotter")

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        df = pd.read_csv(uploaded_file)

        # Display the raw data
        st.subheader("Raw Data")
        st.write(df)

        # Plot the data
        st.subheader("Data Visualization")
        plot_data(df)

def plot_data(df):
    # Plot using Matplotlib
    fig, ax = plt.subplots()
    ax.plot(df.index, df.values)
    ax.set_xlabel("Index")
    ax.set_ylabel("Values")
    st.pyplot(fig)

if __name__ == "__main__":
    main()