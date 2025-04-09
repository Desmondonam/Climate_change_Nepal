import streamlit as st
import sys

sys.path.append("C:/Users/Admin/Desktop/Omdena/Nepal_CBW_Batch2/NIC2_A/Week 12")
from data_utils import prepare_features
from models import split_data, train_model, evaluate_model, save_model
from visualizations import plot_actual_vs_predicted

def show(df):
    """
    Display the model training page
    """
    st.header("Model Training")

    # Prepare the features and the target
    X, y = prepare_features(df)

    # Train test split
    test_size = st.slider("Test data size (%)", 10.0, 40.0, 20.0 / 100.0)
    X_train, X_test, y_train, y_test, = split_data(X, y, test_size)

    st.write(f"Training Data: {len(X_train)} Samples")
    st.write(f"Test data : {len(X_test)} samples")

    # Model selection
    model_type = st.selectbox("Select Model Type", ["Linear Regression", "Random Forest"])

    # train model Button
    if st.button('Train Model'):
        with st.spinner("Trainig in progress..."):
            # Train the model
            model = train_model(X_train, y_train, model_type)

            # Evaluate the model
            metrics = evaluate_model(model, X_train, y_train, X_test, y_test)

            # Display the metrics
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Training Metrics")
                st.write(f"RMSE: {metrics['test_rmse']:.2f} C")
                st.write(f"R2: {metrics['train_r2']:.4f}")

            with col2:
                st.subheader("Testing Metrics")
                st.write(f"RMSE: {metrics['test_rmse']:.2f} C")
                st.write(f"R2: {metrics['test_r2']:.4f}")

            # Plot the actual vs predicted 
            st.subheader("Actual vs Predicted (Test Dta)")
            fig = plot_actual_vs_predicted(metrics['y_test'], metrics['y_pred_test'])
            st.pyplot


            # Save the model
            save_model(model)


            st.success("Model Trained and saved successfully!")
            st.session_state['model'] = model
            st.session_state['model_type'] = model_type