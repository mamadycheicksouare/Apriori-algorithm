# Import required libraries
import streamlit as st
import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Title of the web app
st.title("Apriori Algorithm")


# File uploader widget for the user to upload their CSV
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

# Process the file only if it's uploaded
if uploaded_file is not None:
    # Read CSV into DataFrame
    df = pd.read_csv(uploaded_file)

    # Display a preview of the uploaded data
    st.subheader("First 5 rows of the CSV:")
    st.write(df.head())

    # List of supported metrics for filtering association rules
    metric_options = [
        "support", "confidence", "lift", "representativity", "leverage",
        "conviction", "zhangs_metric", "jaccard", "certainty", "kulczynski"
    ]

    # Define value ranges for each metric (min, max, default)
    metric_ranges = {
        "support": (0.05, 1.0, 0.15),
        "confidence": (0.05, 1.0, 0.15),
        "lift": (0.5, 10.0, 1.0),
        "representativity": (0.05, 1.0, 0.15),
        "leverage": (0.0, 1.0, 0.01),
        "conviction": (0.5, 20.0, 1.0),
        "zhangs_metric": (-1.0, 1.0, 0.0),
        "jaccard": (0.05, 1.0, 0.15),
        "certainty": (0.05, 1.0, 0.15),
        "kulczynski": (0.05, 1.0, 0.15)
    }

    # Slider for user to set the minimum support value for the Apriori algorithm
    min_support_value = st.slider("Minimum Support for Apriori", 0.05, 1.0, 0.15, step=0.01)

    # Create layout with two columns for metric selection and value threshold
    col1, col2 = st.columns([2, 3])

    with col1:
        # Select metric from the dropdown
        selected_metric = st.selectbox("Metric", metric_options)

    # Retrieve the range and default value for the selected metric
    min_val, max_val, default_val = metric_ranges[selected_metric]

    with col2:
        # Slider for setting the minimum threshold value for the selected metric
        metric_value = st.slider("Value", min_val, max_val, default_val, step=0.01)

    # Preprocess the data: transform each row into a list of items (remove NaNs)
    transactions = df.apply(lambda row: row.dropna().tolist(), axis=1).tolist()

    # Use TransactionEncoder to convert list-of-lists into a one-hot encoded DataFrame
    te = TransactionEncoder()
    te_array = te.fit(transactions).transform(transactions)
    df_encoded = pd.DataFrame(te_array, columns=te.columns_)

    # Apply Apriori algorithm to find frequent itemsets
    frequent_itemsets = apriori(df_encoded, min_support=min_support_value, use_colnames=True)

    # Sort itemsets by support in descending order
    frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False).reset_index(drop=True)

    try:
        # Generate association rules based on selected metric and threshold
        rules = association_rules(frequent_itemsets, metric=selected_metric, min_threshold=metric_value)

        # Select and format only necessary columns for display
        filtered_rules = rules[['antecedents', 'consequents', selected_metric]].copy()
        filtered_rules['antecedents'] = filtered_rules['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        filtered_rules['consequents'] = filtered_rules['consequents'].apply(lambda x: ', '.join(sorted(list(x))))

        # Display the filtered rules in a dataframe
        st.subheader("Filtered Association Rules")
        st.dataframe(filtered_rules)

        # Prepare full rule set (with all metrics) for CSV export
        rules_export = rules.copy()
        rules_export['antecedents'] = rules_export['antecedents'].apply(lambda x: ', '.join(sorted(list(x))))
        rules_export['consequents'] = rules_export['consequents'].apply(lambda x: ', '.join(sorted(list(x))))
        csv = rules_export.to_csv(index=False)

        # Provide a download button for exporting the rules
        st.download_button(
            label="Download Full Results as CSV",
            data=csv,
            file_name='apriori_results.csv',
            mime='text/csv'
        )

        # Display readable natural-language format of the rules
        st.subheader("Readable Association Rules")
        for _, row in filtered_rules.iterrows():
            st.write(f"If someone buys **{row['antecedents']}**, they are likely to also buy **{row['consequents']}**.")

    except ValueError:
        # Handle the case where no rules meet the selected threshold
        st.warning("No rules found for the selected metric and threshold.")
