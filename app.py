import time

import streamlit as st

from query_builder import query_suggestion


# Streamlit app interface
def main():
    st.title("Query Suggestion")
    user_input = st.text_area("Enter your request:", height=100)

    if st.button("Submit"):
        # Record start time
        start_time = time.time()
        result = query_suggestion(user_input)
        # Record end time
        end_time = time.time()
        processing_time_ms = (end_time - start_time) * 1000

        # Display the results
        st.subheader("Query suggests for you:")
        for i, item in enumerate(result, 1):
            st.write(f"{i}. {item}")

        # Display processing time
        st.write(f"Processing Time: {processing_time_ms:.2f} ms")


if __name__ == "__main__":
    main()
