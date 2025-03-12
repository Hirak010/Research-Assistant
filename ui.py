import streamlit as st
import requests

# Streamlit App Configuration
st.set_page_config(page_title="LangGraph Article Writer", layout="centered")

# Define API endpoint
API_URL = "http://127.0.0.1:8000/chat"

# Streamlit UI Elements
st.title("Article Writer")
st.write("Generate articles using LangGraph and Tavily with this interface.")

# Input box for subject (with minimum height of 68 pixels)
subject = st.text_area(
    "Write the subject of the article:",
    height=100,  # Adjusted to meet minimum height requirement
    placeholder="e.g., Trends in AI for 2025"
)

# Input box for content details (with minimum height of 68 pixels)
content_details = st.text_area(
    "Enter the content details:",
    height=200,  # Adjusted to meet minimum height requirement
    placeholder="e.g., Cover AI trends, tools, and future predictions"
)

# Button to submit the request
if st.button("Generate Article"):
    if subject.strip() and content_details.strip():
        try:
            # Prepare the payload
            payload = {
                "subject": subject,
                "content_details": content_details
            }

            # Send the request to the FastAPI backend
            response = requests.post(API_URL, json=payload)

            # Handle the response
            if response.status_code == 200:
                response_data = response.json()

                # Display the generated article
                if "article" in response_data:
                    st.subheader("Generated Article")
                    st.markdown(response_data["article"])

                    # Display additional metadata
                    st.write(f"Iterations: {response_data.get('iteration_count', 'N/A')}")
                    if response_data.get("references"):
                        st.subheader("References")
                        for ref in response_data["references"]:
                            st.write(f"- {ref}")
                else:
                    st.error("No article found in the response.")
            else:
                st.error(f"Request failed with status code {response.status_code}.")
        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please provide both a subject and content details before submitting.")