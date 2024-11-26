import streamlit as st
from yt_scrape import get_id, get_video_details, get_comments_in_videos
from sentiment_analysis import initialize_llm, extract_sentiment_insights_no_batch
from googleapiclient.discovery import build
import openai
# import os

# Load API keys
# from dotenv import load_dotenv
# load_dotenv()
# yt_api_key = os.getenv("YOUTUBE_API_KEY")
# openai_api_key = os.getenv("OPENAI_API_KEY")

openai.api_key = st.secrets["OPENAI_API_KEY"]
yt_api_key = st.secrets["YOUTUBE_API_KEY"]

# Initialize APIs
youtube = build('youtube', 'v3', developerKey=yt_api_key, cache_discovery=False)


# Streamlit App UI
st.title("YouTube Video Sentiment Analysis")
st.markdown("""
Enter the URL of a YouTube video, and this tool will:
- Fetch the video's title, description, and tags.
- Retrieve up to 200 top-level comments.
- Analyze the sentiment and themes in the comments.
""")

# User input: YouTube video URL
video_url = st.text_input("Enter YouTube Video URL:", "")
max_comments = st.number_input("Number of Comments to Analyze (max 250):", min_value=1, max_value=250, value=100)

if st.button("Analyze"):
    if not video_url:
        st.error("Please enter a valid YouTube video URL.")
    else:
        # Extract video ID
        video_id = get_id(video_url)
        if not video_id:
            st.error("Invalid YouTube URL.")
        else:
            with st.spinner("Fetching video details..."):
                # Fetch video details
                video_info = get_video_details(youtube, video_id)

                if "error" in video_info:
                    st.error(video_info["error"])
                else:
                    st.subheader("Video Details")
                    st.write(f"**Title:** {video_info['title']}")
                    st.write(f"**Channel:** {video_info['channelTitle']}")
                    # st.write(f"**Description:** {video_info['description']}")
                    # st.write(f"**Tags:** {', '.join(video_info['tags']) if video_info['tags'] else 'No tags available'}")

                    # Fetch comments
                    with st.spinner("Fetching comments..."):
                        comments_df = get_comments_in_videos(youtube, video_id, max_comments=max_comments)

                        if comments_df.empty:
                            st.error("No comments found or comments are disabled for this video.")
                        else:
                            st.write(f"Retrieved {len(comments_df)} comments.")
                            # st.write(comments_df)

                            # Perform sentiment analysis
                            with st.spinner("Analyzing comments..."):
                                llm_chain = initialize_llm()
                                insights = extract_sentiment_insights_no_batch(
                                    llm_chain,
                                    comments_df["comment"].tolist(),
                                    video_info
                                )

                                st.subheader("Sentiment Analysis Results")
                                st.write(insights)