from langchain import PromptTemplate, LLMChain
from langchain.chat_models import ChatOpenAI

# Initialize OpenAI LLM
def initialize_llm():
    """
    Initialize the OpenAI LLM model for sentiment analysis.
    """
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    batch_prompt_template = """
    You are tasked with analyzing a batch of public comments on a YouTube video.

    The video's details are as follows:
    
    - **Title**: "{title}"
    - **Description**: "{description}"
    - **Channel**: "{channelTitle}"
    - **Tags**: "{tags}"

    The analysis must be entirely objective, based only on the content of the comments provided, and must avoid any subjective judgments or external assumptions.

    The comments are as follows:
    {text}
    
    Based on the batch of comments, provide the following insights:

    1. **Sentiment Analysis**:
        - Classify the sentiment as **positive**, **negative**, **neutral**, or **unrelated**.
        - Provide a percentage breakdown for each category, based solely on the tone and content of the comments.

    2. **Recurring Feedback**:
        - Identify recurring themes in the comments, such as praise, criticism, complaints, or common topics of discussion.
        - Quote specific examples to support the identified themes.

    3. **Key Observations**:
        - Highlight notable trends, patterns, or unique reactions in the comments (e.g., mentions of specific aspects of the video like performance, visuals, production, or concept).

    4. **Overall Sentiment**:
        - Summarize the overall perception of the video based exclusively on the comments.
        - Clearly state whether the comments reflect a predominantly positive, negative, or mixed reaction, with reasoning based on the data.

    Remember: Your analysis must rely entirely on the provided comments without making assumptions or including subjective interpretations.
    """

    prompt = PromptTemplate(input_variables=['title', 'description', 'channelTitle', 'tags', 'text'], template=batch_prompt_template)
    return LLMChain(llm=llm, prompt=prompt)

def extract_sentiment_insights_no_batch(llm_chain, comments, video_info):
    """
    Analyze sentiment insights for all comments and a video's metadata.
    """
    comments_str = "\n".join([f"Comment {i+1}: {comment}" for i, comment in enumerate(comments)])
    
    # Handle missing fields for tags or channelTitle
    tags = ", ".join(video_info.get('tags', [])) if video_info.get('tags') else "No tags provided"
    channel_title = video_info.get('channelTitle', "Unknown channel")

    response = llm_chain.run({
        "text": comments_str,
        "title": video_info['title'],
        "description": video_info.get('description', "No description available."),
        "channelTitle": channel_title,
        "tags": tags
    })
    return response.strip()
