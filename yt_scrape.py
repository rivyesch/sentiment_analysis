from urllib.parse import urlparse, parse_qs
import pandas as pd

def get_id(url):
    """
    Extract the video ID from a YouTube URL.
    """
    u_pars = urlparse(url)
    quer_v = parse_qs(u_pars.query).get('v')
    if quer_v:
        return quer_v[0]
    pth = u_pars.path.split('/')
    if pth:
        return pth[-1]
    
def get_video_details(youtube, video_id):
    """
    Get details of a YouTube video (title, description, channel, etc.).
    """
    try:
        request = youtube.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()

        if not response['items']:
            return {"error": "Video not found or unavailable."}

        video = response['items'][0]
        stats_to_keep = {
            'snippet': ['channelTitle', 'title', 'description', 'tags', 'publishedAt'],
            'statistics': ['viewCount', 'likeCount', 'favoriteCount', 'commentCount'],
            'contentDetails': ['duration', 'definition', 'caption']
        }

        video_info = {"video_id": video_id}
        for key, fields in stats_to_keep.items():
            for field in fields:
                video_info[field] = video[key].get(field, None)

        return video_info

    except Exception as e:
        return {"error": f"An error occurred: {e}"}

def get_comments_in_videos(youtube, video_id, max_comments=250):
    """
    Retrieve top-level comments from a YouTube video.
    """
    all_comments = []
    token = None

    while len(all_comments) < max_comments:
        try:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                textFormat="plainText",
                pageToken=token
            )
            response = request.execute()

            for item in response['items']:
                comment = item['snippet']['topLevelComment']['snippet']['textDisplay']
                all_comments.append(comment)
                if len(all_comments) >= max_comments:
                    break

            token = response.get('nextPageToken', None)
            if not token:
                break

        except Exception as e:
            print(f"Could not get comments for video {video_id}. Error: {e}")
            break

    return pd.DataFrame({"comment": all_comments})