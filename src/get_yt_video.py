from youtubesearchpython import VideosSearch

# -------------------------------------------------------
# 1️⃣ Get a single YouTube video link (best match)
# -------------------------------------------------------
def get_yt_video_link(query: str):
    videos_search = VideosSearch(query, limit=1)
    result = videos_search.result()

    if result.get("result"):
        return result["result"][0]["link"]
    return None


# -------------------------------------------------------
# 2️⃣ Get multiple YouTube video titles + links
# -------------------------------------------------------
def get_yt_videos(query: str, limit: int = 5):
    videos_search = VideosSearch(query, limit=limit)
    results = videos_search.result().get("result", [])

    video_titles = [video["title"] for video in results]
    video_links = [video["link"] for video in results]

    return video_titles, video_links
