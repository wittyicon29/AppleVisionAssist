from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader, YoutubeLoader
from langchain_community.document_loaders.merge import MergedDataLoader

def load_data():
    web_loader = WebBaseLoader("https://www.apple.com/apple-vision-pro/")
    loader = YoutubeLoader.from_youtube_url(
        "https://www.youtube.com/watch?v=TX9qSaGXFyg", add_video_info=True
    )
    loader_pdf = PyPDFLoader("Apple_Vision_Pro_Privacy_Overview.pdf")
    loader_all = MergedDataLoader(loaders=[web_loader, loader, loader_pdf])
    return loader_all.load()
