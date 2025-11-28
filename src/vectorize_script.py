from dotenv import load_dotenv
from vectorize_book import vectorize_all_pdfs

load_dotenv()

if __name__ == "__main__":
    vectorize_all_pdfs()
