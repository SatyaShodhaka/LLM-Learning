import requests
from bs4 import BeautifulSoup
import justext


def scrape_webpage(url):
    # Send a GET request to the URL
    headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
    extracted_text = ""
    try:
        response = requests.get(url, headers=headers, timeout=10)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            paragraphs = justext.justext(response.text, justext.get_stoplist("English"))
            for paragraph in paragraphs:
                if not paragraph.is_boilerplate and not paragraph.is_heading:
                    extracted_text += "\n" + paragraph.text
            return extracted_text
        else:
            print(f"Failed to retrieve the webpage. Status code: {response.status_code}")
            return None
    except:
        return None



if __name__ == "__main__":
    # Example usage:
    url_to_scrape = 'https://en.wikipedia.org/wiki/Chancellor_of_Germany#:~:text=Incumbent%0AOlaf%20Scholz'
    extracted_text = scrape_webpage(url_to_scrape)

    # print(extracted_text)


    # import trafilatura
    # downloaded = trafilatura.fetch_url(url_to_scrape)
    # text = trafilatura.html2txt(downloaded)

    # paragraphs = justext.justext(text, justext.get_stoplist("English"))
    # for paragraph in paragraphs:
    #     if not paragraph.is_heading:
    #         extracted_text += "\n" + paragraph.text
    
    # print("*****************FINAL*************************************")
    # print(extracted_text)