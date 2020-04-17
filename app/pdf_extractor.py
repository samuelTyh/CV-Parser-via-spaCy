import re
import requests
import tika
from tika import parser


def parse_from_file(file):
    content_parsed = tika.parser.from_file(filename=file)
    text = content_parsed['content']
    content = text.encode("ascii", "ignore").decode("utf-8")
    output = re.sub(r"[\n\r\s]+", " ", content)
    return output


def parse_from_url(url):
    pdf = requests.get(url, stream=True)
    content_parsed = tika.parser.from_buffer(pdf)
    text = content_parsed['content']
    content = text.encode("ascii", "ignore").decode("utf-8")
    output = re.sub(r"[\n\r\s]+", " ", content)
    return output
