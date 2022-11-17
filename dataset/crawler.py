from icrawler.builtin import GoogleImageCrawler
from icrawler.builtin import BingImageCrawler


def name(search_word, save_dir, max_num=10):    
    # crawler = GoogleImageCrawler(storage={"root_dir": save_dir})
    crawler = BingImageCrawler(storage={"root_dir": save_dir})
    crawler.crawl(keyword=search_word, max_num=max_num)

name("蕪", "kabu", 30)
