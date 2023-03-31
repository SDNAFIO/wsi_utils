import urllib.request
import time
import json
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service as ChromeService
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
import os

from pyvirtualdisplay import Display
display = Display(visible=0, size=(800, 800))
display.start()
driver = webdriver.Chrome()

service = ChromeService(executable_path=ChromeDriverManager().install())

options = Options()
options.add_argument("--window-size=1920,1200")
options.add_argument("--no-sandbox")
options.add_argument("--disable-dev-shm-usage")
#options.add_argument('--headless')
options.add_argument("--remote-debugging-port=9222")
options.add_experimental_option("prefs", {
  "download.default_directory": os.getcwd()
  })

driver = webdriver.Chrome(service=service, options=options)


def every_downloads_chrome(driver):
    if not driver.current_url.startswith("chrome://downloads"):
        driver.get("chrome://downloads/")
    return driver.execute_script("""
        var items = document.querySelector('downloads-manager')
            .shadowRoot.getElementById('downloadsList').items;
        if (items.every(e => e.state === "COMPLETE"))
            return items.map(e => e.fileUrl || e.file_url);
        """)


def download():
    manifest = open('gdc_manifest_kidney.txt')
    entries = [x for x in manifest][1:]

    for idx, entry in enumerate(entries):
        entry_dat = entry.split('\t')
        name = entry_dat[1]
        target_filename = f'{name}.tar.gz'
        md5 = entry_dat[2]

        print(f'Processing: {name}')

        if os.path.exists(target_filename):
            print(f'{target_filename} already exists, skipping download')
        else:
            # Get file ID
            url = 'https://portal.gdc.cancer.gov/auth/api/v0/quick_search?query=' + name
            f = urllib.request.urlopen(url)
            res = f.read().decode('utf-8')
            js = json.loads(res)

            f_id = js['data']['query']['hits'][0]['file_id']
            print(f'\tGot id {f_id}')

            # Get actual download
            url = 'https://portal.gdc.cancer.gov/files/' + f_id
            driver.get(url)
            time.sleep(3)
            btns = driver.find_elements(By.CLASS_NAME, 'button')
            dwnload_btn = [x for x in btns if 'Download' in x.accessible_name][0]

            try:
                accept_btn = [x for x in btns if 'Accept' in x.accessible_name][0]
                accept_btn.click()
            except:
                pass  # Accept already clicked

            time.sleep(3)
            dwnload_btn.click()
            print(f'\tDownloading from: {url}')
            time.sleep(3.)

            # waits for all the files to be completed and returns the paths
            path = WebDriverWait(driver, 120000, 1).until(every_downloads_chrome)[0]
            print(f'\tDownload finished: {url}')
            print(path)
            time.sleep(3.)
            path = path.split('file://')[-1]
            target_path = os.path.join(os.getcwd(), target_filename)
            os.rename(path, target_path)


if __name__ == '__main__':
    download()
