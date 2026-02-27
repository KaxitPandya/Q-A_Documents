import os
import time

from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.common.exceptions import WebDriverException

from webdriver_manager.chrome import ChromeDriverManager

APP_URL = os.environ.get("STREAMLIT_APP_URL")
if not APP_URL:
    raise SystemExit("Missing STREAMLIT_APP_URL env var")

WAKE_PHRASES = [
    "get this app back up",
    "wake",
    "start",
    "resume",
]

def main():
    options = Options()
    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")
    options.add_argument("--disable-dev-shm-usage")

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=options,
    )

    try:
        driver.get(APP_URL)
        time.sleep(6)

        # If the "sleep" interstitial is present, click the wake button.
        try:
            buttons = driver.find_elements(By.TAG_NAME, "button")
            clicked = False
            for b in buttons:
                txt = (b.text or "").strip().lower()
                if any(p in txt for p in WAKE_PHRASES):
                    b.click()
                    clicked = True
                    time.sleep(6)
                    break

            if clicked:
                print("Clicked wake button.")
            else:
                print("No wake button detected (app likely already awake).")
        except WebDriverException:
            print("Could not inspect/click buttons (continuing).")

        # Give it a moment to finish booting/warming
        time.sleep(10)
        print("Done. App should be awake/warmed.")

    finally:
        driver.quit()

if __name__ == "__main__":
    main()
