from playwright.sync_api import sync_playwright


def test_playwright():
    try:
        with sync_playwright() as p:

            browser = p.chromium.launch(headless=False)
            context = browser.new_context()
            page = context.new_page()
            page.goto("https://www.baidu.com")
            print("Title:", page.title())
            browser.close()
        print("Playwright is working correctly!")

    except Exception as e:
        print("Error:", e)
        print("Playwright might not be working as expected.")


if __name__ == "__main__":
    test_playwright()
