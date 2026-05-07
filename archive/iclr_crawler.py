#!/usr/bin/env python3
"""
ICLR论文爬虫 - 使用Playwright爬取2024-2026年论文数据
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import json
import time
import os

def crawl_iclr_papers(year, output_dir="iclr_data"):
    """爬取指定年份的ICLR论文数据"""
    os.makedirs(output_dir, exist_ok=True)
    papers = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        
        # 使用OpenReview的提交列表页面（数据更完整）
        url = f"https://openreview.net/submissions?venue=ICLR.cc%2F{year}%2FConference"
        print(f"正在爬取 ICLR {year}: {url}")
        
        try:
            page.goto(url, timeout=60000)
            page.wait_for_load_state("networkidle", timeout=60000)
            
            # 等待论文列表加载
            page.wait_for_selector(".note", timeout=30000)
            
            # 获取所有论文元素
            notes = page.query_selector_all(".note")
            print(f"找到 {len(notes)} 篇论文")
            
            for note in notes:
                try:
                    # 提取标题
                    title_elem = note.query_selector(".note-content-title a, .note-content-title")
                    title = title_elem.inner_text().strip() if title_elem else ""
                    
                    # 提取作者
                    authors_elem = note.query_selector(".note-content-authors")
                    authors = authors_elem.inner_text().strip() if authors_elem else ""
                    
                    # 提取链接
                    link_elem = note.query_selector("a[href*='/forum?id=']")
                    paper_id = ""
                    if link_elem:
                        href = link_elem.get_attribute("href")
                        if "id=" in href:
                            paper_id = href.split("id=")[1].split("&")[0]
                    
                    # 提取状态
                    status_elem = note.query_selector(".note-content-status, .tag")
                    status = status_elem.inner_text().strip() if status_elem else "Unknown"
                    
                    if title:
                        papers.append({
                            "year": year,
                            "title": title,
                            "authors": authors,
                            "paper_id": paper_id,
                            "url": f"https://openreview.net/forum?id={paper_id}" if paper_id else "",
                            "status": status
                        })
                except Exception as e:
                    print(f"解析论文时出错: {e}")
                    continue
            
            # 尝试点击"Load More"按钮加载更多论文
            max_clicks = 50
            click_count = 0
            
            while click_count < max_clicks:
                try:
                    load_more = page.query_selector("text=Load More, text=Show More, button:has-text('More')")
                    if not load_more or not load_more.is_visible():
                        break
                    load_more.click()
                    page.wait_for_timeout(2000)  # 等待加载
                    click_count += 1
                    
                    # 重新获取所有论文
                    notes = page.query_selector_all(".note")
                    print(f"已加载 {len(notes)} 篇论文...")
                except Exception:
                    break
            
            # 最终重新解析所有论文
            notes = page.query_selector_all(".note")
            papers = []
            for note in notes:
                try:
                    title_elem = note.query_selector(".note-content-title a, .note-content-title")
                    title = title_elem.inner_text().strip() if title_elem else ""
                    
                    authors_elem = note.query_selector(".note-content-authors")
                    authors = authors_elem.inner_text().strip() if authors_elem else ""
                    
                    link_elem = note.query_selector("a[href*='/forum?id=']")
                    paper_id = ""
                    if link_elem:
                        href = link_elem.get_attribute("href")
                        if "id=" in href:
                            paper_id = href.split("id=")[1].split("&")[0]
                    
                    status_elem = note.query_selector(".note-content-status, .tag")
                    status = status_elem.inner_text().strip() if status_elem else "Unknown"
                    
                    if title:
                        papers.append({
                            "year": year,
                            "title": title,
                            "authors": authors,
                            "paper_id": paper_id,
                            "url": f"https://openreview.net/forum?id={paper_id}" if paper_id else "",
                            "status": status
                        })
                except Exception as e:
                    continue
            
        except PlaywrightTimeout as e:
            print(f"页面加载超时: {e}")
        except Exception as e:
            print(f"爬取过程出错: {e}")
        finally:
            browser.close()
    
    # 保存数据
    output_file = os.path.join(output_dir, f"iclr_{year}_papers.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    print(f"ICLR {year}: 保存了 {len(papers)} 篇论文到 {output_file}")
    return papers


def main():
    """主函数"""
    years = [2024, 2025, 2026]
    all_papers = []
    
    for year in years:
        papers = crawl_iclr_papers(year)
        all_papers.extend(papers)
        time.sleep(2)  # 避免请求过快
    
    # 保存合并数据
    with open("iclr_data/all_iclr_papers_2024_2026.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)
    
    print(f"\n总共爬取了 {len(all_papers)} 篇论文")


if __name__ == "__main__":
    # 检查是否安装了playwright
    try:
        import playwright
    except ImportError:
        print("请先安装playwright:")
        print("pip install playwright")
        print("playwright install")
        exit(1)
    
    main()
