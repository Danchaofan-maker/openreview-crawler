#!/usr/bin/env python3
"""
ICLR论文爬虫 - Playwright版本
爬取ICLR 2024-2026论文数据
"""

from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeout
import json
import time
import os

def crawl_iclr_with_playwright(year, output_dir="iclr_data"):
    """使用Playwright爬取ICLR论文"""
    os.makedirs(output_dir, exist_ok=True)
    papers = []
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        )
        page = context.new_page()
        
        # 尝试从ICLR官网virtual页面获取
        url = f"https://iclr.cc/virtual/{year}/papers.html"
        print(f"正在访问: {url}")
        
        try:
            page.goto(url, timeout=60000, wait_until="domcontentloaded")
            page.wait_for_timeout(5000)  # 等待JS执行
            
            # 检查页面内容
            page_text = page.content()
            
            # 尝试从页面中提取论文数据
            # ICLR virtual页面通常包含JSON数据或特定结构
            try:
                # 尝试获取所有论文链接
                paper_links = page.query_selector_all("a[href*='/paper/'], a[href*='openreview.net/forum']")
                
                if paper_links:
                    print(f"找到 {len(paper_links)} 个论文链接")
                    for link in paper_links:
                        try:
                            title = link.inner_text().strip()
                            href = link.get_attribute("href")
                            if title and href:
                                papers.append({
                                    "year": year,
                                    "title": title,
                                    "url": href,
                                    "source": "iclr.cc"
                                })
                        except:
                            continue
                else:
                    print("未找到论文链接，页面可能需要登录或数据加载方式不同")
                    
            except Exception as e:
                print(f"解析页面时出错: {e}")
                
        except Exception as e:
            print(f"访问页面失败: {e}")
        finally:
            browser.close()
    
    # 保存数据
    output_file = os.path.join(output_dir, f"iclr_{year}_playwright.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    print(f"ICLR {year}: 保存了 {len(papers)} 条记录到 {output_file}")
    return papers


def crawl_openreview_api_style(year, output_dir="iclr_data"):
    """
    使用类API方式爬取OpenReview数据
    OpenReview网站有API接口，可以直接请求
    """
    import requests
    
    os.makedirs(output_dir, exist_ok=True)
    papers = []
    
    # OpenReview API v2 endpoint
    api_url = "https://api2.openreview.net/notes"
    
    # 构建查询参数
    venue_id = f"ICLR.cc/{year}/Conference"
    
    params = {
        "invitation": f"{venue_id}/-/Submission",
        "details": "replyCount"
    }
    
    print(f"正在从OpenReview API获取ICLR {year}数据...")
    
    try:
        response = requests.get(api_url, params=params, timeout=30)
        if response.status_code == 200:
            data = response.json()
            notes = data.get("notes", [])
            
            for note in notes:
                papers.append({
                    "year": year,
                    "paper_id": note.get("id"),
                    "forum": note.get("forum"),
                    "title": note.get("content", {}).get("title", ""),
                    "authors": note.get("content", {}).get("authors", []),
                    "abstract": note.get("content", {}).get("abstract", ""),
                    "url": f"https://openreview.net/forum?id={note.get('forum', note.get('id'))}"
                })
            
            print(f"通过API获取到 {len(papers)} 篇论文")
        else:
            print(f"API请求失败: {response.status_code}")
            
    except Exception as e:
        print(f"API请求出错: {e}")
    
    # 保存数据
    output_file = os.path.join(output_dir, f"iclr_{year}_api.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(papers, f, ensure_ascii=False, indent=2)
    
    return papers


if __name__ == "__main__":
    print("ICLR论文爬虫")
    print("=" * 50)
    print("选择爬取方式:")
    print("1. Playwright爬取ICLR官网 (可能数据不完整)")
    print("2. 使用OpenReview API方式 (推荐，数据完整)")
    print("3. 两种方式都尝试")
    print("=" * 50)
    
    choice = input("请输入选择 (1/2/3): ").strip()
    
    years = [2024, 2025, 2026]
    all_papers = []
    
    if choice == "1":
        for year in years:
            papers = crawl_iclr_with_playwright(year)
            all_papers.extend(papers)
    elif choice == "2":
        for year in years:
            papers = crawl_openreview_api_style(year)
            all_papers.extend(papers)
    else:
        for year in years:
            print(f"\n--- 处理 ICLR {year} ---")
            papers1 = crawl_iclr_with_playwright(year)
            papers2 = crawl_openreview_api_style(year)
            all_papers.extend(papers1)
            all_papers.extend(papers2)
    
    # 保存合并数据
    with open("iclr_data/all_papers.json", "w", encoding="utf-8") as f:
        json.dump(all_papers, f, ensure_ascii=False, indent=2)
    
    print(f"\n完成! 总共 {len(all_papers)} 条记录")
