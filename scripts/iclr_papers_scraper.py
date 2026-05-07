#!/usr/bin/env python3
"""
ICLR 2024-2026 已接受论文爬取脚本
使用requests直接调用OpenReview REST API，无需openreview-py库
"""

import requests
import json
import csv
import time
from tqdm import tqdm

class OpenReviewScraper:
    def __init__(self):
        self.base_url = "https://api2.openreview.net"
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
    def get_submissions(self, venue_id):
        """
        获取指定会议的所有投稿
        """
        submissions = []
        offset = 0
        limit = 1000
        
        while True:
            try:
                url = f"{self.base_url}/notes"
                params = {
                    'invitation': f"{venue_id}/-/{self._get_submission_name(venue_id)}",
                    'details': 'replyCount,directReplies',
                    'offset': offset,
                    'limit': limit
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                batch = data.get('notes', [])
                if not batch:
                    break
                    
                submissions.extend(batch)
                offset += len(batch)
                
                if len(batch) < limit:
                    break
                    
                time.sleep(0.5)  # 避免请求过快
                
            except Exception as e:
                print(f"获取投稿时出错: {e}")
                break
                
        return submissions
    
    def _get_submission_name(self, venue_id):
        """
        获取会议的submission名称
        """
        try:
            url = f"{self.base_url}/group"
            params = {'id': venue_id}
            response = requests.get(url, headers=self.headers, params=params, timeout=10)
            response.raise_for_status()
            group = response.json()
            
            # 尝试从group content中获取submission_name
            content = group.get('content', {})
            if isinstance(content, dict):
                submission_name = content.get('submission_name', {})
                if isinstance(submission_name, dict):
                    return submission_name.get('value', 'Submission')
            return 'Submission'
        except:
            return 'Submission'
    
    def get_decisions(self, venue_id):
        """
        获取所有决定（Accept/Reject）
        """
        decisions = {}
        offset = 0
        limit = 1000
        
        while True:
            try:
                url = f"{self.base_url}/notes"
                params = {
                    'invitation': f"{venue_id}/.*/Decision",
                    'offset': offset,
                    'limit': limit
                }
                
                response = requests.get(url, headers=self.headers, params=params, timeout=30)
                response.raise_for_status()
                data = response.json()
                
                batch = data.get('notes', [])
                if not batch:
                    break
                
                for decision_note in batch:
                    forum_id = decision_note.get('forum')
                    content = decision_note.get('content', {})
                    
                    # 提取decision值
                    decision = None
                    if isinstance(content, dict):
                        if 'decision' in content:
                            decision_val = content['decision']
                            if isinstance(decision_val, dict):
                                decision = decision_val.get('value', '')
                            else:
                                decision = str(decision_val)
                    
                    if forum_id and decision:
                        decisions[forum_id] = decision
                
                offset += len(batch)
                if len(batch) < limit:
                    break
                time.sleep(0.5)
                
            except Exception as e:
                print(f"获取决定时出错: {e}")
                break
        
        return decisions
    
    def extract_paper_info(self, note, year, decisions):
        """
        提取论文信息
        """
        info = {}
        
        # 会议信息
        info['conference'] = f"ICLR {year}"
        info['conference_id'] = f"ICLR.cc/{year}/Conference"
        
        # 论文ID
        info['paper_id'] = note.get('id', '')
        info['forum_id'] = note.get('forum', note.get('id', ''))
        
        # 提取content
        content = note.get('content', {})
        
        # 标题
        title = content.get('title', '')
        if isinstance(title, dict):
            title = title.get('value', '')
        info['title'] = title
        
        # 作者
        authors = content.get('authors', [])
        if isinstance(authors, dict):
            authors = authors.get('value', [])
        if isinstance(authors, list):
            info['authors'] = '; '.join(authors)
        else:
            info['authors'] = str(authors)
        
        # 摘要
        abstract = content.get('abstract', '')
        if isinstance(abstract, dict):
            abstract = abstract.get('value', '')
        info['abstract'] = abstract
        
        # 关键词
        keywords = content.get('keywords', [])
        if isinstance(keywords, dict):
            keywords = keywords.get('value', [])
        if isinstance(keywords, list):
            info['keywords'] = '; '.join(keywords)
        else:
            info['keywords'] = str(keywords)
        
        # 领域/主题
        field = content.get('field', '')
        if isinstance(field, dict):
            field = field.get('value', '')
        if not field:
            field = content.get('area', '')
            if isinstance(field, dict):
                field = field.get('value', '')
        if not field:
            field = content.get('subject areas', '')
            if isinstance(field, dict):
                field = field.get('value', '')
        info['field'] = field
        
        # 状态（从decisions中获取）
        forum_id = note.get('forum', note.get('id'))
        decision = decisions.get(forum_id, 'Unknown')
        info['status'] = decision
        
        # 是否录用
        if decision and 'accept' in decision.lower():
            info['accepted'] = 'Yes'
        elif decision and 'reject' in decision.lower():
            info['accepted'] = 'No'
        else:
            info['accepted'] = 'Unknown'
        
        # PDF链接
        pdf_info = content.get('pdf', '')
        if isinstance(pdf_info, dict):
            pdf_url = pdf_info.get('value', '')
        else:
            pdf_url = pdf_info
        
        if pdf_url:
            info['pdf_url'] = pdf_url
        else:
            info['pdf_url'] = f"https://openreview.net/pdf?id={info['paper_id']}"
        
        # OpenReview链接
        info['openreview_url'] = f"https://openreview.net/forum?id={info['forum_id']}"
        
        # 年份
        info['year'] = year
        
        return info

def main():
    """
    主函数
    """
    years = [2024, 2025, 2026]
    scraper = OpenReviewScraper()
    all_papers = []
    
    print("ICLR 2024-2026 已接受论文爬取工具")
    print("=" * 60)
    
    for year in years:
        venue_id = f"ICLR.cc/{year}/Conference"
        print(f"\n正在处理 ICLR {year}...")
        
        # 获取所有决定
        print("获取论文决定...")
        decisions = scraper.get_decisions(venue_id)
        print(f"找到 {len(decisions)} 个决定")
        
        # 获取所有投稿
        print("获取所有投稿...")
        submissions = scraper.get_submissions(venue_id)
        print(f"找到 {len(submissions)} 篇投稿")
        
        # 筛选出已录用的论文
        accepted_count = 0
        for note in tqdm(submissions, desc=f"处理 ICLR {year}"):
            paper_info = scraper.extract_paper_info(note, year, decisions)
            
            # 只保存已录用的论文
            if paper_info['accepted'] == 'Yes':
                all_papers.append(paper_info)
                accepted_count += 1
        
        print(f"ICLR {year}: 找到 {accepted_count} 篇已录用论文")
    
    # 保存到CSV
    if all_papers:
        csv_filename = 'iclr_2024_2026_accepted_papers.csv'
        fieldnames = [
            'conference', 'conference_id', 'year', 'paper_id', 'forum_id',
            'title', 'authors', 'abstract', 'keywords', 'field',
            'status', 'accepted', 'pdf_url', 'openreview_url'
        ]
        
        with open(csv_filename, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for paper in all_papers:
                writer.writerow(paper)
        
        print(f"\n论文信息已保存到: {csv_filename}")
        
        # 同时保存为JSON
        json_filename = 'iclr_2024_2026_accepted_papers.json'
        with open(json_filename, 'w', encoding='utf-8') as jsonfile:
            json.dump(all_papers, jsonfile, ensure_ascii=False, indent=2)
        
        print(f"论文信息已保存到: {json_filename}")
        print(f"\n总计获取 {len(all_papers)} 篇已录用论文")
        
        # 统计信息
        year_stats = {}
        for paper in all_papers:
            year = paper['year']
            year_stats[year] = year_stats.get(year, 0) + 1
        
        print("\n各年份统计:")
        for year in sorted(year_stats.keys()):
            print(f"  ICLR {year}: {year_stats[year]} 篇")
    else:
        print("\n没有找到已录用的论文")
        print("可能的原因：")
        print("1. ICLR 2026的数据尚未公布")
        print("2. API结构可能已更改")
        print("3. 需要检查会议ID是否正确")

if __name__ == "__main__":
    main()
