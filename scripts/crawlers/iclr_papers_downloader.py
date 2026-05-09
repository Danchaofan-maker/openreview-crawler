import openreview
import csv
import json
from tqdm import tqdm

def get_iclr_accepted_papers(year):
    """
    获取指定年份ICLR会议的已接受论文
    """
    venue_id = f"ICLR.cc/{year}/Conference"
    
    print(f"\n正在获取 {year} 年的ICLR论文...")
    
    try:
        # 使用API v2客户端
        client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
        
        # 获取会议group信息
        try:
            venue_group = client.get_group(venue_id)
            submission_name = venue_group.content.get('submission_name', {}).get('value', 'Submission')
        except:
            # 如果获取不到，使用默认值
            submission_name = 'Submission'
        
        # 方法1: 直接通过venueid获取已接受论文（API v2）
        try:
            accepted_papers = client.get_all_notes(content={'venueid': venue_id})
            if accepted_papers:
                print(f"通过venueid找到 {len(accepted_papers)} 篇已接受论文")
                return accepted_papers, client
        except Exception as e:
            print(f"方法1失败: {e}")
        
        # 方法2: 获取所有投稿，然后检查decision
        try:
            submissions = client.get_all_notes(
                invitation=f'{venue_id}/-/{submission_name}',
                details='directReplies'
            )
            
            accepted_papers = []
            for submission in tqdm(submissions, desc="检查论文接受状态"):
                # 检查directReplies中的Decision
                if 'details' in submission and 'directReplies' in submission.details:
                    for reply in submission.details['directReplies']:
                        if 'Decision' in reply.get('invitation', ''):
                            decision = reply.get('content', {}).get('decision', '')
                            if isinstance(decision, dict):
                                decision = decision.get('value', '')
                            if decision and 'accept' in decision.lower():
                                accepted_papers.append(submission)
                                break
            
            print(f"通过decision检查找到 {len(accepted_papers)} 篇已接受论文")
            return accepted_papers, client
            
        except Exception as e:
            print(f"方法2失败: {e}")
            
    except Exception as e:
        print(f"连接OpenReview API失败: {e}")
        return [], None
    
    return [], None

def extract_paper_info(paper, client, year):
    """
    提取论文信息
    """
    info = {
        'year': year,
        'id': paper.id,
        'forum': paper.forum if hasattr(paper, 'forum') else paper.id,
    }
    
    # 提取content中的信息
    content = paper.content if hasattr(paper, 'content') else {}
    
    # 标题
    title = content.get('title', '')
    if isinstance(title, dict):
        title = title.get('value', '')
    info['title'] = title
    
    # 作者
    authors = content.get('authors', [])
    if isinstance(authors, dict):
        authors = authors.get('value', [])
    info['authors'] = ', '.join(authors) if isinstance(authors, list) else str(authors)
    
    # 摘要
    abstract = content.get('abstract', '')
    if isinstance(abstract, dict):
        abstract = abstract.get('value', '')
    info['abstract'] = abstract
    
    # PDF链接
    pdf_info = content.get('pdf', '')
    if isinstance(pdf_info, dict):
        pdf_url = pdf_info.get('value', '')
    else:
        pdf_url = pdf_info
    info['pdf_url'] = f"https://openreview.net/pdf?id={paper.id}" if not pdf_url else pdf_url
    
    # OpenReview链接
    info['openreview_url'] = f"https://openreview.net/forum?id={paper.forum if hasattr(paper, 'forum') else paper.id}"
    
    # 关键词
    keywords = content.get('keywords', [])
    if isinstance(keywords, dict):
        keywords = keywords.get('value', [])
    info['keywords'] = ', '.join(keywords) if isinstance(keywords, list) else str(keywords)
    
    return info

def save_to_csv(papers_info, filename):
    """
    保存论文信息到CSV文件
    """
    if not papers_info:
        print("没有论文数据可保存")
        return
    
    fieldnames = ['year', 'title', 'authors', 'abstract', 'keywords', 'pdf_url', 'openreview_url']
    
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for paper in papers_info:
            writer.writerow({k: paper.get(k, '') for k in fieldnames})
    
    print(f"论文信息已保存到: {filename}")

def save_to_json(papers_info, filename):
    """
    保存论文信息到JSON文件
    """
    with open(filename, 'w', encoding='utf-8') as jsonfile:
        json.dump(papers_info, jsonfile, ensure_ascii=False, indent=2)
    
    print(f"论文信息已保存到: {filename}")

def main():
    """
    主函数
    """
    years = [2024, 2025, 2026]
    all_papers = []
    
    print("ICLR 2024-2026 已接受论文获取工具")
    print("=" * 50)
    
    for year in years:
        papers, client = get_iclr_accepted_papers(year)
        
        if not papers:
            print(f"{year} 年没有找到已接受的论文（可能数据尚未公布或会议ID不正确）")
            continue
        
        print(f"找到 {len(papers)} 篇论文")
        
        # 提取论文信息
        papers_info = []
        for paper in tqdm(papers, desc=f"提取 {year} 年论文信息"):
            try:
                info = extract_paper_info(paper, client, year)
                papers_info.append(info)
            except Exception as e:
                print(f"提取论文信息时出错: {e}")
                continue
        
        all_papers.extend(papers_info)
    
    if all_papers:
        # 保存为CSV
        save_to_csv(all_papers, 'iclr_2024_2026_accepted_papers.csv')
        
        # 保存为JSON
        save_to_json(all_papers, 'iclr_2024_2026_accepted_papers.json')
        
        print(f"\n总计获取 {len(all_papers)} 篇已接受论文")
    else:
        print("\n没有获取到任何论文数据")
        print("\n可能的原因：")
        print("1. ICLR 2026年的数据尚未公布")
        print("2. OpenReview API需要认证")
        print("3. 会议ID格式可能已更改")
        print("\n建议：")
        print("- 检查 https://openreview.net/group?id=ICLR.cc/2025/Conference 确认会议ID")
        print("- 如需认证，取消下面代码中的注释并提供凭据")

def main_with_auth(username=None, password=None):
    """
    使用认证的主函数（如果需要访问非公开数据）
    """
    years = [2024, 2025, 2026]
    all_papers = []
    
    print("使用认证信息连接OpenReview...")
    
    try:
        if username and password:
            client = openreview.api.OpenReviewClient(
                baseurl='https://api2.openreview.net',
                username=username,
                password=password
            )
        else:
            # 使用token方式
            import os
            token = os.getenv('OPENREVIEW_TOKEN')
            if token:
                client = openreview.api.OpenReviewClient(
                    baseurl='https://api2.openreview.net',
                    token=token
                )
            else:
                client = openreview.api.OpenReviewClient(baseurl='https://api2.openreview.net')
    except Exception as e:
        print(f"认证失败: {e}")
        return
    
    for year in years:
        venue_id = f"ICLR.cc/{year}/Conference"
        print(f"\n获取 {year} 年论文...")
        
        try:
            accepted_papers = client.get_all_notes(content={'venueid': venue_id})
            print(f"找到 {len(accepted_papers)} 篇已接受论文")
            
            for paper in tqdm(accepted_papers):
                info = extract_paper_info(paper, client, year)
                all_papers.append(info)
        except Exception as e:
            print(f"获取 {year} 年数据失败: {e}")
    
    if all_papers:
        save_to_csv(all_papers, 'iclr_accepted_papers.csv')
        save_to_json(all_papers, 'iclr_accepted_papers.json')

if __name__ == "__main__":
    # 安装依赖提示
    print("请确保已安装依赖: pip install openreview-py tqdm")
    print()
    
    main()
    
    # 如果需要认证，取消下面的注释
    # main_with_auth(username="your_email@domain.com", password="your_password")
