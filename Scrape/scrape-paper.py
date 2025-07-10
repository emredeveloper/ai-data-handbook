import requests
from bs4 import BeautifulSoup
import os
import time
import fitz  # PyMuPDF kütüphanesi
import subprocess
import sys

BASE_URL = 'https://huggingface.co'
PAPERS_URL = f'{BASE_URL}/papers'

# 'papers' dizini yoksa oluştur
if not os.path.exists('papers'):
    os.makedirs('papers')

# Ana makaleler sayfasını al
response = requests.get(PAPERS_URL)
soup = BeautifulSoup(response.content, 'html.parser')

# En fazla 10 makale linkini bul
articles = soup.find_all('article')
paper_links = []
for article in articles:
    a_tag = article.find('a')
    if a_tag and a_tag.has_attr('href') and a_tag['href'].startswith('/papers/'):
        paper_links.append(BASE_URL + a_tag['href'])
    if len(paper_links) == 10:
        break

# Hangi PDF'lerin zaten mevcut olduğunu kontrol et
existing_pdfs = set(os.listdir('papers'))

# Her makale için PDF'yi bul ve indir
for idx, paper_url in enumerate(paper_links, 1):
    paper_resp = requests.get(paper_url)
    paper_soup = BeautifulSoup(paper_resp.content, 'html.parser')
    pdf_link = None
    # 1. HuggingFace'de doğrudan PDF linki bulmaya çalış
    for a in paper_soup.find_all('a', href=True):
        if a['href'].endswith('.pdf'):
            pdf_link = a['href']
            break
    # 2. If not found, look for arXiv or other external links
    if not pdf_link:
        for a in paper_soup.find_all('a', href=True):
            href = a['href']
            if 'arxiv.org' in href:
                # If the link is to an arXiv abstract, convert to PDF
                if '/abs/' in href:
                    pdf_link = href.replace('/abs/', '/pdf/') + '.pdf' if not href.endswith('.pdf') else href
                elif '/pdf/' in href and href.endswith('.pdf'):
                    pdf_link = href
                break
    # 3. Download if found
    pdf_filename = f'paper_{idx}.pdf'
    pdf_path = os.path.join('papers', pdf_filename)
    if pdf_filename in existing_pdfs and os.path.getsize(pdf_path) > 0:
        print(f"Zaten mevcut, atlanıyor: {pdf_path}")
        continue
    if pdf_link:
        if pdf_link.startswith('/'):
            pdf_link = BASE_URL + pdf_link
        pdf_resp = requests.get(pdf_link)
        if pdf_resp.status_code == 200 and pdf_resp.headers.get('Content-Type', '').startswith('application/pdf'):
            with open(pdf_path, 'wb') as f:
                f.write(pdf_resp.content)
            print(f'Downloaded: {pdf_path}')
        else:
            print(f'PDF link found but could not download PDF for {paper_url}')
        time.sleep(1)
    else:
        print(f'No PDF found for {paper_url}')

# --- PDF Analysis with Ollama (Gemma3:12b, Turkish output) ---
ollama_model = "gemma3:12b"
papers_dir = "papers"
output_dir = "papers/analizler"
os.makedirs(output_dir, exist_ok=True)

def summarize_with_ollama(text):
    prompt = f"Aşağıdaki makaleyi Türkçe özetle ve analiz et:\n\n{text}\n\nKısa ve öz bir özet/analiz yaz."
    result = subprocess.run(
        ["ollama", "run", ollama_model, prompt],
        capture_output=True, text=True, encoding='utf-8'
    )
    return result.stdout.strip()

if len(sys.argv) > 1:
    user_input = sys.argv[1]
    if not user_input.endswith('.pdf'):
        user_input += '.pdf'
    pdf_file = user_input
    pdf_path = os.path.join(papers_dir, pdf_file)
    out_txt = os.path.join(output_dir, pdf_file.replace('.pdf', '.txt'))
    if not os.path.exists(pdf_path):
        print(f"PDF bulunamadı: {pdf_path}")
        sys.exit(1)
    if os.path.exists(out_txt) and os.path.getsize(out_txt) > 0:
        print(f"Analiz zaten mevcut, atlanıyor: {out_txt}")
        sys.exit(0)
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    doc.close()
    summary = summarize_with_ollama(full_text[:2000])
    with open(out_txt, "w", encoding="utf-8") as f:
        f.write(summary)
    print(f"Analiz kaydedildi: {out_txt}")
else:
    print("Analiz için bir PDF dosyası adı belirtin. Örnek: python scrape-paper.py paper_5 veya paper_5.pdf")