import glob
import os
import re

def inspect_pdf(path):
    with open(path, 'rb') as f:
        data = f.read(500000) # first 500KB
    meta = {}
    title_m = re.search(rb'/Title\s*\((.*?)\)', data)
    if title_m:
        meta['title'] = title_m.group(1).decode('latin1', errors='ignore')
    author_m = re.search(rb'/Author\s*\((.*?)\)', data)
    if author_m:
        meta['author'] = author_m.group(1).decode('latin1', errors='ignore')
    doi_m = re.search(rb'10\.\d{4,9}/[-._;()/:A-Za-z0-9]+', data)
    if doi_m:
        meta['doi'] = doi_m.group(0).decode('latin1', errors='ignore')
    return meta

with open('scratch/pdf_metadata.txt', 'w', encoding='utf-8') as out:
    for p in sorted(glob.glob('*.pdf')):
        m = inspect_pdf(p)
        title = m.get('title', 'N/A')
        author = m.get('author', 'N/A')
        doi = m.get('doi', 'N/A')
        line = f"{p} | Title: {title} | Author: {author} | DOI: {doi}\n"
        out.write(line)
        print(line, end='')

print("Done inspecting PDFs.")
