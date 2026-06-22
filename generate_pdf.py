import markdown2
import subprocess
import os
import re

md_path = '/Users/ravijani/Downloads/ee200_q3/Q3_Report.md'
with open(md_path, 'r') as f:
    md_content = f.read()

# Fix image paths
md_content = md_content.replace('figures/', '/Users/ravijani/Downloads/ee200_q3/figures/')

# Remove old headers and links from the main content so we can put them on the title page
md_content = re.sub(r'\*\*Live App:\*\*.*?\n', '', md_content)
md_content = re.sub(r'\*\*Source Code:\*\*.*?\n', '', md_content)
md_content = re.sub(r'# EE200 — Q3 Report:.*?\n', '', md_content)
md_content = re.sub(r'## Audio Fingerprinting System\n', '', md_content)
# Remove lingering horizontal rules at the top
md_content = re.sub(r'^(---\n)+', '', md_content, flags=re.MULTILINE)

# Convert to HTML
html_body = markdown2.markdown(md_content, extras=['tables', 'fenced-code-blocks'])

# Generate TOC
headings = re.findall(r'<h([23])>(.*?)</h[23]>', html_body)
toc_html = "<div class='toc'><h2 style='font-size: 18pt;'>Contents</h2><ul style='list-style-type: none; padding-left: 0;'>"
for level, title in headings:
    if level == '2':
        toc_html += f"<li style='margin-top: 10px; font-weight: bold; font-size: 12pt;'>{title}</li>"
    elif level == '3':
        toc_html += f"<li style='margin-left: 20px; font-size: 11pt; margin-top: 5px;'>{title}</li>"
toc_html += "</ul></div><div class='page-break'></div>"

# First page HTML
first_page = """
<div class="cover-page">
    <div class="cover-titles">
        <h1>EE200: Signals, Systems and Networks</h1>
        <div style="height: 20px;"></div>
        <h2>Professor : Tushar Sandhan</h2>
        <h2>Course Project — Question 3</h2>
        <div style="height: 20px;"></div>
        <h2>Q3: Sonic Signatures &amp; Zapptain America</h2>
    </div>
    
    <div class="cover-links">
        <p><strong>Live App:</strong> <a href="https://ee200-q3-u6rgv2jyrhfqdr4h4lfvje.streamlit.app">https://ee200-q3-u6rgv2jyrhfqdr4h4lfvje.streamlit.app</a></p>
        <p><strong>Source Code:</strong> <a href="https://github.com/Ravi5422/ee200-q3">https://github.com/Ravi5422/ee200-q3</a></p>
    </div>

    <div class="cover-students">
        <div class="student-left">
            Name: Jani Ravi Kailash<br>
            Name: Shreshthraj Bhidodiya
        </div>
        <div class="student-right">
            Roll No.: 240486<br>
            Roll No.: 240996
        </div>
    </div>
</div>
<div class="page-break"></div>
"""

# Assemble HTML
html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  @page {{
      size: A4;
      margin: 2.5cm;
  }}
  body {{ 
      font-family: "Times New Roman", Times, serif; 
      font-size: 11pt; 
      line-height: 1.6; 
      color: #000000 !important; 
      background-color: white;
  }}
  h1, h2, h3, h4, p, a, th, td, li, span, div {{
      color: #000000 !important;
  }}
  a {{ text-decoration: none; }}
  
  /* Cover Page Styles */
  .cover-page {{
      position: relative;
      height: 90vh; 
      display: flex;
      flex-direction: column;
  }}
  .cover-titles {{
      text-align: center;
      margin-top: 80px;
  }}
  .cover-titles h1 {{
      font-size: 20pt;
      font-weight: bold;
      margin-bottom: 10px;
  }}
  .cover-titles h2 {{
      font-size: 16pt;
      font-weight: bold;
      margin: 5px 0;
  }}
  .cover-links {{
      text-align: center;
      margin-top: 60px;
      font-size: 12pt;
  }}
  .cover-students {{
      display: flex;
      justify-content: space-between;
      margin-top: auto;
      font-size: 12pt;
      margin-bottom: 20px;
  }}
  .student-left {{ text-align: left; }}
  .student-right {{ text-align: right; }}

  /* Content Styles */
  h1, h2, h3 {{ font-weight: bold; }}
  h2 {{ font-size: 14pt; margin-top: 30px; margin-bottom: 15px; border-bottom: 1px solid #000; padding-bottom: 5px; }}
  h3 {{ font-size: 12pt; margin-top: 20px; margin-bottom: 10px; }}
  img {{ max-width: 90%; display: block; margin: 20px auto; border: 1px solid #000; }}
  table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
  th, td {{ border: 1px solid #000; padding: 8px 12px; text-align: left; }}
  th {{ font-weight: bold; background-color: white; border-bottom: 2px solid #000; }}
  code {{ padding: 2px 4px; font-size: 10pt; font-family: monospace; border: 1px solid #ccc; }}
  pre {{ padding: 12px; overflow-x: auto; font-family: monospace; border: 1px solid #000; }}
  blockquote {{ border-left: 3px solid #000; margin-left: 0; padding-left: 16px; font-style: italic; }}
  hr {{ border: none; border-top: 1px solid #000; margin: 20px 0; }}
  
  .page-break {{ page-break-after: always; }}
  
  @media print {{
      body {{ margin: 0; padding: 0; }}
  }}
</style>
</head>
<body>
{first_page}
{toc_html}
{html_body}
</body>
</html>
"""

html_out = '/Users/ravijani/Downloads/ee200_q3/Q3_Report_Final.html'
pdf_out = '/Users/ravijani/Downloads/ee200_q3/Q3_Report_Final.pdf'

with open(html_out, 'w') as f:
    f.write(html)

chrome_paths = [
    '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome',
    '/Applications/Chromium.app/Contents/MacOS/Chromium',
]
chrome = next((p for p in chrome_paths if os.path.exists(p)), None)
if chrome:
    print(f"Using Chrome to generate PDF...")
    # Using no-margins might remove the header/footer by default, but let's just use it to be safe 
    # since we added explicit CSS margins (@page { margin: 2.5cm }) which should be respected or overridden.
    # Actually, if we use --no-margins, it kills the @page margin sometimes. Let's NOT use --no-margins, 
    # and use the @page margin so it looks like a real document. Headless chrome does NOT add header/footer 
    # unless --display-header-footer is present.
    subprocess.run([chrome, '--headless', '--disable-gpu', '--no-pdf-header-footer',
                   f'--print-to-pdf={pdf_out}',
                   f'file://{html_out}'])
    print(f"✓ PDF created: {pdf_out}")
else:
    print("Chrome not found")
