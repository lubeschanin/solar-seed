#!/usr/bin/env python3
"""
Generate arXiv-ready PDF from PAPER.md
"""

import markdown
from weasyprint import HTML, CSS
from pathlib import Path
import re

# Read markdown
paper_path = Path(__file__).parent.parent / "PAPER.md"
base_dir = paper_path.parent
md_content = paper_path.read_text()

# Convert relative image paths to absolute file:// URLs
def fix_image_paths(content, base):
    def replace_path(match):
        alt = match.group(1)
        path = match.group(2)
        abs_path = (base / path).resolve()
        return f'![{alt}](file://{abs_path})'
    return re.sub(r'!\[([^\]]*)\]\(([^)]+)\)', replace_path, content)

md_content = fix_image_paths(md_content, base_dir)

# Convert to HTML
md = markdown.Markdown(extensions=['tables', 'fenced_code'])
html_body = md.convert(md_content)

# Academic styling
css = CSS(string='''
@page {
    size: letter;
    margin: 1in;
    @bottom-center {
        content: counter(page);
    }
}

body {
    font-family: "Times New Roman", Times, serif;
    font-size: 11pt;
    line-height: 1.5;
    text-align: justify;
    max-width: 6.5in;
    margin: 0 auto;
}

h1 {
    font-size: 16pt;
    text-align: center;
    margin-top: 0;
    margin-bottom: 0.5em;
}

h2 {
    font-size: 13pt;
    margin-top: 1.5em;
    margin-bottom: 0.5em;
}

h3 {
    font-size: 11pt;
    margin-top: 1em;
    margin-bottom: 0.5em;
}

p {
    margin: 0.5em 0;
}

table {
    border-collapse: collapse;
    margin: 1em auto;
    font-size: 10pt;
}

th, td {
    border: 1px solid #333;
    padding: 0.3em 0.6em;
    text-align: left;
}

th {
    background-color: #f0f0f0;
}

code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f5f5f5;
    padding: 0.1em 0.3em;
}

pre {
    background-color: #f5f5f5;
    padding: 0.5em;
    overflow-x: auto;
    font-size: 9pt;
}

blockquote {
    margin: 1em 2em;
    font-style: italic;
    border-left: 3px solid #ccc;
    padding-left: 1em;
}

img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 1em auto;
}

hr {
    border: none;
    border-top: 1px solid #ccc;
    margin: 2em 0;
}

strong {
    font-weight: bold;
}

em {
    font-style: italic;
}
''')

# Full HTML document
html_doc = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Solar Seed Paper</title>
</head>
<body>
{html_body}
</body>
</html>
'''

# Generate PDF
output_path = Path(__file__).parent.parent / "paper.pdf"
HTML(string=html_doc).write_pdf(output_path, stylesheets=[css])

print(f"PDF generated: {output_path}")
print(f"Size: {output_path.stat().st_size / 1024:.1f} KB")
