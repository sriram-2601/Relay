"""
Script to compile writeup.md into a self-contained, publication-ready HTML report (writeup.html)
with embedded Base64 vector charts, optimized for standard A4 2-page printing and export to PDF.
"""

import os
import re
import base64

CSS_STYLE = """
<style>
    @page {
        size: A4;
        margin: 1.4cm 1.5cm 1.6cm 1.5cm;
    }
    body {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        color: #1f2937;
        line-height: 1.45;
        font-size: 10pt;
        max-width: 840px;
        margin: 0 auto;
        padding: 24px;
        background-color: #ffffff;
    }
    .header-box {
        border-bottom: 2px solid #2563eb;
        padding-bottom: 10px;
        margin-bottom: 16px;
    }
    h1 {
        font-size: 17pt;
        color: #111827;
        margin: 0 0 4px 0;
        font-weight: 700;
    }
    .subtitle {
        color: #4b5563;
        font-size: 9.5pt;
        margin: 0;
    }
    h2 {
        font-size: 11.5pt;
        color: #1e3a8a;
        margin-top: 14px;
        margin-bottom: 4px;
        border-left: 3px solid #3b82f6;
        padding-left: 8px;
        font-weight: 700;
    }
    h3 {
        font-size: 10.5pt;
        color: #1f2937;
        margin-top: 10px;
        margin-bottom: 3px;
        font-weight: 600;
    }
    p, li {
        margin-top: 2px;
        margin-bottom: 4px;
        text-align: justify;
    }
    table {
        width: 100%;
        border-collapse: collapse;
        font-size: 8.5pt;
        margin: 8px 0 12px 0;
    }
    th, td {
        border: 1px solid #d1d5db;
        padding: 4px 7px;
        text-align: left;
    }
    th {
        background-color: #f1f5f9;
        font-weight: 600;
        color: #0f172a;
    }
    tr:nth-child(even) {
        background-color: #f8fafc;
    }
    code {
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
        font-size: 8.5pt;
        background-color: #f1f5f9;
        color: #0f172a;
        padding: 1px 4px;
        border-radius: 3px;
    }
    pre {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        padding: 6px 10px;
        border-radius: 4px;
        font-size: 8pt;
        overflow-x: auto;
        margin: 6px 0;
    }
    .chart-container {
        display: flex;
        justify-content: space-between;
        margin: 10px 0;
        gap: 12px;
    }
    .chart-box {
        flex: 1;
        text-align: center;
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 6px;
        padding: 6px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .chart-box img {
        max-width: 100%;
        height: auto;
        border-radius: 4px;
    }
    .chart-caption {
        font-size: 8pt;
        color: #64748b;
        margin-top: 4px;
        font-weight: 500;
    }
    @media print {
        body { padding: 0; }
        .page-break { page-break-before: always; }
    }
</style>
"""

def get_base64_img(path: str) -> str:
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def md_table_to_html(md_text: str) -> str:
    lines = md_text.split("\n")
    out_lines = []
    in_table = False
    table_lines = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
        else:
            if in_table:
                out_lines.append(parse_table(table_lines))
                table_lines = []
                in_table = False
            out_lines.append(line)

    if in_table:
        out_lines.append(parse_table(table_lines))

    return "\n".join(out_lines)

def parse_table(lines):
    if len(lines) < 2:
        return "\n".join(lines)
    header = [c.strip() for c in lines[0].strip("|").split("|")]
    rows = []
    for l in lines[2:]:
        cols = [c.strip() for c in l.strip("|").split("|")]
        rows.append(cols)
    
    html = ["<table><thead><tr>"]
    for h in header:
        html.append(f"<th>{h}</th>")
    html.append("</tr></thead><tbody>")
    for r in rows:
        html.append("<tr>")
        for cell in r:
            html.append(f"<td>{cell}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)

def convert_markdown_to_html(md_path: str, html_path: str):
    with open(md_path, "r", encoding="utf-8") as f:
        md = f.read()

    md_with_tables = md_table_to_html(md)
    html = md_with_tables

    # Parse Headings
    html = re.sub(r"^### (.*?)$", r"<h3>\1</h3>", html, flags=re.MULTILINE)
    html = re.sub(r"^## (.*?)$", r"<h2>\1</h2>", html, flags=re.MULTILINE)
    html = re.sub(r"^# (.*?)$", r"<div class='header-box'><h1>\1</h1>", html, flags=re.MULTILINE)
    html = html.replace("<div class='header-box'><h1>", "<div class='header-box'><h1>", 1)

    # Subtitle handling
    html = html.replace(
        "**Applicant Submission for ML Engineering Intern — Vaani**",
        "<p class='subtitle'><strong>Applicant Submission for ML Engineering Intern — Vaani</strong> | Library: arXiv 1706.03762v7, 2410.00037v1, 2507.07955v2</p></div>"
    )
    
    # Bold / Italic
    html = re.sub(r"\*\*(.*?)\*\*", r"<strong>\1</strong>", html)
    html = re.sub(r"\*(.*?)\*", r"<em>\1</em>", html)
    
    # Code blocks
    html = re.sub(r"```(?:\w+)?\n([\s\S]*?)```", r"<pre><code>\1</code></pre>", html)
    html = re.sub(r"`([^`]+)`", r"<code>\1</code>", html)
    html = re.sub(r"^---$", r"<hr style='border:0; border-top:1px solid #e5e7eb; margin:12px 0;'/>", html, flags=re.MULTILINE)

    # Encode Base64 charts
    b64_cdf = get_base64_img("output/chart_latency_cdf.png")
    b64_queue = get_base64_img("output/chart_queue_degradation.png")
    b64_cost = get_base64_img("output/chart_quality_vs_cost.png")

    charts_html_1 = f"""
    <div class="chart-container">
        <div class="chart-box">
            <img src="data:image/png;base64,{b64_cdf}" alt="Latency CDF Across Scenarios" />
            <div class="chart-caption">Figure 1: End-to-End Latency CDF Across Load Shapes</div>
        </div>
        <div class="chart-box">
            <img src="data:image/png;base64,{b64_queue}" alt="Queue Contention Degradation" />
            <div class="chart-caption">Figure 2: TTFT vs Concurrent In-Flight Requests (LPU Knee)</div>
        </div>
    </div>
    """

    chart_html_2 = f"""
    <div class="chart-container" style="max-width: 520px; margin: 10px auto;">
        <div class="chart-box">
            <img src="data:image/png;base64,{b64_cost}" alt="Retrieval Pareto Frontier" />
            <div class="chart-caption">Figure 3: Retrieval Quality vs Token Footprint (Pareto Frontier)</div>
        </div>
    </div>
    """

    # Inject charts at Section 3 and Section 5
    if "<h2>3 Results." in html:
        html = html.replace("<h2>3 Results.", charts_html_1 + "<h2>3 Results.")
    if "<h3>Retrieval: quality versus cost.</h3>" in html:
        html = html.replace("<h3>Retrieval: quality versus cost.</h3>", "<h3>Retrieval: quality versus cost.</h3>" + chart_html_2)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>The Relay — Vaani ML Engineering Submission</title>
    {CSS_STYLE}
</head>
<body>
    {html}
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(full_html)
    print(f"Successfully generated self-contained report: {html_path} ({os.path.getsize(html_path):,} bytes)!")

if __name__ == "__main__":
    convert_markdown_to_html("writeup.md", "writeup.html")
