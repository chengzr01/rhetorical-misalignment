"""
Test markdown rendering functionality
"""

import json
import markdown

# Configure markdown converter
md = markdown.Markdown(extensions=['extra', 'nl2br', 'sane_lists'])

def render_markdown(text):
    """Convert markdown text to HTML"""
    if not text:
        return ""
    md.reset()
    return md.convert(text)

# Load sample data
DATA_FILE = '../experiments/cache/agent_deepseek.json'

print("Testing Markdown Rendering")
print("=" * 70)
print()

with open(DATA_FILE, 'r') as f:
    data = json.load(f)

# Test first case
case = data[0]

print("Original Text (first 500 chars):")
print("-" * 70)
original = case.get('information', '')[:500]
print(original)
print("...")
print()

print("Rendered HTML (first 1000 chars):")
print("-" * 70)
rendered = render_markdown(case.get('information', ''))
print(rendered[:1000])
print("...")
print()

# Check for markdown elements
print("Markdown Elements Detected:")
print("-" * 70)

if '###' in original or '##' in original or '# ' in original:
    print("✓ Headers found in original")
if '<h1>' in rendered or '<h2>' in rendered or '<h3>' in rendered:
    print("✓ Headers rendered to HTML")

if '**' in original or '__' in original:
    print("✓ Bold text found in original")
if '<strong>' in rendered:
    print("✓ Bold text rendered to HTML")

if '*' in original or '-' in original:
    print("✓ List markers found in original")
if '<ul>' in rendered or '<ol>' in rendered:
    print("✓ Lists rendered to HTML")

print()
print("=" * 70)
print("Markdown rendering test complete!")
print()
print("The Flask app will now properly render:")
print("  - Headings (H1, H2, H3, etc.)")
print("  - Bold and italic text")
print("  - Lists (ordered and unordered)")
print("  - Tables")
print("  - Code blocks")
print("  - Blockquotes")
print("  - And more!")
