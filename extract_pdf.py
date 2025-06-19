import pdfplumber

pdf_path = "data/3.pdf"
output_path = "data/3.txt"

full_text = ""

with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        full_text += page.extract_text() + "\n"

# Clean empty lines
clean_text = "\n".join([line.strip() for line in full_text.splitlines() if line.strip()])

# Save to file
with open(output_path, "w", encoding="utf-8") as f:
    f.write(clean_text)

print("✅ PDF text saved to:", output_path)
