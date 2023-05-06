import os
import unit_split
import clause_division
import parse
from PyPDF2 import PdfReader

yields = parse.parse_clauses("003.pdf")
#full_doc = unit_split.get_pdf_pages("003.pdf")
#result_doc = unit_split.split_to_unit(full_doc)

for y in yields:
    print(y)

