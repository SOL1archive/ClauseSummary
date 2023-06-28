import os
import unit_split as us
import clause_split as cs
import parse

# yields = parse.parse_clauses("002.pdf")
full_doc = us.get_pdf_pages("./source/002.pdf")
result_doc = us.split_to_unit(full_doc)
clauses_doc = cs.articles2clauses(result_doc)

### Check result
print(clauses_doc[0][0]) 


#for y in yields:
#    print(y)

