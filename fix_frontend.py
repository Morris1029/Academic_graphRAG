import sys

with open('frontend/script.js', 'r', encoding='utf-8') as f:
    text = f.read()

# Try \r\n
old_text_crlf = "function buildAnswerLead(result) {\r\n    return extractFirstSentence(result?.answer || '');\r\n}"
new_text_crlf = "function buildAnswerLead(result) {\r\n    return result?.answer || '';\r\n}"

# Try \n
old_text_lf = "function buildAnswerLead(result) {\n    return extractFirstSentence(result?.answer || '');\n}"
new_text_lf = "function buildAnswerLead(result) {\n    return result?.answer || '';\n}"

if old_text_crlf in text:
    text = text.replace(old_text_crlf, new_text_crlf)
    print("Replaced CRLF")
elif old_text_lf in text:
    text = text.replace(old_text_lf, new_text_lf)
    print("Replaced LF")
else:
    print("Could not find the target text in frontend/script.js")
    
# Also, maybe displayAnswer section title should be "综合回答" for zh.
text = text.replace("'直接答案' : 'Answer'", "'综合回答' : 'Comprehensive Answer'")

with open('frontend/script.js', 'w', encoding='utf-8') as f:
    f.write(text)
