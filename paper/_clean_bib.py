# -*- coding: utf-8 -*-
import re
with open(r'C:\Users\25187\Desktop\×ÔÊÊÓ¦±´Ò¶Ë¹¿ò¼Ü\paper\references.bib', 'r') as f:
    content = f.read()

remove_keys = ['JRSSB2023a','JRSSB2024a','JRSSB2022a','JASA2021a','JRSSB2021a','JRSSB2020a','JRSSB2019a','JRSSB2018a','JASA2022a','JASA2020a','JASA2019a','NeurIPS2022a','ICML2023a']
for key in remove_keys:
    pattern = re.compile(r'@\w+\{' + re.escape(key) + r'[^}]*\n\}', re.DOTALL)
    content = pattern.sub('', content)
content = re.sub(r'\n{3,}', '\n\n', content)

existing = set(re.findall(r'@\w+\{(\w+)', content))

additions = []
additions.append('\n@inproceedings{Guo2017,\n  author    = {Guo, C. and Pleiss, G. and Sun, Y. and Weinberger, K. Q.},\n  title     = {On Calibration of Modern Neural Networks},\n  booktitle = {Proceedings of the 34th International Conference on Machine Learning (ICML)},\n  year      = {2017},\n  pages     = {1321--1330}\n}')
additions.append('\n@inproceedings{Kumar2019,\n  author    = {Kumar, A. and Sarawagi, S. and Jain, U.},\n  title     = {Trainable Calibration Measures for Neural Networks from Kernel Mean Embeddings},\n  booktitle = {Proceedings of the 35th International Conference on Machine Learning (ICML)},\n  year      = {2019},\n  pages     = {2805--2814}\n}')
additions.append('\n@article{Gamerman1998,\n  author  = {Gamerman, D.},\n  title   = {Markov Chain Monte Carlo for Dynamic Generalized Linear Models},\n  journal = {Journal of the Royal Statistical Society: Series B},\n  year    = {1998},\n  volume  = {60},\n  number  = {2},\n  pages   = {329--356}\n}')
additions.append('\n@article{Johnson2016,\n  author  = {Johnson, A. E. W. and Pollard, T. J. and Shen, L. and Lehman, L. H. and Feng, M. and Ghassemi, M. and Moody, B. and Szolovits, P. and Celi, L. A. and Mark, R. G.},\n  title   = {MIMIC-III, a freely accessible critical care database},\n  journal = {Scientific Data},\n  year    = {2016},\n  volume  = {3},\n  pages   = {160035}\n}')
additions.append('\n@book{Anderson1979,\n  author    = {Anderson, B. D. O. and Moore, J. B.},\n  title     = {Optimal Filtering},\n  publisher = {Prentice-Hall},\n  year      = {1979}\n}')

for add in additions:
    m = re.search(r'@\w+\{(\w+)', add)
    if m and m.group(1) not in existing:
        content += add
        existing.add(m.group(1))

with open(r'C:\Users\25187\Desktop\×ÔÊÊÓ¦±´Ò¶Ë¹¿ò¼Ü\paper\references.bib', 'w') as f:
    f.write(content)

keys = ['Guo2017','Kumar2019','Gamerman1998','Johnson2016','Anderson1979','West1985','Dunson2010','Platt1999','Tibshirani1996','Elkan2001','Berger1985','Gneiting2007','Julier1997','Green1995','IEEECIS2019','ZouHastie2005']
for k in keys:
    print(f'OK: {k}' if k in content else f'MISSING: {k}')
for k in ['JRSSB2023a','JRSSB2024a','NeurIPS2022a','ICML2023a']:
    print(f'REMOVED: {k}' if k not in content else f'STILL PRESENT: {k}')
