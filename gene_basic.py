# from __future__ import division
# import re
#
# with open('sequence.fasta') as file:
#     for line in file:
#         print(line)
#
# def get_fasta(fasta_path):
#     fasta = {}
#     with open(fasta_path) as file:
#         sequence = ""
#         for line in file:
#             if line.startswith(">"):
#                 # 去除描述字段行中的\n和>
#                 name = line[1:].rstrip()
#                 fasta[name] = ''
#                 continue
#             # 去除序列字段行中的\n，并将所有字符规范为大写字符
#             fasta[name] += line.rstrip().upper()
#     return fasta
#
#
# # 核苷酸计数
# def nt_count(seq):
#     ntCounts = []
#     for nt in ['A', 'C', 'G', 'T']:
#         ntCounts.append(seq.count(nt))
#     return ntCounts
#
#
# # CG 含量
# def cg_content(seq):
#     total = len(seq)
#     gcCount = seq.count('G') + seq.count('C')
#     gcContent = format(float(gcCount / total * 100), '.6f')
#     return gcContent
#
#
# # DNA 翻译为 RNA
# def dna_trans_rna(seq):
#     rnaSeq = re.sub('T', 'U', seq)
#     # method2: rnaSeq = dnaSeq.replace('T', 'U')
#     return rnaSeq
#
#
# # RNA翻译为蛋白质
# def rna_trans_protein(rnaSeq):
#     codonTable = {
#         'AUA':'I', 'AUC':'I', 'AUU':'I', 'AUG':'M',
#         'ACA':'T', 'ACC':'T', 'ACG':'T', 'ACU':'T',
#         'AAC':'N', 'AAU':'N', 'AAA':'K', 'AAG':'K',
#         'AGC':'S', 'AGU':'S', 'AGA':'R', 'AGG':'R',
#         'CUA':'L', 'CUC':'L', 'CUG':'L', 'CUU':'L',
#         'CCA':'P', 'CCC':'P', 'CCG':'P', 'CCU':'P',
#         'CAC':'H', 'CAU':'H', 'CAA':'Q', 'CAG':'Q',
#         'CGA':'R', 'CGC':'R', 'CGG':'R', 'CGU':'R',
#         'GUA':'V', 'GUC':'V', 'GUG':'V', 'GUU':'V',
#         'GCA':'A', 'GCC':'A', 'GCG':'A', 'GCU':'A',
#         'GAC':'D', 'GAU':'D', 'GAA':'E', 'GAG':'E',
#         'GGA':'G', 'GGC':'G', 'GGG':'G', 'GGU':'G',
#         'UCA':'S', 'UCC':'S', 'UCG':'S', 'UCU':'S',
#         'UUC':'F', 'UUU':'F', 'UUA':'L', 'UUG':'L',
#         'UAC':'Y', 'UAU':'Y', 'UAA':'', 'UAG':'',
#         'UGC':'C', 'UGU':'C', 'UGA':'', 'UGG':'W',
#     }
#     proteinSeq = ""
#     for codonStart in range(0, len(rnaSeq), 3):
#         codon = rnaSeq[codonStart:codonStart + 3]
#         if codon in codonTable:
#             proteinSeq += codonTable[codon]
#     return proteinSeq
#
#
# # 获取反向序列
# def reverse_comple(type, seq):
#     seq = seq[::-1]
#     dnaTable = {
#         "A": "T", "T": "A", "C": "G", "G": "C"
#     }
#     rnaTable = {
#         "A": "T", "U": "A", "C": "G", "G": "C"
#     }
#     res = ""
#     if type == "dna":
#         for ele in seq:
#             if ele in seq:
#                 if type == "dna":
#                     res += dnaTable[ele]
#                 else:
#                     res += rnaTable[ele]
#     return res
#
#
# if __name__ == '__main__':
#     oct4 = get_fasta('sequence.fasta')
#     for name, sequence in oct4.items():
#         print ("name: ", name)
#         print ("sequence: ", sequence)
#         print ("nt_count: ", nt_count(sequence))
#         print ("cg_content: ", cg_content(sequence))
#         rna = dna_trans_rna(sequence)
#         print ("rna: ", rna)
#         protein = rna_trans_protein(rna)
#         print ("protein: ", protein)
#         print ("reverse_comple: ", reverse_comple("dna", sequence))


# 以下为Biopython的简单使用
from Bio.Seq import Seq

# 新建一个DNA序列对象
dna_seq = Seq("GGATGGTTGTCTATTAACTTGTTCAAAAAAGTATCAGGAGTTGTCAAGGCAGAGAAGAGAGTGTTTGCA")
# 序列信息
print("Sequence: ", dna_seq)
# 序列长度
print("Length : ", len(dna_seq))
# 单个核苷酸计数
print("G Counts: ", dna_seq.count("G"))
# 获取反向序列
print("reverse: ", dna_seq[::-1])
# 获取反向互补序列
print("Reverse complement: ", dna_seq.reverse_complement())

# =====转录=====
# 如果序列为编码链，那么直接转换
# transcribe()函数的作用仅仅是将序列中的 T 替换为 U
print("rna: ", dna_seq.transcribe())
# 如果序列为模板链，就需要先转为编码链
transcribe_seq = dna_seq.complement().transcribe()
print("rna: ", transcribe_seq)


# =====翻译=====
print("protein: ", transcribe_seq.translate())
# 如果翻译的是线粒体密码子，那么在参数中需要输入，其他参考 https://www.ncbi.nlm.nih.gov/Taxonomy/Utils/wprintgc.cgi?mode=c
print("protein: ", transcribe_seq.translate(table="Vertebrate Mitochondrial"))
# 在现实生物世界中，一般在遇到终止密码子之后的序列不用翻译
print("protein: ", transcribe_seq.translate(table="Vertebrate Mitochondrial", to_stop=True))
# 如果DNA序列为编码序列，可以直接翻译，DNA序列不是3的倍数时，报错
print("protein: ", dna_seq.translate())
# 在细菌世界中，在细菌遗传密码中 GTG 是个有效的起始密码子，注意第一个密码子（正常情况下 GTG编码缬氨酸， 但是如果作为起始密码子，则翻译成甲硫氨酸）
bacterial_dna = Seq("GTGAAAAAGATGCAATCTATCGTACTCGCACTTTCCCTGGTTCTGGTCGCTCCCATGGCATAA")
print("protein: ", bacterial_dna.translate(table="Bacterial", to_stop=True))
print("protein: ", bacterial_dna.translate(table="Bacterial", cds=True))
