import re

'''
letter -> a     eg: x y z -> a a a
digital -> 0    eg: 1 2 3 -> 0
sqrt(digital) -> 0      eg: sqrt(1 4/2 5) -> 0
digital/digital -> 0    eg: sqrt(1 4)/2 5 -> 0
digital*letter -> a     eg: 4 3 x y z -> a a a
letter/digital -> a     eg: 4 3 x y z / 5 3 -> a a a
'''

LABEL_PATH_RD = '../images/labels_all.txt'
LABEL_PATH_WT = '../images/labels_all_replace.txt'

debug = False

fw = open(LABEL_PATH_WT, 'w', encoding='utf-8')
with open(LABEL_PATH_RD, 'r', encoding='utf-8') as fp:
    cnt = 0
    for line in fp:
        line = line.strip('\n')
        uuid,latex = line.split('\t')
        latex = latex.strip()
        in_str = latex
        len_str = len(latex)
        out_str = ''

        # execute twice to solve simple nested formulas eg( sqrt(1/5) or sqrt(1)/5)
        for i in range(2):
            in_str = re.sub(r'\\sqrt \{ [\d ]+ \}|\\frac \{ [\d ]+ \} \{ [\d ]+ \}','0',in_str)  # sqrt(digital)|digital/digital ->0

        # 4 3 x y z / 5 3 -> a a a
        in_str = re.sub(r'[\d ]+ [a-zA-Z]', ' a', in_str)
        r = re.finditer(r'\\frac \{ ([a-zA-Z ]+)\} \{ [\d ]+ \}', in_str)
        if r:
            for ele in r:
                st, ed = ele.span(1)
                in_str = in_str.replace(ele.group(), 'a ' * ((ed - st) // 2))

        while in_str:
            r = re.match(r'\\\w*|\s+|\^ \{ \w+ \}|\{\w+\}|l o g|l n',in_str)
            if r:
                st,ed = r.span()
                out_str += in_str[st:ed]
                in_str = in_str[ed:]
            else:
                r = re.match(r'[\d ]+',in_str)
                if r:
                    st, ed = r.span()
                    out_str += '0 '
                    in_str = in_str[ed:]
                else:
                    r = re.match(r'[a-zA-Z]+', in_str)
                    if r:
                        st,ed = r.span()
                        out_str += ed*'a'
                        in_str = in_str[ed:]
                    else:
                        out_str += in_str[0]
                        in_str = in_str[1:]
        fw.write(uuid+'\t'+out_str+'\n')

        cnt = cnt + 1
        print(cnt)
        if debug == True:
            if cnt > 5:
                break

