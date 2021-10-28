try:
    import pegtree as pg
except ModuleNotFoundError:
    import os
    os.sytem('pip install pegtree')
    import pegtree as pg
import re

## 前処理

peg = pg.grammar('codemt.pegtree')
parser = pg.generate(peg, start='NLStatement')

def deeppy(text):
    ss = []
    for line in text.split('\n'):
        # A そこで、B はオプション形式
        matched = re.findall('(.*)(そこで、|そのとき、|ここで、|さらに)(.*)', text)
        if len(matched) > 0:
            statement = matched[0][0]
            options = matched[0][2].split('。')
            ss.append((statement, filter_options(options)))
            continue
        for statement in line.split('。'):
            if len(statement) > 0:
                options = []
                ss.append((statement, options))
    return ss
        
def filter_options(options):
    ss = []
    for option in options:
        if len(option) == 0:
            continue
        if option.endswith('ことにする'):
            option = option[:-5]
        ss.append('そこで、'+option)
    return ss

def fix(tree):
    a = [tree.epos_]
    for t in tree:
        a.append(fix(t).epos_)
    for key in tree.keys():
        a.append(fix(tree.get(key)).epos_)
    tree.epos_ = max(a)
    return tree

def preprocess(s):
    tree = parser(s)
    ss = []
    vars = {}
    index = 0
    for t in tree:
        tag = t.getTag()
        if tag == 'NLPChunk' or tag == 'UName':
            ss.append(str(t))
        else:
            key = ('ABCDEFGHIJKLMN')[index]
            key = f'<{key}>'
            vars[key] = str(fix(t))
            ss.append(key)
            index += 1
    return ''.join(ss), vars

def dummy(s):
    return [s], [100.0]

def make_codemt(nmt=dummy):
    def translate(s, **kw):
        ss = []
        for statement, options in deeppy(s):
            s, vars = preprocess(statement)
            cs, _ = nmt(s)
            s = cs[0]
            for key in vars:
                s = s.replace(key, vars[key])
            ss.append(s)
        return '\n'.join(ss)
    return translate


