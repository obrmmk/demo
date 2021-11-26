from .t5 import generate_nmt
from .nlpcode import compose_nmt
import IPython
from google.colab import output

# ダミー関数

TRANSLATOR_HTML = '''
<textarea id="input" style="float: left; width: 48%; height:100px; font-size: large;"></textarea>
<textarea id="output" style="width: 48%; height:100px; font-size: large;"></textarea>
<script>
    var timer = null;
    document.getElementById('input').addEventListener('input', (e) => {
        var text = e.srcElement.value;
        if(timer !== null) {
            clearTimeout(timer);
        }
        timer = setTimeout(() => {
            timer = null;
            (async function() {
                const result = await google.colab.kernel.invokeFunction('notebook.Convert', [text], {});
                const data = result.data['application/json'];
                const textarea = document.getElementById('output');
                textarea.textContent = data.result;
            })();
        }, 600);
    });
</script>
'''

def print_nop(*x):
    pass

def run_corgi(nmt, delay=600, print=print_nop):
    cached = {'':''}
    def convert(text):
        try:
            ss = []
            for line in text.split('\n'):
                if line not in cached:
                    translated = nmt(line, beams=1)
                    print(line, '=>', translated)
                    cached[line] = translated
                else:
                    translated = cached[line]
                ss.append(translated)
            text = '\n'.join(ss)
            return IPython.display.JSON({'result': text})
        except Exception as e:
            print(e)
        return e

    output.register_callback('notebook.Convert', convert)
    HTML = TRANSLATOR_HTML.replace('600', str(delay))
    display(IPython.display.HTML(HTML))


def start_corgi(model_id='1qZmBK0wHO3OZblH8nabuWrrPXU6JInDc', delay=600, print=print_nop):
    nmt = compose_nmt(generate_nmt(model_id=model_id))
    run_corgi(nmt, delay=delay, print=print)
