from .t5 import generate_nmt
from .nlpcode import compose_nmt
import IPython
from google.colab import output

# ダミー関数


def dummy(text: str, **kw):
    # nmt = PyNMT(model, src_vocab, tgt_vocab)
    # translate(nmt, 'もしa+1が偶数ならば')
    return 'ほ'


TRANSLATOR_HTML = '''
<textarea id="input" style="float: left; width: 48%; height:100px; font-size: large;"></textarea>
<textarea id="output" style="width: 48%; height:100px; font-size: large;"></textarea>
<script>
    var timer = null;
    document.getElementById('input').addEventListener('input', (e) => {
    var text = e.srcElement.value;
    if(timer !== null) {
        console.log('clear');
        clearTimeout(timer);
    }
    timer = setTimeout(() => {
        (async function() {
            const result = await google.colab.kernel.invokeFunction('notebook.Convert', [text], {});
            const data = result.data['application/json'];
            const textarea = document.getElementById('output');
            textarea.textContent = data.result;
        })();
        timer = null;
    }, 400);
    });
</script>
'''


def run_corgi(nmt, delay=400):
    def convert(text):
        try:
            text = nmt(text, beams=1)
            return IPython.display.JSON({'result': text})
        except Exception as e:
            print(e)
        return e

    output.register_callback('notebook.Convert', convert)
    HTML = TRANSLATOR_HTML.replace('400', str(delay))
    display(IPython.display.HTML(HTML))


def start_corgi(model_id='1qZmBK0wHO3OZblH8nabuWrrPXU6JInDc', delay=600):
    nmt = compose_nmt(generate_nmt(model_file=model_id))
    run_corgi(nmt, delay)
