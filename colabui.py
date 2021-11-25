import IPython
from google.colab import output
import random
import time


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

import os
import logging
from logging import getLogger, StreamHandler, FileHandler, Formatter, DEBUG, INFO

DIR_PATH = "/content/logging_test"


def make_dir():
    """ Colabのランタイムに（GoogleDriveではなく）ディレクトリを作成する
    """
    os.makedirs(DIR_PATH, exist_ok=True)


def make_logger():
    """ loggerオブジェクトの作成
    """
    global logger

    # loggerオブジェクトを生成する
    logger = getLogger(__name__)
    logger.setLevel(DEBUG)
    logger.propagate = False

    # ログをファイルに記録するためのHandlerを設定する
    fileHandler = FileHandler(f"{DIR_PATH}/test.log", encoding="utf-8")
    fileFormat = Formatter("%(asctime)s - %(levelname)-8s - %(message)s")
    fileHandler.setFormatter(fileFormat)
    fileHandler.setLevel(INFO)
    logger.addHandler(fileHandler)

make_dir()
make_logger()


def start_translator(translate=dummy, html=TRANSLATOR_HTML):
    def convert(text):
        try:
            logger.info(text)
            text = translate(text)
            logger.info(text)
            return IPython.display.JSON({'result': text})
        except Exception as e:
            print(e)
        return e

    output.register_callback('notebook.Convert', convert)
    display(IPython.display.HTML(html))


# IDE Demo


IDE_HTML = '''
<style>
.sample {
    background-color: #b1eeff;
    //padding: 10px;
    //font-size: 18px;
    -webkit-transition: all 0.3s ease;
    -moz-transition: all 0.3s ease;
    -o-transition: all 0.3s ease;
    transition: all  0.3s ease;
    }
.sample:hover {
    background-color: #ffc9d7;
    //padding: 25px;
    font-size: 18px;
    }
</style>
<textarea id="input" style="float: left; width: 48%; height:120px; font-size: large;">print("hello,world")</textarea>
<div style="margin: 10px; padding: 10px;">
<div id="ide" style="background-color: #100066;color: #ffffff"></div>
<div id="ide0" class="sample"></div>
<div id="ide1" class="sample"></div>
<div id="ide2" class="sample"></div>
<div id="ide3" class="sample"></div>
<div id="ide4" class="sample"></div>
</div>
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
            var ide = document.getElementById('ide');
            ide.innerHTML = data.text;
            for(var i = 0; i < 5; i++) {
              ide = document.getElementById(`ide${i}`);
              ide.innerHTML='';
            }
            if(data.text.length>0) {
              for(var i = 0; i < 5; i++) {
                ide = document.getElementById(`ide${i}`);
                ide.innerHTML='';
                if(data.result[i] !== undefined) {
                  ide.innerHTML = data.result[i];
                }
              }
            }
        })();
        timer = null;
    }, 400);
    });
</script>
'''


def beam_search(text):
    return [text, text, text, text, text], [100, 100, 100, 100, 100]


def trim_nlp(text):
    if len(text) == 0:
        return text
    if ord(text[0]) < 127:
        return trim_nlp(text[1:])
    if ord(text[-1]) < 127:
        return trim_nlp(text[:-1])
    return text


def start_ide(select=beam_search, head='そこで、', html=IDE_HTML):
    def convert(text):
        try:
            text = trim_nlp(text)
            pred, prob = select(head+text)
            for i in range(len(pred)):
                pred[i] = f'{pred[i]} ({prob[i]:.2f})'
            return IPython.display.JSON({
                'text': text,
                'result': pred,
            })
        except Exception as e:
            print(e)
        return e

    output.register_callback('notebook.Convert', convert)
    display(IPython.display.HTML(html))


# Chat


BOT_ICON = 'https://4.bp.blogspot.com/-7LcdiJjflkE/XASwYu6DyuI/AAAAAAABQZs/K0EQCKmvDmsVbEES7sAb6_xJhJyQXXLFgCLcBGAs/s800/bluebird_robot_bot.png'
YOUR_ICON = 'https://2.bp.blogspot.com/-WplygmIuX28/VZ-PPsDMOmI/AAAAAAAAvDU/OKG7taU7wXo/s800/girl_think.png'


def start_chatbot(chat=dummy, start='プログラミングで何かお困りかな？', **kw):

    def display_bot(bot_text):
        with output.redirect_to_element('#output'):
            bot_name = kw.get('bot_name', 'ボット')
            bot_icon = kw.get('bot_icon', BOT_ICON)
            display(IPython.display.HTML(f'''
      <div class="sb-box">
        <div class="icon-img icon-img-left">
            <img src="{bot_icon}" width="60px">
        </div><!-- /.icon-img icon-img-left -->
        <div class="icon-name icon-name-left">{bot_name}</div>
        <div class="sb-side sb-side-left">
            <div class="sb-txt sb-txt-left">
              {bot_text}
            </div><!-- /.sb-txt sb-txt-left -->
        </div><!-- /.sb-side sb-side-left -->
    </div><!-- /.sb-box -->
      '''))

    def display_you(your_text):
        with output.redirect_to_element('#output'):
            your_name = kw.get('your_name', 'あなた')
            your_icon = kw.get('your_icon', YOUR_ICON)

            display(IPython.display.HTML(f'''
      <div class="sb-box">
        <div class="icon-img icon-img-right">
            <img src="{your_icon}" width="60px">
        </div><!-- /.icon-img icon-img-right -->
        <div class="icon-name icon-name-right">{your_name}</div>
        <div class="sb-side sb-side-right">
            <div class="sb-txt sb-txt-right">
              {your_text}
            </div><!-- /.sb-txt sb-txt-right -->
        </div><!-- /.sb-side sb-side-right -->
      </div><!-- /.sb-box -->
      '''))

    display(IPython.display.HTML('''
      <style>
        /* 全体 */
        .sb-box {
            position: relative;
            overflow: hidden;
        }

        /* アイコン画像 */
        .icon-img {
            position: absolute;
            overflow: hidden;
            top: 0;
            width: 80px;
            height: 80px;
        }

        /* アイコン画像（左） */
        .icon-img-left {
            left: 0;
        }

        /* アイコン画像（右） */
        .icon-img-right {
            right: 0;
        }

        /* アイコン画像 */
        .icon-img img {
            border-radius: 50%;
            border: 2px solid #eee;
        }

        /* アイコンネーム */
        .icon-name {
            position: absolute;
            width: 80px;
            text-align: center;
            top: 83px;
            color: #fff;
            font-size: 10px;
        }

        /* アイコンネーム（左） */
        .icon-name-left {
            left: 0;
        }

        /* アイコンネーム（右） */
        .icon-name-right {
            right: 0;
        }

        /* 吹き出し */
        .sb-side {
            position: relative;
            float: left;
            margin: 0 105px 40px 105px;
        }

        .sb-side-right {
            float: right;
        }

        /* 吹き出し内のテキスト */
        .sb-txt {
            position: relative;
            border: 2px solid #eee;
            border-radius: 6px;
            background: #eee;
            color: #333;
            font-size: 15px;
            line-height: 1.7;
            padding: 18px;
        }

        .sb-txt>p:last-of-type {
            padding-bottom: 0;
            margin-bottom: 0;
        }

        /* 吹き出しの三角 */
        .sb-txt:before {
            content: "";
            position: absolute;
            border-style: solid;
            top: 16px;
            z-index: 3;
        }

        .sb-txt:after {
            content: "";
            position: absolute;
            border-style: solid;
            top: 15px;
            z-index: 2;
        }

        /* 吹き出しの三角（左） */
        .sb-txt-left:before {
            left: -7px;
            border-width: 7px 10px 7px 0;
            border-color: transparent #eee transparent transparent;
        }

        .sb-txt-left:after {
            left: -10px;
            border-width: 8px 10px 8px 0;
            border-color: transparent #eee transparent transparent;
        }

        /* 吹き出しの三角（右） */
        .sb-txt-right:before {
            right: -7px;
            border-width: 7px 0 7px 10px;
            border-color: transparent transparent transparent #eee;
        }

        .sb-txt-right:after {
            right: -10px;
            border-width: 8px 0 8px 10px;
            border-color: transparent transparent transparent #eee;
        }

        /* 767px（iPad）以下 */

        @media (max-width: 767px) {

            .icon-img {
                width: 60px;
                height: 60px;
            }

            /* アイコンネーム */
            .icon-name {
                width: 60px;
                top: 62px;
                font-size: 9px;
            }

            /* 吹き出し（左） */
            .sb-side-left {
                margin: 0 0 30px 78px;
                /* 吹き出し（左）の上下左右の余白を狭く */
            }

            /* 吹き出し（右） */
            .sb-side-right {
                margin: 0 78px 30px 0;
                /* 吹き出し（右）の上下左右の余白を狭く */
            }

            /* 吹き出し内のテキスト */
            .sb-txt {
                padding: 12px;
                /* 吹き出し内の上下左右の余白を-6px */
            }
        }
    </style>
      <script>
        var inputPane = document.getElementById('input');
        inputPane.addEventListener('keydown', (e) => {
          if(e.keyCode == 13) {
            google.colab.kernel.invokeFunction('notebook.Convert', [inputPane.value], {});
            inputPane.value=''
          }
        });
      </script>
    <div id='output' style='background: #66d;'></div>
    <div style='text-align: right'><textarea id='input' style='width: 100%; background: #eee;'></textarea></div>
      '''))

    def convert(your_text):
        try:
            display_you(your_text)
            bot_text = chat(your_text, **kw)
            time.sleep(random.randint(0, 4))
            display_bot(bot_text)
        except Exception as e:
            print(e)

    output.register_callback('notebook.Convert', convert)
    if start is not None:
        display_bot(start)