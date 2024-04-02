# import logging
import os
from threading import Thread
from tokenize import untokenize

from aiogram import Bot, Dispatcher, executor, types
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request

from main import Main
from src.logger import logging
from src.utils.translate import translate_sentence

# Initialize logging
logging.basicConfig(level=logging.INFO)

app = Flask(__name__)

# Initialize the last_message_id with a starting value
last_message_id = 0

@app.route("/gen", methods=["POST"])
def generate_code():
    if request.method == "POST":
        data = request.get_json()
        src = data.get('src')

        if not src:
            logging.error("No source description provided")
            return jsonify(result="Error: No source description provided"), 400

        logging.info(f"Source description: {src}")
        
        try:
            main_obj = Main(device="cpu")
            model = main_obj.return_model()
            Input, Output = main_obj.setup_fields()
            src_tokens = src.split(" ")
            SRC = Input
            TRG = Output
            translation, attention = translate_sentence(src_tokens, SRC, TRG, model, device="cpu")
            code_str = untokenize(translation[:-1]).decode('utf-8')
            logging.info(f"Generated code: {code_str}")
            return jsonify(result=code_str)
        except Exception as e:
            logging.error(f"Error during code generation: {e}")
            return jsonify(result=f"Error during code generation: {e}"), 500

@app.route("/", methods=["GET"])
def index():
    return render_template('index.html')

def run_flask_app():
    app.run(debug=True, host='0.0.0.0', port=5000, use_reloader=False)

# Load environment variables
load_dotenv()
TOKEN = os.getenv("TOKEN")

bot = Bot(token=TOKEN)
dp = Dispatcher(bot)

async def welcome(message: types.Message):
    await message.reply("Hi\nI am Dragon!\nCreated by Vikalp and Mohd Arsalan.\nHow can I assist you with coding?")

async def clear(message: types.Message):
    # Implement logic to clear past context if necessary
    pass

async def helper(message: types.Message):
    help_command = """
    Hi, I'm Dragon bot! Please follow these commands:\n
    /start - To start the conversation.
    /clear - To clear the past conversation and context.
    /help - To get this help menu.
    
    I hope this helps. :)
    """
    await message.reply(help_command)

async def model(message: types.Message):
    global last_message_id
    current_message_id = message.message_id
    
    # Ensure processing the most recent message
    if current_message_id <= last_message_id:
        return
    last_message_id = current_message_id

    try:
        main_obj = Main(device="cpu")
        model = main_obj.return_model()
        Input, Output = main_obj.setup_fields()
        logging.info(f"From Telegram, processing message ID {current_message_id}: {message.text}")
        src_tokens = message.text.split(" ")
        SRC = Input
        TRG = Output
        translation, attention = translate_sentence(src_tokens, SRC, TRG, model, device="cpu")
        code_str = untokenize(translation[:-1]).decode('utf-8')
        await bot.send_message(chat_id=message.chat.id, text="<pre>" + code_str + "</pre>", parse_mode=types.ParseMode.HTML)
    except Exception as e:
        logging.error("Error generating code: ", e)
        await message.reply("Error generating the code .... ")

if __name__ == "__main__":
    dp.register_message_handler(welcome, commands=['start'])
    dp.register_message_handler(clear, commands=['clear'])
    dp.register_message_handler(helper, commands=['help'])
    dp.register_message_handler(model, lambda message: not message.is_command())
    
    flask_thread = Thread(target=run_flask_app)
    flask_thread.start()
    
    executor.start_polling(dp, skip_updates=True)
