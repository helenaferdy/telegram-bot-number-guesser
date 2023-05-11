import telegram.ext as te
from telegram import Bot
from cnn import cnn_predict, CNN

TOKEN = '5823169125:AAEvsKD5VgW0hJgjfO6_qr5xRUhD1aQDhfs'
IMAGE_PATH = "static/image.png"

updater = te.Updater(TOKEN, use_context=True)
bot = Bot(token=TOKEN)
dispatcher = updater.dispatcher
chat_id = ""

def start(update, context):
    chat_id = update.message.chat_id
    context.bot.send_message(chat_id=chat_id, text="Hello! Your chat ID is: {}".format(chat_id))

def handle_message(update, context):
    update.message.reply_text(f"Send a picture containing number")
    
def handle_image(update, context):
    message = update.message
    chat_id = message.chat_id
    photo = message.photo[-1]  # Get the largest version of the photo
    file_id = photo.file_id
    
    # Get the file object using the file_id
    file_obj = context.bot.get_file(file_id)
    file_obj.download(IMAGE_PATH)
    
    # Pass to ML model
    x = cnn_predict(IMAGE_PATH)
    
    # Reply to the user
    message.reply_text(f'Prediction : {x[1]}\nConfidence : {x[0]}')
    with open(x[2], 'rb') as image_file:
        bot.send_photo(chat_id=chat_id, photo=image_file)



dispatcher.add_handler(te.CommandHandler("start", start))
dispatcher.add_handler(te.MessageHandler(te.Filters.text, handle_message))
dispatcher.add_handler(te.MessageHandler(te.Filters.photo, handle_image))
updater.start_polling()
updater.idle()