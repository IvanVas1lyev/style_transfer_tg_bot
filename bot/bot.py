import logging
import os
from typing import Any, Callable
from aiogram import Bot, Dispatcher, types, F
from aiogram.filters.command import Command
from aiogram.types import FSInputFile
from aiogram.types import InputMediaPhoto

import bot.constants.msgs as msgs
import bot.constants.vars as vars
from model.model import run_model

logging.basicConfig(level=logging.INFO)
bot = Bot(token=vars.TG_TOKEN)
dp = Dispatcher()

# это по-хорошему сделать бд, а не глобальными переменными
images = {}
training_last_msg_id = {}
is_user_start_model = {}
size = vars.DEFAULT_SIZE


def check_is_user_start_model(func: Callable[..., Any]) -> Callable[..., Any]:
    """
    Decorator that checks if the user has started the model.

    Args:
        func: The function to decorate.

    Returns:
        The wrapper function.
    """

    async def wrapper(*args: Any) -> Any:
        """
        Wrapper function that checks if the user has started the model.

        Args:
            *args: Positional arguments.

        Returns:
            The result of the decorated function.
        """
        msg, = args
        chat_id = msg.chat.id
        global is_user_start_model

        if chat_id in is_user_start_model and is_user_start_model[chat_id]:
            await bot.send_message(chat_id, msgs.ERROR_MSG)
            return

        return await func(*args)

    return wrapper


@dp.message(Command('start'))
@check_is_user_start_model
async def process_start_command(message: types.Message) -> None:
    """
    Processes the '/start' command.

    Args:
        message (types.Message): The incoming message.
    """
    await message.reply(msgs.START_MSG)


@dp.message(Command('help'))
@check_is_user_start_model
async def process_help_command(message: types.Message) -> None:
    """
    Processes the '/help' command.

    Args:
        message (types.Message): The incoming message.
    """
    await message.reply(msgs.HELP_MSG)


@dp.message(Command('transfer_style'))
@check_is_user_start_model
async def process_transfer_style_command(message: types.Message) -> None:
    """
    Processes the '/transfer_style' command.

    Args:
        message (types.Message): The incoming message.
    """
    await bot.send_message(message.from_user.id, msgs.PENDING_STYLE_MSG)


@dp.message(Command('show_styles'))
@check_is_user_start_model
async def process_styles_command(message: types.Message) -> None:
    """
    Handles the '/styles' command to display a media group of styles.

    Args:
        message (types.Message): The message object.
    """
    media_group = []

    for file in os.listdir(vars.styles_folder):
        if file.endswith('.png'):
            picture = FSInputFile(os.path.join(vars.styles_folder, file))
            picture_name = file[:-4]

            media_group.append(InputMediaPhoto(
                type='photo',
                media=picture,
                caption=f'№{vars.styles_dict[picture_name]}, '
                        f'{picture_name}')
            )

    await message.answer_media_group(media_group)
    await bot.send_message(message.from_user.id, msgs.STYLES_MSG)


@dp.message(F.photo)
@check_is_user_start_model
async def photo_handler(message: types.Message) -> None:
    """
    Handle photo messages.

    Args:
        message (types.Message): The incoming message.
    """
    if message.chat.id not in images:
        images[message.chat.id] = {}
        images[message.chat.id]['style'] = message.photo[-1]

        await bot.send_message(message.from_user.id, msgs.PENDING_ORIGINAL_MSG)

        return
    else:
        images[message.chat.id]['user_picture'] = message.photo[-1]

        file = await bot.get_file(images[message.chat.id]['style'].file_id)
        await bot.download_file(
            file.file_path,
            f'pictures/user_pictures/{images[message.chat.id]["style"].file_id}.jpeg'

        )
        file = await bot.get_file(images[message.chat.id]['user_picture'].file_id)
        await bot.download_file(
            file.file_path,
            f'pictures/user_pictures/{images[message.chat.id]["user_picture"].file_id}.jpeg'
        )

        output_file_path = f'pictures/results/{message.photo[-1].file_id}.jpg'

        global training_last_msg_id
        global size
        training_last_msg_id[message.from_user.id] = None

        async def func(info):
            if training_last_msg_id[message.from_user.id] is None:
                msg_id = await chat_log(message.from_user.id, info)
            else:
                await bot.delete_message(message.from_user.id, training_last_msg_id[message.from_user.id])
                msg_id = await chat_log(message.from_user.id, info)

            training_last_msg_id[message.from_user.id] = msg_id

        log_func = func
        is_user_start_model[message.from_user.id] = True

        await run_model(
            f'pictures/user_pictures/{images[message.chat.id]["style"].file_id}.jpeg',
            f'pictures/user_pictures/{images[message.chat.id]["user_picture"].file_id}.jpeg',
            output_file_path,
            log_func,
            size
        )
        await send_photo(message.from_user.id, output_file_path)
        is_user_start_model[message.from_user.id] = False
        del images[message.chat.id]
        size = vars.DEFAULT_SIZE


async def chat_log(chat_id: int, message: str) -> int:
    """
    Logs a message in the chat.

    Args:
        chat_id (int): ID of the chat.
        message (str): The message to log.
    """
    msg = await bot.send_message(chat_id, message)

    return msg.message_id


@dp.message()
@check_is_user_start_model
async def message_handler(message: types.Message) -> None:
    """
    Handles incoming messages by sending a predefined message.

    Args:
        message (types.Message): The incoming message.
    """
    if message.text in vars.available_sizes:
        global size
        size = int(message.text)

        await bot.send_message(message.from_user.id, msgs.SUCCESS_CHANGE_SIZE_MSG)
    else:
        await bot.send_message(message.from_user.id, msgs.ONLY_COMMANDS_MSG)


async def send_photo(chat_id: int, file_path: str) -> None:
    """
    Sends a photo to the chat.

    Args:
        chat_id (int): ID of the chat.
        file_path (str): Path to the photo file.
    """
    picture = FSInputFile(file_path)
    await bot.send_photo(chat_id, picture)


async def run_bot() -> None:
    """
    Runs the Telegram bot.

    This function starts the polling of the bot to receive and handle incoming messages.
    """
    await dp.start_polling(bot)
