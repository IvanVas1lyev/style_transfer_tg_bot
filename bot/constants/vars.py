import os


styles_dict = {
    'Vincent van Gogh': 1,
    'Cubism': 2,
    'Picasso2': 3,
    'Mountains': 4,
    'Picasso': 5,
    'Mucha': 6
}

available_sizes = [
    '128',
    '256',
    '512'
]

styles_folder = 'pictures/styles'

TG_TOKEN = os.getenv("TG_BOT_TOKEN")

DEFAULT_SIZE = 256
