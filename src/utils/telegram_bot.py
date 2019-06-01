import telepot

"""
To create a new private channel, you will need 2 things:
1) A token bot: you can create by BotFather inside telegram. From the bot, you have to get the 'token'
    following the on-screen commands
2) A chat id: start the bot on telegram, send any message to the bot (important!) and then go to this address:
    https://api.telegram.org/bot<yourtoken>/getUpdates
    (replace <yourtoken> with the one got at the previous step)
    Copy the id from of the chat object, ex: ... , "chat": {"id": 123456789, "first_name": ...
"""
# stores chat_id and tokens
accounts = {
    'default': (-344193701, '863636934:AAHCVQziRW_MjQBmZXnP1ePXwS49P8Whov4'),
    # <insert your chat_id and token here>
}

# caches created bots per account
bots = {account: None for account in accounts.keys()}


def get_bot(account):
    """ Get or create a new bot and cache it in the dictionary.
        Return bot and chat_id
    """
    if account not in accounts:
        print('Invalid telegram bot account!')
        return None, None

    chat_id, token = accounts[account]
    if bots[account] is None:
        bots[account] = telepot.Bot(token)

    return bots[account], chat_id


def send_message(message, account='default'):
    bot, chat_id = get_bot(account)
    bot.sendMessage(chat_id=chat_id, text=message)

