"""
This file will create training data from scratch for us to use
"""
import logging
import pandas as pd
import random
import string

logging.basicConfig(filename='create_data.log', encoding='utf-8',
                    format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.DEBUG, filemode='w')


def create_list_of_users(num_users=1000):
    """
    Function to create a list of user IDs, allowing a max of 10,000 to keep any data reasonable for now.
    :param num_users: number of unique user IDs to create
    :return: A set of user IDs
    """
    list_of_users = set()
    if num_users > 10000:
        logging.info("Current maximum user limit exceeded, lowering to 10,000 unique users.")
        num_users = 10000

    for i in range(0, num_users):
        unique_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))
        while unique_id in list_of_users:
            unique_id = ''.join(random.choice(string.ascii_lowercase + string.digits) for _ in range(8))

        list_of_users.add(unique_id)

    return list_of_users


def create_list_of_items(num_items=100):
    """
    Functio to create a list of item IDs, allowing for a max of 2000.
    :param num_items: Number of unique item IDs to create
    :return: A set of item IDs
    """
    list_of_items = set()
    if num_items > 2000:
        logging.info("Current maximum item limit exceeded, lowering to 2000 unique items.")
        num_users = 2000

    for i in range(0, num_items):
        unique_id = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))
        while unique_id in list_of_items:
            unique_id = ''.join(random.choice(string.ascii_lowercase) for _ in range(3))

        list_of_items.add(unique_id)

    return list_of_items


def create_transaction_data(list_of_users, list_of_items, num_transactions=0):
    """
    Function to create and save dataframe of our made up transactions between users and items
    :param list_of_users: list of unique user IDs
    :param list_of_items: list of unique item IDs
    :param num_transactions: number of total transactions of users interacting with items
    """
    list_of_users = list(list_of_users)
    list_of_items = list(list_of_items)

    transactions_user = []
    transactions_item = []

    for _ in range(0, num_transactions):
        user = random.choice(list_of_users)
        item = random.choice(list_of_items)
        transactions_user.append(user)
        transactions_item.append(item)

    df = pd.DataFrame(list(zip(transactions_user, transactions_item)), columns=['USER_ID', 'ITEM'])
    df.to_csv(r"..\data\training_data.csv", index=False)
    logging.info("Training data saved!")


if __name__ == "__main__":
    logging.info("Running create_data.py...")
    users = create_list_of_users()
    items = create_list_of_items()
    create_transaction_data(users, items, 1000)


