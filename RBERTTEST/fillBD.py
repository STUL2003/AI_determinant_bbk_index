import psycopg2
import chardet
from psycopg2 import sql
import numpy as np
import pandas as pd

connection = psycopg2.connect(
    host="localhost",
    database="BBK_index",
    user="postgres",
    password="Dima2003",
    port=5432
)

cursor = connection.cursor()
data = []

with connection.cursor() as cursor:
    cursor.itersize = 1000  # сколько строк подгружать за раз
    #cursor.execute(r"SELECT * FROM index_bbk WHERE path::text ~ '^[0-9]+\.[0-9]$';;")
    #cursor.execute(r"SELECT * FROM index_bbk WHERE regexp_replace(path::text, '[^0-9]', '', 'g') ~ '^\d{4}$';")
    cursor.execute(rf"SELECT * FROM index_bbk WHERE path::text ~ '^{28.7}\d$'AND length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 4")
    i = 0
    for row in cursor.fetchall():
        print(row[0], row[1], row[2], print('\n\n'))
        #i+=1







