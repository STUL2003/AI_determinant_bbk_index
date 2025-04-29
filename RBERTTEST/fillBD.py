import psycopg2

connection = psycopg2.connect(
    host="localhost",
    database="BBK_index",
    user="postgres",
    password="Dima2003",
    port=5432
)
