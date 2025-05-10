import psycopg2

connection = psycopg2.connect(
    host='localhost',  # Исправлено на =
    database='BBK_index',  # Исправлено на =
    user='postgres',  # Исправлено на =
    password='Dima2003',  # Исправлено на =
    port=5432  # Исправлено на =
)

data = []

with connection.cursor() as cursor:
    cursor.itersize = 1000
    # Убрана вложенность cursor.execute
    cursor.execute(r"""
        WITH RECURSIVE hierarchy AS (
            SELECT 
                path::text, 
                NULL::text AS parent, 
                0 AS level 
            FROM index_bbk 
            WHERE nlevel(path) = 1

            UNION ALL

            SELECT 
                i.path::text, 
                h.path AS parent, 
                h.level + 1 AS level 
            FROM index_bbk i
            JOIN hierarchy h 
                ON subpath(i.path, 0, -1)::text = h.path
        )
        SELECT path, parent, level FROM hierarchy
    """)

    # Удален лишний commit (не нужен для SELECT)

    i = 0
    for row in cursor.fetchall():
        print(row[0], row[1], row[2])
        i += 1
    print(i)

# Закрытие подключения
connection.close()