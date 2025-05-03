import psycopg2
connection = psycopg2.connect(
    host="localhost",
    database="BBK_index",
    user="postgres",
    password="Dima2003",
    port=5432
)
'''
with connection.cursor() as cursor:
    cursor.execute(rf"""SELECT path, title FROM index_bbk WHERE length(regexp_replace(path::text, '[^0-9]', '', 'g')) = 5;""")
    i=0
    for row in cursor.fetchall():
        print(row[0], row[1])
        i+=1
    print(i)

    for key, value in keywords.items():
        print(key.split()[0])
        for v in value:
            cursor.execute(
                "INSERT INTO keywords_bbk (path, value) VALUES (%s, %s)",
                (key.split()[0], v)
            )
            connection.commit()
'''