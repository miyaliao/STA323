import sqlite3
p=r'C:\Users\86181\.openclaw\workspace\Assignment2\Q3\Q3_data\debit_card_specializing\debit_card_specializing.sqlite'
conn=sqlite3.connect(p)
c=conn.cursor()
print('tables:')
for r in c.execute("select name from sqlite_master where type='table' order by name"):
    print(r[0])

for t in ['transactions','yearmonth','customers','products','gasstations']:
    print(f'\n{t} cols:')
    for r in c.execute(f'pragma table_info({t})'):
        print(r)
