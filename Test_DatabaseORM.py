#Test of the database ORM.# . WORKS 
from DatabaseORM import session, StockPriceMinute
print(session.query(StockPriceMinute.sym).first())

# . WORKS 