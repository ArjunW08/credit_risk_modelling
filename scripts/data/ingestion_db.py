# Import dependencies
import pandas as pd
import os
from sqlalchemy import Engine, create_engine
import logging
import time

logging.basicConfig(
    filename="../../logs/ingestion_db.log",
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filemode="a",
)

engine = create_engine("sqlite:///credit_modelling.db")


def ingest_db(df: pd.DataFrame, table_name: str, engine: Engine) -> None:
    """This function injects the dataframe into the database table."""
    df.to_sql(table_name, con=engine, if_exists="replace", index=False)


def load_raw_data():
    """This funcstion will load CSV's as dataframe and ingest into db"""
    start_time = time.time()
    for file in os.listdir('../../data'):
        if file.endswith('.xlsx'):
            df = pd.read_excel(os.path.join('../../data', file))
            logging.info(f'Ingesting {file} into database')
            print(df.shape)
            ingest_db(df, file[:-5], engine)
    end_time = time.time()
    logging.info("-------------------- Ingestion Complete --------------------")

    logging.info(f"Total time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    load_raw_data()
