#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gzip
import json
import os
import shutil
import time
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from sqlalchemy import create_engine
from tqdm import tqdm


def export_table_as_json(table_name, postgres_connection_string, output_fname):
    engine = create_engine(postgres_connection_string)
    connection = engine.connect()
    query = f'SELECT row_to_json(r) FROM {table_name} AS r'
    data = connection.execute(query).fetchall()
    data = [dict(x['row_to_json']) for x in data]

    if table_name == 'task':
        for x in data:
            x['data']['image'] = x['data'].pop('$undefined$')

    with gzip.open(output_fname, 'wb') as j:
        j.write(json.dumps(data).encode('utf-8'))

    connection.close()
    engine.dispose()
    return data


def _read_gzipped_json(fname):
    with gzip.open(fname, 'rb') as j:
        data = json.loads(j.read())
    return data


def jsonify_pg_tables(tables: Optional[list] = None, output_dir: str = 'tables'):
    if not os.getenv('POSTGRES_CONNECTION_STRING'):
        pg_user = os.environ['POSTGRES_USER']
        pg_pass = os.environ['POSTGRES_PASSWORD']
        pg_host = os.environ['POSTGRES_HOST']
        pg_port = os.environ['POSTGRES_PORT']
        pg_db = os.environ['POSTGRES_NAME']
        pg_conn_str = f'postgresql://{pg_user}:{pg_pass}@{pg_host}:{pg_port}/{pg_db}'  # noqa: E501
    else:
        pg_conn_str = os.environ['POSTGRES_CONNECTION_STRING']

    if not tables:
        tables = [
            'project', 'task', 'prediction', 'task_completion',
            'io_storages_s3importstoragelink', 'io_storages_s3importstorage'
        ]

    timestamp = time.strftime('%Y%m%d%H%M%S')
    Path(output_dir).mkdir(exist_ok=True)
    tables_data = {}

    for table in tqdm(tables):
        print(f'Exporting table: {table}...')
        output_fname = f'{output_dir}/{table}.{timestamp}.json.gz'
        table_data = export_table_as_json(
            table_name=table,
            postgres_connection_string=pg_conn_str,
            output_fname=output_fname)
        tables_data.update({table: table_data})
        shutil.copy2(output_fname, f'{output_dir}/{table}.latest.json.gz')
    return tables_data


if __name__ == '__main__':
    load_dotenv()
    jsonify_pg_tables()
    print('Done!')
