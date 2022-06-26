#!/usr/bin/env python
# coding: utf-8


def _update_query(table: str, field: str, is_jsonb: bool, from_string: str,
                  to_string: str) -> str:
    """Generates a single query to update a field in a PostgresSQL table.

    Args:
        table (str): The table to update.
        field (str): The field to update.
        is_jsonb (bool): Whether the field is jsonb or not.
        from_string (str): The string to replace.
        to_string (str): The string to replace with.

    Returns:
        str: The query to update the field.

    """
    prefix = f'UPDATE {table} SET "{field}" = replace({field}'
    if is_jsonb:
        query = f'{prefix}::TEXT, \'{from_string}\', \'{to_string}\')::jsonb;'
    else:
        query = f'{prefix}, \'{from_string}\', \'{to_string}\');'
    return query


def generate_queries(from_string: str, to_string: str) -> None:
    """Generate queries to update the database for replacing old label names.

    Args:
        from_string (str): The string to be replaced.
        to_string (str): The string to replace with.

    Returns:
        None

    """
    data = {
        "project": [{
            "field": "label_config",
            "is_jsonb": False
        }, {
            "field": "control_weights",
            "is_jsonb": True
        }, {
            "field": "parsed_label_config",
            "is_jsonb": True
        }],
        "prediction": [{
            "field": "result",
            "is_jsonb": True
        }],
        "projects_projectsummary": [{
            "field": "created_labels",
            "is_jsonb": True
        }],
        "task_completion": [{
            "field": "result",
            "is_jsonb": True
        }, {
            "field": "prediction",
            "is_jsonb": True
        }],
        "tasks_annotationdraft": [{
            "field": "result",
            "is_jsonb": True
        }]
    }

    for k, v in data.items():
        for x in v:
            q = _update_query(table=k,
                              field=x['field'],
                              is_jsonb=x['is_jsonb'],
                              from_string=from_string,
                              to_string=to_string)
            print(q + '\n')

    print('--', '-' * 77)

    for k, v in data.items():
        for x in v:
            check_exist = f'SELECT * FROM {k} WHERE {x["field"]}::TEXT ' \
                          f'LIKE \'%{from_string}%\';'
            print(check_exist + '\n')
