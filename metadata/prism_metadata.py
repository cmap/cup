import requests


def make_request_url_filter(endpoint_url, where=None, fields=None):
    clauses = []
    if where:
        where_clause = '"where":{'
        wheres = []
        for k, v in where.items():
            wheres.append('"{k}":"{v}"'.format(k=k, v=v))
        where_clause += ','.join(wheres) + '}'
        clauses.append(where_clause)

    if fields:
        fields_clause = '"fields":{'
        fields_list = []
        if type(fields) == dict:
            for k, v in fields.items():
                fields_list.append('"{k}":"{v}"'.format(k=k, v=v))
        elif type(fields) == list:
            for field in fields:
                fields_list.append('"{k}":"{v}"'.format(k=field, v="true"))
        fields_clause += ','.join(fields_list) + '}'
        clauses.append(fields_clause)

    if len(clauses) > 0:
        # print(endpoint_url.rstrip("/") + '?filter={' +  ','.join(clauses) + '}')
        return endpoint_url.rstrip("/") + '?filter={' + requests.utils.quote(','.join(clauses)) + '}'
    else:
        return endpoint_url


def get_data_from_db(endpoint_url, user_key, where=None, fields=None):
    request_url = make_request_url_filter(endpoint_url, where=where, fields=fields)
    response = requests.get(request_url, headers={'user_key': user_key})
    if response.ok:
        return response.json()
    else:
        response.raise_for_status()
