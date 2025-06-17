from google.cloud import bigquery

# Initialize the BigQuery client
key_path = "C:\Phoenix Portfolio\phoenix-portfolio-461608-e794abe0cf71.json"
client = bigquery.Client.from_service_account_json(key_path)

# Example query
query = """
    SELECT name, SUM(number) as total
    FROM `bigquery-public-data.usa_names.usa_1910_current`
    WHERE state = 'TX'
    GROUP BY name
    ORDER BY total DESC
    LIMIT 10
"""

# Run the query
query_job = client.query(query)

# Print the results
for row in query_job:
    print(f"{row.name}: {row.total}")
