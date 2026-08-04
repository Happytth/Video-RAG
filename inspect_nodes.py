import os
import json
from neo4j import GraphDatabase
from dotenv import load_dotenv
load_dotenv()

uri = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
user = os.environ.get("NEO4J_USERNAME", "neo4j")
pwd = os.environ.get("NEO4J_PASSWORD", "password")
db = os.environ.get("NEO4J_DATABASE")

driver = GraphDatabase.driver(uri, auth=(user, pwd))
with driver.session(database=db) as session:
    res = session.run("MATCH (n:Entity) RETURN n.id, n.node_type, n.description, n.timestamp, n.objects ORDER BY n.time_seconds ASC")
    nodes = [r.data() for r in res]

print(f"Total nodes in Neo4j: {len(nodes)}")
for n in nodes:
    print(f"[{n['n.timestamp']}] ({n['n.node_type']}) {n['n.description']} | Objects: {n['n.objects']}")

driver.close()
